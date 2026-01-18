#!/usr/bin/env python3
"""
Coinbase Advanced API Historical Candle Backfill Script

Fetches historical 5-minute candle data from Coinbase Advanced Trade API
and populates the coinbase-data JSON files.

This gives you MUCH better granularity than CoinGecko (5-min vs hourly).

Usage:
    python backfill_coinbase_candles.py              # Backfill all enabled assets
    python backfill_coinbase_candles.py BTC-USD      # Backfill specific asset
    python backfill_coinbase_candles.py --days 7     # Backfill last 7 days
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()


def load_config(file_path='config.json'):
    """Load configuration from config.json"""
    with open(file_path, 'r') as file:
        return json.load(file)


def load_existing_data(file_path):
    """Load existing data from a JSON file"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file_path}, starting fresh")
                return []
    return []


def save_data(file_path, data):
    """Save data to a JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"✓ Saved {len(data)} data points to {file_path}")


def fetch_coinbase_candles(client, product_id, start_time, end_time, granularity="FIVE_MINUTE"):
    """
    Fetch historical candles from Coinbase Advanced API

    Args:
        client: Coinbase RESTClient instance
        product_id: Product ID (e.g., 'BTC-USD')
        start_time: Start timestamp (Unix timestamp in seconds)
        end_time: End timestamp (Unix timestamp in seconds)
        granularity: Candle size - FIVE_MINUTE, ONE_MINUTE, FIFTEEN_MINUTE, etc.

    Returns:
        List of candle data or None if error

    Note:
        Coinbase API returns candles in format:
        {
            "candles": [
                {
                    "start": "1234567890",  # Unix timestamp
                    "low": "100.00",
                    "high": "105.00",
                    "open": "102.00",
                    "close": "104.00",
                    "volume": "1000.50"  # Volume in base currency (BTC for BTC-USD)
                },
                ...
            ]
        }
    """
    try:
        print(f"  Fetching candles from {datetime.fromtimestamp(start_time, tz=timezone.utc)} to {datetime.fromtimestamp(end_time, tz=timezone.utc)}...")

        # Coinbase API expects timestamps in seconds
        response = client.get_candles(
            product_id=product_id,
            start=int(start_time),
            end=int(end_time),
            granularity=granularity
        )

        # Convert response object to dict
        if hasattr(response, 'to_dict'):
            data = response.to_dict()
        elif hasattr(response, '__dict__'):
            data = response.__dict__
        else:
            data = response

        candles = data.get('candles', [])

        if not candles:
            print(f"  WARNING: No candles returned for {product_id}")
            return []

        print(f"  ✓ Fetched {len(candles)} candles")
        return candles

    except Exception as e:
        print(f"ERROR: Failed to fetch candles from Coinbase: {e}")
        import traceback
        traceback.print_exc()
        return None


def transform_candles_to_data_format(candles, product_id):
    """
    Transform Coinbase candle data to match the existing data format

    Args:
        candles: List of candle dicts from Coinbase API
        product_id: Product ID (e.g., 'BTC-USD')

    Returns:
        List of data points matching existing format:
        {
            "timestamp": 1234567890.0,
            "product_id": "BTC-USD",
            "price": "104.00",
            "volume_24h": "1000.50"
        }
    """
    transformed_data = []

    for candle in candles:
        # Extract candle data
        # Candle might be a dict or an object with attributes
        if isinstance(candle, dict):
            timestamp = candle.get('start')
            close_price = candle.get('close')
            volume = candle.get('volume')
        else:
            timestamp = getattr(candle, 'start', None)
            close_price = getattr(candle, 'close', None)
            volume = getattr(candle, 'volume', None)

        # Skip if missing critical data
        if not timestamp or not close_price:
            continue

        # Convert timestamp string to float (Unix timestamp in seconds)
        try:
            timestamp_float = float(timestamp)
        except (ValueError, TypeError):
            print(f"  WARNING: Invalid timestamp: {timestamp}")
            continue

        data_point = {
            'timestamp': timestamp_float,
            'product_id': product_id,
            'price': str(close_price),
            'volume_24h': str(volume) if volume else "0"
        }

        transformed_data.append(data_point)

    return transformed_data


def merge_data(existing_data, new_data):
    """
    Merge new data with existing data, avoiding duplicates and sorting by timestamp

    Args:
        existing_data: List of existing data points
        new_data: List of new data points to merge

    Returns:
        Merged and sorted list of data points
    """
    # Create a dictionary with timestamp as key to avoid duplicates
    data_dict = {}

    # Add existing data
    for point in existing_data:
        timestamp = point.get('timestamp')
        if timestamp:
            data_dict[timestamp] = point

    # Add new data (will overwrite if timestamp already exists)
    added_count = 0
    for point in new_data:
        timestamp = point.get('timestamp')
        if timestamp and timestamp not in data_dict:
            data_dict[timestamp] = point
            added_count += 1

    # Convert back to list and sort by timestamp
    merged_data = sorted(data_dict.values(), key=lambda x: x.get('timestamp', 0))

    print(f"  ✓ Added {added_count} new data points, total: {len(merged_data)}")
    return merged_data


def backfill_asset_candles(client, product_id, days_back=7):
    """
    Backfill historical 5-minute candles for a single asset

    Args:
        client: Coinbase RESTClient instance
        product_id: Product ID (e.g., 'BTC-USD')
        days_back: Number of days to backfill (default 7)
    """
    print(f"\n=== Backfilling 5-minute candles for {product_id} ===")

    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    print(f"  Requesting {days_back} days of data")
    print(f"  From: {start_time}")
    print(f"  To: {end_time}")

    # Coinbase API limits: Max 300 candles per request
    # For 5-minute candles: 300 candles = 1500 minutes = 25 hours
    # So we need to chunk requests for longer periods

    max_candles_per_request = 300
    minutes_per_candle = 5
    max_minutes_per_request = max_candles_per_request * minutes_per_candle  # 1500 minutes

    all_transformed_data = []
    current_start = start_time
    chunk_num = 0

    while current_start < end_time:
        chunk_num += 1

        # Calculate chunk end time (25 hours from current start, or end_time if sooner)
        current_end = min(
            current_start + timedelta(minutes=max_minutes_per_request),
            end_time
        )

        # Convert to Unix timestamps
        time_start = int(current_start.timestamp())
        time_end = int(current_end.timestamp())

        hours_in_chunk = (current_end - current_start).total_seconds() / 3600
        print(f"\n  Chunk {chunk_num}: {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')} ({hours_in_chunk:.1f} hours)")

        # Fetch candles from Coinbase
        candles = fetch_coinbase_candles(
            client=client,
            product_id=product_id,
            start_time=time_start,
            end_time=time_end,
            granularity="FIVE_MINUTE"
        )

        if candles is None:
            print(f"  WARNING: Failed to fetch candles for chunk {chunk_num}, skipping")
            current_start = current_end
            continue

        if not candles:
            print(f"  WARNING: No candles returned for chunk {chunk_num}")
            current_start = current_end
            continue

        # Transform to data format
        transformed_data = transform_candles_to_data_format(candles, product_id)

        if transformed_data:
            all_transformed_data.extend(transformed_data)
            print(f"  ✓ Transformed {len(transformed_data)} candles")

        # Move to next chunk
        current_start = current_end

        # Add delay between chunks to respect rate limits
        if current_start < end_time:
            time.sleep(0.5)

    print(f"\n  Total candles fetched across all chunks: {len(all_transformed_data)}")

    if not all_transformed_data:
        print(f"  ERROR: No data to save for {product_id}")
        return

    # Save to coinbase-data directory
    print(f"\n  Saving to coinbase-data directory...")
    coinbase_data_file = f"coinbase-data/{product_id}.json"
    existing_coinbase_data = load_existing_data(coinbase_data_file)
    print(f"  Found {len(existing_coinbase_data)} existing data points")
    merged_coinbase_data = merge_data(existing_coinbase_data, all_transformed_data)
    save_data(coinbase_data_file, merged_coinbase_data)

    print(f"\n  ✓ Backfill complete for {product_id}")

    # Calculate expected vs actual
    expected_candles = (days_back * 24 * 60) / 5  # days * hours * minutes / 5-min intervals
    actual_candles = len(all_transformed_data)
    coverage_pct = (actual_candles / expected_candles) * 100 if expected_candles > 0 else 0

    print(f"  📊 Data coverage: {actual_candles}/{int(expected_candles)} candles ({coverage_pct:.1f}%)")


def main():
    """Main backfill execution function"""
    import sys

    print("=" * 70)
    print("Coinbase Advanced API - Historical 5-Minute Candle Backfill")
    print("=" * 70)

    # Load configuration
    config = load_config()

    # Check for API credentials
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')

    if not api_key or not api_secret:
        print("\nERROR: COINBASE_API_KEY and COINBASE_API_SECRET not found in .env file")
        print("Please add your Coinbase Advanced Trade API credentials to .env file")
        return

    # Initialize Coinbase client
    try:
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        print("✓ Coinbase client initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize Coinbase client: {e}")
        return

    # Parse command line arguments
    target_symbol = None
    days_back = 7  # Default to 1 week

    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--days' and i + 2 < len(sys.argv):
            try:
                days_back = int(sys.argv[i + 2])
            except ValueError:
                print(f"WARNING: Invalid --days value, using default {days_back}")
        elif not arg.startswith('--') and arg.isdigit() == False:
            target_symbol = arg

    print(f"\n📅 Backfill period: {days_back} days")
    print(f"📊 Granularity: 5-minute candles")
    print(f"📈 Expected candles per asset: ~{(days_back * 24 * 60) / 5:.0f}")

    # Get enabled wallets
    wallets = config.get('wallets', [])
    enabled_wallets = [wallet for wallet in wallets if wallet.get('enabled', False)]

    # Filter to target symbol if specified
    if target_symbol:
        enabled_wallets = [wallet for wallet in enabled_wallets if wallet.get('symbol') == target_symbol]
        if not enabled_wallets:
            print(f"\nERROR: Symbol '{target_symbol}' not found or not enabled in config.json")
            return
        print(f"\n🎯 Target symbol: {target_symbol}")
    else:
        if not enabled_wallets:
            print("\nNo enabled wallets found in config.json")
            return
        print(f"\nFound {len(enabled_wallets)} enabled wallet(s) to backfill:")
        for wallet in enabled_wallets:
            print(f"  - {wallet.get('symbol')}")

    # Backfill each enabled wallet
    for wallet in enabled_wallets:
        try:
            product_id = wallet.get('symbol')
            backfill_asset_candles(client, product_id, days_back)

            # Add delay between assets to respect rate limits
            time.sleep(1)

        except Exception as e:
            print(f"\nERROR: Failed to backfill {wallet.get('symbol')}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("Backfill process complete!")
    print("=" * 70)
    print("\nYour coinbase-data/*.json files now contain historical 5-minute candles.")
    print("You can now run backtests or start live trading with historical context.")


if __name__ == '__main__':
    main()
