#!/usr/bin/env python3
"""
CoinGecko Historical Data Backfill Script

This script fetches historical price data from CoinGecko API and populates
the coinbase-data JSON files with backfilled data based on the data_retention settings defined in config.json.

Only runs if enable_backfilling_historical_data is set to true in config.json.
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables
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

def calculate_percentage_change_24h(data_points, current_index):
    """
    Calculate 24-hour percentage change based on available data points.
    Returns None if insufficient data.
    """
    if current_index < 96:  # Need at least 24 hours of data (96 15-minute intervals)
        return None

    current_price = float(data_points[current_index]['price'])
    price_24h_ago = float(data_points[current_index - 96]['price'])

    if price_24h_ago == 0:
        return None

    percentage_change = ((current_price - price_24h_ago) / price_24h_ago) * 100
    return str(percentage_change)

def fetch_coingecko_historical_data(coingecko_id, time_start, time_end):
    """
    Fetch historical data from CoinGecko API

    Args:
        coingecko_id: CoinGecko ID for the cryptocurrency (e.g., 'bitcoin', 'ethereum')
        time_start: Start timestamp (Unix timestamp in seconds)
        time_end: End timestamp (Unix timestamp in seconds)

    Returns:
        Dict with 'prices', 'market_caps', 'total_volumes' arrays or None if error
    """
    api_key = os.getenv('COINGECKO_API_KEY')
    if not api_key:
        print("ERROR: COINGECKO_API_KEY not found in environment variables")
        return None

    url = f'https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart/range'

    headers = {
        'x-cg-demo-api-key': api_key,
        'Accept': 'application/json'
    }

    params = {
        'vs_currency': 'usd',
        'from': int(time_start),
        'to': int(time_end)
    }

    try:
        print(f"  Fetching data from {datetime.fromtimestamp(time_start, tz=timezone.utc)} to {datetime.fromtimestamp(time_end, tz=timezone.utc)}...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        # Check if we got the expected data structure
        if 'prices' not in data:
            print("ERROR: Unexpected response format from CoinGecko API")
            return None

        print(f"  ✓ Fetched {len(data['prices'])} data points")
        return data

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch data from CoinGecko: {e}")
        return None

def transform_coingecko_to_coinbase_format(coingecko_data, product_id):
    """
    Transform CoinGecko market chart data to match Coinbase data format

    IMPORTANT: CoinGecko returns total_volumes in USD (because vs_currency='usd'),
    but Coinbase API returns volume_24h in base currency (BTC for BTC-USD).
    We convert CoinGecko USD volumes to BTC by dividing by price to maintain consistency.

    Args:
        coingecko_data: Dict with 'prices', 'market_caps', 'total_volumes' from CoinGecko API
        product_id: Product ID (e.g., 'BTC-USD')

    Returns:
        List of data points in Coinbase format with volume_24h in BTC units
    """
    print(coingecko_data)
    transformed_data = []
    prices = coingecko_data.get('prices', [])
    volumes = coingecko_data.get('total_volumes', [])

    # Create a volume lookup dict by timestamp for easier access
    volume_dict = {int(v[0]): v[1] for v in volumes}

    for i, price_data in enumerate(prices):
        timestamp_ms = price_data[0]
        price = price_data[1]

        # Convert milliseconds to seconds for Unix timestamp
        unix_timestamp = timestamp_ms / 1000

        # Get volume for this timestamp (in USD from CoinGecko)
        volume_usd = volume_dict.get(int(timestamp_ms), 0)

        # Convert USD volume to BTC volume to match Coinbase API format
        # (Coinbase returns volume_24h in base currency, which is BTC for BTC-USD)
        if price > 0 and volume_usd > 0:
            volume_24h_btc = volume_usd / price
        else:
            volume_24h_btc = 0

        data_point = {
            'timestamp': unix_timestamp,
            'product_id': product_id,
            'price': str(price),
            'volume_24h': str(volume_24h_btc)
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

def backfill_wallet_data(wallet, config):
    """
    Backfill historical data for a single wallet

    Args:
        wallet: Wallet configuration dictionary
        config: Full configuration object
    """
    symbol = wallet.get('symbol')
    coingecko_id = wallet.get('coingecko_id')

    if not coingecko_id:
        print(f"  WARNING: No coingecko_id found for {symbol}, skipping")
        return

    print(f"\n=== Backfilling data for {symbol} ===")

    # Get data retention settings
    max_hours = config.get('data_retention', {}).get('max_hours', 730)
    interval_seconds = config.get('data_retention', {}).get('interval_seconds', 900)

    # Calculate start time (max_hours ago from now)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=max_hours)

    # CoinGecko free tier supports up to 365 days of historical data
    max_days = 365
    if max_hours > max_days * 24:
        print(f"  WARNING: Requested {max_hours} hours ({max_hours/24:.1f} days) but CoinGecko free tier only supports {max_days} days")
        print(f"  Limiting to {max_days} days of historical data")
        start_time = end_time - timedelta(days=max_days)

    # CoinGecko API granularity rules:
    # - 1 day: 5-minute intervals
    # - 1-90 days: hourly intervals
    # - Above 90 days: daily intervals
    # To get hourly data, we need to chunk requests into 90-day periods
    chunk_days = 90
    total_days = (end_time - start_time).days

    print(f"  Requesting data from {start_time} to {end_time}")
    print(f"  Total period: {total_days} days")

    if total_days > chunk_days:
        print(f"  Splitting into {chunk_days}-day chunks to get hourly granularity (instead of daily)")
        num_chunks = (total_days // chunk_days) + (1 if total_days % chunk_days else 0)
        print(f"  Will make {num_chunks} API requests")
    else:
        print(f"  Using hourly interval (will fetch ~{total_days * 24} data points)")

    # Fetch data in chunks
    all_transformed_data = []
    current_start = start_time
    chunk_num = 0

    while current_start < end_time:
        chunk_num += 1
        # Calculate chunk end time (90 days from current start, or end_time if sooner)
        current_end = min(current_start + timedelta(days=chunk_days), end_time)

        # Convert to Unix timestamps (in seconds)
        time_start = int(current_start.timestamp())
        time_end = int(current_end.timestamp())

        chunk_days_actual = (current_end - current_start).days
        print(f"  Chunk {chunk_num}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')} ({chunk_days_actual} days)")

        # Fetch historical data from CoinGecko
        coingecko_data = fetch_coingecko_historical_data(
            coingecko_id=coingecko_id,
            time_start=time_start,
            time_end=time_end
        )

        if not coingecko_data:
            print(f"  WARNING: Failed to fetch data for chunk {chunk_num}, skipping")
            current_start = current_end
            continue

        # Transform data to Coinbase format
        transformed_data = transform_coingecko_to_coinbase_format(coingecko_data, symbol)

        if transformed_data:
            all_transformed_data.extend(transformed_data)

        # Move to next chunk
        current_start = current_end

        # Add delay between chunks to respect rate limits (30 calls/min for free tier)
        if current_start < end_time:
            time.sleep(2)

    print(f"  Transforming data to match Coinbase format...")
    print(f"  Total data points fetched across all chunks: {len(all_transformed_data)}")

    if not all_transformed_data:
        print(f"  ERROR: No data to save for {symbol}")
        return

    # Load existing data
    data_file = f"coinbase-data/{symbol}.json"
    existing_data = load_existing_data(data_file)

    print(f"  Found {len(existing_data)} existing data points")

    # Merge with existing data
    merged_data = merge_data(existing_data, all_transformed_data)

    # Save merged data
    save_data(data_file, merged_data)

    print(f"  ✓ Backfill complete for {symbol}")

def main():
    """Main backfill execution function"""
    print("=" * 60)
    print("CoinGecko Historical Data Backfill")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Check if backfilling is enabled
    coingecko_config = config.get('coingecko', {})
    enable_backfilling = coingecko_config.get('enable_backfilling_historical_data', False)

    if not enable_backfilling:
        print("\nBackfilling is DISABLED in config.json")
        print("Set 'coingecko.enable_backfilling_historical_data' to true to enable")
        return

    print("\n✓ Backfilling is ENABLED")

    # Check for API key
    api_key = os.getenv('COINGECKO_API_KEY')
    if not api_key:
        print("\nERROR: COINGECKO_API_KEY not found in .env file")
        print("Please add your CoinGecko API key to .env file:")
        print("COINGECKO_API_KEY=your_api_key_here")
        return

    # Get enabled wallets
    wallets = config.get('wallets', [])
    enabled_wallets = [wallet for wallet in wallets if wallet.get('enabled', False)]

    if not enabled_wallets:
        print("\nNo enabled wallets found in config.json")
        return

    print(f"\nFound {len(enabled_wallets)} enabled wallet(s) to backfill:")
    for wallet in enabled_wallets:
        print(f"  - {wallet.get('symbol')}")

    # Backfill each enabled wallet
    for wallet in enabled_wallets:
        try:
            backfill_wallet_data(wallet, config)
            # Add a small delay between API calls to avoid rate limiting (CoinGecko: 30 calls/min)
            time.sleep(2)
        except Exception as e:
            print(f"\nERROR: Failed to backfill {wallet.get('symbol')}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Backfill process complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
