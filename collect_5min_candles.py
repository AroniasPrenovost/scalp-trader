#!/usr/bin/env python3
"""
5-Minute Candle Collector for Coinbase

Continuously collects 5-minute candles and appends them to coinbase-data/*.json files.
Runs every 5 minutes to fetch the most recent completed candle.

Usage:
    python3 collect_5min_candles.py                 # Run continuously
    python3 collect_5min_candles.py --once          # Fetch once and exit
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

# Configuration
COLLECTION_INTERVAL_SECONDS = 300  # 5 minutes
MAX_RETENTION_HOURS = 4380  # Default from config.json


def load_config(file_path='config.json'):
    """Load configuration from config.json"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


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


def fetch_latest_candle(client, product_id):
    """
    Fetch the most recent 5-minute candle from Coinbase

    Returns:
        Candle dict or None if error
    """
    try:
        # Get the last 2 candles (in case the most recent is still forming)
        end_time = int(datetime.now(timezone.utc).timestamp())
        start_time = end_time - 600  # 10 minutes ago

        response = client.get_candles(
            product_id=product_id,
            start=start_time,
            end=end_time,
            granularity="FIVE_MINUTE"
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
            return None

        # Return the most recent completed candle (second from last if still forming)
        # Coinbase returns candles in descending order (newest first)
        if len(candles) >= 2:
            return candles[1]  # Second candle = most recent completed
        else:
            return candles[0]  # Only one available

    except Exception as e:
        print(f"  ERROR fetching candle for {product_id}: {e}")
        return None


def candle_to_data_point(candle, product_id):
    """
    Transform a candle to data point format

    Args:
        candle: Candle dict or object from Coinbase API
        product_id: Product ID (e.g., 'BTC-USD')

    Returns:
        Data point dict in format:
        {
            "timestamp": 1234567890.0,
            "product_id": "BTC-USD",
            "price": "104.00",
            "volume_24h": "1000.50"
        }
    """
    # Extract candle data
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
        return None

    # Convert timestamp string to float
    try:
        timestamp_float = float(timestamp)
    except (ValueError, TypeError):
        return None

    return {
        'timestamp': timestamp_float,
        'product_id': product_id,
        'price': str(close_price),
        'volume_24h': str(volume) if volume else "0"
    }


def append_candle_to_file(data_point, product_id, max_retention_hours):
    """
    Append a new candle to the data file, avoiding duplicates

    Args:
        data_point: Candle data point to append
        product_id: Product ID
        max_retention_hours: Max hours of data to retain
    """
    file_path = f"coinbase-data/{product_id}.json"

    # Load existing data
    existing_data = load_existing_data(file_path)

    # Check if this timestamp already exists (avoid duplicates)
    timestamp = data_point.get('timestamp')
    timestamps = {d.get('timestamp') for d in existing_data if 'timestamp' in d}

    if timestamp in timestamps:
        print(f"  ℹ️  Candle already exists for {product_id} at {datetime.fromtimestamp(timestamp, tz=timezone.utc)}")
        return False

    # Append new data point
    existing_data.append(data_point)

    # Sort by timestamp
    existing_data.sort(key=lambda x: x.get('timestamp', 0))

    # Clean up old data (keep only max_retention_hours)
    if max_retention_hours > 0:
        cutoff_time = time.time() - (max_retention_hours * 3600)
        original_count = len(existing_data)
        existing_data = [d for d in existing_data if d.get('timestamp', 0) >= cutoff_time]
        removed_count = original_count - len(existing_data)

        if removed_count > 0:
            print(f"  🧹 Removed {removed_count} old candles (retention: {max_retention_hours}h)")

    # Save back to file
    save_data(file_path, existing_data)

    return True


def collect_all_assets(client, config, verbose=True):
    """
    Collect latest 5-minute candle for all enabled assets

    Returns:
        Number of successfully collected candles
    """
    # Get enabled wallets
    wallets = config.get('wallets', [])
    enabled_wallets = [w for w in wallets if w.get('enabled', False)]

    if not enabled_wallets:
        print("❌ No enabled wallets found in config.json")
        return 0

    # Get retention settings
    max_retention_hours = config.get('data_retention', {}).get('max_hours', MAX_RETENTION_HOURS)

    collected_count = 0

    for wallet in enabled_wallets:
        product_id = wallet.get('symbol')

        if verbose:
            print(f"  Fetching {product_id}...", end=' ')

        # Fetch latest candle
        candle = fetch_latest_candle(client, product_id)

        if not candle:
            if verbose:
                print("❌ Failed")
            continue

        # Transform to data point
        data_point = candle_to_data_point(candle, product_id)

        if not data_point:
            if verbose:
                print("❌ Invalid data")
            continue

        # Append to file
        was_added = append_candle_to_file(data_point, product_id, max_retention_hours)

        if was_added:
            collected_count += 1
            if verbose:
                price = data_point.get('price')
                timestamp = data_point.get('timestamp')
                time_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%H:%M:%S')
                print(f"✅ ${price} @ {time_str}")
        else:
            if verbose:
                print("⏭️  Duplicate")

    return collected_count


def print_data_stats(config):
    """Print statistics about collected candle data"""
    wallets = config.get('wallets', [])
    enabled_wallets = [w for w in wallets if w.get('enabled', False)]

    print("\n" + "="*70)
    print("📊 5-MINUTE CANDLE DATA STATISTICS")
    print("="*70 + "\n")

    for wallet in enabled_wallets:
        product_id = wallet.get('symbol')
        file_path = f"coinbase-data/{product_id}.json"

        if not os.path.exists(file_path):
            print(f"{product_id}: No data")
            continue

        data = load_existing_data(file_path)

        if not data:
            print(f"{product_id}: Empty file")
            continue

        # Calculate stats
        num_candles = len(data)
        first_timestamp = data[0].get('timestamp', 0)
        last_timestamp = data[-1].get('timestamp', 0)

        duration_hours = (last_timestamp - first_timestamp) / 3600
        duration_days = duration_hours / 24

        # Expected candles for 5-min intervals
        expected_candles = int(duration_hours * 12)  # 12 candles per hour
        coverage_pct = (num_candles / expected_candles * 100) if expected_candles > 0 else 0

        print(f"{product_id}:")
        print(f"  Candles: {num_candles:,} ({coverage_pct:.1f}% coverage)")
        print(f"  Duration: {duration_days:.1f} days ({duration_hours:.1f} hours)")
        print(f"  Latest: {datetime.fromtimestamp(last_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

        # Ready for strategy check
        # Minimum: 6 candles (LOOKBACK_WINDOW)
        # Recommended: 30+ candles (2.5 hours) for reliable correlation
        if num_candles >= 30:
            print(f"  Status: ✅ Ready for strategy (correlation reliable)")
        elif num_candles >= 6:
            print(f"  Status: 🟡 Minimum met, but need {30 - num_candles} more for reliability")
        else:
            print(f"  Status: 🔴 Need {6 - num_candles} more candles to start")

        print()

    print("="*70 + "\n")


def main():
    """Main execution function"""
    print("="*70)
    print("5-MINUTE CANDLE COLLECTOR")
    print("="*70)

    # Check for --once flag
    run_once = '--once' in sys.argv

    if run_once:
        print("Mode: Single collection (--once)")
    else:
        print(f"Mode: Continuous collection (every {COLLECTION_INTERVAL_SECONDS}s)")

    print()

    # Load config
    config = load_config()

    # Initialize Coinbase client
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')

    if not api_key or not api_secret:
        print("❌ ERROR: COINBASE_API_KEY and COINBASE_API_SECRET not found in .env")
        return

    try:
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        print("✅ Coinbase client initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize Coinbase client: {e}")
        return

    # Get enabled asset count
    enabled_count = len([w for w in config.get('wallets', []) if w.get('enabled', False)])
    print(f"Monitoring {enabled_count} enabled assets\n")
    print("="*70 + "\n")

    if run_once:
        # Single collection
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Collecting candles...\n")
        collected = collect_all_assets(client, config, verbose=True)
        print(f"\n✅ Collected {collected}/{enabled_count} candles")
        print_data_stats(config)
        return

    # Continuous collection
    print("🚀 Starting continuous collection...\n")
    print("💡 Tip: Run in background with: nohup python3 collect_5min_candles.py > candle_collector.log 2>&1 &\n")

    iteration = 0

    try:
        while True:
            iteration += 1
            now = datetime.now()

            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Iteration #{iteration}")
            print()

            # Collect candles
            collected = collect_all_assets(client, config, verbose=True)

            print(f"\n✅ Collected {collected}/{enabled_count} candles")

            # Print stats every 10 iterations (50 minutes)
            if iteration % 10 == 0:
                print_data_stats(config)

            # Calculate next collection time
            next_collection = now + timedelta(seconds=COLLECTION_INTERVAL_SECONDS)
            print(f"\n💤 Next collection at {next_collection.strftime('%H:%M:%S')}")
            print("="*70 + "\n")

            # Sleep until next interval
            time.sleep(COLLECTION_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\n🛑 Collection stopped by user\n")
        print_data_stats(config)
    except Exception as e:
        print(f"\n\n❌ Error in collection loop: {e}\n")
        import traceback
        traceback.print_exc()
        print_data_stats(config)


if __name__ == '__main__':
    main()
