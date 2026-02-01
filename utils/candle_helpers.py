"""
5-Minute Candle Data Collection Helpers

Functions for fetching and storing 5-minute candle data from Coinbase Advanced API.
Designed to integrate with the main index.py loop.

OVERVIEW
========
Your trading bot collects 5-minute candles from Coinbase Advanced API instead of 30-second ticker snapshots.

Benefits:
- Better correlation calculations (consistent intervals)
- Lower API usage (5 min vs 30 sec = 90% reduction)
- Perfect alignment with momentum divergence strategy

DATA FLOW
=========
1. BACKFILL (One-time or as needed)
   backfill_coinbase_candles.py --days 90
   → Populates /coinbase-data/*.json with historical data

2. LIVE COLLECTION (Continuous via index.py)
   Every 5 minutes, index.py fetches latest candle
   → Appends to same /coinbase-data/*.json files
   → Auto-deduplicates (skips existing timestamps)
   → Auto-cleanup (keeps only max_hours from config)

3. STRATEGY EXECUTION (index.py uses this data)
   Your momentum divergence strategy reads candles
   → Calculates correlations over LOOKBACK_WINDOW
   → Identifies divergence opportunities
   → Executes trades

USAGE IN INDEX.PY
=================
from utils.candle_helpers import fetch_latest_5min_candle, candle_to_data_entry

for product_id in enabled_wallets:
    # Fetch most recent completed 5-minute candle
    candle = fetch_latest_5min_candle(coinbase_client, product_id)

    # Transform to same format as before
    data_entry = candle_to_data_entry(candle, product_id)
    # → {timestamp, product_id, price, volume_24h}

    # Append to /coinbase-data/{product_id}.json
    append_crypto_data_to_file(coinbase_data_directory, product_id, data_entry)

DATA FORMAT
===========
Each /coinbase-data/{ASSET}.json file contains:
[
  {
    "timestamp": 1768690800.0,        # Unix timestamp in seconds (5-min intervals)
    "product_id": "BTC-USD",
    "price": "95095.97",              # Close price of the 5-minute candle
    "volume_24h": "10.84506839"       # Volume in base currency
  },
  ...
]

CONFIGURATION
=============
In config.json:
{
  "data_retention": {
    "max_hours": 4380,        // Keep 6 months (auto-cleanup)
    "interval_seconds": 300   // 5 minutes = 300 seconds
  },
  "momentum_divergence": {
    "lookback_window": 6      // 6 candles = 30 minutes
  }
}

COMMANDS
========
# Backfill historical data
python3 backfill_coinbase_candles.py --days 90

# Run live trading (auto-collects candles every 5 minutes)
python3 index.py

# Test integration
python3 test_candle_collection.py

TROUBLESHOOTING
===============
"No candles returned"
  - Asset might be delisted or disabled
  - Check Coinbase status: https://status.coinbase.com/

"Duplicate candle"
  - Normal! index.py skips candles that already exist
  - Prevents data corruption from overlapping runs

"File not found"
  - Run backfill first: python3 backfill_coinbase_candles.py
  - Or wait for index.py to collect first candle (5 min)

Missing candles (coverage < 95%)
  - Some gaps are normal (Coinbase API limits, network issues)
  - Re-run backfill to fill gaps
"""

import time
from datetime import datetime, timedelta, timezone


def fetch_latest_5min_candle(client, product_id):
    """
    Fetch the most recent completed 5-minute candle from Coinbase

    Args:
        client: Coinbase RESTClient instance
        product_id: Product ID (e.g., 'BTC-USD')

    Returns:
        Candle dict or None if error
        Candle format:
        {
            'start': '1234567890',  # Unix timestamp string
            'low': '100.00',
            'high': '105.00',
            'open': '102.00',
            'close': '104.00',
            'volume': '1000.50'  # Volume in base currency
        }
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

        # Return the most recent completed candle (second from last if multiple)
        # Coinbase returns candles in descending order (newest first)
        if len(candles) >= 2:
            return candles[1]  # Second candle = most recent completed
        else:
            return candles[0]  # Only one available

    except Exception as e:
        print(f"  ERROR fetching 5-min candle for {product_id}: {e}")
        return None


def candle_to_data_entry(candle, product_id):
    """
    Transform a Coinbase candle to the data entry format used by index.py

    Args:
        candle: Candle dict or object from Coinbase API
        product_id: Product ID (e.g., 'BTC-USD')

    Returns:
        Data entry dict in format matching existing append_crypto_data_to_file:
        {
            'timestamp': 1234567890.0,
            'product_id': 'BTC-USD',
            'price': '104.00',
            'volume_24h': '1000.50'
        }
        Returns None if candle is invalid
    """
    # Extract candle data (handle both dict and object)
    if isinstance(candle, dict):
        timestamp = candle.get('start')
        close_price = candle.get('close')
        volume = candle.get('volume')
    else:
        timestamp = getattr(candle, 'start', None)
        close_price = getattr(candle, 'close', None)
        volume = getattr(candle, 'volume', None)

    # Validate required fields
    if not timestamp or not close_price:
        return None

    # Convert timestamp string to float (Unix timestamp in seconds)
    try:
        timestamp_float = float(timestamp)
    except (ValueError, TypeError):
        return None

    # Return data entry matching the format in index.py (lines 293-306)
    return {
        'timestamp': timestamp_float,
        'product_id': product_id,
        'price': str(close_price),
        'volume_24h': str(volume) if volume else "0"
    }


def is_candle_duplicate(directory, product_id, timestamp):
    """
    Check if a candle with this timestamp already exists in the data file

    Args:
        directory: Directory containing data files (e.g., 'coinbase-data')
        product_id: Product ID (e.g., 'BTC-USD')
        timestamp: Unix timestamp to check

    Returns:
        True if duplicate exists, False otherwise
    """
    import os
    import json

    file_path = os.path.join(directory, f"{product_id}.json")

    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            return False

        # Check if any entry has this exact timestamp
        timestamps = {d.get('timestamp') for d in data if 'timestamp' in d}
        return timestamp in timestamps

    except (json.JSONDecodeError, Exception):
        return False
