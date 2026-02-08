#!/usr/bin/env python3
"""
Quick test to verify 5-minute candle collection works in index.py

This simulates what happens in the index.py data collection loop.

Run from project root: python3 utils/test_candle_helpers.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dotenv import load_dotenv
from utils.coinbase import get_coinbase_client
from utils.candle_helpers import fetch_latest_5min_candle, candle_to_data_entry, is_candle_duplicate
from utils.file_helpers import append_crypto_data_to_file
from json import load

load_dotenv()

# Load config
with open('config.json', 'r') as f:
    config = load(f)

# Get enabled wallets (same as index.py)
enabled_wallets = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

print("="*70)
print("5-MINUTE CANDLE COLLECTION TEST")
print("="*70)
print(f"\nTesting with {len(enabled_wallets)} enabled assets: {', '.join(enabled_wallets)}\n")

# Initialize Coinbase client (same as index.py)
try:
    client = get_coinbase_client()
    print("✓ Coinbase client initialized\n")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
    exit(1)

# Simulate the data collection loop from index.py
coinbase_data_directory = 'coinbase-data'
candles_collected = 0
candles_skipped = 0

for product_id in enabled_wallets:
    print(f"Processing {product_id}...")

    # Fetch latest 5-minute candle (same as index.py lines 297-301)
    candle = fetch_latest_5min_candle(client, product_id)

    if not candle:
        print(f"  ⚠️  Failed to fetch candle")
        continue

    # Transform to data entry (same as index.py lines 304-308)
    data_entry = candle_to_data_entry(candle, product_id)

    if not data_entry:
        print(f"  ⚠️  Invalid candle data")
        continue

    # Check for duplicates (same as index.py lines 311-313)
    if is_candle_duplicate(coinbase_data_directory, product_id, data_entry['timestamp']):
        print(f"  ⏭️  Duplicate candle (already exists)")
        candles_skipped += 1
        continue

    # Append to file (same as index.py lines 316-317)
    append_crypto_data_to_file(coinbase_data_directory, product_id, data_entry)
    print(f"  ✅ Collected: ${data_entry['price']} @ timestamp {data_entry['timestamp']}")
    candles_collected += 1

print(f"\n{'='*70}")
print(f"RESULTS: Collected {candles_collected}, Skipped {candles_skipped}")
print(f"{'='*70}")
print("\n✓ Integration test complete! This is exactly what index.py will do.\n")
