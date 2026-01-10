#!/usr/bin/env python3
"""
Quick script to backfill BTC-USD data only
"""

import sys
import os

# Add the script directory to path so we can import the backfill module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backfill_historical_data import load_config, backfill_wallet_data

def main():
    print("=" * 60)
    print("BTC-USD Data Backfill")
    print("=" * 60)

    # Load config
    config = load_config('config.json')

    # Find BTC wallet
    btc_wallet = None
    for wallet in config.get('wallets', []):
        if wallet.get('symbol') == 'BTC-USD':
            btc_wallet = wallet
            break

    if not btc_wallet:
        print("ERROR: BTC-USD wallet not found in config.json")
        return

    print(f"\nBackfilling data for {btc_wallet.get('symbol')}...")
    print(f"CoinGecko ID: {btc_wallet.get('coingecko_id')}")

    # Backfill BTC data
    try:
        backfill_wallet_data(btc_wallet, config)
        print("\n✓ BTC-USD backfill complete!")
    except Exception as e:
        print(f"\nERROR: Failed to backfill BTC-USD: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
