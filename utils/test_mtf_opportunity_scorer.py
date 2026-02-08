#!/usr/bin/env python3
"""
Test script for MTF Momentum Breakout Opportunity Scorer

This script tests the MTF opportunity scoring system to ensure it correctly:
1. Scans all enabled assets
2. Scores each one based on MTF strategy signals
3. Returns the best 1-2 opportunities
4. Displays a clear report

Run from project root: python3 utils/test_mtf_opportunity_scorer.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from utils.mtf_opportunity_scorer import (
    score_mtf_opportunity,
    find_best_opportunities,
    print_opportunity_report
)
from utils.coinbase import get_coinbase_client


def load_config(file_path='config.json'):
    """Load configuration file"""
    with open(file_path, 'r') as file:
        return json.load(file)


def main():
    print("="*100)
    print("MTF MOMENTUM BREAKOUT OPPORTUNITY SCORER - TEST")
    print("="*100)
    print()

    # Load config
    print("Loading config...")
    config = load_config()

    # Get enabled wallets
    wallets = config.get('wallets', [])
    enabled_symbols = [w['symbol'] for w in wallets if w.get('enabled', False)]

    print(f"Found {len(enabled_symbols)} enabled assets: {', '.join(enabled_symbols)}")
    print()

    # Initialize Coinbase client
    print("Connecting to Coinbase API...")
    try:
        coinbase_client = get_coinbase_client()
        print("✓ Connected to Coinbase API")
    except Exception as e:
        print(f"✗ Failed to connect to Coinbase API: {e}")
        print("\nPlease check your .env file has valid COINBASE_API_KEY and COINBASE_API_SECRET")
        return

    print()

    # Get market rotation config
    market_rotation_config = config.get('market_rotation', {})
    min_score = market_rotation_config.get('min_score_for_entry', 75)
    max_concurrent = market_rotation_config.get('max_concurrent_orders', 2)

    print(f"Market Rotation Settings:")
    print(f"  - Min Score for Entry: {min_score}")
    print(f"  - Max Concurrent Positions: {max_concurrent}")
    print()

    # Score all enabled assets
    print("="*100)
    print("SCANNING ALL ENABLED ASSETS...")
    print("="*100)
    print()

    all_opportunities = []

    for symbol in enabled_symbols:
        print(f"Scoring {symbol}...", end=" ")
        opp = score_mtf_opportunity(symbol, config, coinbase_client)
        all_opportunities.append(opp)

        if opp['has_signal']:
            print(f"✓ Signal found (score: {opp['score']:.1f}, confidence: {opp['confidence']})")
        elif opp.get('error'):
            print(f"✗ Error: {opp['error']}")
        else:
            print(f"- No signal")

    print()

    # Find best opportunities
    print("="*100)
    print("FINDING BEST OPPORTUNITIES...")
    print("="*100)
    print()

    best_opportunities = find_best_opportunities(
        config=config,
        coinbase_client=coinbase_client,
        min_score=min_score,
        max_opportunities=max_concurrent
    )

    print(f"Found {len(best_opportunities)} opportunities meeting criteria (score ≥ {min_score})")
    print()

    # Print detailed report
    print_opportunity_report(
        all_opportunities=all_opportunities,
        selected_opportunities=best_opportunities,
        active_positions=[]
    )

    # Summary
    print("="*100)
    print("TEST SUMMARY")
    print("="*100)
    print()

    if best_opportunities:
        print(f"✓ SUCCESS: Found {len(best_opportunities)} tradeable opportunities")
        print()
        print("Next steps:")
        print("1. Review the selected opportunities above")
        print("2. Verify the signals make sense based on current market conditions")
        print("3. If ready, set ready_to_trade: true in config.json for desired coins")
        print("4. Run the bot: python3 index.py")
    else:
        print("⚠️  NO OPPORTUNITIES FOUND")
        print()
        print("This could mean:")
        print("- No coins currently have MTF breakout signals")
        print("- All signals scored below minimum threshold")
        print("- Insufficient data (need 200+ days of 5-min candles)")
        print()
        print("Next steps:")
        print("1. Check if you have historical data: ls -lh coinbase-data/")
        print("2. If missing data, run: python3 backfill_coinbase_candles.py --days 200")
        print("3. Wait for better market conditions (BB squeeze + breakout)")
        print("4. Lower min_score_for_entry in config if needed (try 60-70)")

    print()
    print("="*100)


if __name__ == '__main__':
    main()
