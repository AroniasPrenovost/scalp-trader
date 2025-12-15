#!/usr/bin/env python3
"""
Quick test script to verify range support strategy integration

This tests the range strategy against current market data to ensure:
1. It correctly identifies support zones
2. It only triggers buy signals when price is AT support (not mid-range)
3. The configuration parameters work as expected
"""

import json
from utils.range_support_strategy import check_range_support_buy_signal
from utils.file_helpers import get_property_values_from_crypto_file
from utils.coinbase import get_coinbase_client, get_asset_price

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Get range strategy config
range_config = config.get('range_support_strategy', {})
enabled = range_config.get('enabled', True)

print("=" * 80)
print("RANGE SUPPORT STRATEGY - LIVE TEST")
print("=" * 80)
print(f"\nStrategy Enabled: {enabled}")
print(f"Configuration:")
print(f"  Min touches: {range_config.get('min_touches', 2)}")
print(f"  Zone tolerance: {range_config.get('zone_tolerance_percentage', 3.0)}%")
print(f"  Entry tolerance: {range_config.get('entry_tolerance_percentage', 1.5)}%")
print(f"  Extrema order: {range_config.get('extrema_order', 5)}")
print(f"  Lookback window: {range_config.get('lookback_window_hours', 336)} hours")

# Test on all enabled wallets
enabled_wallets = [w for w in config['wallets'] if w['enabled']]
coinbase_client = get_coinbase_client()

for wallet in enabled_wallets:
    symbol = wallet['symbol']

    print(f"\n{'='*80}")
    print(f"Testing: {symbol}")
    print(f"{'='*80}")

    # Get historical price data
    coinbase_data_directory = 'coinbase-data'
    max_age_hours = config['data_retention']['max_hours']
    prices = get_property_values_from_crypto_file(
        coinbase_data_directory,
        symbol,
        'price',
        max_age_hours=max_age_hours
    )

    if not prices or len(prices) < 100:
        print(f"✗ Insufficient price data for {symbol} ({len(prices) if prices else 0} points)")
        continue

    # Get current price
    current_price = get_asset_price(coinbase_client, symbol)

    print(f"\nCurrent price: ${current_price:.2f}")
    print(f"Price data points: {len(prices)}")
    print(f"Price range (last 24h): ${min(prices[-24:]):.2f} - ${max(prices[-24:]):.2f}")

    # Run range strategy
    signal = check_range_support_buy_signal(
        prices=prices,
        current_price=current_price,
        min_touches=range_config.get('min_touches', 2),
        zone_tolerance_percentage=range_config.get('zone_tolerance_percentage', 3.0),
        entry_tolerance_percentage=range_config.get('entry_tolerance_percentage', 1.5),
        extrema_order=range_config.get('extrema_order', 5),
        lookback_window=range_config.get('lookback_window_hours', 336)
    )

    print(f"\n--- RANGE STRATEGY RESULTS ---")
    print(f"Signal: {signal['signal'].upper()}")
    print(f"Reasoning: {signal['reasoning']}")

    if signal['all_zones']:
        print(f"\nIdentified {len(signal['all_zones'])} support zone(s):")
        for i, zone in enumerate(signal['all_zones'][:3], 1):  # Show top 3
            print(f"\n  Zone {i}:")
            print(f"    Price range: ${zone['zone_price_min']:.2f} - ${zone['zone_price_max']:.2f}")
            print(f"    Average: ${zone['zone_price_avg']:.2f}")
            print(f"    Touches: {zone['touches']}")
            print(f"    Strength: {zone['strength']}")

            # Calculate distance from current price
            distance_pct = ((current_price - zone['zone_price_avg']) / zone['zone_price_avg']) * 100
            print(f"    Distance from current price: {distance_pct:+.2f}%")

            # Check if triggered zone
            if signal['signal'] == 'buy' and signal['zone'] == zone:
                print(f"    ✓ ACTIVE ZONE (buy signal triggered)")
    else:
        print(f"\n✗ No support zones identified")

    # Show verdict
    print(f"\n{'='*60}")
    if signal['signal'] == 'buy':
        print(f"✓ BUY SIGNAL - Price is in a valid support zone")
        zone = signal['zone']
        print(f"  Zone: ${zone['zone_price_min']:.2f} - ${zone['zone_price_max']:.2f}")
        print(f"  Strength: {zone['touches']} touches")
    else:
        print(f"✗ NO BUY SIGNAL - Waiting for price to reach support")
        if signal['all_zones']:
            nearest = signal['all_zones'][0]
            print(f"  Nearest zone: ${nearest['zone_price_avg']:.2f} ({signal['distance_from_zone_avg']:+.2f}% away)")
    print(f"{'='*60}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
