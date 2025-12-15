#!/usr/bin/env python3
"""
Test script for Range-Based Support Zone Trading Strategy

This script demonstrates how the range support strategy works by:
1. Loading historical BTC price data
2. Finding support zones (2-3+ bottoms)
3. Checking if current price is in a support zone
4. Calculating entry, stop loss, and profit targets
"""

import json
import matplotlib.pyplot as plt
from utils.range_support_strategy import (
    find_local_extrema,
    identify_support_zones,
    check_range_support_buy_signal,
    calculate_zone_based_targets
)
from utils.file_helpers import get_property_values_from_crypto_file


def visualize_support_zones(prices, signal_result, symbol="BTC-USD"):
    """
    Create a visualization showing price history, identified support zones,
    and whether a buy signal was triggered.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot price line
    x_values = list(range(len(prices)))
    ax.plot(x_values, prices, label='Price', color='#000000', linewidth=1.2, zorder=5)

    # Plot all identified support zones
    for i, zone in enumerate(signal_result['all_zones']):
        zone_color = '#27AE60' if zone == signal_result.get('zone') else '#95A5A6'
        zone_alpha = 0.3 if zone == signal_result.get('zone') else 0.15
        label_prefix = "ACTIVE" if zone == signal_result.get('zone') else f"Zone {i+1}"

        # Draw horizontal band for the zone
        ax.axhspan(
            zone['zone_price_min'],
            zone['zone_price_max'],
            alpha=zone_alpha,
            color=zone_color,
            label=f"{label_prefix}: ${zone['zone_price_avg']:.2f} ({zone['touches']} touches)"
        )

        # Mark each touch point with a marker
        for touch_idx in zone['touch_indices']:
            ax.plot(touch_idx, prices[touch_idx], 'o', color=zone_color, markersize=8, zorder=10)

        # Draw zone average line
        ax.axhline(
            y=zone['zone_price_avg'],
            color=zone_color,
            linewidth=2 if zone == signal_result.get('zone') else 1,
            linestyle='--',
            alpha=0.8 if zone == signal_result.get('zone') else 0.5
        )

    # Mark current price
    current_price = signal_result['current_price']
    ax.axhline(y=current_price, color='#E74C3C', linewidth=2.5, linestyle='-',
               label=f"Current Price: ${current_price:.2f}", alpha=0.9, zorder=15)

    # If buy signal triggered, show entry/stop/target
    if signal_result['signal'] == 'buy' and signal_result['zone']:
        targets = calculate_zone_based_targets(signal_result['zone'])

        ax.axhline(y=targets['entry_price'], color='#17A589', linewidth=2,
                   linestyle='-', label=f"Entry: ${targets['entry_price']:.2f}", alpha=0.9)
        ax.axhline(y=targets['stop_loss'], color='#C0392B', linewidth=2,
                   linestyle=':', label=f"Stop Loss: ${targets['stop_loss']:.2f}", alpha=0.9)
        ax.axhline(y=targets['profit_target'], color='#8E44AD', linewidth=2,
                   linestyle=':', label=f"Target: ${targets['profit_target']:.2f}", alpha=0.9)

        # Add text annotations
        ax.text(len(prices) * 0.02, targets['profit_target'],
                f" R/R: 2.5:1 (+{targets['reward_percentage']:.1f}%)",
                verticalalignment='bottom', fontsize=10, color='#8E44AD', fontweight='bold')
        ax.text(len(prices) * 0.02, targets['stop_loss'],
                f" Risk: -{targets['risk_percentage']:.1f}%",
                verticalalignment='top', fontsize=10, color='#C0392B', fontweight='bold')

    # Configure chart
    ax.set_xlabel('Time (hourly data points)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')

    signal_status = "BUY SIGNAL" if signal_result['signal'] == 'buy' else "NO SIGNAL"
    signal_color = '#27AE60' if signal_result['signal'] == 'buy' else '#E67E22'

    title = f"{symbol} - Range Support Strategy: {signal_status}\n"
    title += f"Reasoning: {signal_result['reasoning']}"
    ax.set_title(title, fontsize=14, fontweight='bold', color=signal_color)

    ax.legend(loc='upper left', fontsize='small', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"./screenshots/{symbol}_range-support-strategy.png"
    plt.savefig(filename, dpi=300)
    print(f"\nVisualization saved to: {filename}")
    plt.close()


def test_single_symbol(symbol, config_name="Moderate"):
    """
    Test the range strategy for a single symbol
    """
    print("=" * 80)
    print(f"TESTING: {symbol}")
    print("=" * 80)

    # Load price data from coinbase-data directory (last 30 days)
    prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=720)

    if not prices or len(prices) == 0:
        print(f"✗ ERROR: Could not load price data for {symbol}")
        print(f"  Make sure you have data in coinbase-data/{symbol}.json\n")
        return None

    current_price = prices[-1]  # Most recent price

    print(f"✓ Loaded {len(prices)} hourly price points")
    print(f"✓ Current price: ${current_price:.4f}")
    print(f"✓ Price range: ${min(prices):.4f} - ${max(prices):.4f}")

    # Use moderate configuration (recommended)
    config_params = {
        'min_touches': 2,
        'zone_tolerance_percentage': 3.0,
        'entry_tolerance_percentage': 1.5,
        'extrema_order': 5,
        'lookback_window': 336  # 14 days
    }

    print(f"\nRunning {config_name} strategy configuration...")
    print("-" * 80)

    signal_result = check_range_support_buy_signal(
        prices=prices,
        current_price=current_price,
        **config_params
    )

    print(f"Signal: {signal_result['signal'].upper()}")
    print(f"Reasoning: {signal_result['reasoning']}")
    print(f"Zones found: {len(signal_result['all_zones'])}")

    if signal_result['all_zones']:
        print(f"\nTop Support Zones Identified:")
        for i, zone in enumerate(signal_result['all_zones'][:3]):  # Show top 3
            print(f"  Zone {i+1}: ${zone['zone_price_avg']:.4f} "
                  f"(${zone['zone_price_min']:.4f} - ${zone['zone_price_max']:.4f}) "
                  f"- {zone['touches']} touches")

    if signal_result['signal'] == 'buy':
        print(f"\n{'='*80}")
        print(f"✓ BUY SIGNAL TRIGGERED FOR {symbol}!")
        print(f"{'='*80}")
        print(f"  Zone strength: {signal_result['zone_strength']} touches")
        print(f"  Distance from zone avg: {signal_result['distance_from_zone_avg']:+.2f}%")

        # Calculate entry/stop/target
        targets = calculate_zone_based_targets(signal_result['zone'])
        print(f"\nTrade Setup:")
        print(f"  Entry: ${targets['entry_price']:.4f}")
        print(f"  Stop Loss: ${targets['stop_loss']:.4f} (-{targets['risk_percentage']:.2f}%)")
        print(f"  Profit Target: ${targets['profit_target']:.4f} (+{targets['reward_percentage']:.2f}%)")
        print(f"  Risk/Reward: 2.5:1")

        # Generate visualization for buy signals
        print(f"\nGenerating chart for {symbol}...")
        visualize_support_zones(prices, signal_result, symbol)
    else:
        print(f"\n✗ No buy signal - waiting for price to reach support zone")

    print("\n")
    return signal_result


def main():
    """
    Main test function - tests the range support strategy on all enabled wallets
    """
    print("=" * 80)
    print("RANGE-BASED SUPPORT ZONE TRADING STRATEGY - MULTI-WALLET TEST")
    print("=" * 80)
    print()

    # Load config to get enabled wallets
    import json
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    print(f"Testing {len(enabled_symbols)} enabled wallets: {', '.join(enabled_symbols)}")
    print()

    all_results = {}

    for symbol in enabled_symbols:
        result = test_single_symbol(symbol, config_name="Moderate")
        if result:
            all_results[symbol] = result

    # Summary across all wallets
    print("=" * 80)
    print("SUMMARY - ALL WALLETS")
    print("=" * 80)

    buy_signals = [sym for sym, result in all_results.items() if result['signal'] == 'buy']

    if buy_signals:
        print(f"\n✓ BUY SIGNALS TRIGGERED: {len(buy_signals)}/{len(all_results)} wallets")
        print(f"\nAssets with buy signals:")
        for symbol in buy_signals:
            result = all_results[symbol]
            print(f"  - {symbol}: Zone strength {result['zone_strength']} touches, "
                  f"distance {result['distance_from_zone_avg']:+.2f}% from zone avg")
        print(f"\nRecommendation: Consider entering positions on assets with buy signals")
        print(f"Charts generated in ./screenshots/ directory")
    else:
        print(f"\n✗ No BUY signals from any wallet")
        print(f"\nRecommendation: Wait for prices to reach support zones before entering")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
