#!/usr/bin/env python3
"""
Test the adaptive mean reversion strategy implementation
"""

from utils.adaptive_mean_reversion import check_adaptive_buy_signal, detect_market_trend
from utils.file_helpers import get_property_values_from_crypto_file
import json

def test_adaptive_strategy():
    print("="*80)
    print("TESTING ADAPTIVE MEAN REVERSION STRATEGY")
    print("="*80)
    print()

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    for symbol in enabled_symbols:
        print(f"\n{'='*80}")
        print(f"Testing {symbol}")
        print(f"{'='*80}\n")

        # Get price data
        prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=200)

        if not prices or len(prices) < 48:
            print(f"✗ Insufficient data for {symbol}")
            continue

        current_price = prices[-1]

        # Test trend detection
        trend = detect_market_trend(prices, lookback=168)
        print(f"📊 Market Trend (1-week): {trend.upper()}")

        # Test buy signal
        signal = check_adaptive_buy_signal(prices, current_price)

        print(f"\n🎯 Strategy Signal: {signal['signal'].upper()}")
        print(f"   Trend: {signal['trend']}")
        print(f"   Deviation from 24h MA: {signal['deviation_from_ma']:+.2f}%")
        print(f"\n   Reasoning: {signal['reasoning']}")

        if signal['signal'] == 'buy':
            print(f"\n✅ BUY SIGNAL DETECTED!")
            print(f"   Entry Price: ${signal['entry_price']:.4f}")
            print(f"   Stop Loss: ${signal['stop_loss']:.4f} (-1.7%)")
            print(f"   Profit Target: ${signal['profit_target']:.4f} (+1.7%)")
        else:
            print(f"\n⚠️  No buy signal - waiting for better setup")

    print("\n" + "="*80)
    print("✅ Adaptive strategy test complete!")
    print("="*80)

if __name__ == "__main__":
    test_adaptive_strategy()
