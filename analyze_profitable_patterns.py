#!/usr/bin/env python3
"""
Analyze the last 5 weeks to find profitable trading patterns
"""

import json
import numpy as np
from utils.file_helpers import get_property_values_from_crypto_file

def analyze_market_patterns(symbol, period_hours=840):
    """Analyze what patterns would have been profitable"""

    print(f"\n{'='*80}")
    print(f"PATTERN ANALYSIS: {symbol} (Last {period_hours//24} days)")
    print(f"{'='*80}\n")

    # Load price data
    all_prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=4380)

    if not all_prices or len(all_prices) < period_hours:
        print(f"✗ Insufficient data")
        return None

    # Get the 5-week period
    prices = all_prices[-period_hours:]

    # Basic stats
    start_price = prices[0]
    end_price = prices[-1]
    min_price = min(prices)
    max_price = max(prices)

    total_return = ((end_price - start_price) / start_price) * 100
    max_drawdown = ((min_price - max_price) / max_price) * 100
    volatility = np.std(prices) / np.mean(prices) * 100

    print(f"📊 Market Overview:")
    print(f"  Start: ${start_price:.4f}")
    print(f"  End: ${end_price:.4f}")
    print(f"  Return: {total_return:+.2f}%")
    print(f"  Range: ${min_price:.4f} - ${max_price:.4f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Volatility: {volatility:.2f}%")

    # Analyze trend direction
    weekly_returns = []
    for i in range(0, len(prices), 168):  # 168 hours = 1 week
        if i + 168 < len(prices):
            week_start = prices[i]
            week_end = prices[i + 168]
            weekly_return = ((week_end - week_start) / week_start) * 100
            weekly_returns.append(weekly_return)

    print(f"\n📈 Weekly Returns:")
    for i, ret in enumerate(weekly_returns, 1):
        print(f"  Week {i}: {ret:+.2f}%")

    # Determine overall trend
    up_weeks = sum(1 for r in weekly_returns if r > 0)
    down_weeks = sum(1 for r in weekly_returns if r < 0)

    if down_weeks > up_weeks:
        trend = "DOWNTREND"
    elif up_weeks > down_weeks:
        trend = "UPTREND"
    else:
        trend = "SIDEWAYS"

    print(f"\n🎯 Market Trend: {trend}")
    print(f"  Up weeks: {up_weeks}/{len(weekly_returns)}")
    print(f"  Down weeks: {down_weeks}/{len(weekly_returns)}")

    # Test simple strategies
    print(f"\n💡 Strategy Simulations:")

    # 1. Buy and Hold
    buy_hold_return = total_return
    print(f"\n  1. Buy & Hold: {buy_hold_return:+.2f}%")

    # 2. Mean Reversion (buy dips, sell rallies)
    mean_reversion_trades = simulate_mean_reversion(prices)
    print(f"  2. Mean Reversion: {mean_reversion_trades['total_return']:+.2f}% ({mean_reversion_trades['total_trades']} trades, {mean_reversion_trades['win_rate']:.1f}% win rate)")

    # 3. Trend Following (buy breakouts, ride momentum)
    trend_following_trades = simulate_trend_following(prices)
    print(f"  3. Trend Following: {trend_following_trades['total_return']:+.2f}% ({trend_following_trades['total_trades']} trades, {trend_following_trades['win_rate']:.1f}% win rate)")

    # 4. Momentum (buy strength, sell weakness)
    momentum_trades = simulate_momentum(prices)
    print(f"  4. Momentum: {momentum_trades['total_return']:+.2f}% ({momentum_trades['total_trades']} trades, {momentum_trades['win_rate']:.1f}% win rate)")

    # Recommendation
    strategies = {
        'Buy & Hold': buy_hold_return,
        'Mean Reversion': mean_reversion_trades['total_return'],
        'Trend Following': trend_following_trades['total_return'],
        'Momentum': momentum_trades['total_return']
    }

    best_strategy = max(strategies.items(), key=lambda x: x[1])

    print(f"\n✅ Best Strategy: {best_strategy[0]} ({best_strategy[1]:+.2f}%)")

    return {
        'symbol': symbol,
        'trend': trend,
        'best_strategy': best_strategy[0],
        'best_return': best_strategy[1],
        'all_strategies': strategies
    }


def simulate_mean_reversion(prices, buy_threshold=-0.02, sell_threshold=0.015):
    """
    Mean reversion: Buy when price drops 2%+ from 24h MA, sell at 1.5% profit
    """
    trades = []
    position = None
    position_size = 1000

    for i in range(24, len(prices)):
        current_price = prices[i]
        ma_24h = np.mean(prices[i-24:i])

        # Check if in position
        if position:
            profit_pct = (current_price - position['entry']) / position['entry']

            # Take profit at 1.5%
            if profit_pct >= sell_threshold:
                trades.append({
                    'entry': position['entry'],
                    'exit': current_price,
                    'return': profit_pct * 100,
                    'duration': i - position['entry_time']
                })
                position = None
            # Stop loss at -2%
            elif profit_pct <= -0.02:
                trades.append({
                    'entry': position['entry'],
                    'exit': current_price,
                    'return': profit_pct * 100,
                    'duration': i - position['entry_time']
                })
                position = None
        else:
            # Enter when price is 2%+ below MA
            deviation = (current_price - ma_24h) / ma_24h
            if deviation <= buy_threshold:
                position = {'entry': current_price, 'entry_time': i}

    # Calculate stats
    if not trades:
        return {'total_return': 0, 'total_trades': 0, 'win_rate': 0}

    wins = [t for t in trades if t['return'] > 0]
    total_return = sum(t['return'] for t in trades) / len(trades)
    win_rate = (len(wins) / len(trades)) * 100

    return {
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate
    }


def simulate_trend_following(prices, lookback=48):
    """
    Trend following: Buy when price crosses above 48h MA, sell when crosses below
    """
    trades = []
    position = None

    for i in range(lookback + 1, len(prices)):
        current_price = prices[i]
        prev_price = prices[i-1]
        ma = np.mean(prices[i-lookback:i])
        prev_ma = np.mean(prices[i-lookback-1:i-1])

        # Check if in position
        if position:
            # Exit when price crosses below MA
            if prev_price >= prev_ma and current_price < ma:
                profit_pct = (current_price - position['entry']) / position['entry']
                trades.append({
                    'entry': position['entry'],
                    'exit': current_price,
                    'return': profit_pct * 100,
                    'duration': i - position['entry_time']
                })
                position = None
        else:
            # Enter when price crosses above MA
            if prev_price < prev_ma and current_price >= ma:
                position = {'entry': current_price, 'entry_time': i}

    # Calculate stats
    if not trades:
        return {'total_return': 0, 'total_trades': 0, 'win_rate': 0}

    wins = [t for t in trades if t['return'] > 0]
    total_return = sum(t['return'] for t in trades) / len(trades)
    win_rate = (len(wins) / len(trades)) * 100

    return {
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate
    }


def simulate_momentum(prices, lookback=12):
    """
    Momentum: Buy when 12h return > 1%, sell when momentum fades
    """
    trades = []
    position = None

    for i in range(lookback + 1, len(prices)):
        current_price = prices[i]
        price_12h_ago = prices[i - lookback]
        momentum = (current_price - price_12h_ago) / price_12h_ago

        # Check if in position
        if position:
            profit_pct = (current_price - position['entry']) / position['entry']

            # Exit when momentum fades (12h return < 0.5%) or profit >= 2%
            if momentum < 0.005 or profit_pct >= 0.02:
                trades.append({
                    'entry': position['entry'],
                    'exit': current_price,
                    'return': profit_pct * 100,
                    'duration': i - position['entry_time']
                })
                position = None
            # Stop loss at -1.5%
            elif profit_pct <= -0.015:
                trades.append({
                    'entry': position['entry'],
                    'exit': current_price,
                    'return': profit_pct * 100,
                    'duration': i - position['entry_time']
                })
                position = None
        else:
            # Enter when 12h momentum > 1%
            if momentum > 0.01:
                position = {'entry': current_price, 'entry_time': i}

    # Calculate stats
    if not trades:
        return {'total_return': 0, 'total_trades': 0, 'win_rate': 0}

    wins = [t for t in trades if t['return'] > 0]
    total_return = sum(t['return'] for t in trades) / len(trades)
    win_rate = (len(wins) / len(trades)) * 100

    return {
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate
    }


def main():
    print("="*80)
    print("PROFITABLE PATTERN ANALYSIS (Last 5 Weeks)")
    print("="*80)

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    all_results = {}

    for symbol in enabled_symbols:
        result = analyze_market_patterns(symbol, period_hours=840)
        if result:
            all_results[symbol] = result

    # Overall recommendation
    print(f"\n{'='*80}")
    print("OVERALL RECOMMENDATION")
    print(f"{'='*80}\n")

    for symbol, result in all_results.items():
        print(f"{symbol}:")
        print(f"  Market: {result['trend']}")
        print(f"  Best Strategy: {result['best_strategy']} ({result['best_return']:+.2f}%)")
        print()


if __name__ == "__main__":
    main()
