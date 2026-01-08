#!/usr/bin/env python3
"""
Simple Backtest for Adaptive AI Strategy
No external dependencies except standard library
"""

import json
import os
from datetime import datetime


def load_price_data(symbol):
    """Load historical price data for a symbol."""
    filepath = f"coinbase-data/{symbol}.json"
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract prices in chronological order
    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    prices = [float(entry['price']) for entry in data_sorted]

    return prices


def detect_trend(prices, lookback=168):
    """Detect market trend (uptrend/downtrend/sideways)."""
    if len(prices) < lookback:
        return 'sideways'

    recent = prices[-lookback:]
    start = recent[0]
    end = recent[-1]
    price_change = (end - start) / start

    # Calculate volatility
    price_range = (max(recent) - min(recent)) / min(recent)

    # Trend detection
    if price_change > 0.05 and price_range > 0.10:
        return 'uptrend'
    elif price_change < -0.05 and price_range > 0.10:
        return 'downtrend'
    else:
        return 'sideways'


def check_buy_signal(prices, current_price):
    """Check for AMR buy signal."""
    if len(prices) < 48:
        return None

    # Detect trend
    trend = detect_trend(prices, lookback=168)

    # Skip downtrends
    if trend == 'downtrend':
        return None

    # Calculate 24h MA
    ma_24h = sum(prices[-24:]) / 24

    # Calculate deviation
    deviation_from_ma = (current_price - ma_24h) / ma_24h

    # BUY SIGNAL: 2-3% below MA in uptrend/sideways
    if -0.03 <= deviation_from_ma <= -0.02:
        return {
            'entry': current_price,
            'stop': current_price * 0.983,  # 1.7% stop
            'target': current_price * 1.017,  # 1.7% profit
            'trend': trend,
            'deviation_pct': deviation_from_ma * 100
        }

    return None


def backtest_symbol(symbol, max_hours=4320):  # 180 days = 4320 hours
    """Backtest a single symbol."""
    prices = load_price_data(symbol)

    if not prices or len(prices) < 200:
        return None

    # Use only last max_hours
    if len(prices) > max_hours:
        prices = prices[-max_hours:]

    trades = []
    position = None

    # Start from hour 168 (need data for trend detection)
    for i in range(168, len(prices)):
        current_price = prices[i]
        historical = prices[:i]

        # ENTRY
        if position is None:
            signal = check_buy_signal(historical, current_price)
            if signal:
                position = {
                    'entry_idx': i,
                    'entry_price': signal['entry'],
                    'stop': signal['stop'],
                    'target': signal['target'],
                    'trend': signal['trend']
                }

        # EXIT
        elif position:
            # Stop loss
            if current_price <= position['stop']:
                pnl = ((current_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl_pct': pnl,
                    'outcome': 'loss',
                    'trend': position['trend'],
                    'hours_held': i - position['entry_idx']
                })
                position = None

            # Profit target
            elif current_price >= position['target']:
                pnl = ((current_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl_pct': pnl,
                    'outcome': 'win',
                    'trend': position['trend'],
                    'hours_held': i - position['entry_idx']
                })
                position = None

    # Close remaining position
    if position:
        final_price = prices[-1]
        pnl = ((final_price - position['entry_price']) / position['entry_price']) * 100
        outcome = 'win' if pnl > 0 else 'loss'
        trades.append({
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'pnl_pct': pnl,
            'outcome': outcome,
            'trend': position['trend'],
            'hours_held': len(prices) - 1 - position['entry_idx']
        })

    if not trades:
        return None

    # Calculate stats
    wins = [t for t in trades if t['outcome'] == 'win']
    losses = [t for t in trades if t['outcome'] == 'loss']

    win_rate = (len(wins) / len(trades)) * 100
    total_pnl = sum(t['pnl_pct'] for t in trades)
    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0

    # By trend
    uptrend = [t for t in trades if t['trend'] == 'uptrend']
    sideways = [t for t in trades if t['trend'] == 'sideways']

    uptrend_wins = len([t for t in uptrend if t['outcome'] == 'win'])
    sideways_wins = len([t for t in sideways if t['outcome'] == 'win'])

    uptrend_wr = (uptrend_wins / len(uptrend)) * 100 if uptrend else 0
    sideways_wr = (sideways_wins / len(sideways)) * 100 if sideways else 0

    avg_hold_hours = sum(t['hours_held'] for t in trades) / len(trades)

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'uptrend_trades': len(uptrend),
        'uptrend_wr': uptrend_wr,
        'sideways_trades': len(sideways),
        'sideways_wr': sideways_wr,
        'avg_hold_hours': avg_hold_hours,
        'trades': trades
    }


def main():
    symbols = [
        'ADA-USD', 'ALGO-USD', 'AVAX-USD', 'BTC-USD', 'DOGE-USD',
        'ETH-USD', 'LINK-USD', 'LTC-USD', 'SOL-USD', 'XRP-USD'
    ]

    print("\n" + "="*120)
    print("ADAPTIVE AI STRATEGY BACKTEST - 180 DAYS")
    print("="*120)

    results = []

    for symbol in symbols:
        print(f"\nBacktesting {symbol}...", end=" ", flush=True)
        result = backtest_symbol(symbol, max_hours=4320)

        if result:
            results.append(result)
            print(f"✓ {result['total_trades']} trades, {result['win_rate']:.1f}% WR, {result['total_pnl']:+.2f}% total P/L")
        else:
            print("⚠️  No trades")

    # Sort by total P/L
    results.sort(key=lambda x: x['total_pnl'], reverse=True)

    # Print comparison table
    print("\n" + "="*120)
    print("RESULTS COMPARISON")
    print("="*120)
    print(f"{'Rank':<6} {'Symbol':<12} {'Trades':<8} {'Win Rate':<10} {'Total P/L':<12} {'Avg Win':<10} {'Avg Loss':<10} {'Avg Hold':<10}")
    print("-"*120)

    for i, r in enumerate(results, 1):
        print(f"#{i:<5} {r['symbol']:<12} {r['total_trades']:<8} {r['win_rate']:>6.1f}%   "
              f"{r['total_pnl']:>+9.2f}%   {r['avg_win']:>+6.2f}%   {r['avg_loss']:>+6.2f}%   "
              f"{r['avg_hold_hours']:>6.1f}h")

    # Overall stats
    total_trades = sum(r['total_trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    overall_wr = (total_wins / total_trades) * 100 if total_trades else 0
    overall_pnl = sum(r['total_pnl'] for r in results)

    print("-"*120)
    print(f"OVERALL: {len(results)} symbols | {total_trades} trades | {overall_wr:.1f}% win rate | {overall_pnl:+.2f}% total P/L")
    print("="*120)

    # Top performers
    print("\n🏆 TOP 5 PERFORMERS (by total P/L):")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r['symbol']:<12} {r['total_pnl']:>+8.2f}% ({r['total_trades']} trades, {r['win_rate']:.1f}% WR)")
        print(f"     └─ Uptrend: {r['uptrend_trades']} trades ({r['uptrend_wr']:.1f}% WR) | "
              f"Sideways: {r['sideways_trades']} trades ({r['sideways_wr']:.1f}% WR)")

    # Worst performers
    print("\n📉 BOTTOM 3 PERFORMERS:")
    for i, r in enumerate(results[-3:], 1):
        print(f"  {i}. {r['symbol']:<12} {r['total_pnl']:>+8.2f}% ({r['total_trades']} trades, {r['win_rate']:.1f}% WR)")

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'period_days': 180,
        'results': results,
        'overall': {
            'total_trades': total_trades,
            'win_rate': overall_wr,
            'total_pnl': overall_pnl
        }
    }

    with open('backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n📊 Detailed results saved to backtest_results.json\n")


if __name__ == '__main__':
    main()
