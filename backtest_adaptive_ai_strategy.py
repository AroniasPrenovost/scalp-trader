#!/usr/bin/env python3
"""
Backtest Adaptive AI Strategy

Tests the unified Adaptive AI strategy against historical wallet data to:
1. Measure performance (win rate, total P/L, risk-adjusted returns)
2. Identify optimal coins to monitor
3. Validate that the unified approach improves on separate strategies

Usage:
    python backtest_adaptive_ai_strategy.py --days 30
    python backtest_adaptive_ai_strategy.py --symbols BTC-USD,ETH-USD
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.adaptive_mean_reversion import check_adaptive_buy_signal, detect_market_trend
from utils.file_helpers import get_property_values_from_crypto_file


def backtest_symbol(symbol, lookback_days=30, data_retention_hours=720):
    """
    Backtest the Adaptive AI strategy on a single symbol.

    Args:
        symbol: Trading pair symbol (e.g. 'BTC-USD')
        lookback_days: Number of days to backtest
        data_retention_hours: Max age of historical data to use

    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*80}")
    print(f"Backtesting {symbol} (last {lookback_days} days)")
    print(f"{'='*80}")

    # Get historical price data
    try:
        prices = get_property_values_from_crypto_file(
            'coinbase-data',
            symbol,
            'price',
            max_age_hours=data_retention_hours
        )
    except Exception as e:
        print(f"⚠️  Error loading data for {symbol}: {e}")
        return None

    if not prices or len(prices) < 168:  # Need at least 1 week for trend detection
        print(f"⚠️  Insufficient data for {symbol}: {len(prices) if prices else 0} points")
        return None

    print(f"✓ Loaded {len(prices)} hourly price points")

    # Backtest parameters
    trades = []
    current_position = None
    lookback_hours = lookback_days * 24

    # Start from hour 168 (need 1 week for trend detection)
    for i in range(168, min(len(prices), lookback_hours)):
        current_price = prices[i]
        historical_prices = prices[:i]

        # Check for AMR signal
        amr_signal = check_adaptive_buy_signal(historical_prices, current_price)

        # ENTRY LOGIC
        if current_position is None and amr_signal and amr_signal['signal'] == 'buy':
            # Enter position
            current_position = {
                'entry_index': i,
                'entry_price': current_price,
                'stop_loss': amr_signal['stop_loss'],
                'profit_target': amr_signal['profit_target'],
                'trend': amr_signal['trend'],
                'deviation_from_ma': amr_signal['deviation_from_ma']
            }
            print(f"  [{i:4d}] ENTRY: ${current_price:.4f} | Trend: {amr_signal['trend']} | Deviation: {amr_signal['deviation_from_ma']:+.2f}%")

        # EXIT LOGIC
        elif current_position is not None:
            entry_price = current_position['entry_price']
            stop_loss = current_position['stop_loss']
            profit_target = current_position['profit_target']

            # Check stop loss
            if current_price <= stop_loss:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                trades.append({
                    'entry_index': current_position['entry_index'],
                    'exit_index': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'exit_reason': 'stop_loss',
                    'pnl_pct': pnl_pct,
                    'outcome': 'loss',
                    'trend': current_position['trend'],
                    'deviation_from_ma': current_position['deviation_from_ma']
                })
                print(f"  [{i:4d}] STOP LOSS: ${current_price:.4f} | P/L: {pnl_pct:+.2f}%")
                current_position = None

            # Check profit target
            elif current_price >= profit_target:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                trades.append({
                    'entry_index': current_position['entry_index'],
                    'exit_index': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'exit_reason': 'profit_target',
                    'pnl_pct': pnl_pct,
                    'outcome': 'win',
                    'trend': current_position['trend'],
                    'deviation_from_ma': current_position['deviation_from_ma']
                })
                print(f"  [{i:4d}] PROFIT TARGET: ${current_price:.4f} | P/L: {pnl_pct:+.2f}%")
                current_position = None

    # Close any remaining position at current price
    if current_position is not None:
        final_price = prices[-1]
        entry_price = current_position['entry_price']
        pnl_pct = ((final_price - entry_price) / entry_price) * 100
        outcome = 'win' if pnl_pct > 0 else 'loss'

        trades.append({
            'entry_index': current_position['entry_index'],
            'exit_index': len(prices) - 1,
            'entry_price': entry_price,
            'exit_price': final_price,
            'exit_reason': 'backtest_end',
            'pnl_pct': pnl_pct,
            'outcome': outcome,
            'trend': current_position['trend'],
            'deviation_from_ma': current_position['deviation_from_ma']
        })
        print(f"  [END] CLOSE POSITION: ${final_price:.4f} | P/L: {pnl_pct:+.2f}%")

    # Calculate summary statistics
    if not trades:
        print(f"\n⚠️  No trades generated for {symbol}")
        return {
            'symbol': symbol,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl_pct': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'max_win_pct': 0,
            'max_loss_pct': 0,
            'trades': []
        }

    wins = [t for t in trades if t['outcome'] == 'win']
    losses = [t for t in trades if t['outcome'] == 'loss']

    win_rate = (len(wins) / len(trades)) * 100 if trades else 0
    total_pnl_pct = sum(t['pnl_pct'] for t in trades)
    avg_win_pct = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss_pct = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
    max_win_pct = max(t['pnl_pct'] for t in wins) if wins else 0
    max_loss_pct = min(t['pnl_pct'] for t in losses) if losses else 0

    # Breakdown by trend
    uptrend_trades = [t for t in trades if t['trend'] == 'uptrend']
    sideways_trades = [t for t in trades if t['trend'] == 'sideways']

    uptrend_wins = len([t for t in uptrend_trades if t['outcome'] == 'win'])
    sideways_wins = len([t for t in sideways_trades if t['outcome'] == 'win'])

    uptrend_win_rate = (uptrend_wins / len(uptrend_trades)) * 100 if uptrend_trades else 0
    sideways_win_rate = (sideways_wins / len(sideways_trades)) * 100 if sideways_trades else 0

    print(f"\n{'─'*80}")
    print(f"RESULTS FOR {symbol}:")
    print(f"{'─'*80}")
    print(f"Total Trades: {len(trades)}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P/L: {total_pnl_pct:+.2f}%")
    print(f"Avg Win: {avg_win_pct:+.2f}% | Avg Loss: {avg_loss_pct:+.2f}%")
    print(f"Max Win: {max_win_pct:+.2f}% | Max Loss: {max_loss_pct:+.2f}%")
    print(f"\nBy Trend:")
    print(f"  Uptrend: {len(uptrend_trades)} trades, {uptrend_win_rate:.1f}% win rate")
    print(f"  Sideways: {len(sideways_trades)} trades, {sideways_win_rate:.1f}% win rate")

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl_pct': total_pnl_pct,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'max_win_pct': max_win_pct,
        'max_loss_pct': max_loss_pct,
        'uptrend_trades': len(uptrend_trades),
        'uptrend_win_rate': uptrend_win_rate,
        'sideways_trades': len(sideways_trades),
        'sideways_win_rate': sideways_win_rate,
        'trades': trades
    }


def backtest_multiple_symbols(symbols, lookback_days=30):
    """
    Backtest multiple symbols and rank them by performance.

    Args:
        symbols: List of trading pair symbols
        lookback_days: Number of days to backtest

    Returns:
        List of backtest results sorted by total P/L
    """
    results = []

    for symbol in symbols:
        result = backtest_symbol(symbol, lookback_days=lookback_days)
        if result and result['total_trades'] > 0:
            results.append(result)

    # Sort by total P/L
    results.sort(key=lambda x: x['total_pnl_pct'], reverse=True)

    return results


def print_comparison_report(results):
    """
    Print a comparison report across all backtested symbols.

    Args:
        results: List of backtest result dictionaries
    """
    print("\n" + "="*100)
    print("BACKTEST COMPARISON - ADAPTIVE AI STRATEGY")
    print("="*100)

    if not results:
        print("No results to display.")
        return

    print(f"{'Rank':<6} {'Symbol':<12} {'Trades':<8} {'Win Rate':<12} {'Total P/L':<12} {'Avg Win':<12} {'Avg Loss':<12}")
    print("-"*100)

    for i, result in enumerate(results, 1):
        rank = f"#{i}"
        symbol = result['symbol']
        trades = result['total_trades']
        win_rate = f"{result['win_rate']:.1f}%"
        total_pnl = f"{result['total_pnl_pct']:+.2f}%"
        avg_win = f"{result['avg_win_pct']:+.2f}%"
        avg_loss = f"{result['avg_loss_pct']:+.2f}%"

        print(f"{rank:<6} {symbol:<12} {trades:<8} {win_rate:<12} {total_pnl:<12} {avg_win:<12} {avg_loss:<12}")

    # Overall stats
    total_trades = sum(r['total_trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    overall_win_rate = (total_wins / total_trades) * 100 if total_trades else 0
    overall_pnl = sum(r['total_pnl_pct'] for r in results)

    print("-"*100)
    print(f"OVERALL: {len(results)} symbols, {total_trades} trades, {overall_win_rate:.1f}% win rate, {overall_pnl:+.2f}% total P/L")
    print("="*100 + "\n")

    # Top performers
    print("🏆 TOP 5 PERFORMERS (by total P/L):")
    for i, result in enumerate(results[:5], 1):
        print(f"  {i}. {result['symbol']}: {result['total_pnl_pct']:+.2f}% ({result['total_trades']} trades, {result['win_rate']:.1f}% win rate)")

    print()


def main():
    parser = argparse.ArgumentParser(description='Backtest Adaptive AI Strategy')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest (default: 30)')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to backtest (e.g., BTC-USD,ETH-USD)')
    parser.add_argument('--all', action='store_true', help='Backtest all available symbols in coinbase-data/')

    args = parser.parse_args()

    # Determine which symbols to backtest
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    elif args.all:
        # Find all available symbols in coinbase-data directory
        data_dir = 'coinbase-data'
        if not os.path.exists(data_dir):
            print(f"Error: {data_dir} directory not found")
            sys.exit(1)

        symbols = []
        for filename in os.listdir(data_dir):
            if filename.endswith('_data.json'):
                symbol = filename.replace('_data.json', '').replace('_', '-')
                symbols.append(symbol)

        if not symbols:
            print(f"No data files found in {data_dir}")
            sys.exit(1)

        print(f"Found {len(symbols)} symbols to backtest: {', '.join(symbols)}")
    else:
        # Default: backtest a few common symbols
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD']
        print(f"Backtesting default symbols: {', '.join(symbols)}")

    # Run backtest
    results = backtest_multiple_symbols(symbols, lookback_days=args.days)

    # Print comparison report
    print_comparison_report(results)

    # Save results to file
    output_file = 'backtest_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'backtest_date': datetime.now().isoformat(),
            'lookback_days': args.days,
            'results': results
        }, f, indent=2)

    print(f"📊 Results saved to {output_file}")


if __name__ == '__main__':
    main()
