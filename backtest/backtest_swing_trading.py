#!/usr/bin/env python3
"""
Backtest Swing Trading Strategy

Strategy:
- Hold 2-10 days for 3-5% moves
- Max 2 concurrent positions ($2,250 each)
- Target: 3% NET profit (~4.5% gross)
- Stop: 1.5% GROSS
- Entry: Pullback to 7-day SMA in strong uptrend
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import sys
sys.path.append(os.path.dirname(__file__))

from utils.swing_trading_strategy import (
    check_swing_entry,
    check_swing_exit,
    get_strategy_info
)

# TRADING COSTS
TAKER_FEE = 0.0025
TAX_RATE = 0.24
POSITION_SIZE_USD = 2250  # Half of $4,500 (max 2 positions)
MIN_NET_PROFIT_USD = 2.00
MAX_CONCURRENT_POSITIONS = 2


def load_price_volume_data(symbol: str) -> Optional[tuple]:
    """Load historical price and volume data."""
    filepath = f"coinbase-data/{symbol}.json"
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    prices = [float(entry['price']) for entry in data_sorted]
    volumes = [float(entry['volume_24h']) for entry in data_sorted]
    timestamps = [entry['timestamp'] for entry in data_sorted]

    return prices, volumes, timestamps


def calculate_net_profit(entry_price: float, exit_price: float, position_size: float) -> Dict:
    """Calculate net profit with fees and taxes."""
    cost_basis = position_size
    shares = position_size / entry_price
    current_value = exit_price * shares

    gross_profit = current_value - cost_basis
    exit_fee = TAKER_FEE * current_value
    capital_gain = current_value - cost_basis
    tax = TAX_RATE * capital_gain if capital_gain > 0 else 0
    net_profit = current_value - cost_basis - exit_fee - tax
    net_pct = (net_profit / cost_basis) * 100
    gross_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'gross_pnl_pct': gross_pct,
        'net_pnl_pct': net_pct,
        'net_pnl_usd': net_profit,
        'profitable': net_profit >= MIN_NET_PROFIT_USD
    }


def backtest_swing_trading(symbols: List[str],
                          target_net_pct: float = 3.0,
                          stop_gross_pct: float = 1.5,
                          max_hold_hours: int = 240,
                          max_hours: int = 4320) -> Optional[Dict]:
    """
    Backtest swing trading strategy across multiple symbols with position limits.

    Args:
        symbols: List of trading pairs
        target_net_pct: NET profit target (default 3%)
        stop_gross_pct: GROSS stop loss (default 1.5%)
        max_hold_hours: Max holding time (default 240h = 10 days)
        max_hours: Historical data to analyze (default 4320h = 180 days)
    """

    # Load data for all symbols
    symbol_data = {}
    for symbol in symbols:
        data = load_price_volume_data(symbol)
        if data:
            prices, volumes, timestamps = data

            # Limit to max_hours
            if len(prices) > max_hours:
                prices = prices[-max_hours:]
                volumes = volumes[-max_hours:]
                timestamps = timestamps[-max_hours:]

            symbol_data[symbol] = {
                'prices': prices,
                'volumes': volumes,
                'timestamps': timestamps
            }

    if not symbol_data:
        return None

    # Find common length (shortest dataset)
    min_length = min(len(data['prices']) for data in symbol_data.values())

    trades = []
    positions = {}  # symbol -> position_dict

    # Track rejection reasons
    signals_rejected = {
        'no_uptrend': 0,
        'weak_trend': 0,
        'no_bounce': 0,
        'too_extended': 0,
        'rsi_extreme': 0,
        'low_volume': 0,
        'position_limit': 0
    }

    # Iterate through time
    for i in range(30 * 24, min_length):  # Start from hour 720 (30 days for SMAs)

        # EXIT CHECK: Check all open positions first
        positions_to_close = []
        for symbol, position in positions.items():
            current_price = symbol_data[symbol]['prices'][i]
            hours_held = i - position['entry_idx']
            historical_prices = symbol_data[symbol]['prices'][:i+1]

            exit_signal = check_swing_exit(
                prices=historical_prices,
                entry_price=position['entry_price'],
                current_price=current_price,
                hours_held=hours_held,
                target_price=position['target'],
                stop_price=position['stop'],
                entry_sma7=position['entry_sma7'],
                max_hold_hours=max_hold_hours
            )

            if exit_signal and exit_signal.get('exit'):
                exit_price = exit_signal['exit_price']
                exit_reason = exit_signal['exit_reason']

                pnl = calculate_net_profit(position['entry_price'], exit_price, POSITION_SIZE_USD)

                trades.append({
                    'symbol': symbol,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': symbol_data[symbol]['timestamps'][i],
                    'gross_pnl_pct': pnl['gross_pnl_pct'],
                    'net_pnl_pct': pnl['net_pnl_pct'],
                    'net_pnl_usd': pnl['net_pnl_usd'],
                    'profitable': pnl['profitable'],
                    'exit_reason': exit_reason,
                    'confidence': position['confidence'],
                    'hours_held': hours_held,
                    'days_held': hours_held / 24
                })
                positions_to_close.append(symbol)

        # Close positions
        for symbol in positions_to_close:
            del positions[symbol]

        # ENTRY CHECK: Look for new entries (only if < 2 positions)
        if len(positions) < MAX_CONCURRENT_POSITIONS:
            for symbol in symbols:
                # Skip if already have position in this symbol
                if symbol in positions:
                    continue

                # Skip if no data
                if symbol not in symbol_data:
                    continue

                current_price = symbol_data[symbol]['prices'][i]
                current_volume = symbol_data[symbol]['volumes'][i]
                historical_prices = symbol_data[symbol]['prices'][:i+1]
                historical_volumes = symbol_data[symbol]['volumes'][:i+1]

                signal = check_swing_entry(
                    prices=historical_prices,
                    volumes=historical_volumes,
                    current_price=current_price,
                    current_volume=current_volume,
                    target_net_pct=target_net_pct,
                    stop_gross_pct=stop_gross_pct,
                    entry_fee_pct=TAKER_FEE * 100,
                    exit_fee_pct=TAKER_FEE * 100,
                    tax_rate_pct=TAX_RATE * 100
                )

                # Track rejections
                if signal and 'reason' in signal:
                    reason = signal['reason'].lower()
                    if 'uptrend' in reason or 'trend' in reason:
                        if 'weak' in reason or 'score' in reason:
                            signals_rejected['weak_trend'] += 1
                        else:
                            signals_rejected['no_uptrend'] += 1
                    elif 'bounce' in reason:
                        signals_rejected['no_bounce'] += 1
                    elif 'extended' in reason or 'above' in reason or 'ran' in reason:
                        signals_rejected['too_extended'] += 1
                    elif 'rsi' in reason:
                        signals_rejected['rsi_extreme'] += 1
                    elif 'volume' in reason:
                        signals_rejected['low_volume'] += 1

                # Enter position if signal is good and we have room
                if signal and signal.get('signal') == 'buy':
                    if len(positions) < MAX_CONCURRENT_POSITIONS:
                        positions[symbol] = {
                            'entry_idx': i,
                            'entry_price': signal['entry_price'],
                            'entry_timestamp': symbol_data[symbol]['timestamps'][i],
                            'stop': signal['stop_loss'],
                            'target': signal['profit_target'],
                            'entry_sma7': signal['metrics']['sma7'],
                            'confidence': signal.get('confidence', 'medium'),
                            'trend_score': signal['trend_score']
                        }
                    else:
                        signals_rejected['position_limit'] += 1
                        break  # Stop checking other symbols this hour

    # Close remaining positions at end
    for symbol, position in positions.items():
        final_price = symbol_data[symbol]['prices'][-1]
        hours_held = len(symbol_data[symbol]['prices']) - 1 - position['entry_idx']
        pnl = calculate_net_profit(position['entry_price'], final_price, POSITION_SIZE_USD)

        trades.append({
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': symbol_data[symbol]['timestamps'][-1],
            'gross_pnl_pct': pnl['gross_pnl_pct'],
            'net_pnl_pct': pnl['net_pnl_pct'],
            'net_pnl_usd': pnl['net_pnl_usd'],
            'profitable': pnl['profitable'],
            'exit_reason': 'eod',
            'confidence': position['confidence'],
            'hours_held': hours_held,
            'days_held': hours_held / 24
        })

    if not trades:
        return {
            'total_trades': 0,
            'signals_rejected': signals_rejected
        }

    # Calculate stats
    total_trades = len(trades)
    profitable_trades = [t for t in trades if t['profitable']]
    wins = [t for t in trades if t['net_pnl_usd'] > 0]
    losses = [t for t in trades if t['net_pnl_usd'] <= 0]

    # By exit reason
    target_exits = [t for t in trades if t['exit_reason'] == 'target']
    stop_exits = [t for t in trades if t['exit_reason'] == 'stop_loss']
    trend_break_exits = [t for t in trades if t['exit_reason'] == 'trend_break']
    timeout_exits = [t for t in trades if t['exit_reason'] == 'max_hold']

    total_net_pnl = sum(t['net_pnl_usd'] for t in trades)
    avg_net_pnl_usd = total_net_pnl / total_trades
    avg_win_usd = sum(t['net_pnl_usd'] for t in wins) / len(wins) if wins else 0
    avg_loss_usd = sum(t['net_pnl_usd'] for t in losses) / len(losses) if losses else 0

    profitability_rate = (len(profitable_trades) / total_trades) * 100
    win_rate = (len(wins) / total_trades) * 100
    target_hit_rate = (len(target_exits) / total_trades) * 100
    stop_hit_rate = (len(stop_exits) / total_trades) * 100
    trend_break_rate = (len(trend_break_exits) / total_trades) * 100

    avg_hold_hours = sum(t['hours_held'] for t in trades) / total_trades
    avg_hold_days = avg_hold_hours / 24

    # By symbol breakdown
    trades_by_symbol = {}
    for trade in trades:
        symbol = trade['symbol']
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = []
        trades_by_symbol[symbol].append(trade)

    symbol_stats = {}
    for symbol, symbol_trades in trades_by_symbol.items():
        symbol_pnl = sum(t['net_pnl_usd'] for t in symbol_trades)
        symbol_wins = len([t for t in symbol_trades if t['net_pnl_usd'] > 0])
        symbol_stats[symbol] = {
            'trades': len(symbol_trades),
            'win_rate': (symbol_wins / len(symbol_trades) * 100) if symbol_trades else 0,
            'total_pnl': symbol_pnl
        }

    return {
        'target_net_pct': target_net_pct,
        'stop_gross_pct': stop_gross_pct,
        'max_hold_days': max_hold_hours / 24,
        'total_trades': total_trades,
        'profitable_trades': len(profitable_trades),
        'profitability_rate': profitability_rate,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_net_pnl_usd': total_net_pnl,
        'avg_net_pnl_usd': avg_net_pnl_usd,
        'avg_win_usd': avg_win_usd,
        'avg_loss_usd': avg_loss_usd,
        'target_exits': len(target_exits),
        'stop_exits': len(stop_exits),
        'trend_break_exits': len(trend_break_exits),
        'timeout_exits': len(timeout_exits),
        'target_hit_rate': target_hit_rate,
        'stop_hit_rate': stop_hit_rate,
        'trend_break_rate': trend_break_rate,
        'avg_hold_hours': avg_hold_hours,
        'avg_hold_days': avg_hold_days,
        'signals_rejected': signals_rejected,
        'symbol_stats': symbol_stats,
        'trades': trades
    }


def main():
    """Run swing trading backtest."""

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [
        wallet['symbol'] for wallet in config['wallets']
        if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
    ]

    strategy_info = get_strategy_info()

    print("="*120)
    print("SWING TRADING STRATEGY BACKTEST")
    print("="*120)
    print(f"\nStrategy: {strategy_info['name']}")
    print(f"Approach: {strategy_info['approach']}")
    print(f"Holding Time: {strategy_info['holding_time']}")
    print(f"Target: {strategy_info['target_profit']}")
    print(f"Stop Loss: {strategy_info['stop_loss']}")
    print(f"Expected Win Rate: {strategy_info['expected_win_rate']}")
    print(f"Max Concurrent Positions: {strategy_info['max_concurrent_positions']}")
    print(f"Position Size: {strategy_info['position_size']}")
    print()
    print("ENTRY CRITERIA:")
    for criterion in strategy_info['entry_criteria']:
        print(f"  • {criterion}")
    print()
    print("EXIT CRITERIA:")
    for criterion in strategy_info['exit_criteria']:
        print(f"  • {criterion}")
    print()
    print(f"Backtesting {len(enabled_symbols)} symbols (180 days)...")
    print()

    result = backtest_swing_trading(
        symbols=enabled_symbols,
        target_net_pct=3.0,
        stop_gross_pct=2.0,  # Wider stop for bounce confirmation strategy
        max_hold_hours=240,
        max_hours=4320
    )

    if not result or result['total_trades'] == 0:
        print("No trades generated.")
        if result and 'signals_rejected' in result:
            print("\nREJECTION BREAKDOWN:")
            for reason, count in result['signals_rejected'].items():
                print(f"  {reason}: {count}")
        return

    # Display results
    print("="*120)
    print("RESULTS")
    print("="*120)
    print()
    print(f"Total trades: {result['total_trades']}")
    print(f"Win rate: {result['win_rate']:.1f}%")
    print(f"Profitability rate (>$2 NET): {result['profitability_rate']:.1f}%")
    print(f"Target hit rate: {result['target_hit_rate']:.1f}%")
    print(f"Stop hit rate: {result['stop_hit_rate']:.1f}%")
    print(f"Trend break rate: {result['trend_break_rate']:.1f}%")
    print()
    print(f"Total P/L: ${result['total_net_pnl_usd']:.2f}")
    print(f"EV per trade: ${result['avg_net_pnl_usd']:.2f}")
    print(f"Avg win: ${result['avg_win_usd']:.2f}")
    print(f"Avg loss: ${result['avg_loss_usd']:.2f}")
    print()
    print(f"Avg hold time: {result['avg_hold_days']:.1f} days ({result['avg_hold_hours']:.0f} hours)")
    print()

    # Symbol breakdown
    print("PERFORMANCE BY SYMBOL:")
    for symbol, stats in sorted(result['symbol_stats'].items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        status = "✅" if stats['total_pnl'] > 0 else "❌"
        print(f"  {status} {symbol}: {stats['trades']} trades, {stats['win_rate']:.1f}% win rate, ${stats['total_pnl']:.2f}")

    print()
    print("EXIT BREAKDOWN:")
    print(f"  Target hits: {result['target_exits']} ({result['target_hit_rate']:.1f}%)")
    print(f"  Stop losses: {result['stop_exits']} ({result['stop_hit_rate']:.1f}%)")
    print(f"  Trend breaks: {result['trend_break_exits']} ({result['trend_break_rate']:.1f}%)")
    print(f"  Timeouts: {result['timeout_exits']}")

    print()
    print("="*120)

    # Comparison to previous strategies
    print()
    print("COMPARISON TO PREVIOUS STRATEGIES:")
    print("  MOMENTUM (0.7% target):    2,610 trades, 35.4% win rate, -$24,158 loss")
    print("  MEAN REVERSION (0.7%):       224 trades, 35.9% win rate,  -$1,884 loss")
    print(f"  SWING TRADING (3.0%):      {result['total_trades']:>5} trades, {result['win_rate']:>4.1f}% win rate, ${result['total_net_pnl_usd']:>7.2f}")
    print()

    if result['win_rate'] >= 55 and result['total_net_pnl_usd'] > 0:
        print("✅ SUCCESS! Strategy is profitable!")
        trades_per_month = result['total_trades'] / 6  # 180 days = 6 months
        monthly_pnl = result['total_net_pnl_usd'] / 6
        print(f"   Monthly trade frequency: {trades_per_month:.1f} trades/month")
        print(f"   Expected monthly profit: ${monthly_pnl:.2f}")
        print()
        print("🚀 READY TO LAUNCH!")
    elif result['win_rate'] >= 50:
        print("⚠️  CLOSE! Win rate is decent but may need tuning.")
    else:
        print("❌ NEEDS WORK: Win rate below target.")

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'strategy': strategy_info,
        'results': result
    }

    output_file = 'backtest/swing_trading_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"📊 Results saved to {output_file}")


if __name__ == "__main__":
    main()
