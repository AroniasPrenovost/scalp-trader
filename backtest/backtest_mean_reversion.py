#!/usr/bin/env python3
"""
Backtest Mean Reversion Strategy

Simple, clean strategy:
- BUY when RSI < 30 (oversold)
- SELL when RSI > 50 (reverted) or hit target
- Target: 0.7% NET
- Stop: 0.5% GROSS
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import sys
sys.path.append(os.path.dirname(__file__))

from utils.mean_reversion_strategy import (
    check_mean_reversion_entry,
    check_mean_reversion_exit,
    get_strategy_info
)

# TRADING COSTS
TAKER_FEE = 0.0025
TAX_RATE = 0.24
POSITION_SIZE_USD = 4609
MIN_NET_PROFIT_USD = 2.00


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


def backtest_mean_reversion(symbol: str,
                            target_net_pct: float = 0.7,
                            stop_gross_pct: float = 0.5,
                            max_hold_hours: int = 24,
                            max_hours: int = 4320) -> Optional[Dict]:
    """
    Backtest mean reversion strategy.

    Args:
        symbol: Trading pair
        target_net_pct: NET profit target (default 0.7%)
        stop_gross_pct: GROSS stop loss (default 0.5%)
        max_hold_hours: Max holding time (default 24h)
        max_hours: Historical data to analyze
    """

    data = load_price_volume_data(symbol)
    if not data:
        return None

    prices, volumes, timestamps = data

    if len(prices) < 48:
        return None

    if len(prices) > max_hours:
        prices = prices[-max_hours:]
        volumes = volumes[-max_hours:]
        timestamps = timestamps[-max_hours:]

    trades = []
    position = None

    # Track rejection reasons
    signals_rejected_rsi = 0
    signals_rejected_volume = 0
    signals_rejected_downtrend = 0
    signals_rejected_not_below_mean = 0

    # Start from hour 24 (need history for indicators)
    for i in range(24, len(prices)):
        current_price = prices[i]
        current_volume = volumes[i]
        historical_prices = prices[:i+1]
        historical_volumes = volumes[:i+1]

        # ENTRY: Check for oversold conditions
        if position is None:
            signal = check_mean_reversion_entry(
                prices=historical_prices,
                volumes=historical_volumes,
                current_price=current_price,
                current_volume=current_volume,
                target_net_pct=target_net_pct,
                stop_gross_pct=stop_gross_pct,
                entry_fee_pct=TAKER_FEE * 100,
                exit_fee_pct=TAKER_FEE * 100,
                tax_rate_pct=TAX_RATE * 100,
                rsi_oversold=30,
                min_volume_ratio=1.0
            )

            # Track rejections
            if signal and 'reason' in signal:
                reason = signal['reason'].lower()
                if 'rsi' in reason or 'oversold' in reason:
                    signals_rejected_rsi += 1
                elif 'volume' in reason:
                    signals_rejected_volume += 1
                elif 'downtrend' in reason or 'falling' in reason:
                    signals_rejected_downtrend += 1
                elif 'mean' in reason or 'below' in reason:
                    signals_rejected_not_below_mean += 1

            if signal and signal.get('signal') == 'buy':
                position = {
                    'entry_idx': i,
                    'entry_price': signal['entry_price'],
                    'entry_timestamp': timestamps[i],
                    'stop': signal['stop_loss'],
                    'target': signal['profit_target'],
                    'mean_target': signal['mean_reversion_target'],
                    'confidence': signal.get('confidence', 'medium'),
                    'entry_rsi': signal['metrics']['rsi']
                }

        # EXIT: Check for mean reversion or stop/target
        elif position:
            hours_held = i - position['entry_idx']
            historical_for_exit = prices[:i+1]

            exit_signal = check_mean_reversion_exit(
                prices=historical_for_exit,
                entry_price=position['entry_price'],
                current_price=current_price,
                hours_held=hours_held,
                target_price=position['target'],
                stop_price=position['stop'],
                max_hold_hours=max_hold_hours,
                rsi_mean=50
            )

            if exit_signal and exit_signal.get('exit'):
                exit_price = exit_signal['exit_price']
                exit_reason = exit_signal['exit_reason']

                pnl = calculate_net_profit(position['entry_price'], exit_price, POSITION_SIZE_USD)

                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': timestamps[i],
                    'gross_pnl_pct': pnl['gross_pnl_pct'],
                    'net_pnl_pct': pnl['net_pnl_pct'],
                    'net_pnl_usd': pnl['net_pnl_usd'],
                    'profitable': pnl['profitable'],
                    'exit_reason': exit_reason,
                    'confidence': position['confidence'],
                    'hours_held': hours_held,
                    'entry_rsi': position['entry_rsi']
                })
                position = None

    # Close remaining position
    if position:
        final_price = prices[-1]
        hours_held = len(prices) - 1 - position['entry_idx']
        pnl = calculate_net_profit(position['entry_price'], final_price, POSITION_SIZE_USD)

        trades.append({
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': timestamps[-1],
            'gross_pnl_pct': pnl['gross_pnl_pct'],
            'net_pnl_pct': pnl['net_pnl_pct'],
            'net_pnl_usd': pnl['net_pnl_usd'],
            'profitable': pnl['profitable'],
            'exit_reason': 'eod',
            'confidence': position['confidence'],
            'hours_held': hours_held,
            'entry_rsi': position['entry_rsi']
        })

    if not trades:
        return {
            'symbol': symbol,
            'total_trades': 0,
            'signals_rejected_rsi': signals_rejected_rsi,
            'signals_rejected_volume': signals_rejected_volume,
            'signals_rejected_downtrend': signals_rejected_downtrend,
            'signals_rejected_not_below_mean': signals_rejected_not_below_mean
        }

    # Calculate stats
    total_trades = len(trades)
    profitable_trades = [t for t in trades if t['profitable']]
    wins = [t for t in trades if t['net_pnl_usd'] > 0]
    losses = [t for t in trades if t['net_pnl_usd'] <= 0]

    # By exit reason
    target_exits = [t for t in trades if t['exit_reason'] == 'target']
    stop_exits = [t for t in trades if t['exit_reason'] == 'stop_loss']
    rsi_reversion_exits = [t for t in trades if t['exit_reason'] == 'rsi_reversion']
    ma_crossover_exits = [t for t in trades if t['exit_reason'] == 'ma_crossover']
    timeout_exits = [t for t in trades if t['exit_reason'] == 'max_hold']

    total_net_pnl = sum(t['net_pnl_usd'] for t in trades)
    avg_net_pnl_usd = total_net_pnl / total_trades
    avg_win_usd = sum(t['net_pnl_usd'] for t in wins) / len(wins) if wins else 0
    avg_loss_usd = sum(t['net_pnl_usd'] for t in losses) / len(losses) if losses else 0

    profitability_rate = (len(profitable_trades) / total_trades) * 100
    win_rate = (len(wins) / total_trades) * 100
    target_hit_rate = (len(target_exits) / total_trades) * 100
    stop_hit_rate = (len(stop_exits) / total_trades) * 100

    avg_hold_hours = sum(t['hours_held'] for t in trades) / total_trades
    avg_entry_rsi = sum(t['entry_rsi'] for t in trades) / total_trades

    return {
        'symbol': symbol,
        'target_net_pct': target_net_pct,
        'stop_gross_pct': stop_gross_pct,
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
        'rsi_reversion_exits': len(rsi_reversion_exits),
        'ma_crossover_exits': len(ma_crossover_exits),
        'timeout_exits': len(timeout_exits),
        'target_hit_rate': target_hit_rate,
        'stop_hit_rate': stop_hit_rate,
        'avg_hold_hours': avg_hold_hours,
        'avg_entry_rsi': avg_entry_rsi,
        'signals_rejected_rsi': signals_rejected_rsi,
        'signals_rejected_volume': signals_rejected_volume,
        'signals_rejected_downtrend': signals_rejected_downtrend,
        'signals_rejected_not_below_mean': signals_rejected_not_below_mean,
        'trades': trades
    }


def main():
    """Run mean reversion backtest."""

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [
        wallet['symbol'] for wallet in config['wallets']
        if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
    ]

    strategy_info = get_strategy_info()

    print("="*120)
    print("MEAN REVERSION STRATEGY BACKTEST")
    print("="*120)
    print(f"\nStrategy: {strategy_info['name']}")
    print(f"Approach: {strategy_info['approach']}")
    print(f"Target: {strategy_info['target_profit']} in {strategy_info['target_holding_time']}")
    print(f"Stop Loss: {strategy_info['max_stop_loss']}")
    print(f"Expected Win Rate: {strategy_info['expected_win_rate']}")
    print()
    print("ENTRY CRITERIA:")
    for criterion in strategy_info['entry_criteria']:
        print(f"  • {criterion}")
    print()
    print(f"Backtesting {len(enabled_symbols)} symbols (180 days)...")
    print()

    results = []

    for symbol in enabled_symbols:
        result = backtest_mean_reversion(symbol, target_net_pct=0.7, stop_gross_pct=0.5,
                                        max_hold_hours=24, max_hours=4320)

        if result and result['total_trades'] > 0:
            results.append(result)

            status = "✅" if result['total_net_pnl_usd'] > 0 else "❌"
            print(f"{symbol}: {status} {result['total_trades']} trades, "
                  f"{result['win_rate']:.1f}% win rate, ${result['total_net_pnl_usd']:.2f} P/L, "
                  f"avg RSI: {result['avg_entry_rsi']:.0f}, "
                  f"avg hold: {result['avg_hold_hours']:.1f}h")

    if not results:
        print("No results to display.")
        return

    # Summary
    print()
    print("="*120)
    print("AGGREGATE RESULTS")
    print("="*120)

    total_trades = sum(r['total_trades'] for r in results)
    total_pnl = sum(r['total_net_pnl_usd'] for r in results)
    avg_win_rate = sum(r['win_rate'] for r in results) / len(results)
    avg_target_hit = sum(r['target_hit_rate'] for r in results) / len(results)
    avg_stop_hit = sum(r['stop_hit_rate'] for r in results) / len(results)
    avg_ev = total_pnl / total_trades if total_trades > 0 else 0
    profitable_symbols = len([r for r in results if r['total_net_pnl_usd'] > 0])

    # Exit reason stats
    total_rsi_exits = sum(r['rsi_reversion_exits'] for r in results)
    total_ma_exits = sum(r['ma_crossover_exits'] for r in results)

    print(f"\nTotal trades: {total_trades}")
    print(f"Win rate: {avg_win_rate:.1f}%")
    print(f"Target hit rate: {avg_target_hit:.1f}%")
    print(f"Stop hit rate: {avg_stop_hit:.1f}%")
    print(f"Total P/L: ${total_pnl:.2f}")
    print(f"EV per trade: ${avg_ev:.2f}")
    print(f"Profitable symbols: {profitable_symbols}/{len(results)}")
    print()
    print(f"EXIT BREAKDOWN:")
    print(f"  RSI reversion exits: {total_rsi_exits} ({total_rsi_exits/total_trades*100:.1f}%)")
    print(f"  MA crossover exits: {total_ma_exits} ({total_ma_exits/total_trades*100:.1f}%)")
    print(f"  Target exits: {sum(r['target_exits'] for r in results)} ({avg_target_hit:.1f}%)")
    print(f"  Stop exits: {sum(r['stop_exits'] for r in results)} ({avg_stop_hit:.1f}%)")

    print()
    print("="*120)

    # Comparison
    print()
    print("COMPARISON TO MOMENTUM STRATEGY:")
    print("  MOMENTUM (0.7% target): 2,610 trades, 35.4% win rate, -$24,158 loss")
    print(f"  MEAN REVERSION (0.7% target): {total_trades} trades, {avg_win_rate:.1f}% win rate, ${total_pnl:.2f}")
    print()

    if avg_win_rate >= 47:
        print("✅ SUCCESS! Win rate above 47% - strategy is profitable!")
        trades_per_day = total_trades / 180
        daily_ev = avg_ev * trades_per_day
        monthly_ev = daily_ev * 30
        print(f"   Expected daily profit: ${daily_ev:.2f}")
        print(f"   Expected monthly profit: ${monthly_ev:.2f}")
        print()
        print("🚀 READY TO LAUNCH!")
    elif avg_win_rate >= 40:
        print("⚠️  CLOSE! Win rate {:.1f}% (need 47%+)".format(avg_win_rate))
        print("   Better than momentum strategy but needs tuning.")
    else:
        print("❌ NEEDS MORE WORK: Win rate still below target.")

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'strategy': strategy_info,
        'results': results
    }

    output_file = 'backtest/mean_reversion_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"📊 Results saved to {output_file}")


if __name__ == "__main__":
    main()
