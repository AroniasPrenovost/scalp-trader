#!/usr/bin/env python3
"""
Backtest Momentum Scalping Strategy
Test the new 0.6-0.8% quick-scalp approach against historical data
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# Import the new strategy
import sys
sys.path.append(os.path.dirname(__file__))
from utils.momentum_scalping_strategy import (
    check_scalp_entry_signal,
    calculate_scalp_targets,
    get_strategy_info
)

# ACTUAL TRADING COSTS
TAKER_FEE = 0.0025  # 0.25% Coinbase taker fee
TAX_RATE = 0.24     # 24% federal tax rate
POSITION_SIZE_USD = 4609

# Min profit needed
MIN_NET_PROFIT_USD = 2.00
net_pct = MIN_NET_PROFIT_USD / POSITION_SIZE_USD
MIN_GAIN_PCT = (net_pct + TAKER_FEE) / (1 - TAX_RATE - TAKER_FEE)  # 0.4%


def load_price_data(symbol: str) -> Optional[tuple]:
    """Load historical price data."""
    filepath = f"coinbase-data/{symbol}.json"
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    prices = [float(entry['price']) for entry in data_sorted]
    timestamps = [entry['timestamp'] for entry in data_sorted]

    return prices, timestamps


def calculate_net_profit(entry_price: float, exit_price: float, position_size: float) -> Dict:
    """Calculate net profit using actual fee structure."""
    cost_basis = position_size
    shares = position_size / entry_price
    current_value = exit_price * shares

    gross_profit = current_value - cost_basis
    exit_fee = TAKER_FEE * current_value
    capital_gain = current_value - cost_basis
    tax = TAX_RATE * capital_gain
    net_profit = current_value - cost_basis - exit_fee - tax
    net_pct = (net_profit / cost_basis) * 100
    gross_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'gross_pnl_pct': gross_pct,
        'net_pnl_pct': net_pct,
        'net_pnl_usd': net_profit,
        'exit_fee_usd': exit_fee,
        'tax_usd': tax,
        'profitable': net_profit >= MIN_NET_PROFIT_USD
    }


def backtest_symbol(symbol: str, max_hold_hours: int = 4, max_hours: int = 4320) -> Optional[Dict]:
    """
    Backtest momentum scalping strategy.

    Args:
        symbol: Trading pair
        max_hold_hours: Maximum hold time (4 hours = 2x expected)
        max_hours: Historical data to analyze
    """

    data = load_price_data(symbol)
    if not data:
        return None

    prices, timestamps = data

    if len(prices) < 200:
        return None

    if len(prices) > max_hours:
        prices = prices[-max_hours:]
        timestamps = timestamps[-max_hours:]

    trades = []
    position = None

    # Start from hour 48 (need historical data)
    for i in range(48, len(prices)):
        current_price = prices[i]
        historical = prices[:i+1]  # Include current price

        # ENTRY
        if position is None:
            signal = check_scalp_entry_signal(historical, current_price)

            if signal and signal.get('signal') == 'buy':
                position = {
                    'entry_idx': i,
                    'entry_price': signal['entry_price'],
                    'entry_timestamp': timestamps[i],
                    'stop': signal['stop_loss'],
                    'target': signal['profit_target'],
                    'strategy': signal['strategy'],
                    'confidence': signal.get('confidence', 'medium'),
                    'metrics': signal.get('metrics', {})
                }

        # EXIT
        elif position:
            hours_held = i - position['entry_idx']
            exit_price = None
            exit_reason = None

            # Check stop loss first
            if current_price <= position['stop']:
                exit_price = position['stop']
                exit_reason = 'stop_loss'

            # Check profit target
            elif current_price >= position['target']:
                exit_price = position['target']
                exit_reason = 'target'

            # Max hold time (2x expected, to give room)
            elif hours_held >= max_hold_hours:
                exit_price = current_price
                exit_reason = 'max_hold'

            if exit_price:
                pnl = calculate_net_profit(position['entry_price'], exit_price, POSITION_SIZE_USD)

                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': timestamps[i],
                    'gross_pnl_pct': pnl['gross_pnl_pct'],
                    'net_pnl_pct': pnl['net_pnl_pct'],
                    'net_pnl_usd': pnl['net_pnl_usd'],
                    'exit_fee_usd': pnl['exit_fee_usd'],
                    'tax_usd': pnl['tax_usd'],
                    'profitable': pnl['profitable'],
                    'exit_reason': exit_reason,
                    'strategy': position['strategy'],
                    'confidence': position['confidence'],
                    'hours_held': hours_held,
                    'entry_metrics': position['metrics']
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
            'exit_fee_usd': pnl['exit_fee_usd'],
            'tax_usd': pnl['tax_usd'],
            'profitable': pnl['profitable'],
            'exit_reason': 'eod',
            'strategy': position['strategy'],
            'confidence': position['confidence'],
            'hours_held': hours_held,
            'entry_metrics': position['metrics']
        })

    if not trades:
        return None

    # Calculate stats
    total_trades = len(trades)
    profitable_trades = [t for t in trades if t['profitable']]
    wins = [t for t in trades if t['net_pnl_usd'] > 0]
    losses = [t for t in trades if t['net_pnl_usd'] <= 0]

    # By exit reason
    target_exits = [t for t in trades if t['exit_reason'] == 'target']
    stop_exits = [t for t in trades if t['exit_reason'] == 'stop_loss']
    timeout_exits = [t for t in trades if t['exit_reason'] == 'max_hold']

    # By strategy type
    support_bounce_trades = [t for t in trades if t['strategy'] == 'support_bounce']
    breakout_trades = [t for t in trades if t['strategy'] == 'breakout']
    consolidation_trades = [t for t in trades if t['strategy'] == 'consolidation_break']

    # P/L metrics
    total_net_pnl = sum(t['net_pnl_usd'] for t in trades)
    avg_net_pnl_usd = total_net_pnl / total_trades
    avg_win_usd = sum(t['net_pnl_usd'] for t in wins) / len(wins) if wins else 0
    avg_loss_usd = sum(t['net_pnl_usd'] for t in losses) / len(losses) if losses else 0

    # Rates
    profitability_rate = (len(profitable_trades) / total_trades) * 100
    win_rate = (len(wins) / total_trades) * 100
    target_hit_rate = (len(target_exits) / total_trades) * 100

    # Avg hold time
    avg_hold_hours = sum(t['hours_held'] for t in trades) / total_trades

    # By strategy stats
    def strategy_stats(strategy_trades):
        if not strategy_trades:
            return None
        prof = [t for t in strategy_trades if t['profitable']]
        return {
            'count': len(strategy_trades),
            'profitability_rate': (len(prof) / len(strategy_trades)) * 100,
            'avg_net_pnl': sum(t['net_pnl_usd'] for t in strategy_trades) / len(strategy_trades)
        }

    return {
        'symbol': symbol,
        'total_trades': total_trades,

        # Profitability
        'profitable_trades': len(profitable_trades),
        'profitability_rate': profitability_rate,

        # Win/loss
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,

        # Net performance
        'total_net_pnl_usd': total_net_pnl,
        'avg_net_pnl_usd': avg_net_pnl_usd,
        'avg_win_usd': avg_win_usd,
        'avg_loss_usd': avg_loss_usd,

        # Exit analysis
        'target_exits': len(target_exits),
        'stop_exits': len(stop_exits),
        'timeout_exits': len(timeout_exits),
        'target_hit_rate': target_hit_rate,

        # Strategy breakdown
        'support_bounce': strategy_stats(support_bounce_trades),
        'breakout': strategy_stats(breakout_trades),
        'consolidation_break': strategy_stats(consolidation_trades),

        'avg_hold_hours': avg_hold_hours,
        'trades': trades
    }


def main():
    """Run momentum scalping backtest."""

    with open('config.json', 'r') as f:
        config = json.load(f)

    symbols = [w['symbol'] for w in config['wallets'] if w.get('enabled', False)]

    strategy_info = get_strategy_info()

    print(f"\n{'='*120}")
    print(f"MOMENTUM SCALPING STRATEGY BACKTEST")
    print(f"{'='*120}")
    print(f"Strategy: {strategy_info['name']}")
    print(f"Target: {strategy_info['target_profit']} profit in {strategy_info['target_holding_time']}")
    print(f"Stop Loss: {strategy_info['max_stop_loss']}")
    print(f"Fee Structure: {TAKER_FEE*100}% + {TAX_RATE*100}% tax")
    print(f"Min Profitable Gain: {MIN_GAIN_PCT*100:.2f}% (${MIN_NET_PROFIT_USD}+)")
    print(f"\nSignal Types:")
    for sig_type, info in strategy_info['signal_types'].items():
        print(f"  - {sig_type}: {info['description']} (Target: {info['target']}, Stop: {info['stop']})")
    print(f"{'='*120}\n")

    print(f"Backtesting {len(symbols)} symbols (180 days)...\n")

    results = []

    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        result = backtest_symbol(symbol, max_hold_hours=4)

        if result:
            results.append(result)
            emoji = "✅" if result['profitability_rate'] >= 50 else "⚠️" if result['profitability_rate'] >= 40 else "❌"
            print(f"{emoji} {result['total_trades']} trades, {result['profitability_rate']:.1f}% profitable, "
                  f"{result['target_hit_rate']:.1f}% hit target, ${result['total_net_pnl_usd']:+.2f}")
        else:
            print("⚠️  No trades")

    if not results:
        print("\n⚠️  No results")
        return

    # Sort by profitability
    results.sort(key=lambda x: x['profitability_rate'], reverse=True)

    # Summary table
    print(f"\n{'='*120}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*120}")
    print(f"{'Symbol':<10} {'Trades':<8} {'Profitable':<12} {'Target Hit':<12} {'Win Rate':<10} {'Net P/L':<12} {'Avg Hold':<10}")
    print(f"{'-'*120}")

    for r in results:
        emoji = "✅" if r['profitability_rate'] >= 50 else "⚠️" if r['profitability_rate'] >= 40 else "❌"
        print(f"{emoji} {r['symbol']:<8} {r['total_trades']:<8} {r['profitability_rate']:>6.1f}%     "
              f"{r['target_hit_rate']:>6.1f}%      {r['win_rate']:>6.1f}%    "
              f"${r['total_net_pnl_usd']:>+9.2f}  {r['avg_hold_hours']:>5.1f}h")

    # Overall
    total_trades = sum(r['total_trades'] for r in results)
    total_profitable = sum(r['profitable_trades'] for r in results)
    overall_pr = (total_profitable / total_trades) * 100 if total_trades else 0
    total_net = sum(r['total_net_pnl_usd'] for r in results)
    avg_ev = total_net / total_trades if total_trades else 0

    total_targets = sum(r['target_exits'] for r in results)
    overall_target_rate = (total_targets / total_trades) * 100 if total_trades else 0

    print(f"{'-'*120}")
    print(f"OVERALL: {total_trades} trades | {overall_pr:.1f}% profitable | "
          f"{overall_target_rate:.1f}% hit target | ${total_net:+.2f} total | ${avg_ev:+.2f} avg EV")
    print(f"{'='*120}\n")

    # Strategy breakdown
    print(f"STRATEGY BREAKDOWN (across all symbols):")
    print(f"{'-'*120}")

    all_support = sum(r['support_bounce']['count'] if r['support_bounce'] else 0 for r in results)
    all_breakout = sum(r['breakout']['count'] if r['breakout'] else 0 for r in results)
    all_consol = sum(r['consolidation_break']['count'] if r['consolidation_break'] else 0 for r in results)

    if all_support > 0:
        support_prof = sum(r['support_bounce']['count'] * r['support_bounce']['profitability_rate'] / 100
                          if r['support_bounce'] else 0 for r in results)
        print(f"  Support Bounce: {all_support} trades ({all_support/total_trades*100:.1f}%), "
              f"{support_prof/all_support*100:.1f}% profitable")

    if all_breakout > 0:
        breakout_prof = sum(r['breakout']['count'] * r['breakout']['profitability_rate'] / 100
                           if r['breakout'] else 0 for r in results)
        print(f"  Breakout: {all_breakout} trades ({all_breakout/total_trades*100:.1f}%), "
              f"{breakout_prof/all_breakout*100:.1f}% profitable")

    if all_consol > 0:
        consol_prof = sum(r['consolidation_break']['count'] * r['consolidation_break']['profitability_rate'] / 100
                         if r['consolidation_break'] else 0 for r in results)
        print(f"  Consolidation Break: {all_consol} trades ({all_consol/total_trades*100:.1f}%), "
              f"{consol_prof/all_consol*100:.1f}% profitable")

    print(f"{'='*120}\n")

    # Verdict
    print(f"VERDICT:")
    if overall_pr >= 55:
        print(f"✅ HIGHLY PROFITABLE: {overall_pr:.1f}% of trades meet ${MIN_NET_PROFIT_USD}+ threshold")
        print(f"   {overall_target_rate:.1f}% hit profit target within 4 hours")
        print(f"   Expected value: ${avg_ev:+.2f} per trade")
    elif overall_pr >= 45:
        print(f"⚠️  MARGINALLY PROFITABLE: {overall_pr:.1f}% profitability")
        print(f"   Consider focusing on best-performing symbols only")
    else:
        print(f"❌ NEEDS IMPROVEMENT: Only {overall_pr:.1f}% profitable")
        print(f"   Strategy parameters may need adjustment")
    print()

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'strategy': strategy_info,
        'fee_structure': {
            'taker_fee_pct': TAKER_FEE * 100,
            'tax_rate_pct': TAX_RATE * 100,
            'min_gain_for_profit_pct': MIN_GAIN_PCT * 100
        },
        'results': results,
        'overall': {
            'total_trades': total_trades,
            'profitable_trades': total_profitable,
            'profitability_rate': overall_pr,
            'target_hit_rate': overall_target_rate,
            'total_net_pnl_usd': total_net,
            'avg_ev_usd': avg_ev
        }
    }

    with open('momentum_scalping_backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"📊 Results saved to momentum_scalping_backtest_results.json\n")


if __name__ == '__main__':
    main()
