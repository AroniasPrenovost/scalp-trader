#!/usr/bin/env python3
"""
Backtest Momentum Scalping Strategy with Intra-Hour Price Simulation

This version simulates 20-second price intervals between hourly candles to properly
test the momentum acceleration filter that's critical to the strategy's performance.
"""

import json
import os
import random
import math
import statistics
from datetime import datetime
from typing import Dict, List, Optional

# Import the strategy
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


def simulate_intra_hour_prices(start_price: float, end_price: float,
                                hourly_volatility: float = None,
                                num_intervals: int = 180) -> List[float]:
    """
    Simulate realistic 20-second price intervals between two hourly prices.

    Uses a constrained random walk that:
    - Starts at start_price
    - Ends at end_price
    - Has realistic intra-hour volatility
    - Models mean-reverting behavior within the hour

    Args:
        start_price: Price at beginning of hour
        end_price: Price at end of hour
        hourly_volatility: Recent hourly volatility (if known), used to scale noise
        num_intervals: Number of 20-second intervals (default 180 = 1 hour)

    Returns:
        List of simulated prices at 20-second intervals
    """
    if num_intervals < 2:
        return [start_price, end_price]

    # Calculate the drift needed to reach end_price
    total_drift = end_price - start_price
    drift_per_step = total_drift / num_intervals

    # Estimate volatility for the random walk
    # If hourly volatility is provided, scale it down to 20-second intervals
    # Otherwise, use the move size as a proxy
    if hourly_volatility is not None:
        # Scale hourly volatility to 20-second intervals
        # hourly_vol = sqrt(num_intervals) * interval_vol
        interval_vol = hourly_volatility / math.sqrt(num_intervals)
    else:
        # Use 0.3% as default intra-hour volatility (conservative estimate)
        interval_vol = start_price * 0.003 / math.sqrt(num_intervals)

    # Generate price path
    prices = [start_price]

    for i in range(1, num_intervals):
        # Random noise component (Gaussian distribution)
        noise = random.gauss(0, interval_vol)

        # Mean-reverting component: pull price towards expected position
        expected_price = start_price + (drift_per_step * i)
        mean_reversion_strength = 0.1  # 10% pull towards expected path
        mean_reversion = (expected_price - prices[-1]) * mean_reversion_strength

        # Next price = previous + drift + noise + mean reversion
        next_price = prices[-1] + drift_per_step + noise + mean_reversion

        # Ensure price stays positive
        next_price = max(next_price, start_price * 0.95)

        prices.append(next_price)

    # Force last price to be exactly end_price
    prices.append(end_price)

    return prices


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
    Backtest momentum scalping strategy with intra-hour price simulation.

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

    # Track stats for signal filtering
    signals_generated = 0
    signals_rejected_by_momentum_filter = 0

    # Start from hour 48 (need historical data)
    for i in range(48, len(prices)):
        current_price = prices[i]
        historical = prices[:i+1]  # Include current price

        # === SIMULATE INTRA-HOUR PRICE MOVEMENTS ===
        # Generate 20-second intervals for the current hour
        # This feeds the momentum acceleration buffer
        if i > 0:
            prev_price = prices[i-1]

            # Calculate recent volatility for realistic simulation
            if i >= 24:
                recent_prices = prices[i-24:i]
                price_changes = [abs((recent_prices[j] - recent_prices[j-1])/recent_prices[j-1])
                                for j in range(1, len(recent_prices))]
                hourly_vol = statistics.stdev(price_changes) * prev_price if len(price_changes) > 1 else None
            else:
                hourly_vol = None

            # Generate intra-hour prices (we'll use last 15 for the momentum buffer)
            intra_hour_prices = simulate_intra_hour_prices(
                start_price=prev_price,
                end_price=current_price,
                hourly_volatility=hourly_vol,
                num_intervals=180  # 3600 seconds / 20 = 180 intervals
            )

            # Feed the last 15 prices (5 minutes of data) into the strategy's buffer
            # This simulates the bot collecting prices every 20 seconds
            from utils.momentum_scalping_strategy import update_intra_hour_buffer

            # Use the last 15 prices to simulate recent momentum
            for price in intra_hour_prices[-15:]:
                update_intra_hour_buffer(symbol, price)

        # ENTRY
        if position is None:
            signal = check_scalp_entry_signal(historical, current_price, symbol=symbol)

            # Track signal generation and filtering
            if signal and 'Momentum' in signal.get('reason', '') and 'rejected' in signal.get('reason', '').lower():
                signals_rejected_by_momentum_filter += 1

            if signal and signal.get('signal') == 'buy':
                signals_generated += 1
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

        # By strategy type
        'support_bounce': strategy_stats(support_bounce_trades),
        'breakout': strategy_stats(breakout_trades),
        'consolidation_break': strategy_stats(consolidation_trades),

        # Timing
        'avg_hold_hours': avg_hold_hours,

        # Signal filtering stats
        'signals_generated': signals_generated,
        'signals_rejected_by_filter': signals_rejected_by_momentum_filter,
        'filter_rejection_rate': (signals_rejected_by_momentum_filter / max(signals_generated, 1)) * 100,

        # All trades
        'trades': trades
    }


def main():
    """Run backtest across all symbols."""

    # Load config to get enabled symbols
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [
        wallet['symbol'] for wallet in config['wallets']
        if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
    ]

    print("=" * 120)
    print("MOMENTUM SCALPING STRATEGY BACKTEST (WITH INTRA-HOUR SIMULATION)")
    print("=" * 120)

    strategy_info = get_strategy_info()
    print(f"Strategy: {strategy_info['name']}")
    print(f"Target: {strategy_info['target_profit']} in {strategy_info['target_holding_time']}")
    print(f"Stop Loss: {strategy_info['max_stop_loss']}")
    print(f"Fee Structure: {TAKER_FEE*100:.2f}% + {TAX_RATE*100:.1f}% tax")
    print(f"Min Profitable Gain: {MIN_GAIN_PCT:.2f}% (${MIN_NET_PROFIT_USD}+)")
    print()
    print("Signal Types:")
    for signal_type, details in strategy_info['signal_types'].items():
        print(f"  - {signal_type}: {details['description']} (Target: {details['target']}, Stop: {details['stop']})")
    print()
    print("KEY DIFFERENCE: This backtest simulates 20-second price intervals to activate")
    print("the momentum acceleration filter, which rejects weak signals.")
    print("=" * 120)
    print()

    print(f"Backtesting {len(enabled_symbols)} symbols (180 days)...")
    print()

    results = []

    for symbol in enabled_symbols:
        result = backtest_symbol(symbol, max_hold_hours=4, max_hours=4320)

        if result:
            results.append(result)

            status = "✅" if result['total_net_pnl_usd'] > 0 else "❌"
            print(f"  {symbol}... {status} {result['total_trades']} trades, "
                  f"{result['profitability_rate']:.1f}% profitable, "
                  f"{result['target_hit_rate']:.1f}% hit target, "
                  f"${result['total_net_pnl_usd']:.2f} "
                  f"({result['signals_rejected_by_filter']} signals filtered)")

    if not results:
        print("No results to display.")
        return

    # Print summary
    print()
    print("=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)
    print(f"{'Symbol':<12} {'Trades':<8} {'Profitable':<12} {'Target Hit':<12} {'Win Rate':<10} "
          f"{'Net P/L':<12} {'Avg Hold':<10} {'Filtered':<10}")
    print("-" * 120)

    # Sort by P/L
    results_sorted = sorted(results, key=lambda x: x['total_net_pnl_usd'], reverse=True)

    for result in results_sorted:
        status = "✅" if result['total_net_pnl_usd'] > 0 else "❌"
        print(f"{status} {result['symbol']:<10} {result['total_trades']:<8} "
              f"{result['profitability_rate']:<11.1f}% {result['target_hit_rate']:<11.1f}% "
              f"{result['win_rate']:<9.1f}% ${result['total_net_pnl_usd']:<11.2f} "
              f"{result['avg_hold_hours']:<9.1f}h {result['signals_rejected_by_filter']:<10}")

    print("-" * 120)

    # Overall stats
    total_trades = sum(r['total_trades'] for r in results)
    total_pnl = sum(r['total_net_pnl_usd'] for r in results)
    avg_profitability = sum(r['profitability_rate'] for r in results) / len(results)
    avg_target_hit = sum(r['target_hit_rate'] for r in results) / len(results)
    avg_ev = total_pnl / total_trades if total_trades > 0 else 0
    total_signals_filtered = sum(r['signals_rejected_by_filter'] for r in results)

    print(f"OVERALL: {total_trades} trades | {avg_profitability:.1f}% profitable | "
          f"{avg_target_hit:.1f}% hit target | ${total_pnl:.2f} total | "
          f"${avg_ev:.2f} avg EV | {total_signals_filtered} filtered")
    print("=" * 120)
    print()

    # Strategy breakdown
    all_sb = sum(r['support_bounce']['count'] if r['support_bounce'] else 0 for r in results)
    all_bo = sum(r['breakout']['count'] if r['breakout'] else 0 for r in results)
    all_cb = sum(r['consolidation_break']['count'] if r['consolidation_break'] else 0 for r in results)

    if all_sb > 0:
        sb_prof = sum((r['support_bounce']['count'] * r['support_bounce']['profitability_rate'] / 100)
                      if r['support_bounce'] else 0 for r in results)
        sb_rate = (sb_prof / all_sb) * 100
    else:
        sb_rate = 0

    if all_bo > 0:
        bo_prof = sum((r['breakout']['count'] * r['breakout']['profitability_rate'] / 100)
                      if r['breakout'] else 0 for r in results)
        bo_rate = (bo_prof / all_bo) * 100
    else:
        bo_rate = 0

    if all_cb > 0:
        cb_prof = sum((r['consolidation_break']['count'] * r['consolidation_break']['profitability_rate'] / 100)
                      if r['consolidation_break'] else 0 for r in results)
        cb_rate = (cb_prof / all_cb) * 100
    else:
        cb_rate = 0

    print("STRATEGY BREAKDOWN (across all symbols):")
    print("-" * 120)
    print(f"  Support Bounce: {all_sb} trades ({all_sb/total_trades*100:.1f}%), {sb_rate:.1f}% profitable")
    print(f"  Breakout: {all_bo} trades ({all_bo/total_trades*100:.1f}%), {bo_rate:.1f}% profitable")
    print(f"  Consolidation Break: {all_cb} trades ({all_cb/total_trades*100:.1f}%), {cb_rate:.1f}% profitable")
    print("=" * 120)
    print()

    # Verdict
    if avg_profitability >= 47:
        print("VERDICT:")
        print("✅ READY TO LAUNCH: Strategy meets expected performance (47%+ profitability)")
    elif avg_profitability >= 40:
        print("VERDICT:")
        print("⚠️  BORDERLINE: Strategy shows promise but below target (40-47% profitability)")
        print("   Consider paper trading first or adjusting parameters")
    else:
        print("VERDICT:")
        print("❌ NEEDS IMPROVEMENT: Only {:.1f}% profitable".format(avg_profitability))
        print("   Strategy parameters may need adjustment")

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'strategy': strategy_info,
        'fee_structure': {
            'taker_fee_pct': TAKER_FEE * 100,
            'tax_rate_pct': TAX_RATE * 100,
            'min_gain_for_profit_pct': MIN_GAIN_PCT
        },
        'intra_hour_simulation': {
            'enabled': True,
            'interval_seconds': 20,
            'intervals_per_hour': 180,
            'buffer_size': 15,
            'description': 'Simulated 20-second price intervals to activate momentum acceleration filter'
        },
        'results': results
    }

    output_file = 'backtest/momentum_scalping_intrahour_backtest_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"📊 Results saved to {output_file}")


if __name__ == "__main__":
    main()
