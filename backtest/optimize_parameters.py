#!/usr/bin/env python3
"""
Parameter Optimization Backtest

Tests multiple target/stop configurations to find optimal settings.
"""

import json
import os
import random
import math
import statistics
from datetime import datetime
from typing import Dict, List, Optional
import sys
sys.path.append(os.path.dirname(__file__))

# ACTUAL TRADING COSTS
TAKER_FEE = 0.0025  # 0.25% Coinbase taker fee
TAX_RATE = 0.24     # 24% federal tax rate
POSITION_SIZE_USD = 4609

# Min profit needed
MIN_NET_PROFIT_USD = 2.00


def simulate_intra_hour_prices(start_price: float, end_price: float,
                                hourly_volatility: float = None,
                                num_intervals: int = 180) -> List[float]:
    """Simulate realistic 20-second price intervals between two hourly prices."""
    if num_intervals < 2:
        return [start_price, end_price]

    total_drift = end_price - start_price
    drift_per_step = total_drift / num_intervals

    if hourly_volatility is not None:
        interval_vol = hourly_volatility / math.sqrt(num_intervals)
    else:
        interval_vol = start_price * 0.003 / math.sqrt(num_intervals)

    prices = [start_price]

    for i in range(1, num_intervals):
        noise = random.gauss(0, interval_vol)
        expected_price = start_price + (drift_per_step * i)
        mean_reversion_strength = 0.1
        mean_reversion = (expected_price - prices[-1]) * mean_reversion_strength
        next_price = prices[-1] + drift_per_step + noise + mean_reversion
        next_price = max(next_price, start_price * 0.95)
        prices.append(next_price)

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


def calculate_net_profit(entry_price: float, exit_price: float, position_size: float,
                          target_net_pct: float) -> Dict:
    """Calculate net profit using actual fee structure."""
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

    # Calculate min profitable threshold
    net_pct_threshold = MIN_NET_PROFIT_USD / POSITION_SIZE_USD
    min_gain_pct = (net_pct_threshold + TAKER_FEE) / (1 - TAX_RATE - TAKER_FEE)

    return {
        'gross_pnl_pct': gross_pct,
        'net_pnl_pct': net_pct,
        'net_pnl_usd': net_profit,
        'exit_fee_usd': exit_fee,
        'tax_usd': tax,
        'profitable': net_profit >= MIN_NET_PROFIT_USD,
        'min_gain_pct': min_gain_pct
    }


# Import strategy functions
from utils.momentum_scalping_strategy import (
    check_scalp_entry_signal,
    update_intra_hour_buffer
)


def backtest_with_parameters(symbol: str,
                              target_net_pct: float,
                              stop_gross_pct: float,
                              max_hold_hours: int = 4,
                              max_hours: int = 4320) -> Optional[Dict]:
    """
    Backtest with specific target/stop parameters.

    Args:
        symbol: Trading pair
        target_net_pct: NET profit target (e.g., 0.6 = 0.6%)
        stop_gross_pct: GROSS stop loss (e.g., 0.4 = 0.4%)
        max_hold_hours: Maximum hold time
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
    signals_rejected = 0

    # Calculate required gross move for target
    # target_net_pct% NET requires solving: net = (gross - fee) * (1 - tax)
    # Simplification: gross ~= target_net_pct / 0.75 (rough approximation for 25% total costs)
    from utils.profit_calculator import calculate_required_price_for_target_profit

    # Start from hour 48 (need historical data)
    for i in range(48, len(prices)):
        current_price = prices[i]
        historical = prices[:i+1]

        # Simulate intra-hour prices
        if i > 0:
            prev_price = prices[i-1]

            if i >= 24:
                recent_prices = prices[i-24:i]
                price_changes = [abs((recent_prices[j] - recent_prices[j-1])/recent_prices[j-1])
                                for j in range(1, len(recent_prices))]
                hourly_vol = statistics.stdev(price_changes) * prev_price if len(price_changes) > 1 else None
            else:
                hourly_vol = None

            intra_hour_prices = simulate_intra_hour_prices(
                start_price=prev_price,
                end_price=current_price,
                hourly_volatility=hourly_vol,
                num_intervals=180
            )

            for price in intra_hour_prices[-15:]:
                update_intra_hour_buffer(symbol, price)

        # ENTRY: Use strategy's entry logic
        if position is None:
            # Get signal with custom target/stop
            signal = check_scalp_entry_signal(
                prices=historical,
                current_price=current_price,
                symbol=symbol,
                entry_fee_pct=TAKER_FEE * 100,  # Convert to percentage
                exit_fee_pct=TAKER_FEE * 100,
                tax_rate_pct=TAX_RATE * 100
            )

            if signal and 'rejected' in signal.get('reason', '').lower():
                signals_rejected += 1

            if signal and signal.get('signal') == 'buy':
                # Override target/stop with our test parameters
                entry_price = signal['entry_price']

                # Calculate custom target
                target_calc = calculate_required_price_for_target_profit(
                    entry_price=entry_price,
                    target_net_profit_pct=target_net_pct,
                    entry_fee_pct=TAKER_FEE * 100,
                    exit_fee_pct=TAKER_FEE * 100,
                    tax_rate_pct=TAX_RATE * 100
                )

                # Custom stop
                stop_price = entry_price * (1 - stop_gross_pct / 100)

                position = {
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'entry_timestamp': timestamps[i],
                    'stop': stop_price,
                    'target': target_calc['required_exit_price'],
                    'strategy': signal['strategy'],
                    'confidence': signal.get('confidence', 'medium'),
                    'metrics': signal.get('metrics', {}),
                    'target_net_pct': target_net_pct,
                    'stop_gross_pct': stop_gross_pct
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

            # Max hold time
            elif hours_held >= max_hold_hours:
                exit_price = current_price
                exit_reason = 'max_hold'

            if exit_price:
                pnl = calculate_net_profit(position['entry_price'], exit_price, POSITION_SIZE_USD,
                                          position['target_net_pct'])

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
                    'target_net_pct': position['target_net_pct'],
                    'stop_gross_pct': position['stop_gross_pct']
                })
                position = None

    # Close remaining position
    if position:
        final_price = prices[-1]
        hours_held = len(prices) - 1 - position['entry_idx']
        pnl = calculate_net_profit(position['entry_price'], final_price, POSITION_SIZE_USD,
                                   position['target_net_pct'])

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
            'target_net_pct': position['target_net_pct'],
            'stop_gross_pct': position['stop_gross_pct']
        })

    if not trades:
        return None

    # Calculate stats
    total_trades = len(trades)
    profitable_trades = [t for t in trades if t['profitable']]
    wins = [t for t in trades if t['net_pnl_usd'] > 0]
    losses = [t for t in trades if t['net_pnl_usd'] <= 0]

    target_exits = [t for t in trades if t['exit_reason'] == 'target']
    stop_exits = [t for t in trades if t['exit_reason'] == 'stop_loss']
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
        'timeout_exits': len(timeout_exits),
        'target_hit_rate': target_hit_rate,
        'stop_hit_rate': stop_hit_rate,
        'avg_hold_hours': avg_hold_hours,
        'signals_rejected': signals_rejected,
        'trades': trades
    }


def main():
    """Test multiple parameter combinations."""

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [
        wallet['symbol'] for wallet in config['wallets']
        if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
    ]

    print("="*120)
    print("PARAMETER OPTIMIZATION BACKTEST")
    print("="*120)
    print("\nTesting different target/stop combinations to find optimal parameters.")
    print(f"Symbols: {', '.join(enabled_symbols)}")
    print()

    # Test configurations
    # Format: (target_net_pct, stop_gross_pct, description)
    test_configs = [
        (0.8, 0.4, "CURRENT - Aggressive target, tight stop"),
        (0.6, 0.6, "BALANCED - Moderate target, moderate stop (1:1)"),
        (0.5, 0.6, "CONSERVATIVE - Lower target, moderate stop"),
        (0.6, 0.8, "WIDE STOP - Moderate target, wide stop"),
        (0.4, 0.5, "TIGHT - Low target, tight stop"),
        (0.5, 0.5, "ULTRA BALANCED - Tight target/stop (1:1)"),
    ]

    all_results = []

    for target_net, stop_gross, description in test_configs:
        print(f"\n{'='*120}")
        print(f"TESTING: Target={target_net}% NET, Stop={stop_gross}% GROSS - {description}")
        print(f"{'='*120}")

        config_results = []

        for symbol in enabled_symbols:
            result = backtest_with_parameters(
                symbol=symbol,
                target_net_pct=target_net,
                stop_gross_pct=stop_gross,
                max_hold_hours=4,
                max_hours=4320
            )

            if result:
                config_results.append(result)
                status = "✅" if result['total_net_pnl_usd'] > 0 else "❌"
                print(f"  {symbol}: {status} {result['total_trades']} trades, "
                      f"{result['win_rate']:.1f}% win rate, "
                      f"${result['total_net_pnl_usd']:.2f} P/L, "
                      f"{result['target_hit_rate']:.1f}% target / {result['stop_hit_rate']:.1f}% stop")

        # Aggregate stats for this configuration
        if config_results:
            total_trades = sum(r['total_trades'] for r in config_results)
            total_pnl = sum(r['total_net_pnl_usd'] for r in config_results)
            avg_win_rate = sum(r['win_rate'] for r in config_results) / len(config_results)
            avg_target_hit = sum(r['target_hit_rate'] for r in config_results) / len(config_results)
            avg_stop_hit = sum(r['stop_hit_rate'] for r in config_results) / len(config_results)
            avg_ev = total_pnl / total_trades if total_trades > 0 else 0
            profitable_symbols = len([r for r in config_results if r['total_net_pnl_usd'] > 0])

            print(f"\n  AGGREGATE: {total_trades} trades, {avg_win_rate:.1f}% win rate, "
                  f"${total_pnl:.2f} total P/L, ${avg_ev:.2f} EV/trade")
            print(f"  Target hit: {avg_target_hit:.1f}% | Stop hit: {avg_stop_hit:.1f}% | "
                  f"{profitable_symbols}/{len(config_results)} symbols profitable")

            all_results.append({
                'target_net_pct': target_net,
                'stop_gross_pct': stop_gross,
                'description': description,
                'total_trades': total_trades,
                'total_pnl_usd': total_pnl,
                'avg_win_rate': avg_win_rate,
                'avg_target_hit_rate': avg_target_hit,
                'avg_stop_hit_rate': avg_stop_hit,
                'avg_ev_per_trade': avg_ev,
                'profitable_symbols': profitable_symbols,
                'total_symbols': len(config_results),
                'symbol_results': config_results
            })

    # Final comparison
    print("\n\n")
    print("="*120)
    print("FINAL COMPARISON - ALL CONFIGURATIONS")
    print("="*120)
    print(f"{'Config':<45} {'Trades':<8} {'Win%':<8} {'Target%':<9} {'Stop%':<8} {'EV/Trade':<10} {'Total P/L':<12} {'Profitable'}")
    print("-"*120)

    # Sort by total P/L
    all_results_sorted = sorted(all_results, key=lambda x: x['total_pnl_usd'], reverse=True)

    for result in all_results_sorted:
        config_label = f"Target: {result['target_net_pct']}%, Stop: {result['stop_gross_pct']}%"
        status = "✅" if result['total_pnl_usd'] > 0 else "❌"

        print(f"{config_label:<45} {result['total_trades']:<8} "
              f"{result['avg_win_rate']:<7.1f}% {result['avg_target_hit_rate']:<8.1f}% "
              f"{result['avg_stop_hit_rate']:<7.1f}% ${result['avg_ev_per_trade']:<9.2f} "
              f"${result['total_pnl_usd']:<11.2f} {result['profitable_symbols']}/{result['total_symbols']} {status}")

    print("="*120)
    print()

    # Recommendation
    best_config = all_results_sorted[0]
    print("RECOMMENDATION:")
    print(f"  Best configuration: Target={best_config['target_net_pct']}% NET, Stop={best_config['stop_gross_pct']}% GROSS")
    print(f"  {best_config['description']}")
    print(f"  Expected win rate: {best_config['avg_win_rate']:.1f}%")
    print(f"  Expected EV per trade: ${best_config['avg_ev_per_trade']:.2f}")
    print(f"  Total P/L over 180 days: ${best_config['total_pnl_usd']:.2f}")
    print()

    if best_config['avg_win_rate'] >= 47 and best_config['total_pnl_usd'] > 0:
        print("✅ READY TO LAUNCH: This configuration meets profitability targets.")
    elif best_config['total_pnl_usd'] > 0:
        print("⚠️  BORDERLINE: Strategy is profitable but below 47% win rate target.")
        print("   Consider paper trading first.")
    else:
        print("❌ NOT READY: All configurations are unprofitable. Strategy needs more work.")

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'configurations_tested': len(test_configs),
        'results': all_results
    }

    output_file = 'backtest/parameter_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"📊 Detailed results saved to {output_file}")


if __name__ == "__main__":
    main()
