#!/usr/bin/env python3
"""
Hybrid Trading Strategy Backtest

Adapts strategy based on market conditions:
- Consolidation: Tight scalping (mean reversion)
- Trending: Momentum following (ride trends)
- Mixed: Conservative parameters
"""

import json
import os
import random
import math
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append(os.path.dirname(__file__))

# ACTUAL TRADING COSTS
TAKER_FEE = 0.0025  # 0.25% Coinbase taker fee
TAX_RATE = 0.24     # 24% federal tax rate
POSITION_SIZE_USD = 4609
MIN_NET_PROFIT_USD = 2.00


def detect_market_condition(prices: List[float], lookback: int = 24) -> Dict:
    """
    Detect current market condition: consolidation, trending, or mixed.

    Returns:
        {
            'condition': 'consolidation' | 'trending' | 'mixed',
            'volatility_24h': float,
            'momentum_6h': float,
            'range_position': float (0-100),
            'trend_strength': float (0-100)
        }
    """
    if len(prices) < lookback:
        return {'condition': 'unknown'}

    recent = prices[-lookback:]
    current_price = prices[-1]

    # Calculate 24h volatility (range as % of price)
    high_24h = max(recent)
    low_24h = min(recent)
    range_24h = high_24h - low_24h
    volatility_24h = (range_24h / low_24h) * 100

    # Calculate 6h momentum
    if len(prices) >= 6:
        price_6h_ago = prices[-7]
        momentum_6h = ((current_price - price_6h_ago) / price_6h_ago) * 100
    else:
        momentum_6h = 0

    # Calculate position in range
    range_position = ((current_price - low_24h) / range_24h * 100) if range_24h > 0 else 50

    # Calculate trend strength (using simple momentum magnitude)
    trend_strength = abs(momentum_6h) * 10  # Scale to 0-100
    trend_strength = min(trend_strength, 100)

    # Determine condition
    condition = 'mixed'

    # CONSOLIDATION: Low volatility + weak momentum
    if volatility_24h < 3.0 and abs(momentum_6h) < 1.0:
        condition = 'consolidation'

    # TRENDING: High volatility + strong momentum
    elif volatility_24h > 5.0 and abs(momentum_6h) > 2.0:
        condition = 'trending'

    # MIXED: Everything else
    else:
        condition = 'mixed'

    return {
        'condition': condition,
        'volatility_24h': volatility_24h,
        'momentum_6h': momentum_6h,
        'range_position': range_position,
        'trend_strength': trend_strength
    }


def get_strategy_parameters(condition: str) -> Dict:
    """
    Get strategy parameters based on market condition.

    Args:
        condition: 'consolidation', 'trending', or 'mixed'

    Returns:
        Dictionary with target_net_pct, stop_gross_pct, max_hold_hours, entry_types
    """

    if condition == 'consolidation':
        # TIGHT SCALPING: Quick in/out in range-bound markets
        return {
            'target_net_pct': 0.4,
            'stop_gross_pct': 0.4,
            'max_hold_hours': 2,
            'entry_types': ['support_bounce'],  # Mean reversion only
            'description': 'Consolidation scalp (tight range)'
        }

    elif condition == 'trending':
        # MOMENTUM FOLLOWING: Ride the trend with room to breathe
        return {
            'target_net_pct': 0.8,
            'stop_gross_pct': 0.8,
            'max_hold_hours': 6,
            'entry_types': ['breakout'],  # Momentum only
            'description': 'Trend following (momentum)'
        }

    else:  # mixed
        # CONSERVATIVE: Balanced approach
        return {
            'target_net_pct': 0.5,
            'stop_gross_pct': 0.6,
            'max_hold_hours': 4,
            'entry_types': ['support_bounce', 'breakout'],  # Both
            'description': 'Mixed market (conservative)'
        }


def simulate_intra_hour_prices(start_price: float, end_price: float,
                                hourly_volatility: float = None,
                                num_intervals: int = 180) -> List[float]:
    """Simulate realistic 20-second price intervals."""
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


def calculate_net_profit(entry_price: float, exit_price: float, position_size: float) -> Dict:
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

    return {
        'gross_pnl_pct': gross_pct,
        'net_pnl_pct': net_pct,
        'net_pnl_usd': net_profit,
        'exit_fee_usd': exit_fee,
        'tax_usd': tax,
        'profitable': net_profit >= MIN_NET_PROFIT_USD
    }


# Import strategy functions
from utils.momentum_scalping_strategy import (
    check_scalp_entry_signal,
    update_intra_hour_buffer
)
from utils.profit_calculator import calculate_required_price_for_target_profit


def backtest_hybrid_strategy(symbol: str, max_hours: int = 4320) -> Optional[Dict]:
    """
    Backtest hybrid strategy that adapts to market conditions.

    Args:
        symbol: Trading pair
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

    # Track market conditions
    condition_counts = {'consolidation': 0, 'trending': 0, 'mixed': 0}
    trades_by_condition = {'consolidation': [], 'trending': [], 'mixed': []}

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

        # DETECT MARKET CONDITION
        market_condition = detect_market_condition(historical, lookback=24)
        condition = market_condition['condition']

        if condition != 'unknown':
            condition_counts[condition] = condition_counts.get(condition, 0) + 1

        # GET STRATEGY PARAMETERS FOR THIS CONDITION
        strategy_params = get_strategy_parameters(condition)

        # ENTRY: Use strategy's entry logic
        if position is None:
            # Get signal
            signal = check_scalp_entry_signal(
                prices=historical,
                current_price=current_price,
                symbol=symbol,
                entry_fee_pct=TAKER_FEE * 100,
                exit_fee_pct=TAKER_FEE * 100,
                tax_rate_pct=TAX_RATE * 100
            )

            if signal and 'rejected' in signal.get('reason', '').lower():
                signals_rejected += 1

            # Check if this signal type is allowed in current market condition
            if signal and signal.get('signal') == 'buy':
                signal_strategy = signal.get('strategy')

                # Only trade if signal type matches the market condition strategy
                if signal_strategy in strategy_params['entry_types']:
                    entry_price = signal['entry_price']

                    # Calculate custom target for this condition
                    target_calc = calculate_required_price_for_target_profit(
                        entry_price=entry_price,
                        target_net_profit_pct=strategy_params['target_net_pct'],
                        entry_fee_pct=TAKER_FEE * 100,
                        exit_fee_pct=TAKER_FEE * 100,
                        tax_rate_pct=TAX_RATE * 100
                    )

                    # Custom stop for this condition
                    stop_price = entry_price * (1 - strategy_params['stop_gross_pct'] / 100)

                    position = {
                        'entry_idx': i,
                        'entry_price': entry_price,
                        'entry_timestamp': timestamps[i],
                        'stop': stop_price,
                        'target': target_calc['required_exit_price'],
                        'strategy': signal_strategy,
                        'confidence': signal.get('confidence', 'medium'),
                        'market_condition': condition,
                        'condition_metrics': market_condition,
                        'strategy_params': strategy_params,
                        'max_hold_hours': strategy_params['max_hold_hours']
                    }

        # EXIT
        elif position:
            hours_held = i - position['entry_idx']
            exit_price = None
            exit_reason = None

            # Check stop loss
            if current_price <= position['stop']:
                exit_price = position['stop']
                exit_reason = 'stop_loss'

            # Check profit target
            elif current_price >= position['target']:
                exit_price = position['target']
                exit_reason = 'target'

            # Max hold time (specific to market condition)
            elif hours_held >= position['max_hold_hours']:
                exit_price = current_price
                exit_reason = 'max_hold'

            if exit_price:
                pnl = calculate_net_profit(position['entry_price'], exit_price, POSITION_SIZE_USD)

                trade = {
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
                    'market_condition': position['market_condition'],
                    'condition_volatility': position['condition_metrics']['volatility_24h'],
                    'condition_momentum': position['condition_metrics']['momentum_6h'],
                    'strategy_description': position['strategy_params']['description']
                }

                trades.append(trade)
                trades_by_condition[position['market_condition']].append(trade)
                position = None

    # Close remaining position
    if position:
        final_price = prices[-1]
        hours_held = len(prices) - 1 - position['entry_idx']
        pnl = calculate_net_profit(position['entry_price'], final_price, POSITION_SIZE_USD)

        trade = {
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
            'market_condition': position['market_condition'],
            'condition_volatility': position['condition_metrics']['volatility_24h'],
            'condition_momentum': position['condition_metrics']['momentum_6h'],
            'strategy_description': position['strategy_params']['description']
        }

        trades.append(trade)
        trades_by_condition[position['market_condition']].append(trade)

    if not trades:
        return None

    # Calculate overall stats
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

    # Calculate stats by condition
    def condition_stats(condition_trades):
        if not condition_trades:
            return None

        ct_total = len(condition_trades)
        ct_profitable = [t for t in condition_trades if t['profitable']]
        ct_wins = [t for t in condition_trades if t['net_pnl_usd'] > 0]
        ct_target = [t for t in condition_trades if t['exit_reason'] == 'target']
        ct_stop = [t for t in condition_trades if t['exit_reason'] == 'stop_loss']
        ct_pnl = sum(t['net_pnl_usd'] for t in condition_trades)

        return {
            'total_trades': ct_total,
            'win_rate': (len(ct_wins) / ct_total) * 100,
            'profitability_rate': (len(ct_profitable) / ct_total) * 100,
            'target_hit_rate': (len(ct_target) / ct_total) * 100,
            'stop_hit_rate': (len(ct_stop) / ct_total) * 100,
            'total_pnl_usd': ct_pnl,
            'avg_pnl_usd': ct_pnl / ct_total
        }

    return {
        'symbol': symbol,
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
        'condition_counts': condition_counts,
        'consolidation_stats': condition_stats(trades_by_condition['consolidation']),
        'trending_stats': condition_stats(trades_by_condition['trending']),
        'mixed_stats': condition_stats(trades_by_condition['mixed']),
        'trades': trades
    }


def main():
    """Run hybrid strategy backtest."""

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [
        wallet['symbol'] for wallet in config['wallets']
        if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
    ]

    print("="*120)
    print("HYBRID STRATEGY BACKTEST")
    print("="*120)
    print("\nAdaptive strategy based on market conditions:")
    print("  • CONSOLIDATION (<3% volatility, <1% momentum): Tight scalping (0.4% target/stop, 2h hold)")
    print("  • TRENDING (>5% volatility, >2% momentum): Momentum following (0.8% target/stop, 6h hold)")
    print("  • MIXED (everything else): Conservative (0.5% target, 0.6% stop, 4h hold)")
    print()

    results = []

    for symbol in enabled_symbols:
        result = backtest_hybrid_strategy(symbol, max_hours=4320)

        if result:
            results.append(result)

            status = "✅" if result['total_net_pnl_usd'] > 0 else "❌"
            print(f"{symbol}: {status} {result['total_trades']} trades, "
                  f"{result['win_rate']:.1f}% win rate, ${result['total_net_pnl_usd']:.2f} P/L")

            # Show breakdown by condition
            if result['consolidation_stats']:
                cs = result['consolidation_stats']
                print(f"    Consolidation: {cs['total_trades']} trades, {cs['win_rate']:.1f}% win, ${cs['total_pnl_usd']:.2f}")
            if result['trending_stats']:
                ts = result['trending_stats']
                print(f"    Trending: {ts['total_trades']} trades, {ts['win_rate']:.1f}% win, ${ts['total_pnl_usd']:.2f}")
            if result['mixed_stats']:
                ms = result['mixed_stats']
                print(f"    Mixed: {ms['total_trades']} trades, {ms['win_rate']:.1f}% win, ${ms['total_pnl_usd']:.2f}")

    if not results:
        print("No results to display.")
        return

    # Aggregate stats
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

    print(f"\nTotal trades: {total_trades}")
    print(f"Win rate: {avg_win_rate:.1f}%")
    print(f"Target hit rate: {avg_target_hit:.1f}%")
    print(f"Stop hit rate: {avg_stop_hit:.1f}%")
    print(f"Total P/L: ${total_pnl:.2f}")
    print(f"EV per trade: ${avg_ev:.2f}")
    print(f"Profitable symbols: {profitable_symbols}/{len(results)}")

    # Breakdown by condition
    print()
    print("PERFORMANCE BY MARKET CONDITION:")
    print("-"*120)

    for condition in ['consolidation', 'trending', 'mixed']:
        condition_results = [r[f'{condition}_stats'] for r in results if r[f'{condition}_stats']]

        if condition_results:
            total_cond_trades = sum(cr['total_trades'] for cr in condition_results)
            avg_cond_win_rate = sum(cr['win_rate'] for cr in condition_results) / len(condition_results)
            total_cond_pnl = sum(cr['total_pnl_usd'] for cr in condition_results)
            avg_cond_ev = total_cond_pnl / total_cond_trades if total_cond_trades > 0 else 0

            print(f"\n{condition.upper()}:")
            print(f"  Trades: {total_cond_trades}")
            print(f"  Win rate: {avg_cond_win_rate:.1f}%")
            print(f"  Total P/L: ${total_cond_pnl:.2f}")
            print(f"  EV per trade: ${avg_cond_ev:.2f}")

    print()
    print("="*120)

    # Verdict
    if avg_win_rate >= 47 and total_pnl > 0:
        print("✅ READY TO LAUNCH: Hybrid strategy meets profitability targets!")
        print("   Win rate above 47% and positive total P/L.")
    elif total_pnl > 0:
        print("⚠️  BORDERLINE: Strategy is profitable but win rate below 47%.")
        print("   Consider paper trading first to validate results.")
    else:
        print("❌ NEEDS MORE WORK: Strategy still unprofitable with hybrid approach.")
        print("   May need to refine market condition detection or entry signals.")

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'strategy_type': 'hybrid',
        'description': 'Adaptive strategy based on market conditions',
        'results': results
    }

    output_file = 'backtest/hybrid_strategy_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"📊 Results saved to {output_file}")


if __name__ == "__main__":
    main()
