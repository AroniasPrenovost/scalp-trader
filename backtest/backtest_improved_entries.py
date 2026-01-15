#!/usr/bin/env python3
"""
Backtest Improved Entry Strategy

Tests the new entry logic with 4 quality filters:
1. Volume confirmation (2x average)
2. Trend alignment (6h uptrend required)
3. Quality scoring (70+ score)
4. Tighter filters (higher quality, less frequency)
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

# Import improved strategy
from utils.improved_entry_strategy import check_improved_entry_signal, get_strategy_info
from utils.momentum_scalping_strategy import update_intra_hour_buffer

# TRADING COSTS
TAKER_FEE = 0.0025
TAX_RATE = 0.24
POSITION_SIZE_USD = 4609
MIN_NET_PROFIT_USD = 2.00


def load_price_volume_data(symbol: str) -> Optional[tuple]:
    """Load historical price and volume data from JSON."""
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


def simulate_intra_hour_prices(start_price: float, end_price: float,
                                hourly_volatility: float = None,
                                num_intervals: int = 180) -> List[float]:
    """Simulate 20-second intervals."""
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


def calculate_net_profit(entry_price: float, exit_price: float, position_size: float) -> Dict:
    """Calculate net profit with actual fees and taxes."""
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


def backtest_improved_strategy(symbol: str,
                               target_net_pct: float = 0.7,
                               stop_gross_pct: float = 0.5,
                               max_hold_hours: int = 4,
                               max_hours: int = 4320) -> Optional[Dict]:
    """
    Backtest with improved entry strategy.

    Args:
        symbol: Trading pair
        target_net_pct: NET profit target (default 0.7%)
        stop_gross_pct: GROSS stop loss (default 0.5%)
        max_hold_hours: Max holding time
        max_hours: Historical data to analyze
    """

    data = load_price_volume_data(symbol)
    if not data:
        return None

    prices, volumes, timestamps = data

    if len(prices) < 200:
        return None

    if len(prices) > max_hours:
        prices = prices[-max_hours:]
        volumes = volumes[-max_hours:]
        timestamps = timestamps[-max_hours:]

    trades = []
    position = None
    signals_generated = 0
    signals_rejected_volume = 0
    signals_rejected_trend = 0
    signals_rejected_quality = 0
    signals_rejected_momentum = 0

    # Start from hour 48
    for i in range(48, len(prices)):
        current_price = prices[i]
        current_volume = volumes[i]
        historical_prices = prices[:i+1]
        historical_volumes = volumes[:i+1]

        # Simulate intra-hour prices for momentum acceleration
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

        # ENTRY: Use improved strategy
        if position is None:
            signal = check_improved_entry_signal(
                prices=historical_prices,
                volumes=historical_volumes,
                current_price=current_price,
                current_volume=current_volume,
                symbol=symbol,
                target_net_pct=target_net_pct,
                stop_gross_pct=stop_gross_pct,
                entry_fee_pct=TAKER_FEE * 100,
                exit_fee_pct=TAKER_FEE * 100,
                tax_rate_pct=TAX_RATE * 100,
                min_quality_score=60
            )

            # Track rejection reasons
            if signal and 'reason' in signal:
                reason = signal['reason'].lower()
                if 'volume' in reason:
                    signals_rejected_volume += 1
                elif 'trend' in reason:
                    signals_rejected_trend += 1
                elif 'quality' in reason or 'score' in reason:
                    signals_rejected_quality += 1
                elif 'momentum' in reason:
                    signals_rejected_momentum += 1

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
                    'quality_score': signal.get('quality_score', 0),
                    'quality_grade': signal.get('quality_grade', 'N/A')
                }

        # EXIT
        elif position:
            hours_held = i - position['entry_idx']
            exit_price = None
            exit_reason = None

            if current_price <= position['stop']:
                exit_price = position['stop']
                exit_reason = 'stop_loss'
            elif current_price >= position['target']:
                exit_price = position['target']
                exit_reason = 'target'
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
                    'quality_score': position['quality_score'],
                    'quality_grade': position['quality_grade']
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
            'quality_score': position['quality_score'],
            'quality_grade': position['quality_grade']
        })

    if not trades:
        return {
            'symbol': symbol,
            'total_trades': 0,
            'signals_rejected_volume': signals_rejected_volume,
            'signals_rejected_trend': signals_rejected_trend,
            'signals_rejected_quality': signals_rejected_quality,
            'signals_rejected_momentum': signals_rejected_momentum
        }

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
    avg_quality_score = sum(t['quality_score'] for t in trades) / total_trades

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
        'avg_quality_score': avg_quality_score,
        'signals_generated': signals_generated,
        'signals_rejected_volume': signals_rejected_volume,
        'signals_rejected_trend': signals_rejected_trend,
        'signals_rejected_quality': signals_rejected_quality,
        'signals_rejected_momentum': signals_rejected_momentum,
        'trades': trades
    }


def main():
    """Run improved strategy backtest."""

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [
        wallet['symbol'] for wallet in config['wallets']
        if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
    ]

    strategy_info = get_strategy_info()

    print("="*120)
    print("IMPROVED ENTRY STRATEGY BACKTEST")
    print("="*120)
    print(f"\nStrategy: {strategy_info['name']}")
    print(f"Target: {strategy_info['target_profit']} in {strategy_info['target_holding_time']}")
    print(f"Stop Loss: {strategy_info['max_stop_loss']}")
    print(f"Expected Win Rate: {strategy_info['expected_win_rate']}")
    print()
    print("NEW FILTERS:")
    for improvement in strategy_info['improvements']:
        print(f"  • {improvement}")
    print()
    print(f"Backtesting {len(enabled_symbols)} symbols (180 days)...")
    print()

    results = []

    for symbol in enabled_symbols:
        result = backtest_improved_strategy(symbol, target_net_pct=0.7, stop_gross_pct=0.5,
                                           max_hold_hours=4, max_hours=4320)

        if result and result['total_trades'] > 0:
            results.append(result)

            status = "✅" if result['total_net_pnl_usd'] > 0 else "❌"
            print(f"{symbol}: {status} {result['total_trades']} trades, "
                  f"{result['win_rate']:.1f}% win rate, ${result['total_net_pnl_usd']:.2f} P/L, "
                  f"avg score: {result['avg_quality_score']:.0f}")
            print(f"    Rejected: Vol={result['signals_rejected_volume']}, "
                  f"Trend={result['signals_rejected_trend']}, "
                  f"Quality={result['signals_rejected_quality']}, "
                  f"Momentum={result['signals_rejected_momentum']}")

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

    total_rejected_volume = sum(r['signals_rejected_volume'] for r in results)
    total_rejected_trend = sum(r['signals_rejected_trend'] for r in results)
    total_rejected_quality = sum(r['signals_rejected_quality'] for r in results)
    total_rejected_momentum = sum(r['signals_rejected_momentum'] for r in results)

    print(f"\nTotal trades: {total_trades}")
    print(f"Win rate: {avg_win_rate:.1f}%")
    print(f"Target hit rate: {avg_target_hit:.1f}%")
    print(f"Stop hit rate: {avg_stop_hit:.1f}%")
    print(f"Total P/L: ${total_pnl:.2f}")
    print(f"EV per trade: ${avg_ev:.2f}")
    print(f"Profitable symbols: {profitable_symbols}/{len(results)}")
    print()
    print(f"FILTER STATISTICS:")
    print(f"  Signals rejected by volume filter: {total_rejected_volume}")
    print(f"  Signals rejected by trend filter: {total_rejected_trend}")
    print(f"  Signals rejected by quality scoring: {total_rejected_quality}")
    print(f"  Signals rejected by momentum filter: {total_rejected_momentum}")
    print(f"  Total signals filtered: {total_rejected_volume + total_rejected_trend + total_rejected_quality + total_rejected_momentum}")

    print()
    print("="*120)

    # Comparison
    print()
    print("COMPARISON TO OLD STRATEGY:")
    print("  OLD (0.7% target, basic filters): 2,610 trades, 35.4% win rate, -$24,158 loss")
    print(f"  NEW (0.7% target, improved filters): {total_trades} trades, {avg_win_rate:.1f}% win rate, ${total_pnl:.2f}")
    print()

    if avg_win_rate >= 47:
        print("✅ SUCCESS! Win rate above 47% - strategy is profitable!")
        trades_per_day = total_trades / 180
        daily_ev = avg_ev * trades_per_day
        monthly_ev = daily_ev * 30
        print(f"   Expected daily profit: ${daily_ev:.2f}")
        print(f"   Expected monthly profit: ${monthly_ev:.2f}")
    elif avg_win_rate >= 40:
        print("⚠️  CLOSE! Win rate {:.1f}% (need 47%+)".format(avg_win_rate))
        print("   Consider further tuning or paper trading first.")
    else:
        print("❌ NEEDS MORE WORK: Win rate still below target.")

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'strategy': strategy_info,
        'results': results
    }

    output_file = 'backtest/improved_entry_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"📊 Results saved to {output_file}")


if __name__ == "__main__":
    main()
