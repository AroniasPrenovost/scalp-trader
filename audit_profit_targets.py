#!/usr/bin/env python3
"""
Audit Price Action - Find Optimal Profit Targets
Analyze how often different profit targets are actually hit after entry signals
"""

import json
import os
from typing import Dict, List, Optional
import sys
sys.path.append(os.path.dirname(__file__))
from utils.momentum_scalping_strategy import check_scalp_entry_signal


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


def analyze_price_movement_after_signal(prices: List[float], entry_idx: int,
                                        entry_price: float, max_hours: int = 6) -> Dict:
    """
    After an entry signal, analyze how far price actually moves.

    Returns what % targets would have been hit and when.
    """
    targets_to_test = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    stop_loss = 0.4  # Fixed 0.4% stop

    results = {
        'entry_price': entry_price,
        'max_upside': 0,
        'max_downside': 0,
        'hit_stop': False,
        'hours_to_stop': None,
        'targets_hit': {}
    }

    # Check price movement for next max_hours
    for h in range(1, min(max_hours + 1, len(prices) - entry_idx)):
        future_price = prices[entry_idx + h]
        pct_change = ((future_price - entry_price) / entry_price) * 100

        # Track max movement
        results['max_upside'] = max(results['max_upside'], pct_change)
        results['max_downside'] = min(results['max_downside'], pct_change)

        # Check stop loss
        if pct_change <= -stop_loss and not results['hit_stop']:
            results['hit_stop'] = True
            results['hours_to_stop'] = h

        # Check each target
        for target_pct in targets_to_test:
            if target_pct not in results['targets_hit'] and pct_change >= target_pct:
                results['targets_hit'][target_pct] = {
                    'hours': h,
                    'price': future_price,
                    'before_stop': not results['hit_stop']
                }

    return results


def audit_symbol(symbol: str, max_hours_analysis: int = 4320) -> Optional[Dict]:
    """Audit price action after entry signals for a symbol."""

    data = load_price_data(symbol)
    if not data:
        return None

    prices, timestamps = data

    if len(prices) < 200:
        return None

    if len(prices) > max_hours_analysis:
        prices = prices[-max_hours_analysis:]
        timestamps = timestamps[-max_hours_analysis:]

    signals_analyzed = []

    # Find all entry signals
    for i in range(48, len(prices) - 6):  # Need 6 hours ahead to analyze
        current_price = prices[i]
        historical = prices[:i+1]

        signal = check_scalp_entry_signal(historical, current_price)

        if signal and signal.get('signal') == 'buy':
            # Analyze what happened after this signal
            movement = analyze_price_movement_after_signal(
                prices, i, signal['entry_price'], max_hours=6
            )

            signals_analyzed.append({
                'timestamp': timestamps[i],
                'strategy': signal['strategy'],
                'entry_price': signal['entry_price'],
                'movement': movement
            })

    if not signals_analyzed:
        return None

    # Calculate statistics
    total_signals = len(signals_analyzed)
    hit_stop = sum(1 for s in signals_analyzed if s['movement']['hit_stop'])

    # For each target level, calculate hit rate
    target_stats = {}
    targets_to_test = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

    for target_pct in targets_to_test:
        hits = [s for s in signals_analyzed if target_pct in s['movement']['targets_hit']]
        hits_before_stop = [s for s in hits if s['movement']['targets_hit'][target_pct]['before_stop']]

        if hits:
            avg_hours = sum(s['movement']['targets_hit'][target_pct]['hours'] for s in hits) / len(hits)
        else:
            avg_hours = None

        target_stats[target_pct] = {
            'total_hits': len(hits),
            'hit_rate': (len(hits) / total_signals) * 100,
            'hits_before_stop': len(hits_before_stop),
            'hit_rate_before_stop': (len(hits_before_stop) / total_signals) * 100,
            'avg_hours_to_hit': avg_hours
        }

    # By strategy type
    support_bounces = [s for s in signals_analyzed if s['strategy'] == 'support_bounce']
    breakouts = [s for s in signals_analyzed if s['strategy'] == 'breakout']

    def strategy_target_stats(strategy_signals):
        if not strategy_signals:
            return None
        stats = {}
        for target_pct in targets_to_test:
            hits = [s for s in strategy_signals if target_pct in s['movement']['targets_hit']]
            hits_before_stop = [s for s in hits if s['movement']['targets_hit'][target_pct]['before_stop']]
            stats[target_pct] = {
                'hit_rate': (len(hits) / len(strategy_signals)) * 100,
                'hit_rate_before_stop': (len(hits_before_stop) / len(strategy_signals)) * 100
            }
        return stats

    return {
        'symbol': symbol,
        'total_signals': total_signals,
        'stop_loss_rate': (hit_stop / total_signals) * 100,
        'target_stats': target_stats,
        'support_bounce_stats': strategy_target_stats(support_bounces),
        'breakout_stats': strategy_target_stats(breakouts),
        'support_bounce_count': len(support_bounces),
        'breakout_count': len(breakouts)
    }


def main():
    """Audit all symbols to find optimal targets."""

    with open('config.json', 'r') as f:
        config = json.load(f)

    symbols = [w['symbol'] for w in config['wallets'] if w.get('enabled', False)]

    print(f"\n{'='*120}")
    print(f"PRICE ACTION AUDIT - OPTIMAL PROFIT TARGET ANALYSIS")
    print(f"{'='*120}")
    print(f"Analyzing entry signals to see how often different profit targets are actually hit")
    print(f"Fixed Stop Loss: 0.4%")
    print(f"Testing Targets: 0.4%, 0.6%, 0.8%, 1.0%, 1.2%, 1.5%, 2.0%")
    print(f"{'='*120}\n")

    results = []

    for symbol in symbols:
        print(f"Auditing {symbol}...", end=" ", flush=True)
        result = audit_symbol(symbol)

        if result:
            results.append(result)
            print(f"✓ {result['total_signals']} signals analyzed")
        else:
            print("⚠️  No data")

    if not results:
        print("\n⚠️  No results")
        return

    # Print detailed results
    print(f"\n{'='*120}")
    print(f"TARGET HIT RATES (Before Stop Loss)")
    print(f"{'='*120}")
    print(f"{'Symbol':<12} {'Signals':<10} {'0.4%':<8} {'0.6%':<8} {'0.8%':<8} {'1.0%':<8} "
          f"{'1.2%':<8} {'1.5%':<8} {'2.0%':<8}")
    print(f"{'-'*120}")

    for r in results:
        print(f"{r['symbol']:<12} {r['total_signals']:<10} ", end="")
        for target in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
            hit_rate = r['target_stats'][target]['hit_rate_before_stop']
            print(f"{hit_rate:>5.1f}%  ", end="")
        print()

    # Overall averages
    print(f"{'-'*120}")
    print(f"{'AVERAGE':<12} {sum(r['total_signals'] for r in results):<10} ", end="")
    for target in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        total_signals = sum(r['total_signals'] for r in results)
        total_hits = sum(r['target_stats'][target]['hits_before_stop'] for r in results)
        avg_hit_rate = (total_hits / total_signals) * 100
        print(f"{avg_hit_rate:>5.1f}%  ", end="")
    print()
    print(f"{'='*120}\n")

    # Average time to hit targets
    print(f"AVERAGE HOURS TO HIT TARGET")
    print(f"{'-'*120}")
    print(f"{'Target':<10} {'Avg Hours':<12} {'Hit Rate':<12}")
    print(f"{'-'*120}")

    for target in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        all_times = []
        for r in results:
            if r['target_stats'][target]['avg_hours_to_hit']:
                all_times.append(r['target_stats'][target]['avg_hours_to_hit'])

        total_signals = sum(r['total_signals'] for r in results)
        total_hits = sum(r['target_stats'][target]['hits_before_stop'] for r in results)
        hit_rate = (total_hits / total_signals) * 100

        if all_times:
            avg_time = sum(all_times) / len(all_times)
            print(f"{target}%{'':<7} {avg_time:<12.1f} {hit_rate:.1f}%")
        else:
            print(f"{target}%{'':<7} {'N/A':<12} {hit_rate:.1f}%")

    print(f"{'='*120}\n")

    # Strategy breakdown
    print(f"BY STRATEGY TYPE")
    print(f"{'-'*120}")
    print(f"\nSUPPORT BOUNCE (Bottom 30% of range):")
    print(f"{'Target':<10} {'Hit Rate Before Stop':<25}")
    print(f"{'-'*120}")

    for target in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        total_support = sum(r['support_bounce_count'] for r in results if r['support_bounce_stats'])
        if total_support > 0:
            support_stats = [r['support_bounce_stats'][target] for r in results if r['support_bounce_stats']]
            avg_hit_rate = sum(s['hit_rate_before_stop'] for s in support_stats) / len(support_stats)
            print(f"{target}%{'':<7} {avg_hit_rate:.1f}%")

    print(f"\nBREAKOUT (Top 30% of range):")
    print(f"{'Target':<10} {'Hit Rate Before Stop':<25}")
    print(f"{'-'*120}")

    for target in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        total_breakout = sum(r['breakout_count'] for r in results if r['breakout_stats'])
        if total_breakout > 0:
            breakout_stats = [r['breakout_stats'][target] for r in results if r['breakout_stats']]
            avg_hit_rate = sum(s['hit_rate_before_stop'] for s in breakout_stats) / len(breakout_stats)
            print(f"{target}%{'':<7} {avg_hit_rate:.1f}%")

    print(f"\n{'='*120}\n")

    # Recommendations
    print(f"RECOMMENDATIONS:")
    print(f"{'-'*120}")

    total_signals = sum(r['total_signals'] for r in results)

    print(f"\nBased on {total_signals} entry signals across all symbols:\n")

    # Find optimal targets
    for target in [0.6, 0.8, 1.0, 1.2, 1.5]:
        total_hits = sum(r['target_stats'][target]['hits_before_stop'] for r in results)
        hit_rate = (total_hits / total_signals) * 100

        # Calculate expected value at this target
        # Assume 0.4% stop loss hit at inverse rate
        stop_rate = 100 - hit_rate  # Simplified

        # Net profit calculation (after 0.25% fees + 24% tax)
        # Win: target% - 0.25% fee - (target% * 0.24 tax)
        # Loss: -0.4% - 0.25% fee
        win_net_pct = target - 0.25 - (target * 0.24)
        loss_net_pct = -0.4 - 0.25

        expected_value_pct = (hit_rate/100 * win_net_pct) + ((100-hit_rate)/100 * loss_net_pct)
        expected_value_usd = expected_value_pct * 46.09  # Per $4609 position

        profitable_rate = hit_rate if win_net_pct >= 0.4 else 0  # Need 0.4% net for $2+

        print(f"  {target}% Target:")
        print(f"    Hit Rate: {hit_rate:.1f}%")
        print(f"    Win Net: {win_net_pct:+.2f}%, Loss Net: {loss_net_pct:+.2f}%")
        print(f"    Expected Value: {expected_value_pct:+.3f}% (${expected_value_usd:+.2f} per trade)")
        print(f"    Estimated Profitability: {profitable_rate:.1f}%")

        if expected_value_usd > 0:
            print(f"    ✅ POSITIVE EV")
        else:
            print(f"    ❌ NEGATIVE EV")
        print()

    print(f"{'='*120}\n")

    # Save results
    output = {
        'audit_date': '2026-01-10',
        'results': results
    }

    with open('analysis/profit_target_audit_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"📊 Detailed results saved to analysis/profit_target_audit_results.json\n")


if __name__ == '__main__':
    main()
