#!/usr/bin/env python3
"""
Advanced analysis to determine optimal early profit threshold.
Focus: How much profit did we "give back" by not exiting early?
"""

import json
import statistics

def load_trades(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_optimal_threshold(trades):
    """
    Determine the sweet spot for early profit rotation.
    Key insight: We want to capture gains before they evaporate.
    """

    print("=" * 80)
    print("OPTIMAL EARLY PROFIT THRESHOLD ANALYSIS")
    print("=" * 80)

    # Analyze trades by final outcome
    profitable_trades = [t for t in trades if t['total_profit'] > 0]
    breakeven_trades = [t for t in trades if -5 <= t['total_profit'] <= 5]  # Near breakeven
    losing_trades = [t for t in trades if t['total_profit'] < -5]

    print(f"\nTrade outcomes:")
    print(f"  Profitable (>$0): {len(profitable_trades)}")
    print(f"  Near breakeven (-$5 to +$5): {len(breakeven_trades)}")
    print(f"  Losing (<-$5): {len(losing_trades)}")

    # Key metric: If we had exited at different % thresholds, what would we have captured?
    print("\n" + "=" * 80)
    print("SCENARIO ANALYSIS: Early Exit Performance")
    print("=" * 80)

    thresholds = [0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    for threshold in thresholds:
        # Simulate: exit when profit >= threshold
        total_captured = 0
        exits_taken = 0
        trades_held = 0

        for trade in trades:
            potential_pct = trade['potential_profit_percentage']
            final_profit = trade['total_profit']

            # Would we have exited at this threshold?
            # Note: potential_profit_percentage is the final result
            # For simulation, assume if final >= threshold, we exited there
            # If final < threshold, we held and got the final result

            if potential_pct >= threshold:
                # We would exit at threshold (assume we capture that %)
                position_value = trade.get('position_sizing', {}).get('entry_position_value', 1000)
                # Estimate profit at threshold (rough approximation)
                estimated_profit = (position_value * threshold / 100) - (position_value * 0.018)  # Rough fees/tax
                total_captured += max(estimated_profit, final_profit * (threshold / potential_pct))
                exits_taken += 1
            else:
                # Hold and take final result
                total_captured += final_profit
                trades_held += 1

        avg_per_trade = total_captured / len(trades) if trades else 0

        print(f"\n{threshold}% threshold:")
        print(f"  Early exits taken: {exits_taken} ({exits_taken/len(trades)*100:.1f}%)")
        print(f"  Trades held to target: {trades_held} ({trades_held/len(trades)*100:.1f}%)")
        print(f"  Total P&L: ${total_captured:.2f}")
        print(f"  Avg per trade: ${avg_per_trade:.2f}")
        print(f"  vs. Actual: ${sum(t['total_profit'] for t in trades):.2f}")

    # Analyze the winning trades more carefully
    print("\n" + "=" * 80)
    print("WINNING TRADES ANALYSIS")
    print("=" * 80)

    winning_details = []
    for trade in profitable_trades:
        pct = trade['potential_profit_percentage']
        profit = trade['total_profit']
        position = trade.get('position_sizing', {}).get('entry_position_value', 0)

        winning_details.append({
            'symbol': trade['symbol'],
            'pct': pct,
            'profit_usd': profit,
            'position': position,
            'date': trade['timestamp'][:10]
        })

    # Sort by profit percentage
    winning_details.sort(key=lambda x: x['pct'])

    print(f"\nAll {len(winning_details)} winning trades (sorted by %):")
    for i, w in enumerate(winning_details, 1):
        print(f"  {i:2d}. {w['symbol']:10s} {w['date']} | {w['pct']:5.2f}% | ${w['profit_usd']:7.2f} | Pos: ${w['position']:.0f}")

    # Find the inflection point
    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLD CALCULATION")
    print("=" * 80)

    # Strategy: Capture most winning trades while avoiding giving back gains
    winning_pcts = [w['pct'] for w in winning_details]

    if len(winning_pcts) > 4:
        p10 = statistics.quantiles(winning_pcts, n=10)[0]  # 10th percentile
        p25 = statistics.quantiles(winning_pcts, n=4)[0]   # 25th percentile
        p33 = statistics.quantiles(winning_pcts, n=3)[0]   # 33rd percentile
        p50 = statistics.median(winning_pcts)               # 50th percentile

        print(f"\nWinning trade percentiles:")
        print(f"  10th: {p10:.2f}% (captures 90% of winners)")
        print(f"  25th: {p25:.2f}% (captures 75% of winners)")
        print(f"  33rd: {p33:.2f}% (captures 67% of winners)")
        print(f"  50th: {p50:.2f}% (captures 50% of winners)")

        # Calculate expected value at each threshold
        print(f"\nExpected value analysis:")
        for name, pct_threshold in [("10th pctl", p10), ("25th pctl", p25), ("33rd pctl", p33), ("50th pctl", p50)]:
            qualifying = [w for w in winning_details if w['pct'] >= pct_threshold]
            total = sum(w['profit_usd'] for w in qualifying)
            print(f"  {name} ({pct_threshold:.2f}%): ${total:.2f} from {len(qualifying)} trades")

    # Final recommendation
    print("\n" + "=" * 80)
    print("🎯 FINAL RECOMMENDATION")
    print("=" * 80)

    # Based on your current $5 min_profit_usd with median position of ~$985
    # That's about 0.51%

    # But looking at data:
    # - 25th percentile of winners is 0.40%
    # - Median position is $985
    # - You have 51% win rate

    # Sweet spot calculation:
    median_position = statistics.median([t.get('position_sizing', {}).get('entry_position_value', 0)
                                        for t in trades if t.get('position_sizing', {}).get('entry_position_value', 0) > 0])

    current_usd_threshold = 5.0
    current_pct_equivalent = (current_usd_threshold / median_position) * 100

    print(f"\nCurrent setup:")
    print(f"  min_profit_usd: ${current_usd_threshold}")
    print(f"  Median position: ${median_position:.2f}")
    print(f"  Equivalent %: {current_pct_equivalent:.2f}%")

    if len(winning_pcts) > 4:
        p25 = statistics.quantiles(winning_pcts, n=4)[0]
        p33 = statistics.quantiles(winning_pcts, n=3)[0]

        print(f"\n💡 Recommended min_profit_percentage: {p25:.2f}% to {p33:.2f}%")
        print(f"\nWhy this range?")
        print(f"  ✓ {p25:.2f}% captures 75% of your winning trades")
        print(f"  ✓ {p33:.2f}% captures 67% of your winning trades")
        print(f"  ✓ Current ${current_usd_threshold} = ~{current_pct_equivalent:.2f}%, which is close to 25th percentile")
        print(f"  ✓ Lower threshold = lock in gains faster, reduce risk of reversals")
        print(f"  ✓ Your median winning trade is {statistics.median(winning_pcts):.2f}%")
        print(f"\n⚡ SUGGESTED VALUE: {p25:.2f}%")
        print(f"   This will trigger early rotation when you have ~75% of typical winning gains,")
        print(f"   protecting you from the negative → positive → negative scenario.")

if __name__ == '__main__':
    trades = load_trades('/Users/arons_stuff/Documents/scalp-scripts/transactions/data.json')
    analyze_optimal_threshold(trades)
