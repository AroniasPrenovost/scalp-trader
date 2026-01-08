#!/usr/bin/env python3
"""
Analyze trade history to determine optimal min_profit_percentage for early profit rotation.
This will replace the fixed min_profit_usd with a percentage-based approach.
"""

import json
import statistics
from datetime import datetime
from typing import List, Dict

def load_trades(filepath: str) -> List[Dict]:
    """Load trade history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_profit_distribution(trades: List[Dict]):
    """Analyze the distribution of profits from all trades."""

    # Separate winning and losing trades
    winning_trades = [t for t in trades if t['potential_profit_percentage'] > 0]
    losing_trades = [t for t in trades if t['potential_profit_percentage'] <= 0]

    # Get profit percentages
    all_profits_pct = [t['potential_profit_percentage'] for t in trades]
    winning_profits_pct = [t['potential_profit_percentage'] for t in winning_trades]
    losing_profits_pct = [t['potential_profit_percentage'] for t in losing_trades]

    # Get actual total profit after costs
    all_total_profit_usd = [t['total_profit'] for t in trades]
    winning_total_profit_usd = [t['total_profit'] for t in winning_trades]

    print("=" * 80)
    print("TRADE HISTORY ANALYSIS")
    print("=" * 80)
    print(f"\nTotal trades: {len(trades)}")
    print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("PROFIT PERCENTAGE DISTRIBUTION (potential_profit_percentage)")
    print("=" * 80)

    if all_profits_pct:
        print(f"\nAll trades:")
        print(f"  Mean: {statistics.mean(all_profits_pct):.2f}%")
        print(f"  Median: {statistics.median(all_profits_pct):.2f}%")
        print(f"  Min: {min(all_profits_pct):.2f}%")
        print(f"  Max: {max(all_profits_pct):.2f}%")
        print(f"  Std Dev: {statistics.stdev(all_profits_pct) if len(all_profits_pct) > 1 else 0:.2f}%")

    if winning_profits_pct:
        print(f"\nWinning trades only:")
        print(f"  Mean: {statistics.mean(winning_profits_pct):.2f}%")
        print(f"  Median: {statistics.median(winning_profits_pct):.2f}%")
        print(f"  Min: {min(winning_profits_pct):.2f}%")
        print(f"  Max: {max(winning_profits_pct):.2f}%")
        print(f"  25th percentile: {statistics.quantiles(winning_profits_pct, n=4)[0]:.2f}%")
        print(f"  75th percentile: {statistics.quantiles(winning_profits_pct, n=4)[2]:.2f}%")

    print("\n" + "=" * 80)
    print("ACTUAL PROFIT USD DISTRIBUTION (total_profit after fees & taxes)")
    print("=" * 80)

    if all_total_profit_usd:
        print(f"\nAll trades:")
        print(f"  Mean: ${statistics.mean(all_total_profit_usd):.2f}")
        print(f"  Median: ${statistics.median(all_total_profit_usd):.2f}")
        print(f"  Total: ${sum(all_total_profit_usd):.2f}")
        print(f"  Min: ${min(all_total_profit_usd):.2f}")
        print(f"  Max: ${max(all_total_profit_usd):.2f}")

    if winning_total_profit_usd:
        print(f"\nWinning trades only:")
        print(f"  Mean: ${statistics.mean(winning_total_profit_usd):.2f}")
        print(f"  Median: ${statistics.median(winning_total_profit_usd):.2f}")
        print(f"  Total: ${sum(winning_total_profit_usd):.2f}")

    # Analyze profit thresholds
    print("\n" + "=" * 80)
    print("PROFIT THRESHOLD ANALYSIS")
    print("=" * 80)
    print("\nIf we used different min_profit_percentage thresholds for early exit:")

    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for threshold in thresholds:
        # Trades that would qualify for early exit
        qualifying_trades = [t for t in trades if t['potential_profit_percentage'] >= threshold]
        qualifying_profit = sum([t['total_profit'] for t in qualifying_trades])

        # Trades that peaked above threshold but closed below
        peaked_above = [t for t in trades
                       if t['potential_profit_percentage'] < threshold
                       and t.get('exit_analysis', {}).get('exit_trigger') != 'profit_target']

        print(f"\n  {threshold:.1f}% threshold:")
        print(f"    Trades qualifying: {len(qualifying_trades)} ({len(qualifying_trades)/len(trades)*100:.1f}%)")
        print(f"    Total profit captured: ${qualifying_profit:.2f}")
        if qualifying_trades:
            print(f"    Avg profit per qualifying trade: ${qualifying_profit/len(qualifying_trades):.2f}")

    # Analyze by position size
    print("\n" + "=" * 80)
    print("PROFIT % BY POSITION SIZE")
    print("=" * 80)

    # Calculate profit percentage relative to position size
    for trade in trades[:10]:  # Show first 10 as examples
        position_value = trade.get('position_sizing', {}).get('entry_position_value', 0)
        total_profit_usd = trade['total_profit']
        if position_value > 0:
            profit_pct_of_position = (total_profit_usd / position_value) * 100
            print(f"\n{trade['symbol']} - {trade['timestamp'][:10]}")
            print(f"  Position value: ${position_value:.2f}")
            print(f"  Total profit USD: ${total_profit_usd:.2f}")
            print(f"  Profit % of position: {profit_pct_of_position:.2f}%")
            print(f"  potential_profit_percentage: {trade['potential_profit_percentage']:.2f}%")

    # Calculate relationship between position size and min profit percentage
    print("\n" + "=" * 80)
    print("CALCULATING OPTIMAL MIN_PROFIT_PERCENTAGE")
    print("=" * 80)

    # Current config uses min_profit_usd = 5.0
    # Let's see what percentage that represents across different position sizes
    typical_position_sizes = [t.get('position_sizing', {}).get('entry_position_value', 0)
                             for t in trades if t.get('position_sizing', {}).get('entry_position_value', 0) > 0]

    if typical_position_sizes:
        avg_position = statistics.mean(typical_position_sizes)
        median_position = statistics.median(typical_position_sizes)

        print(f"\nTypical position sizes:")
        print(f"  Average: ${avg_position:.2f}")
        print(f"  Median: ${median_position:.2f}")
        print(f"  Min: ${min(typical_position_sizes):.2f}")
        print(f"  Max: ${max(typical_position_sizes):.2f}")

        print(f"\nIf min_profit_usd = $5.00:")
        print(f"  On ${avg_position:.2f} position = {(5.0/avg_position)*100:.2f}%")
        print(f"  On ${median_position:.2f} position = {(5.0/median_position)*100:.2f}%")
        print(f"  On ${min(typical_position_sizes):.2f} position = {(5.0/min(typical_position_sizes))*100:.2f}%")
        print(f"  On ${max(typical_position_sizes):.2f} position = {(5.0/max(typical_position_sizes))*100:.2f}%")

    # Recommendation based on winning trade distribution
    if winning_profits_pct:
        percentiles = statistics.quantiles(winning_profits_pct, n=4)
        q1 = percentiles[0]  # 25th percentile
        median_win = statistics.median(winning_profits_pct)
        q3 = percentiles[2]  # 75th percentile

        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print(f"\nBased on winning trade distribution:")
        print(f"  25th percentile (conservative): {q1:.2f}%")
        print(f"  50th percentile (median): {median_win:.2f}%")
        print(f"  75th percentile (aggressive): {q3:.2f}%")
        print(f"\nRecommended min_profit_percentage: {q1:.2f}% to {median_win:.2f}%")
        print(f"\nRationale:")
        print(f"  - {q1:.2f}% captures 75% of winning trades")
        print(f"  - {median_win:.2f}% captures 50% of winning trades")
        print(f"  - Lower threshold = more early exits, less risk of giving back gains")
        print(f"  - Higher threshold = fewer exits, potentially higher profits but more risk")

if __name__ == '__main__':
    trades = load_trades('/Users/arons_stuff/Documents/scalp-scripts/transactions/data.json')
    analyze_profit_distribution(trades)
