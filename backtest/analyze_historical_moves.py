#!/usr/bin/env python3
"""
Analyze Historical Price Movements

This script analyzes raw price data to understand what size moves
actually occur within different timeframes, helping us set realistic
profit targets and stop losses.
"""

import json
import os
from typing import Dict, List

def load_price_data(symbol: str) -> List[float]:
    """Load historical price data."""
    filepath = f"coinbase-data/{symbol}.json"
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    prices = [float(entry['price']) for entry in data_sorted]

    return prices


def analyze_forward_moves(prices: List[float], max_hours: int = 4) -> Dict:
    """
    Analyze forward price movements from any given point.

    For each hour, calculate:
    - Max favorable move within next N hours
    - Min unfavorable move within next N hours
    - Where price ends up after N hours

    This tells us what targets/stops are realistic.
    """

    results = {
        'max_favorable_1h': [],
        'max_favorable_2h': [],
        'max_favorable_3h': [],
        'max_favorable_4h': [],
        'min_unfavorable_1h': [],
        'min_unfavorable_2h': [],
        'min_unfavorable_3h': [],
        'min_unfavorable_4h': [],
        'end_price_1h': [],
        'end_price_2h': [],
        'end_price_3h': [],
        'end_price_4h': [],
    }

    # Analyze from each starting point
    for i in range(len(prices) - max_hours):
        entry_price = prices[i]

        for hours in range(1, max_hours + 1):
            if i + hours >= len(prices):
                break

            # Get all prices within the next N hours
            future_prices = prices[i:i+hours+1]

            # Calculate max favorable move (highest point)
            max_price = max(future_prices)
            max_favorable_pct = ((max_price - entry_price) / entry_price) * 100

            # Calculate min unfavorable move (lowest point)
            min_price = min(future_prices)
            min_unfavorable_pct = ((min_price - entry_price) / entry_price) * 100

            # Calculate ending move
            end_price = prices[i + hours]
            end_pct = ((end_price - entry_price) / entry_price) * 100

            results[f'max_favorable_{hours}h'].append(max_favorable_pct)
            results[f'min_unfavorable_{hours}h'].append(min_unfavorable_pct)
            results[f'end_price_{hours}h'].append(end_pct)

    return results


def print_percentiles(data: List[float], label: str):
    """Print percentile distribution of data."""
    sorted_data = sorted(data)
    n = len(sorted_data)

    percentiles = [10, 25, 50, 75, 90, 95, 99]

    print(f"\n{label}:")
    for p in percentiles:
        idx = int(n * p / 100)
        if idx < n:
            print(f"  {p}th percentile: {sorted_data[idx]:.2f}%")


def analyze_symbol(symbol: str, max_hours_data: int = 4320) -> Dict:
    """Analyze a single symbol's price movements."""

    prices = load_price_data(symbol)
    if not prices:
        return None

    # Use last 180 days
    if len(prices) > max_hours_data:
        prices = prices[-max_hours_data:]

    if len(prices) < 100:
        return None

    print(f"\n{'='*100}")
    print(f"ANALYZING: {symbol} ({len(prices)} hours of data)")
    print(f"{'='*100}")

    # Analyze forward moves
    moves = analyze_forward_moves(prices, max_hours=4)

    # Print analysis for each timeframe
    for hours in [1, 2, 3, 4]:
        print(f"\n--- {hours}-HOUR FORWARD ANALYSIS ---")

        max_favorable = moves[f'max_favorable_{hours}h']
        min_unfavorable = moves[f'min_unfavorable_{hours}h']
        end_moves = moves[f'end_price_{hours}h']

        print(f"\nData points analyzed: {len(max_favorable)}")

        # Max favorable (upside potential)
        print_percentiles(max_favorable, f"Max Favorable Move (highest point reached in {hours}h)")

        # Min unfavorable (downside risk)
        print_percentiles(min_unfavorable, f"Min Unfavorable Move (lowest point reached in {hours}h)")

        # Ending position
        print_percentiles(end_moves, f"Ending Position (where price is after {hours}h)")

        # Calculate realistic target/stop levels
        print(f"\n*** REALISTIC PARAMETERS FOR {hours}H HOLDING PERIOD ***")

        # Target: What % of trades reach at least X% favorable move?
        targets = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5]
        print("\nTarget Hit Rate (% of trades that reach target):")
        for target in targets:
            hit_count = len([m for m in max_favorable if m >= target])
            hit_rate = (hit_count / len(max_favorable)) * 100
            print(f"  {target}% target: {hit_rate:.1f}% of trades reach it")

        # Stop: What % of trades drop below X%?
        stops = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        print("\nStop Hit Rate (% of trades that drop to stop level):")
        for stop in stops:
            hit_count = len([m for m in min_unfavorable if m <= -stop])
            hit_rate = (hit_count / len(min_unfavorable)) * 100
            print(f"  -{stop}% stop: {hit_rate:.1f}% of trades hit it")

    return {
        'symbol': symbol,
        'data_points': len(prices),
        'moves': moves
    }


def main():
    """Analyze all enabled symbols."""

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [
        wallet['symbol'] for wallet in config['wallets']
        if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
    ]

    print("="*100)
    print("HISTORICAL PRICE MOVEMENT ANALYSIS")
    print("="*100)
    print("\nThis analysis helps determine realistic profit targets and stop losses")
    print("by examining what actually happens to price in the hours following any entry.")
    print()

    results = []

    for symbol in enabled_symbols:
        result = analyze_symbol(symbol, max_hours_data=4320)
        if result:
            results.append(result)

    # Aggregate analysis across all symbols
    print("\n\n")
    print("="*100)
    print("AGGREGATE ANALYSIS (ALL SYMBOLS)")
    print("="*100)

    if results:
        for hours in [1, 2, 4]:
            print(f"\n{'='*100}")
            print(f"{hours}-HOUR HOLDING PERIOD - RECOMMENDED PARAMETERS")
            print(f"{'='*100}")

            # Aggregate all moves
            all_max_favorable = []
            all_min_unfavorable = []

            for result in results:
                all_max_favorable.extend(result['moves'][f'max_favorable_{hours}h'])
                all_min_unfavorable.extend(result['moves'][f'min_unfavorable_{hours}h'])

            # Calculate what % hit various targets
            print("\nTARGET RECOMMENDATIONS (based on hit rates):")
            targets = [0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
            for target in targets:
                hit_count = len([m for m in all_max_favorable if m >= target])
                hit_rate = (hit_count / len(all_max_favorable)) * 100

                # Color code based on viability
                if hit_rate >= 50:
                    status = "✅ VERY ACHIEVABLE"
                elif hit_rate >= 35:
                    status = "✓ ACHIEVABLE"
                elif hit_rate >= 20:
                    status = "⚠️  CHALLENGING"
                else:
                    status = "❌ TOO AGGRESSIVE"

                print(f"  {target}% gross target: {hit_rate:.1f}% hit rate - {status}")

            print("\nSTOP LOSS RECOMMENDATIONS (based on hit rates):")
            stops = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
            for stop in stops:
                hit_count = len([m for m in all_min_unfavorable if m <= -stop])
                hit_rate = (hit_count / len(all_min_unfavorable)) * 100

                # Color code based on viability
                if hit_rate <= 20:
                    status = "✅ RARELY HIT"
                elif hit_rate <= 35:
                    status = "✓ ACCEPTABLE"
                elif hit_rate <= 50:
                    status = "⚠️  FREQUENTLY HIT"
                else:
                    status = "❌ TOO TIGHT"

                print(f"  -{stop}% stop: {hit_rate:.1f}% hit rate - {status}")

            print("\n*** OPTIMAL COMBINATIONS ***")
            print(f"\nFor {hours}h holding period, recommended configurations:")

            # Find optimal combos
            configs = [
                (0.4, 0.4, "Ultra-tight, quick scalp"),
                (0.5, 0.5, "Tight, balanced 1:1"),
                (0.6, 0.6, "Moderate, balanced 1:1"),
                (0.8, 0.6, "Conservative target, moderate stop"),
                (0.5, 0.6, "Tight target, moderate stop"),
                (0.6, 0.8, "Moderate target, wide stop"),
            ]

            for target, stop, desc in configs:
                target_hit_rate = len([m for m in all_max_favorable if m >= target]) / len(all_max_favorable) * 100
                stop_hit_rate = len([m for m in all_min_unfavorable if m <= -stop]) / len(all_min_unfavorable) * 100

                # Calculate expected value (simplified)
                # Assume: if target hit first = win, if stop hit first = loss
                # This is rough but gives us an idea
                win_rate_proxy = target_hit_rate / (target_hit_rate + stop_hit_rate) * 100 if (target_hit_rate + stop_hit_rate) > 0 else 0

                print(f"\n  Target: {target}%, Stop: {stop}% ({desc})")
                print(f"    Target hit rate: {target_hit_rate:.1f}%")
                print(f"    Stop hit rate: {stop_hit_rate:.1f}%")
                print(f"    Estimated win rate: ~{win_rate_proxy:.1f}%")

                # Rough EV calculation (not accounting for fees/taxes yet)
                ev = (win_rate_proxy/100 * target) - ((100-win_rate_proxy)/100 * stop)
                print(f"    Rough expected value: {ev:.2f}% per trade")


if __name__ == "__main__":
    main()
