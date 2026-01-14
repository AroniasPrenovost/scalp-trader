#!/usr/bin/env python3
"""
Analyze Historical Data for Scalping Opportunities
Find where 0.5-1.0% moves actually happen and what conditions create them
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# Scalping targets
MIN_SCALP_TARGET = 0.005  # 0.5%
IDEAL_SCALP_TARGET = 0.008  # 0.8%
MAX_SCALP_TARGET = 0.010  # 1.0%

# For profitability
MIN_PROFITABLE_MOVE = 0.004  # 0.4% (minimum to make $2+)


def load_price_data(symbol: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Load price data from Coinbase and global volume from CoinGecko."""

    # Load Coinbase price data
    coinbase_filepath = f"coinbase-data/{symbol}.json"
    if not os.path.exists(coinbase_filepath):
        return None, None, None, None

    with open(coinbase_filepath, 'r') as f:
        coinbase_data = json.load(f)

    coinbase_sorted = sorted(coinbase_data, key=lambda x: x['timestamp'])
    prices = [float(entry['price']) for entry in coinbase_sorted]
    coinbase_volumes = [float(entry.get('volume_24h', 0)) for entry in coinbase_sorted]
    timestamps = [entry['timestamp'] for entry in coinbase_sorted]

    # Load CoinGecko global volume (matches index.py:881-882)
    global_volumes = []
    coingecko_filepath = f"coingecko-global-volume/{symbol}.json"

    if os.path.exists(coingecko_filepath):
        with open(coingecko_filepath, 'r') as f:
            coingecko_data = json.load(f)

        coingecko_sorted = sorted(coingecko_data, key=lambda x: x['timestamp'])

        # Match timestamps between coinbase and coingecko data
        coingecko_dict = {entry['timestamp']: float(entry.get('volume_24h', 0))
                         for entry in coingecko_sorted}

        # Use global volume if available, otherwise fall back to Coinbase volume
        for ts in timestamps:
            # Find closest coingecko timestamp (within 1 hour tolerance)
            closest_vol = None
            min_diff = float('inf')
            for cg_ts, vol in coingecko_dict.items():
                diff = abs(cg_ts - ts)
                if diff < min_diff and diff < 3600:  # Within 1 hour
                    min_diff = diff
                    closest_vol = vol

            global_volumes.append(closest_vol if closest_vol is not None else 0)
    else:
        # Fall back to Coinbase volumes if CoinGecko data not available
        print(f"  ⚠️  No CoinGecko volume data, using Coinbase volume")
        global_volumes = coinbase_volumes

    return prices, coinbase_volumes, global_volumes, timestamps


def calculate_hourly_metrics(prices: List[float], coinbase_volumes: List[float],
                             global_volumes: List[float], timestamps: List[float],
                             lookback: int = 24) -> List[Dict]:
    """
    For each hour, calculate:
    - Price change from previous hour
    - High/low range in that hour
    - Volume change
    - Volatility (recent price range)
    - Time of day
    - Success rate of entries at that point
    """
    metrics = []

    for i in range(lookback, len(prices)):
        current_price = prices[i]
        prev_price = prices[i-1]

        # Price change from last hour
        hourly_change_pct = ((current_price - prev_price) / prev_price) * 100

        # Look at next few hours to see if scalp opportunity existed
        future_high = current_price
        future_low = current_price
        hours_to_check = min(4, len(prices) - i - 1)  # Check next 4 hours

        for j in range(1, hours_to_check + 1):
            future_price = prices[i + j]
            future_high = max(future_high, future_price)
            future_low = min(future_low, future_price)

        # Calculate potential moves from current price
        max_upside_pct = ((future_high - current_price) / current_price) * 100
        max_downside_pct = ((future_low - current_price) / current_price) * 100

        # Did a 0.5%+ move happen?
        had_scalp_opportunity = max_upside_pct >= MIN_SCALP_TARGET * 100
        had_ideal_opportunity = max_upside_pct >= IDEAL_SCALP_TARGET * 100
        had_profitable_move = max_upside_pct >= MIN_PROFITABLE_MOVE * 100

        # How many hours until target hit?
        hours_to_target = None
        if had_scalp_opportunity:
            target_price = current_price * (1 + MIN_SCALP_TARGET)
            for j in range(1, hours_to_check + 1):
                if prices[i + j] >= target_price:
                    hours_to_target = j
                    break

        # Recent volatility
        recent_prices = prices[max(0, i-24):i+1]
        volatility_24h = ((max(recent_prices) - min(recent_prices)) / min(recent_prices)) * 100

        # Volume change (using global volume from CoinGecko)
        current_global_volume = global_volumes[i]
        prev_global_volume = global_volumes[i-1] if i > 0 else current_global_volume
        global_volume_change_pct = ((current_global_volume - prev_global_volume) / prev_global_volume * 100) if prev_global_volume > 0 else 0

        # Also track Coinbase volume for comparison
        current_coinbase_volume = coinbase_volumes[i]
        prev_coinbase_volume = coinbase_volumes[i-1] if i > 0 else current_coinbase_volume
        coinbase_volume_change_pct = ((current_coinbase_volume - prev_coinbase_volume) / prev_coinbase_volume * 100) if prev_coinbase_volume > 0 else 0

        # Time of day (hour in UTC)
        hour_of_day = datetime.fromtimestamp(timestamps[i]).hour

        # Recent momentum
        prices_last_3h = prices[max(0, i-3):i+1]
        momentum_3h = ((prices_last_3h[-1] - prices_last_3h[0]) / prices_last_3h[0]) * 100 if len(prices_last_3h) > 1 else 0

        # Is price near recent support/resistance?
        recent_24h = prices[max(0, i-24):i+1]
        price_range_24h = max(recent_24h) - min(recent_24h)
        distance_from_low = ((current_price - min(recent_24h)) / price_range_24h) * 100 if price_range_24h > 0 else 50
        distance_from_high = ((max(recent_24h) - current_price) / price_range_24h) * 100 if price_range_24h > 0 else 50

        metrics.append({
            'index': i,
            'timestamp': timestamps[i],
            'price': current_price,
            'hourly_change_pct': hourly_change_pct,
            'max_upside_pct': max_upside_pct,
            'max_downside_pct': max_downside_pct,
            'had_scalp_opportunity': had_scalp_opportunity,
            'had_ideal_opportunity': had_ideal_opportunity,
            'had_profitable_move': had_profitable_move,
            'hours_to_target': hours_to_target,
            'volatility_24h': volatility_24h,
            'global_volume_change_pct': global_volume_change_pct,
            'coinbase_volume_change_pct': coinbase_volume_change_pct,
            'global_volume': current_global_volume,
            'hour_of_day': hour_of_day,
            'momentum_3h': momentum_3h,
            'distance_from_low_pct': distance_from_low,
            'distance_from_high_pct': distance_from_high,
        })

    return metrics


def analyze_conditions(metrics: List[Dict]) -> Dict:
    """Analyze what conditions lead to scalping opportunities."""

    total_hours = len(metrics)
    opportunities = [m for m in metrics if m['had_scalp_opportunity']]
    ideal_opportunities = [m for m in metrics if m['had_ideal_opportunity']]
    profitable_moves = [m for m in metrics if m['had_profitable_move']]

    print(f"  Total hours analyzed: {total_hours}")
    print(f"  Hours with 0.5%+ move: {len(opportunities)} ({len(opportunities)/total_hours*100:.1f}%)")
    print(f"  Hours with 0.8%+ move: {len(ideal_opportunities)} ({len(ideal_opportunities)/total_hours*100:.1f}%)")
    print(f"  Hours with 0.4%+ move: {len(profitable_moves)} ({len(profitable_moves)/total_hours*100:.1f}%)")

    if not opportunities:
        return None

    # Analyze conditions
    analysis = {}

    # 1. Time of day patterns
    hour_distribution = defaultdict(int)
    for opp in opportunities:
        hour_distribution[opp['hour_of_day']] += 1

    best_hours = sorted(hour_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
    analysis['best_hours'] = best_hours

    # 2. Volatility ranges that produce opportunities
    volatilities = [m['volatility_24h'] for m in opportunities]
    analysis['avg_volatility_when_opportunity'] = sum(volatilities) / len(volatilities)
    analysis['min_volatility'] = min(volatilities)
    analysis['max_volatility'] = max(volatilities)

    # 3. Momentum patterns
    momentum_positive = [m for m in opportunities if m['momentum_3h'] > 0]
    momentum_negative = [m for m in opportunities if m['momentum_3h'] < 0]
    analysis['opportunities_with_positive_momentum'] = len(momentum_positive)
    analysis['opportunities_with_negative_momentum'] = len(momentum_negative)

    # 4. Position in range (bounces vs breakouts)
    near_low = [m for m in opportunities if m['distance_from_low_pct'] < 30]  # Bottom 30% of range
    near_high = [m for m in opportunities if m['distance_from_high_pct'] < 30]  # Top 30% of range
    mid_range = [m for m in opportunities if 30 <= m['distance_from_low_pct'] <= 70]

    analysis['opportunities_near_low'] = len(near_low)
    analysis['opportunities_near_high'] = len(near_high)
    analysis['opportunities_mid_range'] = len(mid_range)

    # 5. Speed to target
    with_target_data = [m for m in opportunities if m['hours_to_target'] is not None]
    if with_target_data:
        speeds = [m['hours_to_target'] for m in with_target_data]
        analysis['avg_hours_to_target'] = sum(speeds) / len(speeds)
        analysis['fast_moves_1h'] = len([s for s in speeds if s == 1])
        analysis['fast_moves_2h'] = len([s for s in speeds if s <= 2])

    # 6. Volume patterns (global CoinGecko volume)
    global_volume_changes = [m['global_volume_change_pct'] for m in opportunities]
    analysis['avg_global_volume_change'] = sum(global_volume_changes) / len(global_volume_changes)
    global_volume_spikes = [m for m in opportunities if m['global_volume_change_pct'] > 10]
    analysis['opportunities_with_global_volume_spike'] = len(global_volume_spikes)

    # Also track Coinbase-only volume
    coinbase_volume_changes = [m['coinbase_volume_change_pct'] for m in opportunities]
    analysis['avg_coinbase_volume_change'] = sum(coinbase_volume_changes) / len(coinbase_volume_changes)
    coinbase_volume_spikes = [m for m in opportunities if m['coinbase_volume_change_pct'] > 10]
    analysis['opportunities_with_coinbase_volume_spike'] = len(coinbase_volume_spikes)

    # 7. Hourly price action that preceded opportunity
    hourly_changes = [m['hourly_change_pct'] for m in opportunities]
    analysis['avg_hourly_change_before_opp'] = sum(hourly_changes) / len(hourly_changes)

    small_dips = [m for m in opportunities if -1.0 < m['hourly_change_pct'] < -0.2]
    small_pumps = [m for m in opportunities if 0.2 < m['hourly_change_pct'] < 1.0]
    consolidation = [m for m in opportunities if -0.2 <= m['hourly_change_pct'] <= 0.2]

    analysis['after_small_dip'] = len(small_dips)
    analysis['after_small_pump'] = len(small_pumps)
    analysis['after_consolidation'] = len(consolidation)

    return analysis


def print_analysis(symbol: str, analysis: Dict, metrics: List[Dict]):
    """Print detailed analysis."""
    if not analysis:
        print(f"  ⚠️  No scalping opportunities found")
        return

    total_opps = len([m for m in metrics if m['had_scalp_opportunity']])

    print(f"\n  📊 SCALPING OPPORTUNITY ANALYSIS")
    print(f"  {'-'*60}")

    print(f"\n  ⏰ TIME PATTERNS:")
    print(f"     Best hours (UTC): ", end="")
    for hour, count in analysis['best_hours']:
        pct = (count / total_opps) * 100
        print(f"{hour:02d}:00 ({count}, {pct:.0f}%)  ", end="")
    print()

    print(f"\n  📈 VOLATILITY RANGES:")
    print(f"     Avg volatility when opportunity: {analysis['avg_volatility_when_opportunity']:.2f}%")
    print(f"     Range: {analysis['min_volatility']:.2f}% - {analysis['max_volatility']:.2f}%")

    print(f"\n  🎯 MOMENTUM PATTERNS:")
    pos_pct = (analysis['opportunities_with_positive_momentum'] / total_opps) * 100
    neg_pct = (analysis['opportunities_with_negative_momentum'] / total_opps) * 100
    print(f"     Positive 3h momentum: {analysis['opportunities_with_positive_momentum']} ({pos_pct:.1f}%)")
    print(f"     Negative 3h momentum: {analysis['opportunities_with_negative_momentum']} ({neg_pct:.1f}%)")

    print(f"\n  📍 PRICE POSITION:")
    low_pct = (analysis['opportunities_near_low'] / total_opps) * 100
    high_pct = (analysis['opportunities_near_high'] / total_opps) * 100
    mid_pct = (analysis['opportunities_mid_range'] / total_opps) * 100
    print(f"     Near 24h low (bounces): {analysis['opportunities_near_low']} ({low_pct:.1f}%)")
    print(f"     Near 24h high: {analysis['opportunities_near_high']} ({high_pct:.1f}%)")
    print(f"     Mid-range: {analysis['opportunities_mid_range']} ({mid_pct:.1f}%)")

    if 'avg_hours_to_target' in analysis:
        print(f"\n  ⚡ SPEED TO TARGET (0.5%+):")
        print(f"     Avg hours to hit target: {analysis['avg_hours_to_target']:.1f}h")
        print(f"     Hit in 1 hour: {analysis['fast_moves_1h']} ({analysis['fast_moves_1h']/total_opps*100:.1f}%)")
        print(f"     Hit in ≤2 hours: {analysis['fast_moves_2h']} ({analysis['fast_moves_2h']/total_opps*100:.1f}%)")

    print(f"\n  📊 VOLUME PATTERNS:")
    print(f"     Global volume (CoinGecko):")
    print(f"       Avg change: {analysis['avg_global_volume_change']:+.2f}%")
    global_spike_pct = (analysis['opportunities_with_global_volume_spike'] / total_opps) * 100
    print(f"       With 10%+ spike: {analysis['opportunities_with_global_volume_spike']} ({global_spike_pct:.1f}%)")
    print(f"     Coinbase volume:")
    print(f"       Avg change: {analysis['avg_coinbase_volume_change']:+.2f}%")
    cb_spike_pct = (analysis['opportunities_with_coinbase_volume_spike'] / total_opps) * 100
    print(f"       With 10%+ spike: {analysis['opportunities_with_coinbase_volume_spike']} ({cb_spike_pct:.1f}%)")

    print(f"\n  📉 PRECEDING PRICE ACTION:")
    dip_pct = (analysis['after_small_dip'] / total_opps) * 100
    pump_pct = (analysis['after_small_pump'] / total_opps) * 100
    cons_pct = (analysis['after_consolidation'] / total_opps) * 100
    print(f"     After small dip (-0.2 to -1.0%): {analysis['after_small_dip']} ({dip_pct:.1f}%)")
    print(f"     After small pump (+0.2 to +1.0%): {analysis['after_small_pump']} ({pump_pct:.1f}%)")
    print(f"     After consolidation (±0.2%): {analysis['after_consolidation']} ({cons_pct:.1f}%)")


def main():
    """Analyze all symbols for scalping opportunities."""

    with open('config.json', 'r') as f:
        config = json.load(f)

    symbols = [w['symbol'] for w in config['wallets'] if w.get('enabled', False)]

    print(f"\n{'='*80}")
    print(f"SCALPING OPPORTUNITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Analyzing historical data to find 0.5-1.0% move patterns")
    print(f"Period: Last 180 days (4320 hours)")
    print(f"{'='*80}\n")

    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"📈 {symbol}")
        print(f"{'='*80}")

        result = load_price_data(symbol)
        if not result or result[0] is None:
            print(f"  ⚠️  No price data available")
            continue

        prices, coinbase_volumes, global_volumes, timestamps = result

        if not prices or len(prices) < 200:
            print(f"  ⚠️  Insufficient data")
            continue

        # Limit to last 4320 hours (180 days)
        if len(prices) > 4320:
            prices = prices[-4320:]
            coinbase_volumes = coinbase_volumes[-4320:]
            global_volumes = global_volumes[-4320:]
            timestamps = timestamps[-4320:]

        metrics = calculate_hourly_metrics(prices, coinbase_volumes, global_volumes, timestamps)
        analysis = analyze_conditions(metrics)

        if analysis:
            print_analysis(symbol, analysis, metrics)
            all_results[symbol] = {
                'analysis': analysis,
                'total_opportunities': len([m for m in metrics if m['had_scalp_opportunity']]),
                'total_hours': len(metrics)
            }

    # Overall summary
    print(f"\n\n{'='*80}")
    print(f"OVERALL SUMMARY - BEST SYMBOLS FOR SCALPING")
    print(f"{'='*80}\n")

    ranked = sorted(all_results.items(),
                   key=lambda x: x[1]['total_opportunities'] / x[1]['total_hours'],
                   reverse=True)

    print(f"{'Symbol':<12} {'Total Opps':<15} {'Opp Frequency':<20} {'Fast Moves (≤2h)':<20}")
    print(f"{'-'*80}")

    for symbol, data in ranked:
        opp_freq = (data['total_opportunities'] / data['total_hours']) * 100
        fast_moves = data['analysis'].get('fast_moves_2h', 0)
        fast_pct = (fast_moves / data['total_opportunities'] * 100) if data['total_opportunities'] > 0 else 0

        print(f"{symbol:<12} {data['total_opportunities']:<15} {opp_freq:>6.1f}%             "
              f"{fast_moves} ({fast_pct:.1f}%)")

    print(f"\n{'='*80}\n")

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'results': all_results
    }

    with open('analysis/scalping_opportunity_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"📊 Detailed results saved to analysis/scalping_opportunity_analysis.json\n")


if __name__ == '__main__':
    main()
