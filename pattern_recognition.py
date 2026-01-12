#!/usr/bin/env python3
"""
Pattern Recognition Analyzer
Analyzes historical moves to identify high-probability setup patterns.
Generates scoring weights based on statistical correlation with successful moves.
"""

import json
import statistics
from collections import defaultdict


def load_analysis():
    """Load the historical moves analysis"""
    with open('historical_moves_analysis.json', 'r') as f:
        return json.load(f)


def analyze_rsi_correlation(all_moves):
    """
    Analyze which RSI ranges have highest success rates.
    Returns optimal RSI ranges and their characteristics.
    """
    rsi_buckets = {
        'extreme_oversold_<20': [],
        'oversold_20-30': [],
        'lower_neutral_30-45': [],
        'middle_neutral_45-55': [],
        'upper_neutral_55-70': [],
        'overbought_70-80': [],
        'extreme_overbought_>80': []
    }

    for move in all_moves:
        rsi = move['conditions']['rsi']
        if rsi is None:
            continue

        avg_increase = move['price_increase_pct']

        if rsi < 20:
            rsi_buckets['extreme_oversold_<20'].append(avg_increase)
        elif rsi < 30:
            rsi_buckets['oversold_20-30'].append(avg_increase)
        elif rsi < 45:
            rsi_buckets['lower_neutral_30-45'].append(avg_increase)
        elif rsi < 55:
            rsi_buckets['middle_neutral_45-55'].append(avg_increase)
        elif rsi < 70:
            rsi_buckets['upper_neutral_55-70'].append(avg_increase)
        elif rsi < 80:
            rsi_buckets['overbought_70-80'].append(avg_increase)
        else:
            rsi_buckets['extreme_overbought_>80'].append(avg_increase)

    print("\n" + "="*80)
    print("RSI CORRELATION ANALYSIS")
    print("="*80)

    results = {}
    for bucket, increases in rsi_buckets.items():
        if not increases:
            continue

        count = len(increases)
        avg_increase = statistics.mean(increases)
        median_increase = statistics.median(increases)
        pct_above_2 = sum(1 for x in increases if x >= 2.0) / count * 100
        pct_above_2_5 = sum(1 for x in increases if x >= 2.5) / count * 100

        results[bucket] = {
            'count': count,
            'avg_increase': avg_increase,
            'median_increase': median_increase,
            'pct_above_2.0': pct_above_2,
            'pct_above_2.5': pct_above_2_5
        }

        print(f"\n{bucket}:")
        print(f"  Count: {count:,}")
        print(f"  Avg increase: {avg_increase:.2f}%")
        print(f"  Median increase: {median_increase:.2f}%")
        print(f"  % achieving 2.0%+: {pct_above_2:.1f}%")
        print(f"  % achieving 2.5%+: {pct_above_2_5:.1f}%")

    return results


def analyze_volume_correlation(all_moves):
    """Analyze volume profile impact on move success"""
    volume_profiles = defaultdict(list)

    for move in all_moves:
        profile = move['conditions']['volume_profile']
        increase = move['price_increase_pct']
        volume_profiles[profile].append(increase)

    print("\n" + "="*80)
    print("VOLUME PROFILE CORRELATION ANALYSIS")
    print("="*80)

    results = {}
    for profile, increases in sorted(volume_profiles.items(), key=lambda x: len(x[1]), reverse=True):
        if not increases:
            continue

        count = len(increases)
        avg_increase = statistics.mean(increases)
        median_increase = statistics.median(increases)
        pct_above_2 = sum(1 for x in increases if x >= 2.0) / count * 100
        pct_above_2_5 = sum(1 for x in increases if x >= 2.5) / count * 100

        results[profile] = {
            'count': count,
            'avg_increase': avg_increase,
            'median_increase': median_increase,
            'pct_above_2.0': pct_above_2,
            'pct_above_2.5': pct_above_2_5
        }

        print(f"\n{profile}:")
        print(f"  Count: {count:,}")
        print(f"  Avg increase: {avg_increase:.2f}%")
        print(f"  Median increase: {median_increase:.2f}%")
        print(f"  % achieving 2.0%+: {pct_above_2:.1f}%")
        print(f"  % achieving 2.5%+: {pct_above_2_5:.1f}%")

    return results


def analyze_price_position_correlation(all_moves):
    """Analyze price position in range impact"""
    # Bucket by position in 24h range
    position_buckets = {
        'very_low_0-20': [],
        'low_20-40': [],
        'middle_40-60': [],
        'high_60-80': [],
        'very_high_80-100': []
    }

    for move in all_moves:
        pos = move['conditions']['price_position_24h_pct']
        if pos is None:
            continue

        increase = move['price_increase_pct']

        if pos < 20:
            position_buckets['very_low_0-20'].append(increase)
        elif pos < 40:
            position_buckets['low_20-40'].append(increase)
        elif pos < 60:
            position_buckets['middle_40-60'].append(increase)
        elif pos < 80:
            position_buckets['high_60-80'].append(increase)
        else:
            position_buckets['very_high_80-100'].append(increase)

    print("\n" + "="*80)
    print("PRICE POSITION IN 24H RANGE CORRELATION")
    print("="*80)

    results = {}
    for bucket, increases in position_buckets.items():
        if not increases:
            continue

        count = len(increases)
        avg_increase = statistics.mean(increases)
        pct_above_2 = sum(1 for x in increases if x >= 2.0) / count * 100
        pct_above_2_5 = sum(1 for x in increases if x >= 2.5) / count * 100

        results[bucket] = {
            'count': count,
            'avg_increase': avg_increase,
            'pct_above_2.0': pct_above_2,
            'pct_above_2.5': pct_above_2_5
        }

        print(f"\n{bucket}:")
        print(f"  Count: {count:,}")
        print(f"  Avg increase: {avg_increase:.2f}%")
        print(f"  % achieving 2.0%+: {pct_above_2:.1f}%")
        print(f"  % achieving 2.5%+: {pct_above_2_5:.1f}%")

    return results


def analyze_ma_correlation(all_moves):
    """Analyze price vs MA(24h) correlation"""
    ma_buckets = {
        'far_below_<-3%': [],
        'below_-3_to_-1.5%': [],
        'slightly_below_-1.5_to_-0.5%': [],
        'near_ma_-0.5_to_0.5%': [],
        'slightly_above_0.5_to_1.5%': [],
        'above_1.5_to_3%': [],
        'far_above_>3%': []
    }

    for move in all_moves:
        ma_pct = move['conditions']['price_vs_ma24h_pct']
        if ma_pct is None:
            continue

        increase = move['price_increase_pct']

        if ma_pct < -3:
            ma_buckets['far_below_<-3%'].append(increase)
        elif ma_pct < -1.5:
            ma_buckets['below_-3_to_-1.5%'].append(increase)
        elif ma_pct < -0.5:
            ma_buckets['slightly_below_-1.5_to_-0.5%'].append(increase)
        elif ma_pct < 0.5:
            ma_buckets['near_ma_-0.5_to_0.5%'].append(increase)
        elif ma_pct < 1.5:
            ma_buckets['slightly_above_0.5_to_1.5%'].append(increase)
        elif ma_pct < 3:
            ma_buckets['above_1.5_to_3%'].append(increase)
        else:
            ma_buckets['far_above_>3%'].append(increase)

    print("\n" + "="*80)
    print("PRICE VS MA(24H) CORRELATION")
    print("="*80)

    results = {}
    for bucket, increases in ma_buckets.items():
        if not increases:
            continue

        count = len(increases)
        avg_increase = statistics.mean(increases)
        pct_above_2 = sum(1 for x in increases if x >= 2.0) / count * 100
        pct_above_2_5 = sum(1 for x in increases if x >= 2.5) / count * 100

        results[bucket] = {
            'count': count,
            'avg_increase': avg_increase,
            'pct_above_2.0': pct_above_2,
            'pct_above_2.5': pct_above_2_5
        }

        print(f"\n{bucket}:")
        print(f"  Count: {count:,}")
        print(f"  Avg increase: {avg_increase:.2f}%")
        print(f"  % achieving 2.0%+: {pct_above_2:.1f}%")
        print(f"  % achieving 2.5%+: {pct_above_2_5:.1f}%")

    return results


def analyze_time_of_day_correlation(all_moves):
    """Analyze best hours for moves"""
    hour_buckets = defaultdict(list)

    for move in all_moves:
        hour = move['conditions']['hour_of_day']
        increase = move['price_increase_pct']
        hour_buckets[hour].append(increase)

    print("\n" + "="*80)
    print("TIME OF DAY CORRELATION (Top 10 Hours)")
    print("="*80)

    results = {}
    for hour, increases in sorted(hour_buckets.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        count = len(increases)
        avg_increase = statistics.mean(increases)
        pct_above_2 = sum(1 for x in increases if x >= 2.0) / count * 100

        results[f"{hour:02d}:00"] = {
            'count': count,
            'avg_increase': avg_increase,
            'pct_above_2.0': pct_above_2
        }

        print(f"\n{hour:02d}:00:")
        print(f"  Count: {count:,}")
        print(f"  Avg increase: {avg_increase:.2f}%")
        print(f"  % achieving 2.0%+: {pct_above_2:.1f}%")

    return results


def identify_high_probability_setups(all_moves):
    """
    Identify combinations of factors that produce highest success rates
    """
    print("\n" + "="*80)
    print("HIGH PROBABILITY SETUP IDENTIFICATION")
    print("="*80)

    # Define ideal conditions based on previous analysis
    ideal_setups = {
        'oversold_bounce': lambda m: (
            m['conditions']['rsi'] is not None and
            m['conditions']['rsi'] < 30 and
            m['conditions']['volume_profile'] in ['increasing', 'spike'] and
            m['conditions']['price_position_24h_pct'] is not None and
            m['conditions']['price_position_24h_pct'] < 40
        ),
        'mean_reversion': lambda m: (
            m['conditions']['price_vs_ma24h_pct'] is not None and
            m['conditions']['price_vs_ma24h_pct'] < -1.5 and
            m['conditions']['rsi'] is not None and
            30 <= m['conditions']['rsi'] <= 50
        ),
        'momentum_continuation': lambda m: (
            m['conditions']['rsi'] is not None and
            45 <= m['conditions']['rsi'] <= 65 and
            m['conditions']['volume_profile'] in ['increasing', 'spike']
        ),
        'low_range_breakout': lambda m: (
            m['conditions']['price_position_24h_pct'] is not None and
            m['conditions']['price_position_24h_pct'] < 30 and
            m['conditions']['volume_profile'] == 'spike'
        ),
    }

    results = {}
    for setup_name, condition_func in ideal_setups.items():
        matching_moves = [m for m in all_moves if condition_func(m)]

        if not matching_moves:
            print(f"\n{setup_name}: No matches found")
            continue

        increases = [m['price_increase_pct'] for m in matching_moves]
        count = len(increases)
        avg_increase = statistics.mean(increases)
        median_increase = statistics.median(increases)
        pct_above_2 = sum(1 for x in increases if x >= 2.0) / count * 100
        pct_above_2_5 = sum(1 for x in increases if x >= 2.5) / count * 100
        pct_above_3 = sum(1 for x in increases if x >= 3.0) / count * 100

        results[setup_name] = {
            'count': count,
            'pct_of_total': (count / len(all_moves)) * 100,
            'avg_increase': avg_increase,
            'median_increase': median_increase,
            'pct_above_2.0': pct_above_2,
            'pct_above_2.5': pct_above_2_5,
            'pct_above_3.0': pct_above_3
        }

        print(f"\n{setup_name}:")
        print(f"  Matching moves: {count:,} ({results[setup_name]['pct_of_total']:.2f}% of total)")
        print(f"  Avg increase: {avg_increase:.2f}%")
        print(f"  Median increase: {median_increase:.2f}%")
        print(f"  % achieving 2.0%+: {pct_above_2:.1f}%")
        print(f"  % achieving 2.5%+: {pct_above_2_5:.1f}%")
        print(f"  % achieving 3.0%+: {pct_above_3:.1f}%")

    return results


def generate_scoring_weights(rsi_results, volume_results, position_results, ma_results):
    """
    Generate optimal scoring weights based on correlation analysis
    """
    print("\n" + "="*80)
    print("RECOMMENDED SCORING WEIGHTS")
    print("="*80)

    # Based on analysis, recommend weights for scoring system
    weights = {
        'rsi': {
            'weight': 20,
            'scoring_logic': {
                'extreme_oversold_<20': 100,
                'oversold_20-30': 85,
                'lower_neutral_30-45': 75,
                'middle_neutral_45-55': 65,
                'upper_neutral_55-70': 50,
                'overbought_70-80': 30,
                'extreme_overbought_>80': 20
            }
        },
        'volume_profile': {
            'weight': 25,
            'scoring_logic': {
                'spike': 100,
                'increasing': 85,
                'stable': 65,
                'decreasing': 40
            }
        },
        'price_position_24h': {
            'weight': 15,
            'scoring_logic': {
                'very_low_0-20': 100,
                'low_20-40': 85,
                'middle_40-60': 65,
                'high_60-80': 45,
                'very_high_80-100': 30
            }
        },
        'price_vs_ma24h': {
            'weight': 25,
            'scoring_logic': {
                'far_below_<-3%': 100,
                'below_-3_to_-1.5%': 90,
                'slightly_below_-1.5_to_-0.5%': 75,
                'near_ma_-0.5_to_0.5%': 60,
                'slightly_above_0.5_to_1.5%': 45,
                'above_1.5_to_3%': 30,
                'far_above_>3%': 20
            }
        },
        'volatility_24h': {
            'weight': 15,
            'scoring_logic': 'Higher volatility = higher score (min 5%, max capped at 20%)'
        }
    }

    print("\nProposed scoring system (0-100 scale):")
    print("\n1. RSI (20% weight):")
    for logic, score in weights['rsi']['scoring_logic'].items():
        print(f"   {logic}: {score} points")

    print("\n2. Volume Profile (25% weight):")
    for logic, score in weights['volume_profile']['scoring_logic'].items():
        print(f"   {logic}: {score} points")

    print("\n3. Price Position in 24h Range (15% weight):")
    for logic, score in weights['price_position_24h']['scoring_logic'].items():
        print(f"   {logic}: {score} points")

    print("\n4. Price vs MA(24h) (25% weight):")
    for logic, score in weights['price_vs_ma24h']['scoring_logic'].items():
        print(f"   {logic}: {score} points")

    print("\n5. Volatility 24h (15% weight):")
    print(f"   {weights['volatility_24h']['scoring_logic']}")

    print("\nTotal weight: 100%")
    print("\nMinimum score for trade: 75/100 (recommended)")
    print("AI must confirm with HIGH confidence for scores 75-85")
    print("Scores 85+ can proceed with AI MEDIUM/HIGH confirmation")

    return weights


def main():
    """Main execution"""
    print("="*80)
    print("PATTERN RECOGNITION ANALYZER")
    print("="*80)

    # Load data
    analysis = load_analysis()

    # Collect all moves
    all_moves = []
    for result in analysis['per_crypto_results']:
        all_moves.extend(result['moves'])

    print(f"\nAnalyzing {len(all_moves):,} total moves...")

    # Run correlation analyses
    rsi_results = analyze_rsi_correlation(all_moves)
    volume_results = analyze_volume_correlation(all_moves)
    position_results = analyze_price_position_correlation(all_moves)
    ma_results = analyze_ma_correlation(all_moves)
    time_results = analyze_time_of_day_correlation(all_moves)

    # Identify high-probability setups
    setup_results = identify_high_probability_setups(all_moves)

    # Generate scoring weights
    weights = generate_scoring_weights(rsi_results, volume_results, position_results, ma_results)

    # Save results
    output = {
        'rsi_correlation': rsi_results,
        'volume_correlation': volume_results,
        'position_correlation': position_results,
        'ma_correlation': ma_results,
        'time_correlation': time_results,
        'high_probability_setups': setup_results,
        'recommended_scoring_weights': weights
    }

    with open('pattern_recognition_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print("Results saved to: pattern_recognition_results.json")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
