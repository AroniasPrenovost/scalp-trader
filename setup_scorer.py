#!/usr/bin/env python3
"""
Setup Scorer
Real-time scoring system for trading setups based on historical pattern analysis.
Scores 0-100 based on probability of 1.5-2.5% price increase.
"""

import json
import statistics
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))


def get_crypto_data_from_file(directory, product_id, max_age_hours=None):
    """Load crypto data from file"""
    import time
    file_name = f"{product_id}.json"
    file_path = os.path.join(directory, file_name)

    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            if not isinstance(data, list):
                return []
        except json.JSONDecodeError:
            return []

    if max_age_hours is not None:
        cutoff_time = time.time() - (max_age_hours * 3600)
        data = [entry for entry in data if entry.get('timestamp', 0) >= cutoff_time]

    return data


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]

    avg_gain = statistics.mean(gains[-period:]) if gains else 0
    avg_loss = statistics.mean(losses[-period:]) if losses else 0

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def score_rsi(rsi):
    """
    Score RSI based on historical correlation (20% weight).
    Based on pattern recognition: oversold and lower neutral have highest success.
    """
    if rsi is None:
        return 0, "No RSI data"

    if rsi < 20:
        return 100, "Extreme oversold"
    elif rsi < 30:
        return 85, "Oversold"
    elif rsi < 45:
        return 75, "Lower neutral"
    elif rsi < 55:
        return 65, "Middle neutral"
    elif rsi < 70:
        return 50, "Upper neutral"
    elif rsi < 80:
        return 30, "Overbought"
    else:
        return 20, "Extreme overbought"


def analyze_volume_profile(volumes, current_volume):
    """
    Analyze current volume vs recent average.
    Returns profile type and score.
    """
    if not volumes or len(volumes) < 24:
        return 'insufficient_data', 0, "Not enough volume history"

    recent_avg = statistics.mean(volumes[-24:])

    if recent_avg == 0:
        return 'insufficient_data', 0, "Zero average volume"

    ratio = current_volume / recent_avg

    # Determine profile
    if ratio > 1.5:
        return 'spike', 100, f"Volume spike ({ratio:.2f}x avg)"
    elif ratio > 1.2:
        return 'increasing', 85, f"Volume increasing ({ratio:.2f}x avg)"
    elif ratio > 0.8:
        return 'stable', 65, f"Volume stable ({ratio:.2f}x avg)"
    else:
        return 'decreasing', 40, f"Volume decreasing ({ratio:.2f}x avg)"


def score_volume_profile(volumes, current_volume):
    """
    Score volume profile (25% weight).
    Based on pattern recognition: spikes and increasing volume perform best.
    """
    profile, base_score, description = analyze_volume_profile(volumes, current_volume)
    return base_score, description


def get_price_position_in_range(current_price, prices, lookback=24):
    """
    Calculate where price is in recent range (0-100%).
    """
    if len(prices) < lookback:
        return None

    recent_prices = prices[-lookback:]
    price_min = min(recent_prices)
    price_max = max(recent_prices)

    if price_max == price_min:
        return 50.0

    position = ((current_price - price_min) / (price_max - price_min)) * 100
    return position


def score_price_position(current_price, prices):
    """
    Score price position in 24h range (15% weight).
    Based on pattern recognition: lower in range is slightly better.
    """
    position = get_price_position_in_range(current_price, prices, lookback=24)

    if position is None:
        return 0, "Insufficient price history"

    if position < 20:
        return 100, f"Very low in range ({position:.1f}%)"
    elif position < 40:
        return 85, f"Low in range ({position:.1f}%)"
    elif position < 60:
        return 65, f"Middle of range ({position:.1f}%)"
    elif position < 80:
        return 45, f"High in range ({position:.1f}%)"
    else:
        return 30, f"Very high in range ({position:.1f}%)"


def calculate_ma(prices, period=24):
    """Calculate moving average"""
    if len(prices) < period:
        return None
    return statistics.mean(prices[-period:])


def score_price_vs_ma(current_price, prices):
    """
    Score price vs MA(24h) (25% weight).
    Based on pattern recognition: below MA has strong edge (mean reversion).
    """
    ma_24h = calculate_ma(prices, period=24)

    if ma_24h is None or ma_24h == 0:
        return 0, "Insufficient data for MA"

    pct_vs_ma = ((current_price - ma_24h) / ma_24h) * 100

    if pct_vs_ma < -3:
        return 100, f"Far below MA ({pct_vs_ma:.2f}%)"
    elif pct_vs_ma < -1.5:
        return 90, f"Below MA ({pct_vs_ma:.2f}%)"
    elif pct_vs_ma < -0.5:
        return 75, f"Slightly below MA ({pct_vs_ma:.2f}%)"
    elif pct_vs_ma < 0.5:
        return 60, f"Near MA ({pct_vs_ma:.2f}%)"
    elif pct_vs_ma < 1.5:
        return 45, f"Slightly above MA ({pct_vs_ma:.2f}%)"
    elif pct_vs_ma < 3:
        return 30, f"Above MA ({pct_vs_ma:.2f}%)"
    else:
        return 20, f"Far above MA ({pct_vs_ma:.2f}%)"


def calculate_volatility(prices, lookback=24):
    """Calculate 24h volatility as percentage range"""
    if len(prices) < lookback:
        return None

    recent_prices = prices[-lookback:]
    price_min = min(recent_prices)
    price_max = max(recent_prices)

    if price_min == 0:
        return None

    volatility = ((price_max - price_min) / price_min) * 100
    return volatility


def score_volatility(prices):
    """
    Score volatility (15% weight).
    Higher volatility (5-20%) gives better scores.
    """
    volatility = calculate_volatility(prices, lookback=24)

    if volatility is None:
        return 0, "Insufficient data for volatility"

    if volatility < 5:
        return 30, f"Very low volatility ({volatility:.1f}%)"
    elif volatility < 10:
        return 60, f"Low volatility ({volatility:.1f}%)"
    elif volatility < 15:
        return 85, f"Moderate volatility ({volatility:.1f}%)"
    elif volatility < 20:
        return 100, f"High volatility ({volatility:.1f}%)"
    else:
        # Cap at 20% to avoid overly chaotic markets
        return 90, f"Very high volatility ({volatility:.1f}%)"


def calculate_setup_score(product_id, coinbase_dir='coinbase-data', coingecko_dir='coingecko-global-volume'):
    """
    Calculate overall setup score (0-100) for a cryptocurrency.
    """
    # Load data
    data = get_crypto_data_from_file(coinbase_dir, product_id)

    if not data or len(data) < 50:
        return {
            'product_id': product_id,
            'score': 0,
            'grade': 'F',
            'recommendation': 'PASS',
            'reason': 'Insufficient data'
        }

    # Extract arrays
    prices = [float(entry['price']) for entry in data]
    volumes = [float(entry.get('volume_24h', 0)) for entry in data]
    current_price = prices[-1]
    current_volume = volumes[-1] if volumes else 0

    # Calculate RSI
    rsi = calculate_rsi(prices)

    # Score each component
    rsi_score, rsi_reason = score_rsi(rsi)
    volume_score, volume_reason = score_volume_profile(volumes, current_volume)
    position_score, position_reason = score_price_position(current_price, prices)
    ma_score, ma_reason = score_price_vs_ma(current_price, prices)
    volatility_score, volatility_reason = score_volatility(prices)

    # Weighted average (total: 100%)
    weights = {
        'rsi': 0.20,
        'volume': 0.25,
        'position': 0.15,
        'ma': 0.25,
        'volatility': 0.15
    }

    overall_score = (
        rsi_score * weights['rsi'] +
        volume_score * weights['volume'] +
        position_score * weights['position'] +
        ma_score * weights['ma'] +
        volatility_score * weights['volatility']
    )

    # Determine grade and recommendation
    if overall_score >= 85:
        grade = 'A'
        recommendation = 'STRONG BUY'
    elif overall_score >= 75:
        grade = 'B'
        recommendation = 'BUY (if AI confirms HIGH)'
    elif overall_score >= 65:
        grade = 'C'
        recommendation = 'WAIT (needs AI HIGH + volume spike)'
    elif overall_score >= 50:
        grade = 'D'
        recommendation = 'WEAK - Pass'
    else:
        grade = 'F'
        recommendation = 'PASS'

    # Identify setup type
    setup_type = identify_setup_type(rsi, ma_score, volume_score)

    return {
        'product_id': product_id,
        'score': round(overall_score, 1),
        'grade': grade,
        'recommendation': recommendation,
        'setup_type': setup_type,
        'components': {
            'rsi': {
                'value': round(rsi, 2) if rsi else None,
                'score': rsi_score,
                'weight': weights['rsi'],
                'weighted_score': round(rsi_score * weights['rsi'], 1),
                'reason': rsi_reason
            },
            'volume': {
                'score': volume_score,
                'weight': weights['volume'],
                'weighted_score': round(volume_score * weights['volume'], 1),
                'reason': volume_reason
            },
            'price_position': {
                'score': position_score,
                'weight': weights['position'],
                'weighted_score': round(position_score * weights['position'], 1),
                'reason': position_reason
            },
            'price_vs_ma': {
                'score': ma_score,
                'weight': weights['ma'],
                'weighted_score': round(ma_score * weights['ma'], 1),
                'reason': ma_reason
            },
            'volatility': {
                'score': volatility_score,
                'weight': weights['volatility'],
                'weighted_score': round(volatility_score * weights['volatility'], 1),
                'reason': volatility_reason
            }
        },
        'current_price': current_price,
        'analyzed_at': datetime.now().isoformat()
    }


def identify_setup_type(rsi, ma_score, volume_score):
    """Identify the type of setup based on conditions (prioritized by historical win rate)"""
    # PRIORITY 1: Mean reversion (76.9% win rate when price <-1.5% below MA)
    if ma_score >= 75:  # Below MA
        return 'mean_reversion'
    # PRIORITY 2: Volume breakout
    elif volume_score == 100:  # Spike
        return 'volume_breakout'
    # PRIORITY 3: Oversold bounce (lower priority - lower win rate)
    elif rsi is not None and rsi < 30 and volume_score >= 85:
        return 'oversold_bounce'
    # PRIORITY 4: Momentum continuation
    elif rsi is not None and 45 <= rsi <= 65 and volume_score >= 85:
        return 'momentum_continuation'
    else:
        return 'general'


def print_score_report(result):
    """Print formatted score report"""
    print(f"\n{'='*80}")
    print(f"SETUP SCORE: {result['product_id']}")
    print(f"{'='*80}")
    print(f"\nOverall Score: {result['score']}/100  (Grade: {result['grade']})")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Setup Type: {result['setup_type']}")
    print(f"Current Price: ${result['current_price']:.2f}")

    print(f"\nComponent Breakdown:")
    print(f"{'Component':<20} {'Score':<10} {'Weight':<10} {'Contribution':<15} {'Reason'}")
    print(f"{'-'*80}")

    comps = result['components']
    for name, data in comps.items():
        contrib = data['weighted_score']
        print(f"{name:<20} {data['score']:<10} {int(data['weight']*100)}%{'':<7} {contrib:<15} {data['reason']}")

    print(f"{'='*80}\n")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Score cryptocurrency trading setups')
    parser.add_argument('--symbol', type=str, help='Symbol to score (e.g., BTC-USD)')
    parser.add_argument('--all', action='store_true', help='Score all enabled cryptos')
    parser.add_argument('--test', action='store_true', help='Test mode - show detailed breakdown')

    args = parser.parse_args()

    if args.symbol:
        # Score single symbol
        result = calculate_setup_score(args.symbol)
        if args.test:
            print_score_report(result)
        else:
            print(f"{result['product_id']}: {result['score']}/100 ({result['grade']}) - {result['recommendation']}")

    elif args.all:
        # Score all enabled cryptos
        with open('config.json', 'r') as f:
            config = json.load(f)

        enabled_cryptos = [
            wallet['symbol']
            for wallet in config['wallets']
            if wallet.get('enabled', False)
        ]

        results = []
        for symbol in enabled_cryptos:
            result = calculate_setup_score(symbol)
            results.append(result)

        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n{'='*80}")
        print(f"SETUP SCORES - ALL ENABLED CRYPTOS")
        print(f"{'='*80}\n")
        print(f"{'Rank':<6} {'Symbol':<12} {'Score':<10} {'Grade':<8} {'Setup Type':<20} {'Recommendation'}")
        print(f"{'-'*80}")

        for rank, result in enumerate(results, 1):
            print(f"{rank:<6} {result['product_id']:<12} {result['score']:<10} {result['grade']:<8} {result['setup_type']:<20} {result['recommendation']}")

        print(f"\n{'='*80}")
        print(f"Trade-worthy setups (score >= 75): {sum(1 for r in results if r['score'] >= 75)}")
        print(f"{'='*80}\n")

        # Show detailed breakdown for top 3
        print(f"\nTop 3 Opportunities:\n")
        for result in results[:3]:
            print_score_report(result)

    else:
        print("Usage: python3 setup_scorer.py --symbol BTC-USD")
        print("       python3 setup_scorer.py --all")
        print("       python3 setup_scorer.py --symbol LINK-USD --test")


if __name__ == '__main__':
    main()
