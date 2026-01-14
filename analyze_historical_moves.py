#!/usr/bin/env python3
"""
Historical Move Analyzer
Identifies all 1.5%+ price increases in historical data and tags conditions that preceded them.
"""

import json
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import only what we need, avoiding dotenv dependencies
# We'll copy the necessary functions to avoid the import chain


def get_crypto_data_from_file(directory, product_id, max_age_hours=None):
    """
    Reads all data entries for a specific crypto from its dedicated JSON file.
    """
    file_name = f"{product_id}.json"
    file_path = os.path.join(directory, file_name)

    # Return empty list if file doesn't exist
    if not os.path.exists(file_path):
        return []

    # Read the JSON array
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            if not isinstance(data, list):
                return []
        except json.JSONDecodeError:
            return []

    # Filter by age if specified
    if max_age_hours is not None:
        cutoff_time = time.time() - (max_age_hours * 3600)
        data = [entry for entry in data if entry.get('timestamp', 0) >= cutoff_time]

    return data


def calculate_rsi(prices, period=14):
    """
    Calculate RSI (Relative Strength Index).
    """
    if len(prices) < period + 1:
        return None

    # Calculate price changes
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

    # Separate gains and losses
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]

    # Calculate average gains and losses
    avg_gain = statistics.mean(gains[-period:]) if gains else 0
    avg_loss = statistics.mean(losses[-period:]) if losses else 0

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_moving_average(prices, window):
    """Calculate simple moving average"""
    if len(prices) < window:
        return None
    return statistics.mean(prices[-window:])


def analyze_volume_profile(volumes, current_idx, lookback=24):
    """
    Analyze volume trend leading up to a move.
    Returns: 'increasing', 'decreasing', 'spike', or 'stable'
    """
    if current_idx < lookback:
        return 'insufficient_data'

    recent_volumes = volumes[max(0, current_idx-lookback):current_idx]
    if not recent_volumes or len(recent_volumes) < 3:
        return 'insufficient_data'

    avg_volume = statistics.mean(recent_volumes)
    current_volume = volumes[current_idx] if current_idx < len(volumes) else 0

    # Spike detection
    if current_volume > avg_volume * 1.5:
        return 'spike'

    # Trend detection (compare first half vs second half)
    mid = len(recent_volumes) // 2
    first_half_avg = statistics.mean(recent_volumes[:mid])
    second_half_avg = statistics.mean(recent_volumes[mid:])

    if second_half_avg > first_half_avg * 1.2:
        return 'increasing'
    elif second_half_avg < first_half_avg * 0.8:
        return 'decreasing'
    else:
        return 'stable'


def get_price_position_in_range(current_price, prices, lookback_hours):
    """
    Calculate where price is relative to recent range.
    Returns percentage from min (0%) to max (100%)
    """
    if len(prices) < lookback_hours:
        return None

    recent_prices = prices[-lookback_hours:]
    price_min = min(recent_prices)
    price_max = max(recent_prices)

    if price_max == price_min:
        return 50.0  # Middle if no range

    position = ((current_price - price_min) / (price_max - price_min)) * 100
    return position


def calculate_volatility(prices, lookback=24):
    """Calculate price volatility as percentage range"""
    if len(prices) < lookback:
        return None

    recent_prices = prices[-lookback:]
    price_min = min(recent_prices)
    price_max = max(recent_prices)

    if price_min == 0:
        return None

    volatility = ((price_max - price_min) / price_min) * 100
    return volatility


def find_price_moves(prices, timestamps, threshold_pct=1.5, window_hours=[1, 2, 4, 8, 12, 24]):
    """
    Find all price moves exceeding threshold within various time windows.
    Returns list of moves with metadata.
    """
    moves = []

    for i in range(len(prices)):
        for hours in window_hours:
            # Look ahead to find if price increased by threshold within time window
            end_idx = i + hours
            if end_idx >= len(prices):
                continue

            start_price = prices[i]
            start_time = timestamps[i]

            # Find max price within the window
            window_prices = prices[i:end_idx+1]
            max_price = max(window_prices)
            max_idx = i + window_prices.index(max_price)

            price_increase = ((max_price - start_price) / start_price) * 100

            if price_increase >= threshold_pct:
                time_to_peak_hours = (timestamps[max_idx] - start_time) / 3600

                moves.append({
                    'start_idx': i,
                    'start_time': start_time,
                    'start_price': start_price,
                    'peak_idx': max_idx,
                    'peak_time': timestamps[max_idx],
                    'peak_price': max_price,
                    'price_increase_pct': price_increase,
                    'window_hours': hours,
                    'time_to_peak_hours': time_to_peak_hours
                })

    return moves


def analyze_move_conditions(move, prices, volumes, timestamps, global_volumes=None):
    """
    Analyze market conditions at the start of a move.
    Returns dict of conditions.
    """
    idx = move['start_idx']
    start_time = move['start_time']
    start_price = move['start_price']

    # Time of day and day of week
    dt = datetime.fromtimestamp(start_time)
    hour_of_day = dt.hour
    day_of_week = dt.strftime('%A')

    # Calculate RSI at start (need at least 14 data points)
    rsi = None
    if idx >= 14:
        recent_prices = prices[max(0, idx-14):idx+1]
        if len(recent_prices) >= 14:
            try:
                rsi = calculate_rsi(recent_prices)
            except:
                rsi = None

    # Volume profile
    volume_profile = analyze_volume_profile(volumes, idx, lookback=24)

    # Price position in ranges
    position_24h = get_price_position_in_range(start_price, prices[:idx+1], 24)
    position_7d = get_price_position_in_range(start_price, prices[:idx+1], 168)  # 7 days
    position_30d = get_price_position_in_range(start_price, prices[:idx+1], 720)  # 30 days

    # Volatility in 24h before move
    volatility_24h = calculate_volatility(prices[:idx+1], lookback=24)

    # Moving average position
    ma_24h = calculate_moving_average(prices[:idx+1], 24)
    price_vs_ma_pct = None
    if ma_24h and ma_24h != 0:
        price_vs_ma_pct = ((start_price - ma_24h) / ma_24h) * 100

    # Global vs Coinbase volume ratio
    volume_ratio = None
    if global_volumes and idx < len(global_volumes):
        coinbase_vol = volumes[idx]
        global_vol = global_volumes[idx]
        if coinbase_vol > 0:
            volume_ratio = global_vol / coinbase_vol

    # Average volume (24h)
    avg_volume_24h = None
    if idx >= 24:
        avg_volume_24h = statistics.mean(volumes[max(0, idx-24):idx])

    current_volume = volumes[idx] if idx < len(volumes) else None
    volume_vs_avg_pct = None
    if avg_volume_24h and avg_volume_24h > 0 and current_volume:
        volume_vs_avg_pct = ((current_volume - avg_volume_24h) / avg_volume_24h) * 100

    return {
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'rsi': round(rsi, 2) if rsi else None,
        'volume_profile': volume_profile,
        'price_position_24h_pct': round(position_24h, 2) if position_24h else None,
        'price_position_7d_pct': round(position_7d, 2) if position_7d else None,
        'price_position_30d_pct': round(position_30d, 2) if position_30d else None,
        'volatility_24h_pct': round(volatility_24h, 2) if volatility_24h else None,
        'price_vs_ma24h_pct': round(price_vs_ma_pct, 2) if price_vs_ma_pct else None,
        'volume_vs_avg_24h_pct': round(volume_vs_avg_pct, 2) if volume_vs_avg_pct else None,
        'global_coinbase_volume_ratio': round(volume_ratio, 2) if volume_ratio else None,
    }


def analyze_crypto(product_id, coinbase_dir, coingecko_dir, min_increase_pct=1.5):
    """
    Analyze historical moves for a single cryptocurrency.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {product_id}")
    print(f"{'='*80}")

    # Load Coinbase data
    coinbase_data = get_crypto_data_from_file(coinbase_dir, product_id)
    if not coinbase_data or len(coinbase_data) < 50:
        print(f"Insufficient data for {product_id} (need at least 50 data points)")
        return None

    # Load CoinGecko global volume data
    coingecko_data = get_crypto_data_from_file(coingecko_dir, product_id)

    # Extract arrays
    timestamps = [entry['timestamp'] for entry in coinbase_data]
    prices = [float(entry['price']) for entry in coinbase_data]
    volumes = [float(entry.get('volume_24h', 0)) for entry in coinbase_data]

    global_volumes = None
    if coingecko_data and len(coingecko_data) == len(coinbase_data):
        global_volumes = [float(entry.get('volume_24h', 0)) for entry in coingecko_data]

    print(f"Data points: {len(prices)}")
    print(f"Date range: {datetime.fromtimestamp(timestamps[0])} to {datetime.fromtimestamp(timestamps[-1])}")

    # Find all moves
    print(f"\nSearching for {min_increase_pct}%+ price increases...")
    moves = find_price_moves(prices, timestamps, threshold_pct=min_increase_pct)

    print(f"Found {len(moves)} moves of {min_increase_pct}%+")

    if not moves:
        return None

    # Analyze conditions for each move
    analyzed_moves = []
    for move in moves:
        conditions = analyze_move_conditions(move, prices, volumes, timestamps, global_volumes)
        analyzed_move = {**move, 'conditions': conditions}
        analyzed_moves.append(analyzed_move)

    return {
        'product_id': product_id,
        'data_points': len(prices),
        'date_range_start': timestamps[0],
        'date_range_end': timestamps[-1],
        'total_moves_found': len(analyzed_moves),
        'moves': analyzed_moves
    }


def generate_summary_statistics(results):
    """
    Generate summary statistics across all moves.
    """
    all_moves = []
    for result in results:
        if result:
            all_moves.extend(result['moves'])

    if not all_moves:
        return None

    # Aggregate statistics
    stats = {
        'total_moves': len(all_moves),
        'rsi_distribution': defaultdict(int),
        'volume_profile_distribution': defaultdict(int),
        'hour_of_day_distribution': defaultdict(int),
        'day_of_week_distribution': defaultdict(int),
        'price_position_24h_avg': None,
        'price_vs_ma24h_avg': None,
        'volume_vs_avg_avg': None,
    }

    # Collect values for averaging
    rsi_values = []
    price_pos_24h = []
    price_vs_ma = []
    volume_vs_avg = []

    for move in all_moves:
        cond = move['conditions']

        # RSI buckets
        if cond['rsi'] is not None:
            rsi_values.append(cond['rsi'])
            if cond['rsi'] <= 30:
                stats['rsi_distribution']['oversold_<30'] += 1
            elif cond['rsi'] >= 70:
                stats['rsi_distribution']['overbought_>70'] += 1
            else:
                stats['rsi_distribution']['neutral_30-70'] += 1

        # Volume profile
        stats['volume_profile_distribution'][cond['volume_profile']] += 1

        # Time distributions
        stats['hour_of_day_distribution'][cond['hour_of_day']] += 1
        stats['day_of_week_distribution'][cond['day_of_week']] += 1

        # Collect for averaging
        if cond['price_position_24h_pct'] is not None:
            price_pos_24h.append(cond['price_position_24h_pct'])
        if cond['price_vs_ma24h_pct'] is not None:
            price_vs_ma.append(cond['price_vs_ma24h_pct'])
        if cond['volume_vs_avg_24h_pct'] is not None:
            volume_vs_avg.append(cond['volume_vs_avg_24h_pct'])

    # Calculate averages
    if price_pos_24h:
        stats['price_position_24h_avg'] = round(statistics.mean(price_pos_24h), 2)
    if price_vs_ma:
        stats['price_vs_ma24h_avg'] = round(statistics.mean(price_vs_ma), 2)
    if volume_vs_avg:
        stats['volume_vs_avg_avg'] = round(statistics.mean(volume_vs_avg), 2)
    if rsi_values:
        stats['rsi_avg'] = round(statistics.mean(rsi_values), 2)

    return stats


def main():
    """Main execution function"""
    print("="*80)
    print("HISTORICAL MOVE ANALYZER")
    print("Identifying 1.5%+ price increases and their conditions")
    print("="*80)

    # Directories
    coinbase_dir = 'coinbase-data'
    coingecko_dir = 'coingecko-global-volume'
    output_file = 'analysis/historical_moves_analysis.json'

    # Load config to get enabled cryptos
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_cryptos = [
        wallet['symbol']
        for wallet in config['wallets']
        if wallet.get('enabled', False)
    ]

    print(f"\nAnalyzing {len(enabled_cryptos)} enabled cryptocurrencies...")
    print(f"Cryptos: {', '.join(enabled_cryptos)}\n")

    # Analyze each crypto
    results = []
    for product_id in enabled_cryptos:
        result = analyze_crypto(product_id, coinbase_dir, coingecko_dir, min_increase_pct=1.5)
        if result:
            results.append(result)

    # Generate summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    summary_stats = generate_summary_statistics(results)

    if summary_stats:
        print(f"\nTotal moves found: {summary_stats['total_moves']}")
        print(f"\nRSI Distribution:")
        for bucket, count in sorted(summary_stats['rsi_distribution'].items()):
            pct = (count / summary_stats['total_moves']) * 100
            print(f"  {bucket}: {count} ({pct:.1f}%)")

        print(f"\nVolume Profile Distribution:")
        for profile, count in sorted(summary_stats['volume_profile_distribution'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / summary_stats['total_moves']) * 100
            print(f"  {profile}: {count} ({pct:.1f}%)")

        print(f"\nTop Hours of Day:")
        sorted_hours = sorted(summary_stats['hour_of_day_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        for hour, count in sorted_hours:
            pct = (count / summary_stats['total_moves']) * 100
            print(f"  {hour:02d}:00: {count} ({pct:.1f}%)")

        print(f"\nAverages:")
        print(f"  RSI: {summary_stats.get('rsi_avg', 'N/A')}")
        print(f"  Price position in 24h range: {summary_stats.get('price_position_24h_avg', 'N/A')}%")
        print(f"  Price vs MA(24h): {summary_stats.get('price_vs_ma24h_avg', 'N/A')}%")
        print(f"  Volume vs 24h avg: {summary_stats.get('volume_vs_avg_avg', 'N/A')}%")

    # Save results
    output_data = {
        'generated_at': time.time(),
        'analysis_date': datetime.now().isoformat(),
        'min_increase_threshold_pct': 1.5,
        'summary_statistics': summary_stats,
        'per_crypto_results': results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
