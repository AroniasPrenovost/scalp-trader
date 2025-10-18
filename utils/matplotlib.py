import os
import time
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display windows
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from utils.price_helpers import calculate_percentage_from_min

def format_price(x, p):
    """Format price labels, removing .00 for whole numbers"""
    if x == int(x):
        return f'${int(x)}'
    else:
        return f'${x:.2f}'

def format_volume(x, p):
    """Format volume labels, removing .00 for whole numbers"""
    if x >= 1e9:
        value = x / 1e9
        if value == int(value):
            return f'{int(value)}B'
        else:
            return f'{value:.2f}B'
    elif x >= 1e6:
        value = x / 1e6
        if value == int(value):
            return f'{int(value)}M'
        else:
            return f'{value:.2f}M'
    elif x >= 1e3:
        value = x / 1e3
        if value == int(value):
            return f'{int(value)}K'
        else:
            return f'{value:.1f}K'
    else:
        if x == int(x):
            return f'{int(x)}'
        else:
            return f'{x:.0f}'

def create_time_labels(num_points, interval_minutes, current_timestamp):
    """
    Create time labels for x-axis based on data points and interval.

    Args:
        num_points: Number of data points
        interval_minutes: Minutes between each data point
        current_timestamp: Current timestamp (most recent data point)

    Returns:
        tuple: (positions, labels) for x-axis ticks
    """
    # Calculate total time span
    total_minutes = (num_points - 1) * interval_minutes
    start_time = datetime.fromtimestamp(current_timestamp) - timedelta(minutes=total_minutes)

    # Determine appropriate number of labels (aim for 6-10 labels)
    target_num_labels = 8
    step = max(1, num_points // target_num_labels)

    positions = []
    labels = []

    for i in range(0, num_points, step):
        positions.append(i)
        point_time = start_time + timedelta(minutes=i * interval_minutes)

        # Format label as "Mon 10/17" for all time spans
        label = point_time.strftime('%a %-m/%-d')  # e.g., "Mon 10/17"

        labels.append(label)

    # Always include the last point, but check for duplicate labels
    if positions[-1] != num_points - 1:
        end_time = datetime.fromtimestamp(current_timestamp)
        last_label = end_time.strftime('%a %-m/%-d')  # e.g., "Mon 10/17"

        # Only add if it's not a duplicate of the previous label
        if last_label != labels[-1]:
            positions.append(num_points - 1)
            labels.append(last_label)

    return positions, labels

def calculate_moving_average(prices, period):
    """Calculate simple moving average"""
    if len(prices) < period:
        return [None] * len(prices)

    ma = []
    for i in range(len(prices)):
        if i < period - 1:
            ma.append(None)
        else:
            ma.append(np.mean(prices[i - period + 1:i + 1]))
    return ma


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return [None] * len(prices), [None] * len(prices), [None] * len(prices)

    ma = calculate_moving_average(prices, period)
    upper = []
    lower = []

    for i in range(len(prices)):
        if i < period - 1 or ma[i] is None:
            upper.append(None)
            lower.append(None)
        else:
            std = np.std(prices[i - period + 1:i + 1])
            upper.append(ma[i] + (std_dev * std))
            lower.append(ma[i] - (std_dev * std))

    return ma, upper, lower


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return [None] * len(prices)

    deltas = np.diff(prices)
    gains = []
    losses = []

    for delta in deltas:
        if delta > 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(delta))

    rsi_values = [None]  # First value is None since we need deltas

    for i in range(len(gains)):
        if i < period - 1:
            rsi_values.append(None)
        else:
            avg_gain = np.mean(gains[max(0, i - period + 1):i + 1])
            avg_loss = np.mean(losses[max(0, i - period + 1):i + 1])

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

    return rsi_values


def plot_simple_snapshot(
    current_timestamp,
    interval,
    symbol,
    price_data,
    min_price,
    max_price,
    range_percentage_from_min,
    volume_data=None,
    analysis=None
):

    """
    Enhanced snapshot plot with technical indicators and optional AI analysis.
    For use before trading logic when entry_price and analysis are not yet available.

    Args:
        current_timestamp: timestamp for filename
        interval: data interval in minutes
        symbol: trading pair symbol
        price_data: list of price values
        min_price: minimum price in range
        max_price: maximum price in range
        range_percentage_from_min: percentage change from min to max price
        volume_data: optional volume data for volume subplot
        analysis: optional AI analysis dictionary with support/resistance/buy/sell levels
    """

    # Ensure all data is numeric - handle string conversions and filter invalid data
    clean_price_data = []
    for p in price_data:
        try:
            if p is None:
                continue
            if isinstance(p, str):
                p = p.strip()
                if p == '':
                    continue
            clean_price_data.append(float(p))
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid price value in snapshot: {p}")
            continue
    price_data = clean_price_data

    if not price_data:
        print("Error: No valid price data for snapshot")
        return

    # Note: Lookback window filtering is handled by the caller (index.py)
    # Do not apply additional filtering here to avoid double-filtering

    # Recalculate min/max for the data
    min_price = float(min(price_data))
    max_price = float(max(price_data))
    range_percentage_from_min = calculate_percentage_from_min(min_price, max_price)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))

    # Determine number of subplots
    num_subplots = 1
    has_rsi = len(price_data) >= 15
    if has_rsi:
        num_subplots += 1

    # Create subplots with height ratios
    if num_subplots == 2:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    else:
        gs = fig.add_gridspec(1, 1)

    # Main price chart
    ax1 = fig.add_subplot(gs[0])

    # Plot price line
    x_values = list(range(len(price_data)))
    ax1.plot(x_values, price_data, marker=',', label='Price', c='#000000', linewidth=0.8, zorder=5)

    # Reduce left/right margins with small padding (half of default)
    if len(price_data) > 0:
        padding = len(price_data) * 0.01
        ax1.set_xlim(-padding, len(price_data) - 1 + padding)

    # Calculate and plot moving averages
    if len(price_data) >= 20:
        ma20 = calculate_moving_average(price_data, 20)
        ma20_clean = [val if val is not None else np.nan for val in ma20]
        ax1.plot(x_values, ma20_clean, label='MA(20)', c='#1565C0', linewidth=2.0, alpha=1.0, linestyle='--')

    if len(price_data) >= 50:
        ma50 = calculate_moving_average(price_data, 50)
        ma50_clean = [val if val is not None else np.nan for val in ma50]
        ax1.plot(x_values, ma50_clean, label='MA(50)', c='#D32F2F', linewidth=2.0, alpha=1.0, linestyle='--')

    # Calculate and plot Bollinger Bands
    if len(price_data) >= 20:
        bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(price_data, period=20)
        bb_upper_clean = [val if val is not None else np.nan for val in bb_upper]
        bb_lower_clean = [val if val is not None else np.nan for val in bb_lower]
        ax1.plot(x_values, bb_upper_clean, label='BB Upper', c='#607D8B', linewidth=1.8, alpha=0.9, linestyle=':')
        ax1.plot(x_values, bb_lower_clean, label='BB Lower', c='#607D8B', linewidth=1.8, alpha=0.9, linestyle=':')
        ax1.fill_between(x_values, bb_upper_clean, bb_lower_clean, alpha=0.15, color='#607D8B')

    # Plot AI analysis levels if available
    if analysis:
        if analysis.get('major_support'):
            ax1.axhline(y=analysis['major_support'], color='#27AE60', linewidth=1.5,
                       linestyle='--', label=f"Major Support (${analysis['major_support']:.4f})", alpha=0.85)
        if analysis.get('minor_support'):
            ax1.axhline(y=analysis['minor_support'], color='#58D68D', linewidth=1,
                       linestyle=':', label=f"Minor Support (${analysis['minor_support']:.4f})", alpha=0.7)
        if analysis.get('major_resistance'):
            ax1.axhline(y=analysis['major_resistance'], color='#C0392B', linewidth=1.5,
                       linestyle='--', label=f"Major Resistance (${analysis['major_resistance']:.4f})", alpha=0.85)
        if analysis.get('minor_resistance'):
            ax1.axhline(y=analysis['minor_resistance'], color='#EC7063', linewidth=1,
                       linestyle=':', label=f"Minor Resistance (${analysis['minor_resistance']:.4f})", alpha=0.7)
        if analysis.get('buy_in_price'):
            ax1.axhline(y=analysis['buy_in_price'], color='#17A589', linewidth=1.8,
                       linestyle='-', label=f"AI Buy Target (${analysis['buy_in_price']:.4f})", alpha=0.9)
        if analysis.get('sell_price'):
            ax1.axhline(y=analysis['sell_price'], color='#8E44AD', linewidth=1.8,
                       linestyle='-', label=f"AI Sell Target (${analysis['sell_price']:.4f})", alpha=0.9)

    # Configure main chart
    price_range = max_price - min_price
    buffer = price_range * 0.1
    ax1.set_ylim(min_price - buffer, max_price + buffer)

    # Set time-based x-axis labels
    positions, labels = create_time_labels(len(price_data), interval, current_timestamp)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_price))
    ax1.set_ylabel('Price (USD)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.5, linewidth=0.8, which='major')
    ax1.grid(True, alpha=0.25, linewidth=0.5, which='minor')
    ax1.minorticks_on()
    ax1.legend(loc='upper left', fontsize='x-small', ncol=2)

    # Title with AI analysis info
    title = f"{symbol} - Range: {range_percentage_from_min:.2f}% (low to high)"
    if analysis:
        title += f" | Trend: {analysis.get('market_trend', 'N/A')} | Confidence: {analysis.get('confidence_level', 'N/A')}"
    ax1.set_title(title, fontsize=12, fontweight='bold')

    # RSI subplot
    if has_rsi:
        ax2 = fig.add_subplot(gs[1])
        rsi = calculate_rsi(price_data, period=14)
        rsi_clean = [val if val is not None else np.nan for val in rsi]
        ax2.plot(x_values, rsi_clean, label='RSI(14)', c='#6A1B9A', linewidth=1.5)
        ax2.axhline(y=70, color='#C62828', linewidth=2.0, linestyle='--', alpha=0.8, label='Overbought (>70)')
        ax2.axhline(y=30, color='#2E7D32', linewidth=2.0, linestyle='--', alpha=0.8, label='Oversold (<30)')
        ax2.fill_between(x_values, 70, 100, alpha=0.2, color='#C62828')
        ax2.fill_between(x_values, 0, 30, alpha=0.2, color='#2E7D32')
        ax2.set_ylim(0, 100)
        padding = len(price_data) * 0.01
        ax2.set_xlim(-padding, len(price_data) - 1 + padding)
        ax2.set_ylabel('RSI', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.5, linewidth=0.8)
        ax2.legend(loc='upper left', fontsize='x-small')
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_xlabel(f"Time", fontsize=10, fontweight='bold')

    # X-axis label on bottom subplot
    if not has_rsi:
        ax1.set_xlabel(f"Time", fontsize=10, fontweight='bold')

    # Save figure
    # Convert interval to candle interval label
    if interval >= 10080:  # 1 week or more
        weeks = int(interval / 10080)
        candle_label = "1w" if weeks == 1 else f"{weeks}w"
    elif interval >= 1440:  # 1 day or more
        days = int(interval / 1440)
        candle_label = "1d" if days == 1 else f"{days}d"
    elif interval >= 60:  # 1 hour or more
        hours = int(interval / 60)
        candle_label = "1h" if hours == 1 else f"{hours}h"
    else:  # minutes
        candle_label = f"{int(interval)}m"

    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))
    filename = os.path.join("./screenshots", f"{symbol}_{candle_label}-candles_snapshot_{timestamp_str}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the specific figure
    plt.close('all')  # Ensure all figures are closed
    print(f"Snapshot saved as {filename}")


def resample_volume_data_to_timeframe(volume_data, interval_minutes, target_timeframe_hours):
    """
    Resample volume data by SUMMING volumes across each timeframe period.

    Args:
        volume_data: List of volume values at regular intervals
        interval_minutes: The interval in minutes between each volume data point
        target_timeframe_hours: Target timeframe in hours (e.g., 1 for 1h, 4 for 4h, 24 for 1d)

    Returns:
        List of summed volumes for each resampled period
    """
    if not volume_data or len(volume_data) == 0:
        return []

    # Ensure all volume data is numeric
    clean_volume_data = []
    for v in volume_data:
        try:
            if isinstance(v, str):
                v = v.strip()
                if v == '':
                    continue
            clean_volume_data.append(float(v))
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid volume value in resample: {v}")
            continue

    volume_data = clean_volume_data
    if not volume_data or len(volume_data) == 0:
        return []

    # Calculate how many data points to group together
    target_timeframe_minutes = target_timeframe_hours * 60
    points_per_period = int(target_timeframe_minutes / interval_minutes)

    # If no resampling needed, return original data
    if points_per_period <= 1:
        return volume_data

    # Sum volumes for each period
    resampled = []
    for i in range(0, len(volume_data), points_per_period):
        period_data = volume_data[i:i + points_per_period]
        if len(period_data) > 0:
            resampled.append(sum(period_data))

    return resampled


def resample_price_data_to_timeframe(price_data, interval_minutes, target_timeframe_hours):
    """
    Resample price data from one interval to a different timeframe using OHLC aggregation.

    Args:
        price_data: List of price values at regular intervals
        interval_minutes: The interval in minutes between each price data point
        target_timeframe_hours: Target timeframe in hours (e.g., 1 for 1h, 4 for 4h, 24 for 1d, 168 for 1w)

    Returns:
        Dictionary with resampled data:
        {
            'close': list of closing prices (for line charts and indicators),
            'open': list of opening prices,
            'high': list of high prices,
            'low': list of low prices
        }
    """
    if not price_data or len(price_data) == 0:
        return {'close': [], 'open': [], 'high': [], 'low': []}

    # Ensure all price data is numeric (convert strings to float, filter invalid data)
    clean_price_data = []
    for p in price_data:
        try:
            if isinstance(p, str):
                p = p.strip()
                if p == '':
                    continue
            clean_price_data.append(float(p))
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid price value in resample: {p}")
            continue

    # Use cleaned data
    price_data = clean_price_data
    if not price_data or len(price_data) == 0:
        return {'close': [], 'open': [], 'high': [], 'low': []}

    # Calculate how many data points to group together
    target_timeframe_minutes = target_timeframe_hours * 60
    points_per_candle = int(target_timeframe_minutes / interval_minutes)

    # If target timeframe is smaller than or equal to our data interval, just return original data
    # (no resampling needed when points_per_candle <= 1)
    if points_per_candle <= 1:
        return {
            'close': price_data,
            'open': price_data,
            'high': price_data,
            'low': price_data
        }

    # Resample into OHLC candles
    resampled = {'close': [], 'open': [], 'high': [], 'low': []}

    for i in range(0, len(price_data), points_per_candle):
        candle_data = price_data[i:i + points_per_candle]

        if len(candle_data) > 0:
            resampled['open'].append(candle_data[0])
            resampled['high'].append(max(candle_data))
            resampled['low'].append(min(candle_data))
            resampled['close'].append(candle_data[-1])

    return resampled


def plot_multi_timeframe_charts(
    current_timestamp,
    interval,
    symbol,
    price_data,
    volume_data=None,
    analysis=None
):
    """
    Generate multiple charts at different timeframes for LLM analysis by resampling data.
    Creates 5 charts: 24h, 7d, 30d, 90d, and 6mo views using ALL available data.

    Args:
        current_timestamp: timestamp for filename
        interval: data interval in minutes (e.g., 5 for 5-minute data)
        symbol: trading pair symbol
        price_data: full list of price values (all available historical data)
        volume_data: optional full list of volume values (will be summed when resampled)
        analysis: optional AI analysis dictionary

    Returns:
        Dictionary with paths to generated charts:
        {
            '24h': path to 24-hour chart,
            '7d': path to 7-day chart,
            '30d': path to 30-day chart,
            '90d': path to 90-day chart,
            '6mo': path to 6-month chart
        }
    """
    # Ensure all data is numeric - handle string conversions and filter invalid data
    clean_price_data = []
    for p in price_data:
        try:
            if p is None:
                continue
            if isinstance(p, str):
                p = p.strip()
                if p == '':
                    continue
            clean_price_data.append(float(p))
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid price value in multi-timeframe chart: {p}")
            continue
    price_data = clean_price_data

    # Skip if we don't have enough data
    if len(price_data) < 10:
        print(f"  Skipping all charts: insufficient data ({len(price_data)} points, need at least 10)")
        return {}

    # Optimized timeframes based on senior developer recommendations for effective AI analysis:
    # 1. Full 6-Month View: Macro trends, major support/resistance, long-term patterns
    # 2. 90-Day View: Extended trend analysis, quarterly patterns
    # 3. 30-Day View: Recent trend analysis, medium-term momentum
    # 4. 7-Day View: Short-term price action, immediate trading context
    # 5. 24-Hour View: Immediate market conditions, entry/exit timing
    timeframes = {
        '6mo': {
            'hours': 1,  # Use 1h candles for full 6-month historical view
            'label': '6mo',
            'title_suffix': '6 Month Full History - Macro Trends',
            'lookback_hours': 4380  # Full 6 months (182.5 days)
        },
        '90d': {
            'hours': 1,  # Use 1h candles for 90-day view
            'label': '90d',
            'title_suffix': '90 Day View - Extended Trend Analysis',
            'lookback_hours': 2160  # 90 days
        },
        '30d': {
            'hours': 1,  # Use 1h candles for 30-day view
            'label': '30d',
            'title_suffix': '30 Day View - Recent Trend Analysis',
            'lookback_hours': 720  # 30 days
        },
        '7d': {
            'hours': 1,  # Use 1h candles for 7-day view
            'label': '7d',
            'title_suffix': '7 Day View - Short-term Price Action',
            'lookback_hours': 168  # 7 days
        },
        '24h': {
            'hours': 1,  # Use 1h candles for intraday view
            'label': '24h',
            'title_suffix': '24 Hour View - Immediate Market Conditions',
            'lookback_hours': 24  # 1 day
        }
    }

    chart_paths = {}
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))

    for timeframe_key, timeframe_config in timeframes.items():
        # Filter price data based on lookback window
        lookback_hours = timeframe_config.get('lookback_hours')
        if lookback_hours is not None:
            # Calculate how many data points to keep based on lookback window
            lookback_data_points = int((lookback_hours * 60) / interval)
            filtered_price_data = price_data[-lookback_data_points:] if len(price_data) > lookback_data_points else price_data
        else:
            # Use all available data
            filtered_price_data = price_data

        # Resample the filtered price data into this timeframe
        resampled_data = resample_price_data_to_timeframe(
            filtered_price_data,
            interval,
            timeframe_config['hours']
        )

        # Use close prices for the chart (this is what technical indicators need)
        resampled_prices = resampled_data['close']

        # Skip if resampling resulted in too few candles
        if len(resampled_prices) < 3:
            print(f"  Skipping {timeframe_key} chart: insufficient data after resampling ({len(resampled_prices)} candles, need at least 3)")
            continue

        # Only include volume for 24h and 7d charts
        # Volume is summed across each resampled period to show total trading activity
        resampled_volumes = None
        if timeframe_key in ['24h', '7d'] and volume_data:
            volume_data_clean = [float(v) if v is not None else 0.0 for v in volume_data]

            # Apply the same lookback window filter to volume data
            if lookback_hours is not None:
                filtered_volume_data = volume_data_clean[-lookback_data_points:] if len(volume_data_clean) > lookback_data_points else volume_data_clean
            else:
                filtered_volume_data = volume_data_clean

            # Use the volume-specific resampling function that SUMS volume
            resampled_volumes = resample_volume_data_to_timeframe(
                filtered_volume_data,
                interval,
                timeframe_config['hours']
            )

        # Calculate min/max for this resampled timeframe
        local_min = min(resampled_prices)
        local_max = max(resampled_prices)
        local_range_pct = calculate_percentage_from_min(local_min, local_max)

        lookback_desc = f"{lookback_hours}h lookback" if lookback_hours else "all data"
        print(f"  Generating {timeframe_key} chart ({len(resampled_prices)} candles from {len(filtered_price_data)} data points, {lookback_desc})")

        # Generate chart for this timeframe
        chart_path = _generate_single_timeframe_chart(
            current_timestamp=current_timestamp,
            interval=timeframe_config['hours'] * 60,  # Convert hours to minutes for the chart
            symbol=symbol,
            price_data=resampled_prices,
            min_price=local_min,
            max_price=local_max,
            range_percentage_from_min=local_range_pct,
            volume_data=resampled_volumes,
            analysis=analysis,
            timeframe_label=timeframe_config['label'],
            timeframe_title=timeframe_config['title_suffix'],
            timestamp_str=timestamp_str
        )

        chart_paths[timeframe_key] = chart_path

    return chart_paths


def _generate_single_timeframe_chart(
    current_timestamp,
    interval,
    symbol,
    price_data,
    min_price,
    max_price,
    range_percentage_from_min,
    volume_data=None,
    analysis=None,
    timeframe_label='',
    timeframe_title='',
    timestamp_str=None
):
    """
    Internal function to generate a single chart for a specific timeframe.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))

    # Determine number of subplots
    num_subplots = 1
    if volume_data:
        num_subplots += 1

    has_rsi = len(price_data) >= 15
    if has_rsi:
        num_subplots += 1

    # Create subplots with height ratios
    if num_subplots == 3:
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    elif num_subplots == 2:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    else:
        gs = fig.add_gridspec(1, 1)

    # Main price chart
    ax1 = fig.add_subplot(gs[0])
    x_values = list(range(len(price_data)))
    ax1.plot(x_values, price_data, marker=',', label='Price', c='#000000', linewidth=0.8, zorder=5)

    # Reduce left/right margins with small padding (half of default)
    padding = len(price_data) * 0.01
    ax1.set_xlim(-padding, len(price_data) - 1 + padding)

    # Calculate and plot moving averages
    if len(price_data) >= 20:
        ma20 = calculate_moving_average(price_data, 20)
        ma20_clean = [val if val is not None else np.nan for val in ma20]
        ax1.plot(x_values, ma20_clean, label='MA(20)', c='#1565C0', linewidth=2.0, alpha=1.0, linestyle='--')

    if len(price_data) >= 50:
        ma50 = calculate_moving_average(price_data, 50)
        ma50_clean = [val if val is not None else np.nan for val in ma50]
        ax1.plot(x_values, ma50_clean, label='MA(50)', c='#D32F2F', linewidth=2.0, alpha=1.0, linestyle='--')

    # Calculate and plot Bollinger Bands
    if len(price_data) >= 20:
        bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(price_data, period=20)
        # Filter out None values for plotting - matplotlib requires numeric values
        bb_upper_clean = [val if val is not None else np.nan for val in bb_upper]
        bb_lower_clean = [val if val is not None else np.nan for val in bb_lower]
        ax1.plot(x_values, bb_upper_clean, label='BB Upper', c='#607D8B', linewidth=1.8, alpha=0.9, linestyle=':')
        ax1.plot(x_values, bb_lower_clean, label='BB Lower', c='#607D8B', linewidth=1.8, alpha=0.9, linestyle=':')
        ax1.fill_between(x_values, bb_upper_clean, bb_lower_clean, alpha=0.15, color='#607D8B')

    # Plot AI analysis levels if available
    if analysis:
        if analysis.get('major_support'):
            ax1.axhline(y=analysis['major_support'], color='#27AE60', linewidth=1.5,
                       linestyle='--', label=f"Major Support (${analysis['major_support']:.4f})", alpha=0.85)
        if analysis.get('minor_support'):
            ax1.axhline(y=analysis['minor_support'], color='#58D68D', linewidth=1,
                       linestyle=':', label=f"Minor Support (${analysis['minor_support']:.4f})", alpha=0.7)
        if analysis.get('major_resistance'):
            ax1.axhline(y=analysis['major_resistance'], color='#C0392B', linewidth=1.5,
                       linestyle='--', label=f"Major Resistance (${analysis['major_resistance']:.4f})", alpha=0.85)
        if analysis.get('minor_resistance'):
            ax1.axhline(y=analysis['minor_resistance'], color='#EC7063', linewidth=1,
                       linestyle=':', label=f"Minor Resistance (${analysis['minor_resistance']:.4f})", alpha=0.7)
        if analysis.get('buy_in_price'):
            ax1.axhline(y=analysis['buy_in_price'], color='#17A589', linewidth=1.8,
                       linestyle='-', label=f"AI Buy Target (${analysis['buy_in_price']:.4f})", alpha=0.9)
        if analysis.get('sell_price'):
            ax1.axhline(y=analysis['sell_price'], color='#8E44AD', linewidth=1.8,
                       linestyle='-', label=f"AI Sell Target (${analysis['sell_price']:.4f})", alpha=0.9)

    # Configure main chart
    price_range = max_price - min_price
    buffer = price_range * 0.1
    ax1.set_ylim(min_price - buffer, max_price + buffer)

    # Set time-based x-axis labels
    positions, labels = create_time_labels(len(price_data), interval, current_timestamp)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_price))
    ax1.set_ylabel('Price (USD)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.5, linewidth=0.8, which='major')
    ax1.grid(True, alpha=0.25, linewidth=0.5, which='minor')
    ax1.minorticks_on()
    ax1.legend(loc='upper left', fontsize='x-small', ncol=2)

    # Title
    title = f"{symbol} - {timeframe_title} - Range: {range_percentage_from_min:.2f}% (low to high)"
    if analysis:
        title += f" | Trend: {analysis.get('market_trend', 'N/A')}"
    ax1.set_title(title, fontsize=12, fontweight='bold')

    # RSI subplot
    subplot_idx = 1
    if has_rsi:
        ax2 = fig.add_subplot(gs[subplot_idx])
        rsi = calculate_rsi(price_data, period=14)
        # Filter out None values for plotting - matplotlib requires numeric values
        rsi_clean = [val if val is not None else np.nan for val in rsi]
        ax2.plot(x_values, rsi_clean, label='RSI(14)', c='#6A1B9A', linewidth=1.5)
        ax2.axhline(y=70, color='#C62828', linewidth=2.0, linestyle='--', alpha=0.8, label='Overbought (>70)')
        ax2.axhline(y=30, color='#2E7D32', linewidth=2.0, linestyle='--', alpha=0.8, label='Oversold (<30)')
        ax2.fill_between(x_values, 70, 100, alpha=0.2, color='#C62828')
        ax2.fill_between(x_values, 0, 30, alpha=0.2, color='#2E7D32')
        ax2.set_ylim(0, 100)
        padding = len(price_data) * 0.01
        ax2.set_xlim(-padding, len(price_data) - 1 + padding)
        ax2.set_ylabel('RSI', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.5, linewidth=0.8)
        ax2.legend(loc='upper left', fontsize='x-small')
        # Share x-axis with main chart
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        subplot_idx += 1

    # Volume subplot
    if volume_data:
        ax3 = fig.add_subplot(gs[subplot_idx])
        # Ensure volume_data contains only numeric values
        volume_data_clean = [float(val) if val is not None else 0.0 for val in volume_data]

        # Color bars based on volume increase/decrease (not price)
        colors = []
        for i in range(len(volume_data_clean)):
            if i == 0 or volume_data_clean[i] >= volume_data_clean[i-1]:
                colors.append('#2E7D32')  # Green for volume increase
            else:
                colors.append('#C62828')  # Red for volume decrease

        ax3.bar(x_values, volume_data_clean, color=colors, alpha=0.85, width=0.8, edgecolor='none')
        padding = len(price_data) * 0.01
        ax3.set_xlim(-padding, len(price_data) - 1 + padding)
        ax3.set_ylabel('Volume', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.5, linewidth=0.8, axis='y')

        # Format volume labels for better readability
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(format_volume))
        # Share x-axis with main chart
        ax3.set_xticks(positions)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        subplot_idx += 1

    # X-axis label on bottom subplot
    if num_subplots > 1:
        fig.get_axes()[-1].set_xlabel(f"Time", fontsize=10, fontweight='bold')
    else:
        ax1.set_xlabel(f"Time", fontsize=10, fontweight='bold')

    # Save figure
    if not timestamp_str:
        timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))

    # Calculate candle interval label from interval (in minutes)
    if interval >= 1440:  # 1 day or more
        days = int(interval / 1440)
        candle_label = "1d" if days == 1 else f"{days}d"
    elif interval >= 60:  # 1 hour or more
        hours = int(interval / 60)
        candle_label = "1h" if hours == 1 else f"{hours}h"
    else:  # minutes
        candle_label = f"{int(interval)}m"

    filename = os.path.join("./screenshots", f"{symbol}_{timeframe_label}-window_{candle_label}-candles_{timestamp_str}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')

    return filename


def plot_graph(
    current_timestamp,
    interval,
    symbol,
    price_data,
    min_price,
    max_price,
    range_percentage_from_min,
    entry_price,
    volume_data=None,
    analysis=None,
    buy_event=False
):
    """
    Enhanced plot with technical indicators and AI analysis
    NOTE: volume_data parameter is deprecated and ignored (rolling 24h volume is misleading on long timeframes)

    Args:
        current_timestamp: timestamp for filename
        interval: data interval in minutes
        symbol: trading pair symbol
        price_data: list of price values
        min_price: minimum price in range
        max_price: maximum price in range
        range_percentage_from_min: percentage change from min to max price
        entry_price: entry price if in position
        volume_data: DEPRECATED - ignored (use plot_multi_timeframe_charts for 24h volume)
        analysis: optional AI analysis dictionary with support/resistance/buy/sell levels
    """

    # Ensure all data is numeric - handle string conversions and filter invalid data
    clean_price_data = []
    for p in price_data:
        try:
            if p is None:
                continue
            if isinstance(p, str):
                p = p.strip()
                if p == '':
                    continue
            clean_price_data.append(float(p))
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid price value in plot_graph: {p}")
            continue
    price_data = clean_price_data

    if not price_data:
        print("Error: No valid price data for plot_graph")
        return None

    # Note: Lookback window filtering is handled by the caller (index.py)
    # Do not apply additional filtering here to avoid double-filtering

    # Recalculate min/max for the data
    min_price = float(min(price_data))
    max_price = float(max(price_data))
    range_percentage_from_min = calculate_percentage_from_min(min_price, max_price)
    entry_price = float(entry_price)

    # Create figure with subplots (main chart + RSI only, NO volume due to 24h rolling data issue)
    fig = plt.figure(figsize=(14, 10))

    # Determine number of subplots based on available data
    num_subplots = 1

    # Calculate if we have enough data for RSI
    has_rsi = len(price_data) >= 15
    if has_rsi:
        num_subplots += 1

    # Create subplots with height ratios
    if num_subplots == 2:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    else:
        gs = fig.add_gridspec(1, 1)

    # Main price chart
    ax1 = fig.add_subplot(gs[0])

    # Plot price line
    x_values = list(range(len(price_data)))
    ax1.plot(x_values, price_data, marker=',', label='Price', c='#000000', linewidth=0.8, zorder=5)

    # Calculate and plot moving averages
    if len(price_data) >= 20:
        ma20 = calculate_moving_average(price_data, 20)
        ma20_clean = [val if val is not None else np.nan for val in ma20]
        ax1.plot(x_values, ma20_clean, label='MA(20)', c='#1565C0', linewidth=2.0, alpha=1.0, linestyle='--')

    if len(price_data) >= 50:
        ma50 = calculate_moving_average(price_data, 50)
        ma50_clean = [val if val is not None else np.nan for val in ma50]
        ax1.plot(x_values, ma50_clean, label='MA(50)', c='#D32F2F', linewidth=2.0, alpha=1.0, linestyle='--')

    # Calculate and plot Bollinger Bands
    if len(price_data) >= 20:
        bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(price_data, period=20)
        # Filter out None values for plotting - matplotlib requires numeric values
        bb_upper_clean = [val if val is not None else np.nan for val in bb_upper]
        bb_lower_clean = [val if val is not None else np.nan for val in bb_lower]
        ax1.plot(x_values, bb_upper_clean, label='BB Upper', c='#607D8B', linewidth=1.8, alpha=0.9, linestyle=':')
        ax1.plot(x_values, bb_lower_clean, label='BB Lower', c='#607D8B', linewidth=1.8, alpha=0.9, linestyle=':')
        ax1.fill_between(x_values, bb_upper_clean, bb_lower_clean, alpha=0.15, color='#607D8B')

    # Plot AI analysis levels if available
    if analysis:
        # Support levels
        if analysis.get('major_support'):
            ax1.axhline(y=analysis['major_support'], color='#27AE60', linewidth=1.5,
                       linestyle='--', label=f"Major Support (${analysis['major_support']:.4f})", alpha=0.85)
        if analysis.get('minor_support'):
            ax1.axhline(y=analysis['minor_support'], color='#58D68D', linewidth=1,
                       linestyle=':', label=f"Minor Support (${analysis['minor_support']:.4f})", alpha=0.7)

        # Resistance levels
        if analysis.get('major_resistance'):
            ax1.axhline(y=analysis['major_resistance'], color='#C0392B', linewidth=1.5,
                       linestyle='--', label=f"Major Resistance (${analysis['major_resistance']:.4f})", alpha=0.85)
        if analysis.get('minor_resistance'):
            ax1.axhline(y=analysis['minor_resistance'], color='#EC7063', linewidth=1,
                       linestyle=':', label=f"Minor Resistance (${analysis['minor_resistance']:.4f})", alpha=0.7)

        # Buy/Sell targets
        if analysis.get('buy_in_price'):
            ax1.axhline(y=analysis['buy_in_price'], color='#17A589', linewidth=1.8,
                       linestyle='-', label=f"AI Buy Target (${analysis['buy_in_price']:.4f})", alpha=0.9)
        if analysis.get('sell_price'):
            ax1.axhline(y=analysis['sell_price'], color='#8E44AD', linewidth=1.8,
                       linestyle='-', label=f"AI Sell Target (${analysis['sell_price']:.4f})", alpha=0.9)

    # Plot entry price if in position
    if entry_price > 0:
        ax1.axhline(y=entry_price, color='#D35400', linewidth=2,
                   linestyle='-', label=f"Entry Price (${entry_price:.4f})", zorder=10)

    # Configure main chart
    price_range = max_price - min_price
    buffer = price_range * 0.1
    ax1.set_ylim(min_price - buffer, max_price + buffer)

    # Set time-based x-axis labels
    positions, labels = create_time_labels(len(price_data), interval, current_timestamp)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_price))
    ax1.set_ylabel('Price (USD)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.5, linewidth=0.8, which='major')
    ax1.grid(True, alpha=0.25, linewidth=0.5, which='minor')
    ax1.minorticks_on()
    ax1.legend(loc='upper left', fontsize='x-small', ncol=2)

    # Title with AI analysis info
    title = f"{symbol} - Range: {range_percentage_from_min:.2f}% (low to high)"
    if analysis:
        title += f" | Trend: {analysis.get('market_trend', 'N/A')} | Confidence: {analysis.get('confidence_level', 'N/A')}"
    ax1.set_title(title, fontsize=12, fontweight='bold')

    # RSI subplot
    if has_rsi:
        ax2 = fig.add_subplot(gs[1])
        rsi = calculate_rsi(price_data, period=14)
        # Filter out None values for plotting - matplotlib requires numeric values
        rsi_clean = [val if val is not None else np.nan for val in rsi]
        ax2.plot(x_values, rsi_clean, label='RSI(14)', c='#6A1B9A', linewidth=1.5)
        ax2.axhline(y=70, color='#C62828', linewidth=2.0, linestyle='--', alpha=0.8, label='Overbought (>70)')
        ax2.axhline(y=30, color='#2E7D32', linewidth=2.0, linestyle='--', alpha=0.8, label='Oversold (<30)')
        ax2.fill_between(x_values, 70, 100, alpha=0.2, color='#C62828')
        ax2.fill_between(x_values, 0, 30, alpha=0.2, color='#2E7D32')
        ax2.set_ylim(0, 100)
        padding = len(price_data) * 0.01
        ax2.set_xlim(-padding, len(price_data) - 1 + padding)
        ax2.set_ylabel('RSI', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.5, linewidth=0.8)
        ax2.legend(loc='upper left', fontsize='x-small')
        # Share x-axis with main chart
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_xlabel(f"Time", fontsize=10, fontweight='bold')

    # X-axis label on bottom subplot
    if not has_rsi:
        ax1.set_xlabel(f"Time", fontsize=10, fontweight='bold')

    # Save figure
    event_type = 'sell'
    if buy_event:
        event_type = 'buy'

    # Convert interval to candle interval label
    if interval >= 10080:  # 1 week or more
        weeks = int(interval / 10080)
        candle_label = "1w" if weeks == 1 else f"{weeks}w"
    elif interval >= 1440:  # 1 day or more
        days = int(interval / 1440)
        candle_label = "1d" if days == 1 else f"{days}d"
    elif interval >= 60:  # 1 hour or more
        hours = int(interval / 60)
        candle_label = "1h" if hours == 1 else f"{hours}h"
    else:  # minutes
        candle_label = f"{int(interval)}m"

    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))
    filename = os.path.join("./screenshots", f"{symbol}_{candle_label}-candles_{event_type}_{timestamp_str}.png")
    print(f"Generating market snapshot: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the specific figure
    plt.close('all')  # Ensure all figures are closed
    print(f"Chart saved as {filename}")
    return filename
