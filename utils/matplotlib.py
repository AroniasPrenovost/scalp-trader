import os
import time
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display windows
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from utils.price_helpers import calculate_percentage_from_min, calculate_rsi

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
    total_hours = total_minutes / 60
    start_time = datetime.fromtimestamp(current_timestamp) - timedelta(minutes=total_minutes)

    # Get timezone abbreviation (PST/PDT etc.)
    import time as time_module
    tz_name = time_module.tzname[time_module.localtime(current_timestamp).tm_isdst]

    positions = []
    labels = []

    # Use 2-hour intervals for all timespans
    label_interval_minutes = 120

    # Calculate step (how many data points between labels)
    step = max(1, int(label_interval_minutes / interval_minutes))

    # Limit total number of labels to avoid overcrowding
    max_labels = 15
    if num_points // step > max_labels:
        step = max(1, num_points // max_labels)

    for i in range(0, num_points, step):
        positions.append(i)
        point_time = start_time + timedelta(minutes=i * interval_minutes)

        # Format based on timespan
        if total_hours <= 24:
            # Short timespan: show date and time with AM/PM
            # Format: "1/21 10:00 AM PST"
            label = point_time.strftime(f'%-m/%-d %-I:%M %p {tz_name}')
        elif total_hours <= 168:  # 1 week
            # Medium timespan: show date and hour
            # Format: "1/21 10AM PST"
            label = point_time.strftime(f'%-m/%-d %-I%p {tz_name}')
        else:
            # Long timespan: show date only
            # Format: "Mon 1/21"
            label = point_time.strftime('%-m/%-d')

        labels.append(label)

    # Always include the last point, but check for duplicate labels
    if positions[-1] != num_points - 1:
        end_time = datetime.fromtimestamp(current_timestamp)
        if total_hours <= 24:
            last_label = end_time.strftime(f'%-m/%-d %-I:%M %p {tz_name}')
        elif total_hours <= 168:
            last_label = end_time.strftime(f'%-m/%-d %-I%p {tz_name}')
        else:
            last_label = end_time.strftime('%-m/%-d')

        # Only add if it's not a duplicate of the previous label
        if last_label != labels[-1]:
            positions.append(num_points - 1)
            labels.append(last_label)

    return positions, labels


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


def plot_volume_trend_chart(
    current_timestamp,
    interval,
    symbol,
    volume_data
):
    """
    Generate a volume trend chart showing the rolling 24-hour volume over time.
    This correctly visualizes Coinbase's rolling 24h volume snapshots as a trend line.
    Uses ALL available volume data - the rolling 24h metric is independent of timeframes.

    Args:
        current_timestamp: timestamp for filename
        interval: data interval in minutes (e.g., 60 for hourly snapshots)
        symbol: trading pair symbol
        volume_data: list of ALL rolling 24h volume values (complete historical data)

    Returns:
        Path to generated chart, or None if insufficient data
    """
    # Clean and validate volume data
    clean_volume_data = []
    for v in volume_data:
        try:
            if v is None:
                continue
            if isinstance(v, str):
                v = v.strip()
                if v == '':
                    continue
            clean_volume_data.append(float(v))
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid volume value: {v}")
            continue

    if not clean_volume_data or len(clean_volume_data) < 3:
        print(f"Insufficient volume data for trend chart ({len(clean_volume_data)} points)")
        return None

    # Use ALL available volume data
    filtered_volume = clean_volume_data

    # Calculate the actual timespan covered by the data
    total_hours = (len(filtered_volume) * interval) / 60
    total_days = total_hours / 24

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x_values = list(range(len(filtered_volume)))

    # Plot volume as a line with area fill
    ax.plot(x_values, filtered_volume, color='#1565C0', linewidth=1.5, label='Rolling 24h Volume (snapshots)', zorder=3)
    ax.fill_between(x_values, 0, filtered_volume, alpha=0.2, color='#1565C0', zorder=2)

    # Calculate volume statistics
    avg_volume = np.mean(filtered_volume)
    max_volume = max(filtered_volume)
    min_volume = min(filtered_volume)
    current_volume = filtered_volume[-1]

    # Add average volume line
    ax.axhline(y=avg_volume, color='#757575', linewidth=1.5,
               linestyle=':', label=f'Average: {avg_volume:,.0f} BTC', alpha=0.7)

    # Format and configure chart - use custom formatter for BTC volumes
    def format_btc_volume(x, p):
        """Format BTC volume labels"""
        if x >= 1e6:
            return f'{x/1e6:.2f}M BTC'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K BTC'
        else:
            return f'{x:.0f} BTC'

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_btc_volume))
    ax.set_ylabel('24h Volume (BTC)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')

    # Set time-based x-axis labels
    positions, labels = create_time_labels(len(filtered_volume), interval, current_timestamp)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add padding to x-axis
    padding = len(filtered_volume) * 0.01
    ax.set_xlim(-padding, len(filtered_volume) - 1 + padding)

    # Set y-axis limits with some headroom
    y_range = max_volume - min_volume
    ax.set_ylim(max(0, min_volume - y_range * 0.1), max_volume + y_range * 0.1)

    # Grid and legend
    ax.grid(True, alpha=0.4, linewidth=0.8, which='major', axis='y')
    ax.grid(True, alpha=0.2, linewidth=0.5, which='minor', axis='y')
    ax.minorticks_on()
    ax.legend(loc='upper left', fontsize='small')

    # Calculate volume change percentage from start to current
    if len(filtered_volume) > 1:
        volume_change_pct = ((current_volume - filtered_volume[0]) / filtered_volume[0]) * 100
        change_indicator = "↑" if volume_change_pct > 0 else "↓"
    else:
        volume_change_pct = 0
        change_indicator = ""

    # Create title that accurately describes what's shown
    if total_days >= 30:
        timespan_label = f"{total_days:.0f}d"
    else:
        timespan_label = f"{total_hours:.0f}h"

    title = f"{symbol} - Rolling 24h Volume Snapshots ({len(filtered_volume)} points over {timespan_label})\n"
    title += f"Current: {current_volume:,.0f} BTC | Avg: {avg_volume:,.0f} BTC"
    if volume_change_pct != 0:
        title += f" | Change: {change_indicator}{abs(volume_change_pct):.1f}%"

    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)

    # Save figure with descriptive filename
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))
    filename = os.path.join("./screenshots", f"{symbol}_volume-24h-snapshots_{timestamp_str}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')

    print(f"  Volume snapshot chart saved: {filename}")
    return filename


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
    # Create figure with subplots (price + optional volume, no RSI)
    fig = plt.figure(figsize=(14, 10))

    # Determine number of subplots
    num_subplots = 1
    if volume_data:
        num_subplots += 1

    # Create subplots with height ratios
    if num_subplots == 2:
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

    # Plot analysis levels if available
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
                       linestyle='-', label='_nolegend_', alpha=0.9)
        if analysis.get('sell_price'):
            ax1.axhline(y=analysis['sell_price'], color='#8E44AD', linewidth=1.8,
                       linestyle='-', label='_nolegend_', alpha=0.9)

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

    # Volume subplot
    if volume_data:
        ax3 = fig.add_subplot(gs[1])
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
    event_type=None,
    screenshot_type=None,
    timeframe_label=None
):
    """
    Enhanced plot with technical indicators and market analysis
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
        analysis: optional analysis dictionary with support/resistance/buy/sell levels
        event_type: optional event type ('buy', 'sell') to include in filename
        screenshot_type: optional screenshot type ('iteration') for special formatting
        timeframe_label: optional timeframe label for iteration screenshots
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

    # Create figure with single chart (no indicators)
    fig = plt.figure(figsize=(14, 8))

    # Single chart - no subplots
    ax1 = fig.add_subplot(111)

    # Plot price line
    x_values = list(range(len(price_data)))
    ax1.plot(x_values, price_data, marker=',', label='Price', c='#000000', linewidth=0.8, zorder=5)

    # Plot analysis levels if available
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
                       linestyle='-', label='_nolegend_', alpha=0.9)
        if analysis.get('sell_price'):
            ax1.axhline(y=analysis['sell_price'], color='#8E44AD', linewidth=1.8,
                       linestyle='-', label='_nolegend_', alpha=0.9)

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

    # Title with symbol and info
    if analysis:
        title = f"{symbol} - Range: {range_percentage_from_min:.2f}% | Trend: {analysis.get('market_trend', 'N/A')} | Confidence: {analysis.get('confidence_level', 'N/A')}"
    else:
        title = f"{symbol} - Range: {range_percentage_from_min:.2f}% (low to high)"
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # X-axis label
    ax1.set_xlabel(f"Time", fontsize=10, fontweight='bold')

    # Save figure
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))

    # Generate filename based on screenshot type
    if screenshot_type == 'iteration' and timeframe_label:
        # Iteration screenshot format: {symbol}_{timeframe}_{timestamp}
        filename = os.path.join("./screenshots", f"{symbol}_{timeframe_label}_{timestamp_str}.png")
    elif event_type:
        # Buy/sell event screenshot format: {symbol}_{timeframe}_{event}_{timestamp}
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

        filename = os.path.join("./screenshots", f"{symbol}_{candle_label}_{event_type}_{timestamp_str}.png")
    else:
        # Generic snapshot format (fallback)
        if interval >= 10080:
            weeks = int(interval / 10080)
            candle_label = "1w" if weeks == 1 else f"{weeks}w"
        elif interval >= 1440:
            days = int(interval / 1440)
            candle_label = "1d" if days == 1 else f"{days}d"
        elif interval >= 60:
            hours = int(interval / 60)
            candle_label = "1h" if hours == 1 else f"{hours}h"
        else:
            candle_label = f"{int(interval)}m"

        filename = os.path.join("./screenshots", f"{symbol}_{candle_label}_snapshot_{timestamp_str}.png")

    print(f"Generating market snapshot: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the specific figure
    plt.close('all')  # Ensure all figures are closed
    print(f"Chart saved as {filename}")
    return filename


def format_price(x, p):
    """Format price labels"""
    if x >= 1000:
        return f'${x:,.0f}'
    elif x >= 1:
        return f'${x:.2f}'
    else:
        return f'${x:.4f}'


def format_volume(x, p):
    """Format volume labels"""
    if x >= 1e9:
        return f'{x/1e9:.2f}B'
    elif x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.1f}K'
    else:
        return f'{x:.0f}'


def plot_coinbase_metrics_chart(
    current_timestamp,
    interval,
    symbol,
    price_data,
    volume_24h_data,
    price_pct_change_24h_data,
    volume_pct_change_24h_data,
    event_type=None,
    screenshot_type=None,
    timeframe_label=None
):
    """
    Generate a multi-panel chart showing all Coinbase metrics to visualize correlations.

    Args:
        current_timestamp: timestamp for filename
        interval: data interval in minutes
        symbol: trading pair symbol
        price_data: list of price values
        volume_24h_data: list of 24h volume values
        price_pct_change_24h_data: list of 24h price percentage change values
        volume_pct_change_24h_data: list of 24h volume percentage change values
        event_type: optional event type ('buy', 'sell') to include in filename
        screenshot_type: optional screenshot type ('iteration') for special formatting
        timeframe_label: optional timeframe label for iteration screenshots

    Returns:
        Path to generated chart, or None if insufficient data
    """
    # Clean and validate data
    def clean_numeric_data(data):
        clean_data = []
        for v in data:
            try:
                if v is None:
                    clean_data.append(None)
                elif isinstance(v, str):
                    v = v.strip()
                    if v == '':
                        clean_data.append(None)
                    else:
                        clean_data.append(float(v))
                else:
                    clean_data.append(float(v))
            except (ValueError, TypeError):
                clean_data.append(None)
        return clean_data

    price_data = clean_numeric_data(price_data)
    volume_24h_data = clean_numeric_data(volume_24h_data)
    price_pct_change_24h_data = clean_numeric_data(price_pct_change_24h_data)
    volume_pct_change_24h_data = clean_numeric_data(volume_pct_change_24h_data)

    # Ensure all data lists have the same length
    min_length = min(len(price_data), len(volume_24h_data),
                     len(price_pct_change_24h_data), len(volume_pct_change_24h_data))

    if min_length < 3:
        print(f"Insufficient data for Coinbase metrics chart ({min_length} points)")
        return None

    # Truncate all lists to the same length
    price_data = price_data[:min_length]
    volume_24h_data = volume_24h_data[:min_length]
    price_pct_change_24h_data = price_pct_change_24h_data[:min_length]
    volume_pct_change_24h_data = volume_pct_change_24h_data[:min_length]

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)

    x_values = list(range(min_length))

    # Set time-based x-axis labels
    positions, labels = create_time_labels(min_length, interval, current_timestamp)

    # Subplot 1: Price
    ax1 = fig.add_subplot(gs[0])
    # Filter out None values for plotting
    price_clean = [val if val is not None else np.nan for val in price_data]
    ax1.plot(x_values, price_clean, color='#000000', linewidth=1.5, label='Price', zorder=3)

    ax1.set_ylabel('Price (USD)', fontsize=10, fontweight='bold')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_price))
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.legend(loc='upper left', fontsize='small')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    padding = min_length * 0.01
    ax1.set_xlim(-padding, min_length - 1 + padding)
    ax1.set_title(f"{symbol} - Coinbase Metrics Correlation Analysis", fontsize=12, fontweight='bold')

    # Subplot 2: Volume 24h
    ax2 = fig.add_subplot(gs[1])
    volume_clean = [val if val is not None else np.nan for val in volume_24h_data]
    ax2.plot(x_values, volume_clean, color='#1565C0', linewidth=1.5, label='24h Volume', zorder=3)
    ax2.fill_between(x_values, 0, volume_clean, alpha=0.2, color='#1565C0', zorder=2)

    ax2.set_ylabel('24h Volume', fontsize=10, fontweight='bold')
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_volume))
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.legend(loc='upper left', fontsize='small')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_xlim(-padding, min_length - 1 + padding)

    # Subplot 3: Price % Change 24h
    ax3 = fig.add_subplot(gs[2])
    price_pct_clean = [val if val is not None else np.nan for val in price_pct_change_24h_data]
    # Color based on positive/negative
    colors_price_pct = ['#2E7D32' if (v is not None and not np.isnan(v) and v >= 0) else '#C62828'
                        for v in price_pct_clean]
    ax3.bar(x_values, price_pct_clean, color=colors_price_pct, alpha=0.7, width=0.8, edgecolor='none')
    ax3.axhline(y=0, color='#000000', linewidth=1, linestyle='-', alpha=0.5)

    ax3.set_ylabel('Price Change 24h (%)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.set_xlim(-padding, min_length - 1 + padding)

    # Subplot 4: Volume % Change 24h
    ax4 = fig.add_subplot(gs[3])
    volume_pct_clean = [val if val is not None else np.nan for val in volume_pct_change_24h_data]
    # Color based on positive/negative
    colors_vol_pct = ['#1565C0' if (v is not None and not np.isnan(v) and v >= 0) else '#FF6F00'
                      for v in volume_pct_clean]
    ax4.bar(x_values, volume_pct_clean, color=colors_vol_pct, alpha=0.7, width=0.8, edgecolor='none')
    ax4.axhline(y=0, color='#000000', linewidth=1, linestyle='-', alpha=0.5)

    ax4.set_ylabel('Volume Change 24h (%)', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Time', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax4.set_xticks(positions)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_xlim(-padding, min_length - 1 + padding)

    # Save figure
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))

    # Generate filename based on screenshot type
    if screenshot_type == 'iteration' and timeframe_label:
        filename = os.path.join("./screenshots", f"{symbol}_{timeframe_label}_metrics_{timestamp_str}.png")
    elif event_type:
        if interval >= 10080:
            weeks = int(interval / 10080)
            candle_label = "1w" if weeks == 1 else f"{weeks}w"
        elif interval >= 1440:
            days = int(interval / 1440)
            candle_label = "1d" if days == 1 else f"{days}d"
        elif interval >= 60:
            hours = int(interval / 60)
            candle_label = "1h" if hours == 1 else f"{hours}h"
        else:
            candle_label = f"{int(interval)}m"
        filename = os.path.join("./screenshots", f"{symbol}_{candle_label}_metrics_{event_type}_{timestamp_str}.png")
    else:
        if interval >= 10080:
            weeks = int(interval / 10080)
            candle_label = "1w" if weeks == 1 else f"{weeks}w"
        elif interval >= 1440:
            days = int(interval / 1440)
            candle_label = "1d" if days == 1 else f"{days}d"
        elif interval >= 60:
            hours = int(interval / 60)
            candle_label = "1h" if hours == 1 else f"{hours}h"
        else:
            candle_label = f"{int(interval)}m"
        filename = os.path.join("./screenshots", f"{symbol}_{candle_label}_metrics_snapshot_{timestamp_str}.png")

    print(f"Generating Coinbase metrics chart: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    print(f"Coinbase metrics chart saved as {filename}")
    return filename
