import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from utils.price_helpers import calculate_trading_range_percentage

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
    trade_range_percentage,
    volume_data=None
):

    """
    Simple snapshot plot with just price and optional volume data.
    For use before trading logic when entry_price and analysis are not yet available.

    Args:
        current_timestamp: timestamp for filename
        interval: data interval in minutes
        symbol: trading pair symbol
        price_data: list of price values
        min_price: minimum price in range
        max_price: maximum price in range
        trade_range_percentage: trading range percentage
        volume_data: optional list of volume values
    """

    # Create figure with subplots
    if volume_data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
        fig.subplots_adjust(hspace=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot price line
    x_values = list(range(len(price_data)))
    ax1.plot(x_values, price_data, marker=',', label='Price', c='black', linewidth=1.5)

    # Plot min/max lines
    ax1.axhline(y=min_price, color='green', linewidth=1, linestyle='--',
                label=f"Min (${min_price:.4f})", alpha=0.6)
    ax1.axhline(y=max_price, color='red', linewidth=1, linestyle='--',
                label=f"Max (${max_price:.4f})", alpha=0.6)

    # Configure main chart
    price_range = max_price - min_price
    buffer = price_range * 0.1
    ax1.set_ylim(min_price - buffer, max_price + buffer)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax1.set_ylabel('Price (USD)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize='small')
    ax1.set_title(f"{symbol} - Range: {trade_range_percentage}%", fontsize=12, fontweight='bold')

    if not volume_data:
        ax1.set_xlabel(f"Data Points (Interval: {interval} min)", fontsize=10, fontweight='bold')

    # Volume subplot (if data provided)
    if volume_data:
        colors = ['green' if i == 0 or price_data[i] >= price_data[i-1] else 'red'
                 for i in range(len(volume_data))]
        ax2.bar(x_values, volume_data, color=colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('Volume (24h)', fontsize=10, fontweight='bold')
        ax2.set_xlabel(f"Data Points (Interval: {interval} min)", fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

    # Save figure
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))
    filename = os.path.join("./screenshots", f"{symbol}_snapshot_{timestamp_str}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Snapshot saved as {filename}")


def plot_graph(
    current_timestamp,
    interval,
    symbol,
    price_data,
    min_price,
    max_price,
    trade_range_percentage,
    entry_price,
    volume_data=None,
    analysis=None,
    buy_event=False
):
    """
    Enhanced plot with technical indicators and AI analysis

    Args:
        current_timestamp: timestamp for filename
        interval: data interval in minutes
        symbol: trading pair symbol
        price_data: list of price values
        min_price: minimum price in range
        max_price: maximum price in range
        trade_range_percentage: trading range percentage
        entry_price: entry price if in position
        volume_data: optional list of volume values
        analysis: optional AI analysis dictionary with support/resistance/buy/sell levels
    """

    # Create figure with subplots (main chart + RSI + volume)
    fig = plt.figure(figsize=(14, 10))

    # Determine number of subplots based on available data
    num_subplots = 1
    if volume_data:
        num_subplots += 1

    # Calculate if we have enough data for RSI
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

    # Plot price line
    x_values = list(range(len(price_data)))
    ax1.plot(x_values, price_data, marker=',', label='Price', c='black', linewidth=1.5, zorder=5)

    # Calculate and plot moving averages
    if len(price_data) >= 20:
        ma20 = calculate_moving_average(price_data, 20)
        ax1.plot(x_values, ma20, label='MA(20)', c='blue', linewidth=1, alpha=0.7, linestyle='--')

    if len(price_data) >= 50:
        ma50 = calculate_moving_average(price_data, 50)
        ax1.plot(x_values, ma50, label='MA(50)', c='orange', linewidth=1, alpha=0.7, linestyle='--')

    # Calculate and plot Bollinger Bands
    if len(price_data) >= 20:
        bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(price_data, period=20)
        ax1.plot(x_values, bb_upper, label='BB Upper', c='gray', linewidth=0.8, alpha=0.5, linestyle=':')
        ax1.plot(x_values, bb_lower, label='BB Lower', c='gray', linewidth=0.8, alpha=0.5, linestyle=':')
        ax1.fill_between(x_values, bb_upper, bb_lower, alpha=0.1, color='gray')

    # Plot AI analysis levels if available
    if analysis:
        # Support levels
        if analysis.get('major_support'):
            ax1.axhline(y=analysis['major_support'], color='green', linewidth=1.5,
                       linestyle='--', label=f"Major Support (${analysis['major_support']:.4f})", alpha=0.7)
        if analysis.get('minor_support'):
            ax1.axhline(y=analysis['minor_support'], color='lightgreen', linewidth=1,
                       linestyle=':', label=f"Minor Support (${analysis['minor_support']:.4f})", alpha=0.6)

        # Resistance levels
        if analysis.get('major_resistance'):
            ax1.axhline(y=analysis['major_resistance'], color='red', linewidth=1.5,
                       linestyle='--', label=f"Major Resistance (${analysis['major_resistance']:.4f})", alpha=0.7)
        if analysis.get('minor_resistance'):
            ax1.axhline(y=analysis['minor_resistance'], color='lightcoral', linewidth=1,
                       linestyle=':', label=f"Minor Resistance (${analysis['minor_resistance']:.4f})", alpha=0.6)

        # Buy/Sell targets
        if analysis.get('buy_in_price'):
            ax1.axhline(y=analysis['buy_in_price'], color='cyan', linewidth=1.8,
                       linestyle='-', label=f"AI Buy Target (${analysis['buy_in_price']:.4f})", alpha=0.8)
        if analysis.get('sell_price'):
            ax1.axhline(y=analysis['sell_price'], color='purple', linewidth=1.8,
                       linestyle='-', label=f"AI Sell Target (${analysis['sell_price']:.4f})", alpha=0.8)

    # Plot entry price if in position
    if entry_price > 0:
        ax1.axhline(y=entry_price, color='magenta', linewidth=2,
                   linestyle='-', label=f"Entry Price (${entry_price:.4f})", zorder=10)

    # Configure main chart
    price_range = max_price - min_price
    buffer = price_range * 0.1
    ax1.set_ylim(min_price - buffer, max_price + buffer)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax1.set_ylabel('Price (USD)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize='x-small', ncol=2)

    # Title with AI analysis info
    title = f"{symbol} - Range: {trade_range_percentage}%"
    if analysis:
        title += f" | Trend: {analysis.get('market_trend', 'N/A')} | Confidence: {analysis.get('confidence_level', 'N/A')}"
    ax1.set_title(title, fontsize=12, fontweight='bold')

    # RSI subplot
    subplot_idx = 1
    if has_rsi:
        ax2 = fig.add_subplot(gs[subplot_idx])
        rsi = calculate_rsi(price_data, period=14)
        ax2.plot(x_values, rsi, label='RSI(14)', c='purple', linewidth=1.5)
        ax2.axhline(y=70, color='red', linewidth=0.8, linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(y=30, color='green', linewidth=0.8, linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.fill_between(x_values, 70, 100, alpha=0.1, color='red')
        ax2.fill_between(x_values, 0, 30, alpha=0.1, color='green')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize='x-small')
        subplot_idx += 1

    # Volume subplot
    if volume_data:
        ax3 = fig.add_subplot(gs[subplot_idx])
        colors = ['green' if i == 0 or price_data[i] >= price_data[i-1] else 'red'
                 for i in range(len(volume_data))]
        ax3.bar(x_values, volume_data, color=colors, alpha=0.6, width=0.8)
        ax3.set_ylabel('Volume (24h)', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

    # X-axis label on bottom subplot
    if num_subplots > 1:
        fig.get_axes()[-1].set_xlabel(f"Data Points (Interval: {interval} min)", fontsize=10, fontweight='bold')
    else:
        ax1.set_xlabel(f"Data Points (Interval: {interval} min)", fontsize=10, fontweight='bold')

    # Save figure
    event_type = 'sell'
    if buy_event:
        event_type = 'buy'
    timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(current_timestamp))
    filename = os.path.join("./screenshots", f"{symbol}_chart_{event_type}_{timestamp_str}.png")
    print(f"Generating market snapshot: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Chart saved as {filename}")
