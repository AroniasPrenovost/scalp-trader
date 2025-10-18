#!/usr/bin/env python3
"""Quick test script to generate multi-timeframe charts"""

import time
from utils.matplotlib import plot_multi_timeframe_charts
from utils.file_helpers import get_property_values_from_crypto_file

def calculate_volume_change_percentage(volume_list):
    """
    Calculate the percentage change in volume from one data point to the next.

    Args:
        volume_list: List of volume values (e.g., 24h volume)

    Returns:
        List of percentage changes (first element will be 0.0)
    """
    if not volume_list or len(volume_list) < 2:
        return []

    volume_changes = [0.0]  # First data point has no previous value to compare

    for i in range(1, len(volume_list)):
        prev_volume = volume_list[i - 1]
        current_volume = volume_list[i]

        if prev_volume == 0 or prev_volume is None:
            # Avoid division by zero
            volume_changes.append(0.0)
        else:
            # Calculate percentage change: ((current - previous) / previous) * 100
            change_pct = ((current_volume - prev_volume) / prev_volume) * 100
            volume_changes.append(change_pct)

    return volume_changes


# Configuration
symbol = "BTC-USD"
interval_minutes = 60  # 1 hour intervals
data_retention_hours = 4380  # 6 months
coinbase_data_directory = 'coinbase-data'

# Get price data
print(f"Loading price data for {symbol}...")
coin_prices_LIST = get_property_values_from_crypto_file(
    coinbase_data_directory,
    symbol,
    'price',
    max_age_hours=data_retention_hours
)
coin_prices_LIST = [float(price) for price in coin_prices_LIST if price is not None]

# Get volume data
coin_volume_24h_LIST = get_property_values_from_crypto_file(
    coinbase_data_directory,
    symbol,
    'volume_24h',
    max_age_hours=data_retention_hours
)
coin_volume_24h_LIST = [float(volume) for volume in coin_volume_24h_LIST if volume is not None]

# Calculate volume change percentage
print("Calculating volume change percentages...")
coin_volume_change_LIST = calculate_volume_change_percentage(coin_volume_24h_LIST)

print(f"Loaded {len(coin_prices_LIST)} price data points")
print(f"Loaded {len(coin_volume_24h_LIST)} volume data points")
print(f"Calculated {len(coin_volume_change_LIST)} volume change data points")

# Show a few examples of volume changes
if len(coin_volume_change_LIST) > 5:
    print(f"\nSample volume changes (last 5):")
    for i in range(-5, 0):
        print(f"  Volume: ${coin_volume_24h_LIST[i]/1e6:.2f}M -> Change: {coin_volume_change_LIST[i]:+.1f}%")

# Generate charts
print("\nGenerating multi-timeframe charts...")
chart_paths = plot_multi_timeframe_charts(
    current_timestamp=time.time(),
    interval=interval_minutes,
    symbol=symbol,
    price_data=coin_prices_LIST,
    volume_data=coin_volume_24h_LIST,
    analysis=None
)

print("\n✓ Charts generated:")
for timeframe, path in chart_paths.items():
    print(f"  {timeframe}: {path}")
