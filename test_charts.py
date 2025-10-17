#!/usr/bin/env python3
"""Quick test script to generate multi-timeframe charts"""

import time
from utils.matplotlib import plot_multi_timeframe_charts
from utils.file_helpers import get_property_values_from_crypto_file

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

print(f"Loaded {len(coin_prices_LIST)} price data points")
print(f"Loaded {len(coin_volume_24h_LIST)} volume data points")

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
