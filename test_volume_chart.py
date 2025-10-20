#!/usr/bin/env python3
"""
Test script to generate a volume trend chart using existing data
"""
import time
from utils.matplotlib import plot_volume_trend_chart
from utils.file_helpers import get_property_values_from_crypto_file

# Configuration
global_volume_directory = "./coingecko-global-volume"  # Use global volume data
symbol = "BTC-USD"  # Change to test different symbols
data_retention_hours = 4380  # Should match your config
interval_minutes = 60  # 1 hour intervals

print(f"Testing volume trend chart generation for {symbol}")
print(f"Data directory: {global_volume_directory}")
print(f"Interval: {interval_minutes} minutes")
print()

# Get volume data from file (global volume from CoinGecko)
print("Loading global volume data (CoinGecko)...")
coin_volume_24h_LIST = get_property_values_from_crypto_file(
    global_volume_directory,
    symbol,
    'volume_24h',
    max_age_hours=data_retention_hours
)

# Convert to float
coin_volume_24h_LIST = [float(volume) for volume in coin_volume_24h_LIST if volume is not None]

print(f"Loaded {len(coin_volume_24h_LIST)} volume data points")

if len(coin_volume_24h_LIST) > 0:
    print(f"Latest volume: ${coin_volume_24h_LIST[-1]/1e6:.2f}M")
    print(f"Average volume: ${sum(coin_volume_24h_LIST)/len(coin_volume_24h_LIST)/1e6:.2f}M")
    print()

# Generate volume snapshot chart (uses ALL available data)
print("Generating volume snapshot chart...")
chart_path = plot_volume_trend_chart(
    current_timestamp=time.time(),
    interval=interval_minutes,
    symbol=symbol,
    volume_data=coin_volume_24h_LIST
)

if chart_path:
    print(f"\n✓ Volume snapshot chart generated successfully!")
    print(f"  Path: {chart_path}")
else:
    print("\n✗ Failed to generate volume snapshot chart")
    print("  Check that you have sufficient data (at least 3 data points)")

print("\nTest complete!")
