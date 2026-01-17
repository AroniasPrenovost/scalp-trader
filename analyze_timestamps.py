#!/usr/bin/env python3
import json
import glob
from datetime import datetime, timedelta

def analyze_timestamps():
    # Get all JSON files in coinbase-data
    files = glob.glob('coinbase-data/*.json')

    # Get current time and 20 minutes ago
    current_time = datetime.now().timestamp()
    twenty_min_ago = current_time - (20 * 60)

    print(f"Current time: {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analyzing data from last 20 minutes (since {datetime.fromtimestamp(twenty_min_ago).strftime('%Y-%m-%d %H:%M:%S')})\n")
    print("=" * 80)

    for file_path in sorted(files):
        crypto_pair = file_path.split('/')[-1].replace('.json', '')

        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Filter for last 20 minutes
        recent_data = [entry for entry in data if entry.get('timestamp', 0) >= twenty_min_ago]

        if len(recent_data) < 2:
            print(f"\n{crypto_pair}: Not enough data points in last 20 minutes (only {len(recent_data)})")
            continue

        # Sort by timestamp
        recent_data.sort(key=lambda x: x['timestamp'])

        # Calculate intervals
        intervals = []
        timestamps = []
        for i in range(1, len(recent_data)):
            interval = recent_data[i]['timestamp'] - recent_data[i-1]['timestamp']
            intervals.append(interval)
            timestamps.append((
                datetime.fromtimestamp(recent_data[i-1]['timestamp']).strftime('%H:%M:%S'),
                datetime.fromtimestamp(recent_data[i]['timestamp']).strftime('%H:%M:%S'),
                interval
            ))

        if not intervals:
            continue

        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)

        print(f"\n{crypto_pair}:")
        print(f"  Data points in last 20 min: {len(recent_data)}")
        print(f"  Average interval: {avg_interval:.2f} seconds")
        print(f"  Min interval: {min_interval:.2f} seconds")
        print(f"  Max interval: {max_interval:.2f} seconds")
        print(f"  Expected (60s): {'✓' if 55 <= avg_interval <= 65 else '✗'}")

        # Show individual intervals
        print(f"  Individual intervals:")
        for start_time, end_time, interval in timestamps:
            deviation = abs(interval - 60)
            status = "✓" if deviation <= 5 else ("⚠" if deviation <= 10 else "✗")
            print(f"    {start_time} -> {end_time}: {interval:6.2f}s {status}")

if __name__ == "__main__":
    analyze_timestamps()
