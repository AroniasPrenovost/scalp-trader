#!/usr/bin/env python3
"""
High-Frequency Data Collector for Coinbase

Collects ticker data every 20-30 seconds and stores ALL available fields:
- Timestamp
- Price
- Volume (24h)
- Best bid/ask
- Recent trades
- Bid-ask spread
- Trade sides (buy/sell pressure)

Run this for 1-2 days to build dataset, then backtest with high granularity.
"""

import os
import sys
import time
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.coinbase import get_coinbase_client

# Configuration
COLLECTION_INTERVAL_SECONDS = 30  # Collect every 30 seconds
DATA_DIRECTORY = 'coinbase-data-hf'  # High-frequency data directory
MAX_AGE_DAYS = 7  # Keep 7 days of high-frequency data


def fetch_ticker_data(client, product_id):
    """
    Fetch comprehensive ticker data from Coinbase Advanced Trade API

    Returns all available fields:
    - trades: Recent trades with price, size, side, time
    - best_bid: Current best bid price
    - best_ask: Current best ask price
    """
    try:
        # Get product ticker (includes trades, best bid/ask)
        ticker = client.get_market_trades(product_id=product_id, limit=10)

        if not ticker or not hasattr(ticker, 'trades'):
            print(f"  ⚠️  No ticker data for {product_id}")
            return None

        # Extract trade data
        trades = []
        if hasattr(ticker, 'trades') and ticker.trades:
            for trade in ticker.trades[:10]:  # Last 10 trades
                trades.append({
                    'trade_id': trade.trade_id if hasattr(trade, 'trade_id') else None,
                    'price': str(trade.price) if hasattr(trade, 'price') else None,
                    'size': str(trade.size) if hasattr(trade, 'size') else None,
                    'side': trade.side if hasattr(trade, 'side') else None,
                    'time': trade.time if hasattr(trade, 'time') else None
                })

        # Get best bid/ask
        best_bid = str(ticker.best_bid) if hasattr(ticker, 'best_bid') else None
        best_ask = str(ticker.best_ask) if hasattr(ticker, 'best_ask') else None

        # Calculate derived metrics
        last_price = trades[0]['price'] if trades else None
        bid_ask_spread = None
        bid_ask_spread_pct = None

        if best_bid and best_ask and last_price:
            try:
                bid_float = float(best_bid)
                ask_float = float(best_ask)
                price_float = float(last_price)

                bid_ask_spread = ask_float - bid_float
                bid_ask_spread_pct = (bid_ask_spread / price_float) * 100
            except (ValueError, TypeError):
                pass

        # Calculate buy/sell pressure from recent trades
        buy_volume = 0
        sell_volume = 0
        total_volume = 0

        for trade in trades:
            if trade['size'] and trade['side']:
                try:
                    size = float(trade['size'])
                    total_volume += size

                    if trade['side'] == 'BUY':
                        buy_volume += size
                    elif trade['side'] == 'SELL':
                        sell_volume += size
                except (ValueError, TypeError):
                    pass

        buy_pressure = (buy_volume / total_volume * 100) if total_volume > 0 else 50.0
        sell_pressure = (sell_volume / total_volume * 100) if total_volume > 0 else 50.0

        return {
            'timestamp': time.time(),
            'product_id': product_id,
            'price': last_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'bid_ask_spread': str(bid_ask_spread) if bid_ask_spread else None,
            'bid_ask_spread_pct': str(bid_ask_spread_pct) if bid_ask_spread_pct else None,
            'recent_trades': trades,
            'buy_pressure_pct': str(buy_pressure),
            'sell_pressure_pct': str(sell_pressure),
            'num_recent_trades': len(trades)
        }

    except Exception as e:
        print(f"  ❌ Error fetching ticker for {product_id}: {e}")
        return None


def append_to_hf_data_file(data_entry):
    """
    Append high-frequency data entry to its dedicated file
    Format: coinbase-data-hf/{product_id}.json
    """
    if not data_entry or 'product_id' not in data_entry:
        return

    # Ensure directory exists
    os.makedirs(DATA_DIRECTORY, exist_ok=True)

    product_id = data_entry['product_id']
    file_path = os.path.join(DATA_DIRECTORY, f"{product_id}.json")

    # Read existing data
    existing_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        except json.JSONDecodeError:
            print(f"  ⚠️  Corrupted file {file_path}, starting fresh")
            existing_data = []

    # Append new entry
    existing_data.append(data_entry)

    # Write back to file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)


def cleanup_old_hf_data(product_id, max_age_days):
    """Remove data points older than max_age_days"""
    file_path = os.path.join(DATA_DIRECTORY, f"{product_id}.json")

    if not os.path.exists(file_path):
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            return

        # Filter out old entries
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        filtered_data = [entry for entry in data if entry.get('timestamp', 0) >= cutoff_time]

        removed_count = len(data) - len(filtered_data)

        if removed_count > 0:
            # Write back filtered data
            with open(file_path, 'w') as f:
                json.dump(filtered_data, f, indent=2)

            print(f"  🧹 Cleaned up {removed_count} old entries from {product_id}")

    except Exception as e:
        print(f"  ⚠️  Error cleaning up {product_id}: {e}")


def load_enabled_symbols():
    """Load enabled symbols from config.json"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        symbols = [
            wallet['symbol']
            for wallet in config.get('wallets', [])
            if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
        ]

        return symbols
    except Exception as e:
        print(f"Error loading config: {e}")
        return []


def print_collection_stats():
    """Print stats about collected data"""
    if not os.path.exists(DATA_DIRECTORY):
        print("\n📊 No data collected yet\n")
        return

    print("\n" + "="*80)
    print("📊 HIGH-FREQUENCY DATA COLLECTION STATS")
    print("="*80 + "\n")

    for filename in sorted(os.listdir(DATA_DIRECTORY)):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(DATA_DIRECTORY, filename)
        product_id = filename.replace('.json', '')

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list) or len(data) == 0:
                continue

            # Calculate stats
            num_points = len(data)
            first_timestamp = data[0].get('timestamp', 0)
            last_timestamp = data[-1].get('timestamp', 0)

            duration_hours = (last_timestamp - first_timestamp) / 3600
            duration_days = duration_hours / 24

            # Calculate average interval
            intervals = []
            for i in range(1, len(data)):
                if 'timestamp' in data[i] and 'timestamp' in data[i-1]:
                    interval = data[i]['timestamp'] - data[i-1]['timestamp']
                    intervals.append(interval)

            avg_interval = sum(intervals) / len(intervals) if intervals else 0

            print(f"{product_id}:")
            print(f"  Data points: {num_points:,}")
            print(f"  Duration: {duration_days:.2f} days ({duration_hours:.1f} hours)")
            print(f"  Avg interval: {avg_interval:.1f}s")

            # Show if ready for backtesting (need at least 1 day)
            if duration_days >= 1:
                print(f"  Status: ✅ Ready for backtesting")
            elif duration_hours >= 6:
                print(f"  Status: 🟡 {24 - duration_hours:.1f} hours until ready")
            else:
                print(f"  Status: 🔴 Need {1 - duration_days:.1f} more days")

            print()

        except Exception as e:
            print(f"  ⚠️  Error reading {filename}: {e}\n")

    print("="*80 + "\n")


def main():
    """Main collection loop"""
    print("="*80)
    print("HIGH-FREQUENCY DATA COLLECTOR")
    print("="*80)
    print(f"\nCollection interval: {COLLECTION_INTERVAL_SECONDS}s")
    print(f"Data retention: {MAX_AGE_DAYS} days")
    print(f"Storage location: {DATA_DIRECTORY}/\n")

    # Load symbols
    symbols = load_enabled_symbols()

    if not symbols:
        print("❌ No enabled symbols found in config.json")
        return

    print(f"Monitoring {len(symbols)} symbols: {', '.join(symbols)}\n")
    print("="*80 + "\n")

    # Initialize Coinbase client
    try:
        client = get_coinbase_client()
        print("✅ Connected to Coinbase Advanced Trade API\n")
    except Exception as e:
        print(f"❌ Failed to initialize Coinbase client: {e}")
        return

    # Collection loop
    iteration = 0
    cleanup_interval = 3600  # Clean up every hour
    last_cleanup_time = 0
    stats_interval = 300  # Print stats every 5 minutes
    last_stats_time = 0

    print("🚀 Starting data collection...\n")

    try:
        while True:
            iteration += 1
            current_time = time.time()

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iteration #{iteration}")

            # Collect data for each symbol
            collected_count = 0
            for symbol in symbols:
                ticker_data = fetch_ticker_data(client, symbol)

                if ticker_data:
                    append_to_hf_data_file(ticker_data)
                    collected_count += 1

                    # Print summary
                    price = ticker_data.get('price', 'N/A')
                    spread = ticker_data.get('bid_ask_spread_pct', 'N/A')
                    buy_pressure = ticker_data.get('buy_pressure_pct', 'N/A')

                    if spread != 'N/A' and buy_pressure != 'N/A':
                        print(f"  ✓ {symbol}: ${price} | Spread: {float(spread):.3f}% | Buy pressure: {float(buy_pressure):.1f}%")
                    else:
                        print(f"  ✓ {symbol}: ${price}")

            print(f"✅ Collected {collected_count}/{len(symbols)} symbols\n")

            # Periodic cleanup
            if current_time - last_cleanup_time >= cleanup_interval:
                print("🧹 Running cleanup...")
                for symbol in symbols:
                    cleanup_old_hf_data(symbol, MAX_AGE_DAYS)
                print()
                last_cleanup_time = current_time

            # Periodic stats
            if current_time - last_stats_time >= stats_interval:
                print_collection_stats()
                last_stats_time = current_time

            # Sleep until next collection
            print(f"💤 Sleeping {COLLECTION_INTERVAL_SECONDS}s...\n")
            time.sleep(COLLECTION_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\n🛑 Collection stopped by user\n")
        print_collection_stats()
    except Exception as e:
        print(f"\n\n❌ Error in collection loop: {e}\n")
        print_collection_stats()


if __name__ == "__main__":
    main()
