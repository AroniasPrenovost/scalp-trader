#!/usr/bin/env python3
"""
CoinGecko Live Data Collection for Global Volume Tracking

This module collects real-time global volume data (across all exchanges)
from CoinGecko API to maintain consistency with backfilled historical data.

Why separate from Coinbase data:
- Coinbase API returns Coinbase-only volume (~5-7% of global volume)
- CoinGecko returns global volume across all exchanges
- For accurate volume charts, we need consistent data sources

Usage:
    from utils.coingecko_live import fetch_coingecko_current_data, should_update_coingecko_data
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def fetch_coingecko_current_data(coingecko_id='bitcoin', vs_currency='usd'):
    """
    Fetch current price and global volume from CoinGecko API.

    This uses the /coins/{id} endpoint which returns current market data
    including global volume across all exchanges.

    Args:
        coingecko_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
        vs_currency: Currency to price against (default: 'usd')

    Returns:
        dict with 'price' and 'volume_24h' (global), or None if error
    """
    api_key = os.getenv('COINGECKO_API_KEY')
    if not api_key:
        print("WARNING: COINGECKO_API_KEY not found in environment variables")
        return None

    # Use the /coins/{id} endpoint for current data
    url = f'https://api.coingecko.com/api/v3/coins/{coingecko_id}'

    headers = {
        'x-cg-demo-api-key': api_key,
        'Accept': 'application/json'
    }

    params = {
        'localization': 'false',
        'tickers': 'false',
        'market_data': 'true',
        'community_data': 'false',
        'developer_data': 'false',
        'sparkline': 'false'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract market data
        market_data = data.get('market_data', {})

        # Get current price in USD
        current_price = market_data.get('current_price', {}).get(vs_currency.lower())

        # Get total volume (global across all exchanges) in USD
        total_volume_usd = market_data.get('total_volume', {}).get(vs_currency.lower())

        if current_price is None or total_volume_usd is None:
            print(f"WARNING: Missing price or volume data from CoinGecko for {coingecko_id}")
            return None

        # Convert USD volume to BTC (base currency) to match our data format
        volume_24h_btc = total_volume_usd / current_price if current_price > 0 else 0

        result = {
            'price': str(current_price),
            'volume_24h': str(volume_24h_btc),  # Global volume in BTC
            'volume_24h_usd': str(total_volume_usd)  # Store USD too for reference
        }

        return result

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch data from CoinGecko for {coingecko_id}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"ERROR: Failed to parse CoinGecko response for {coingecko_id}: {e}")
        return None


def should_update_coingecko_data(last_update_timestamp, update_interval_seconds=3600):
    """
    Check if it's time to update CoinGecko data based on interval.

    Args:
        last_update_timestamp: Unix timestamp of last update
        update_interval_seconds: How often to update (default: 3600 = 1 hour)

    Returns:
        bool: True if update is needed
    """
    if last_update_timestamp is None:
        return True

    current_time = time.time()
    elapsed = current_time - last_update_timestamp

    return elapsed >= update_interval_seconds


def get_last_coingecko_update_time(data_file_path):
    """
    Get the timestamp of the last CoinGecko data entry in a file.

    Args:
        data_file_path: Path to the JSON data file

    Returns:
        float: Unix timestamp of last entry, or None if no data
    """
    import json

    if not os.path.exists(data_file_path):
        return None

    try:
        with open(data_file_path, 'r') as f:
            data = json.load(f)

        if not data or len(data) == 0:
            return None

        # Return timestamp of last entry
        return data[-1].get('timestamp')

    except (json.JSONDecodeError, KeyError, IndexError):
        return None


if __name__ == '__main__':
    # Test the functions
    print("Testing CoinGecko live data collection...\n")

    print("Fetching current Bitcoin data from CoinGecko:")
    btc_data = fetch_coingecko_current_data('bitcoin')

    if btc_data:
        print(f"✓ Success!")
        print(f"  Price: ${float(btc_data['price']):,.2f}")
        print(f"  Global Volume (24h): {float(btc_data['volume_24h']):,.2f} BTC")
        print(f"  Global Volume (24h): ${float(btc_data['volume_24h_usd']):,.2f} USD")
    else:
        print("✗ Failed to fetch data")

    print("\n" + "="*60)
    print("Test complete!")
