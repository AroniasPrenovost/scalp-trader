import os
import base64
from dotenv import load_dotenv
from json import dumps, load
import json
import math
import time
from pprint import pprint
from collections import deque
import numpy as np
import pandas as pd
import requests # supports CoinMarketCap
# coinbase api
from coinbase.rest import RESTClient
# from mailjet_rest import Client
# parse CLI args
import argparse
# related to price change % logic
import glob



# custom imports
from utils.email import send_email_notification
from utils.file_helpers import count_files_in_directory, delete_files_older_than_x_hours, is_most_recent_file_older_than_x_minutes
from utils.price_helpers import calculate_trading_range_percentage, calculate_current_price_position_within_trading_range, calculate_offset_price, calculate_price_change_percentage
from utils.technical_indicators import calculate_market_cap_efficiency, calculate_fibonacci_levels
from utils.time_helpers import print_local_time
# coinbase api
from utils.coinbase import get_coinbase_client, get_coinbase_order_by_order_id, place_market_buy_order, place_market_sell_order, get_asset_price, calculate_exchange_fee
coinbase_client = get_coinbase_client()
# custom coinbase listings check
from utils.new_coinbase_listings import check_for_new_coinbase_listings, save_listed_coins_to_file
# load .env s
load_dotenv()

def load_config(file_path):
    with open(file_path, 'r') as file:
        return load(file)

#
#
# Initialize a dictionary to store price data for each asset
#

LOCAL_PRICE_DATA = {}

#
#
# Initialize a dictionary to store volume-based recommendations for each asset
#

VOLUME_BASED_RECOMMENDATIONS = {}

#
#
# Initialize a dictionary to store volume-based recommendations based on each provider data
#

COINBASE_DATA_RECOMMENDATIONS = {}
CMC_DATA_RECOMMENDATIONS = {}
TREND_NOTIFICATIONS = {}

#
#
# Initialize a dictionary to store support and resistance levels for each asset
#

SUPPORT_RESISTANCE_LEVELS = {}
last_calculated_support_resistance_pivot_prices = {}  # Store the last calculated price for each asset

#
#
# Time tracking
#

APP_START_TIME_DATA = {} # global data to help manage time

#
#
# Define time intervals
#

# ------------------
INTERVAL_SECONDS = 30
INTERVAL_MINUTES = 15
# ------------------
# INTERVAL_SECONDS = 2
# INTERVAL_MINUTES = 360
# ------------------
# INTERVAL_SECONDS = 5
# INTERVAL_MINUTES = 240 # 4 hours
# ------------------
# INTERVAL_SECONDS = 5
# INTERVAL_MINUTES = 480 # 8 hours
# ------------------
# INTERVAL_SECONDS = 5
# INTERVAL_MINUTES = 720 # 12 hours
# ------------------

DATA_POINTS_FOR_X_MINUTES = int((60 / INTERVAL_SECONDS) * INTERVAL_MINUTES)

#
#
# Store the last error and manage number of errors before killing program
#

LAST_EXCEPTION_ERROR = None
LAST_EXCEPTION_ERROR_COUNT = 0
MAX_LAST_EXCEPTION_ERROR_COUNT = 8


#
#
# Initialize a dictionary to store trend data for each asset
#

LOCAL_TREND_1_DATA = {}
TEST_TREND_1_DATA = {}
TREND_1_PRICE_OFFSET_PERCENT = 0.05 # for visibility on graph

LOCAL_TREND_2_DATA = {}
TEST_TREND_2_DATA = {}
TREND_2_PRICE_OFFSET_PERCENT = 0.1 # for visibility on graph

# for mapping the divergent outcomes between these 2 ^
LOCAL_UPWARD_TREND_DIVERGENCE_DATA = {}
TEST_UPWARD_TREND_DIVERGENCE_DATA = {}

LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA = {}
TEST_DOWNWARD_TREND_DIVERGENCE_DATA = {}



coinbase_spot_maker_fee = float(os.environ.get('COINBASE_SPOT_MAKER_FEE'))
coinbase_spot_taker_fee = float(os.environ.get('COINBASE_SPOT_TAKER_FEE'))
coinbase_stable_pair_spot_maker_fee = float(os.environ.get('COINBASE_STABLE_PAIR_SPOT_MAKER_FEE'))
coinbase_stable_pair_spot_taker_fee = float(os.environ.get('COINBASE_STABLE_PAIR_SPOT_TAKER_FEE'))
federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))


#
#
# Get trend
#

def determine_trend_1(prices, data_points_for_entire_interval, timeframe_percent):
    """
    Determine the trend in the given price data.

    :param prices: deque of stock prices
    :param data_points_for_entire_interval: total number of data points for the timeframe
    :return: string indicating the trend ('downward', 'upward', or 'neutral')
    """
    # Calculate the period as 17% of the total data points
    period = max(1, int(data_points_for_entire_interval * timeframe_percent))

    if len(prices) < period:
        return 'neutral'

    # Calculate the moving average for the specified period
    moving_average = np.mean(list(prices)[-period:])

    # Compare the current price to the moving average
    current_price = prices[-1]

    if current_price < moving_average:
        return 'downward'
    elif current_price > moving_average:
        return 'upward'
    else:
        return 'neutral'

#
#
# Get trend 2
#

def determine_trend_2(prices, lookback_period=30, short_window=20, long_window=50, support_resistance_window=30, atr_window=14):
    """
    Detects a trend in the price data using moving averages, support/resistance levels, and ATR.

    :param prices: deque of stock prices
    :param lookback_period: number of periods to look back for highs and lows
    :param short_window: window size for the short moving average
    :param long_window: window size for the long moving average
    :param support_resistance_window: window size for calculating support and resistance
    :param atr_window: window size for calculating the Average True Range
    :return: string indicating the change ('bullish', 'bearish', or 'none')
    """
    if len(prices) < max(lookback_period, long_window, support_resistance_window, atr_window):
        return 'none'

    # Convert deque to pandas Series for easier manipulation
    prices_series = pd.Series(list(prices))

    # Calculate moving averages
    short_ma = prices_series.rolling(window=short_window).mean()
    long_ma = prices_series.rolling(window=long_window).mean()

    # Calculate support and resistance
    support = prices_series.rolling(window=support_resistance_window).min()
    resistance = prices_series.rolling(window=support_resistance_window).max()

    # Calculate ATR
    high_low = prices_series.rolling(window=2).apply(lambda x: x.max() - x.min(), raw=True)
    high_close = prices_series.rolling(window=2).apply(lambda x: abs(x[1] - x[0]), raw=True)
    low_close = prices_series.rolling(window=2).apply(lambda x: abs(x[1] - x[0]), raw=True)
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_window).mean()

    # Determine trend
    trend = np.where(short_ma > long_ma, 'upward', 'downward')

    # Get recent prices for lookback period
    recent_prices = prices_series[-lookback_period:]
    recent_high = recent_prices.max()
    recent_low = recent_prices.min()
    current_price = prices_series.iloc[-1]

    if trend[-1] == 'upward':
        return 'bullish'
    elif trend[-1] == 'downward':
        return 'bearish'

    # Determine trend using ATR
    if current_price > recent_high + atr.iloc[-1] and trend[-1] == 'upward':
        return 'bullish'
    elif current_price < recent_low - atr.iloc[-1] and trend[-1] == 'downward':
        return 'bearish'
    else:
        return 'none'




#
#
# Generating test data
#

def generate_test_price_data(start_price, data_points, trend=0.001, volatility=0.025):
    """
    Generate a list of simulated price data for testing.

    :param start_price: The starting price for the simulation.
    :param data_points: The number of minutes (data points) to generate.
    :param trend: The average change in price per minute (positive for upward trend, negative for downward).
    :param volatility: The standard deviation of the price changes (higher values for more volatility).
    :return: A deque containing the simulated price data.
    """
    prices = deque(maxlen=data_points)
    current_price = start_price

    for _ in range(data_points):
        # Simulate a random price change with a trend
        price_change = np.random.normal(loc=trend, scale=volatility)
        current_price += price_change
        prices.append(current_price)

    return prices


#
#
# TEST mode
#

IS_TEST_MODE = False
parser = argparse.ArgumentParser(description='Process some command-line arguments.')
parser.add_argument('mode', nargs='?', default='default', help='Mode of operation (e.g., test)')
args = parser.parse_args()
mode = args.mode
if mode == 'test':
    cfig = load_config('config.json');
    for asset in cfig['assets']:
        enabled = asset['enabled']
        symbol = asset['symbol']
        if enabled:
            IS_TEST_MODE = True
            print(f"Generating test data for {symbol}...")

            start_price = get_asset_price(coinbase_client, symbol)

            symbol = asset['symbol']
            SHARES_TO_ACQUIRE = asset['shares_to_acquire']
            TARGET_PROFIT_PERCENTAGE = asset['target_profit_percentage']
            TEST_DATA_TREND_RATE = asset['test_data_trend_rate']
            TEST_DATA_VOLATILITY_RATE = asset['test_data_volatility_rate']
            TREND_1_TIMEFRAME_PERCENT = asset['trend_1_timeframe_percent']
            TREND_1_DISPLAY = asset['trend_1_display']
            TREND_2_TIMEFRAME_PERCENT = asset['trend_2_timeframe_percent']
            TREND_2_DISPLAY = asset['trend_2_display']
            READY_TO_TRADE = asset['ready_to_trade']
            ENABLE_GRAPH_DISPLAY = asset['enable_graph_display']
            ENABLE_GRAPH_SCREENSHOT = asset['enable_graph_screenshot']

            #
            # Initialize price data storage

            if symbol not in TEST_TREND_1_DATA:
                TEST_TREND_1_DATA[symbol] = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)

            if symbol not in TEST_TREND_2_DATA:
                TEST_TREND_2_DATA[symbol] = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)

            if symbol not in TEST_UPWARD_TREND_DIVERGENCE_DATA:
                TEST_UPWARD_TREND_DIVERGENCE_DATA[symbol] = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)

            if symbol not in TEST_DOWNWARD_TREND_DIVERGENCE_DATA:
                TEST_DOWNWARD_TREND_DIVERGENCE_DATA[symbol] = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)

            raw_test_data = generate_test_price_data(start_price, DATA_POINTS_FOR_X_MINUTES, TEST_DATA_TREND_RATE, TEST_DATA_VOLATILITY_RATE)
            raw_test_data.reverse()

            # populate local price data array with generated test prices
            if symbol not in LOCAL_PRICE_DATA:
                LOCAL_PRICE_DATA[symbol] = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)

            for price in raw_test_data:
                LOCAL_PRICE_DATA[symbol].append(price)

                #
                #
                # TREND #1
                #
                trend_1 = determine_trend_1(LOCAL_PRICE_DATA[symbol], DATA_POINTS_FOR_X_MINUTES, TREND_1_TIMEFRAME_PERCENT)
                trend_1_offset_price = calculate_offset_price(price, trend_1, TREND_1_PRICE_OFFSET_PERCENT)
                TEST_TREND_1_DATA[symbol].append(trend_1_offset_price)
                #
                #
                # TREND #2
                #
                trend_2 = determine_trend_2(LOCAL_PRICE_DATA[symbol], TREND_2_TIMEFRAME_PERCENT)
                trend_2_offset_price = calculate_offset_price(price, trend_2, TREND_2_PRICE_OFFSET_PERCENT)
                TEST_TREND_2_DATA[symbol].append(trend_2_offset_price)
                #
                #
                # visualize indicator divergences
                #
                upward_divergence = trend_1 == 'upward' and trend_2 == 'bearish'
                if upward_divergence == True:
                    TEST_UPWARD_TREND_DIVERGENCE_DATA[symbol].append(price)
                downward_divergence = trend_1 == 'downward' and trend_2 == 'bullish'
                if downward_divergence == True:
                    TEST_DOWNWARD_TREND_DIVERGENCE_DATA[symbol].append(price)

            print('Upward trend divergences: ', f"{len(TEST_UPWARD_TREND_DIVERGENCE_DATA[symbol])}/{len(LOCAL_PRICE_DATA[symbol])}")
            print('Downward trend divergences: ', f"{len(TEST_DOWNWARD_TREND_DIVERGENCE_DATA[symbol])}/{len(LOCAL_PRICE_DATA[symbol])}")
else:
    print(f"Running in {mode} mode")


#
#
# Coinmarketcap API and data
#

coinmarketcap_api_key = os.environ.get('COINMARKETCAP_API_KEY')

# Initialize a dictionary to store Coinmarketcap volume data and its timestamp
CMC_VOLUME_DATA_CACHE = {}
CMC_VOLUME_DATA_TIMESTAMP = {}

def fetch_coinmarketcap_volume_data(symbol):
    global CMC_VOLUME_DATA_CACHE, CMC_VOLUME_DATA_TIMESTAMP

    CMC_SYMBOL = symbol.split('-')[0] # NOTE symbol modification
    current_time = time.time()

    # Check if we have cached data and if it's still valid (less than 15 minutes old)
    if symbol in CMC_VOLUME_DATA_CACHE and (current_time - CMC_VOLUME_DATA_TIMESTAMP[symbol] < 900):
        return CMC_VOLUME_DATA_CACHE[symbol]

    # If not, fetch new data
    LATEST_PRICE_API_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    headers = {
        'X-CMC_PRO_API_KEY': coinmarketcap_api_key,
        'Accept': 'application/json',
    }
    params = {
        'symbol': CMC_SYMBOL,
    }

    try:
        response = requests.get(LATEST_PRICE_API_URL, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        CMC_VOLUME_DATA_CACHE[symbol] = data['data'][CMC_SYMBOL]
        CMC_VOLUME_DATA_TIMESTAMP[symbol] = current_time
        return CMC_VOLUME_DATA_CACHE[symbol]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest data for {symbol}: {e}")
        return None


#
#
# Save order data to local json ledger
#

def save_order_data_to_local_json_ledger(symbol, order_data):
    """
    Save order data to a local file specific to the symbol.
    """
    file_name = f"{symbol}_orders.json"
    try:
        # Load existing data if the file exists and is not empty
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        # Append the new order data
        existing_data.append(order_data)

        # Save the updated data back to the file
        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

        print(f"Order data saved to {file_name}.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_name}. The file might be corrupted.")
        # Attempt to overwrite the file with the new order data
        with open(file_name, 'w') as file:
            json.dump([order_data], file, indent=4)
        print(f"Order data saved to {file_name} after resetting the file.")
    except Exception as e:
        print(f"Error saving order data for {symbol}: {e}")

#
#
# get_last_order_from_local_json_ledger
#

def get_last_order_from_local_json_ledger(symbol):
    """
    Retrieve the most recent order from the local JSON ledger for the given symbol.
    """
    file_name = f"{symbol}_orders.json"
    try:
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, 'r') as file:
                orders = json.load(file)
                if orders:
                    # Return the most recent order
                    return orders[-1]
        print(f"No orders found in ledger for {symbol}.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_name}. The file might be corrupted.")
        return None
    except Exception as e:
        print(f"Error retrieving order from ledger for {symbol}: {e}")
        return None

#
#
#
#

def reset_json_ledger_file(symbol):
    # Construct the file name based on the symbol
    file_name = f"{symbol}_orders.json"

    # Write an empty list to the file to reset it
    with open(file_name, 'w') as file:
        json.dump([], file)

#
#
#
#


def detect_stored_coinbase_order_type(last_order):
    if last_order is None:
        return 'none'
    if 'order' in last_order: # if 'order' key exists, it's a finalized order
        if 'side' in last_order['order']:
            type = last_order['order']['side']
            return type.lower()
    else:
        return 'placeholder'



#
#
# Determine support and resistance levels
#

def calculate_support_resistance_1(prices, window_size):
    """
    Calculate support, resistance, and pivot levels using local maxima/minima and traditional pivot calculation.

    :param prices: deque of stock prices
    :param window_size: number of periods to consider for local maxima/minima
    :return: tuple containing pivot, support, and resistance levels
    """
    if len(prices) < window_size:
        raise ValueError("Prices deque must contain at least 'window_size' elements.")

    prices_series = pd.Series(list(prices))

    # Calculate local maxima and minima
    local_max = prices_series.rolling(window=window_size, center=True).max()
    local_min = prices_series.rolling(window=window_size, center=True).min()

    # Determine resistance as the average of local maxima
    resistance = local_max.mean()

    # Determine support as the average of local minima
    support = local_min.mean()

    # Calculate pivot point using the last high, low, and close prices
    high = max(prices)
    low = min(prices)
    close = prices[-1]  # Assuming the last price is the closing price
    pivot = (high + low + close) / 3

    return pivot, support, resistance


def calculate_support_resistance_2(prices):
        """
        Calculate support and resistance levels using the 15-minute rule for a given set of stock prices.

        :param prices: deque of stock prices
        :return: tuple containing pivot, support, and resistance levels
        """
        if not prices or len(prices) < 15:
            raise ValueError("Prices deque must contain at least fifteen elements for the 15-minute rule.")

        # Use the first 15 minutes to determine high and low
        first_15_min_prices = list(prices)[:15]
        high = max(first_15_min_prices)
        low = min(first_15_min_prices)

        # Calculate pivot point
        pivot = (high + low) / 2

        # Calculate support and resistance levels
        resistance = high
        support = low

        return pivot, support, resistance

#
#
# Check if support and resistance levels should be recalculated
#

def should_recalculate_support_resistance_1(prices, last_calculated_price, price_change_threshold=1.0):
    """
    Determine if support and resistance levels should be recalculated based on significant price movement.

    :param prices: deque of stock prices
    :param last_calculated_price: the price at the last calculation of support/resistance
    :param price_change_threshold: the percentage change required to trigger a recalculation
    :return: boolean indicating whether to recalculate support and resistance
    """
    if not prices:
        return False

    current_price = prices[-1]
    price_change_percentage = abs((current_price - last_calculated_price) / last_calculated_price) * 100

    return price_change_percentage >= price_change_threshold


#
#
# Calculate Simple Moving Average (SMA)
#

def calculate_sma(prices, period):
    if len(prices) < period:
        return None
    # Convert deque to a list before slicing
    prices_list = list(prices)
    return np.mean(prices_list[-period:])

#
#
# Calculate MACD
#

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    if len(prices) < long_period:
        return None, None
    prices_list = list(prices)  # Convert deque to list
    short_ema = np.mean(prices_list[-short_period:])
    long_ema = np.mean(prices_list[-long_period:])
    macd_line = short_ema - long_ema
    signal_line = np.mean(prices_list[-signal_period:])
    return macd_line, signal_line

#
#
# Calculate Bollinger Bands
#

def calculate_bollinger_bands(prices, period=20, num_std_dev=2):
    if len(prices) < period:
        return None, None, None
    prices_list = list(prices)  # Convert deque to list
    sma = calculate_sma(prices_list, period)
    std_dev = np.std(prices_list[-period:])
    upper_bollinger_band = sma + (num_std_dev * std_dev)
    lower_bollinger_band = sma - (num_std_dev * std_dev)
    return upper_bollinger_band, lower_bollinger_band, sma

#
#
# Volume Analysis Functions
#

def volume_strength_index(volume_24h, volume_change_24h, price_change_24h):
    """
    Calculate a volume strength index based on 24-hour metrics.
    :param volume_24h: Total volume in 24 hours
    :param volume_change_24h: Percentage change in volume
    :param price_change_24h: Percentage change in price
    :return: Volume strength score (-1 to 1)
    """
    volume_momentum = volume_change_24h / 100
    price_volume_correlation = (price_change_24h / 100) * (abs(volume_change_24h) / 100) if volume_24h > 0 else 0
    strength_score = (volume_momentum + price_volume_correlation) / 2
    return max(min(strength_score, 1), -1)

def generate_volume_signal(volume_24h, volume_change_24h, price_change_24h, volume_threshold=5000000, volume_change_threshold=-20):
    """
    Generate trading signals based on volume characteristics.
    :param volume_24h: Total volume in 24 hours
    :param volume_change_24h: Percentage change in volume
    :param price_change_24h: Percentage change in price
    :param volume_threshold: Minimum volume for consideration
    :param volume_change_threshold: Minimum volume change threshold
    :return: Trading signal (-1: sell, 0: hold, 1: buy)
    """
    strength_score = volume_strength_index(volume_24h, volume_change_24h, price_change_24h)
    if volume_24h < volume_threshold:
        return 0  # Insufficient volume
    if volume_change_24h < volume_change_threshold:
        return -1 if price_change_24h > 0 else 0  # Potential bearish divergence
    if strength_score > 0.5:
        return 1  # Strong bullish signal
    elif strength_score < -0.5:
        return -1  # Strong bearish signal
    return 0  # Neutral signal

def volume_volatility_indicator(volume_24h, volume_change_24h, price_change_1h):
    """
    Calculate volume volatility.
    :param volume_24h: Total volume in 24 hours
    :param volume_change_24h: Percentage change in volume
    :param price_change_1h: Percentage change in 1 hour
    :return: Volatility score
    """
    volume_volatility = abs(volume_change_24h)
    price_momentum = abs(price_change_1h)
    volatility_score = (volume_volatility * price_momentum) / 100
    return volatility_score

# Scalp Trading Strategy Function (using volume analysis)
#

def volume_based_strategy_recommendation(data):
    """
    Scalp trading strategy using volume analysis.
    :param data: Dictionary containing volume and price change data
    :return: Trading recommendation
    """
    volume_24h = data['quote']['USD']['volume_24h']
    volume_change_24h = data['quote']['USD']['volume_change_24h']
    price_change_24h = data['quote']['USD']['percent_change_24h']
    price_change_1h = data['quote']['USD']['percent_change_1h']

    volume_signal = generate_volume_signal(volume_24h, volume_change_24h, price_change_24h)
    volatility = volume_volatility_indicator(volume_24h, volume_change_24h, price_change_1h)

    if volume_signal == 1 and volatility < 2:
        return 'buy'
        # Strong volume support with low volatility"
    elif volume_signal == -1 and volatility > 3:
        return 'sell'
        # Weakening volume and high volatility"
    else:
        return 'hold'
        # Insufficient clear signals"

# volume_data = fetch_coinmarketcap_volume_data(symbol)
# if volume_data:
#     volume_based_strategy = volume_based_strategy_recommendation(volume_data)
#     print(f"volume-based trading strategy recommendation for {symbol}: {volume_based_strategy}")





#
#
#
#
#
#

# Function to save current listed coins to a timestamped file
def save_new_coinbase_data(coins, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_name = f"listed_coins_{timestamp}.json"
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'w') as file:
        json.dump(coins, file, indent=4)
    print(f"Listed coins saved to {file_path}.")

#
#
#
#

# Function to convert Product objects to dictionaries
def convert_products_to_dicts(products):
    return [product.to_dict() if hasattr(product, 'to_dict') else product for product in products]



#
#
#
#

# def get_price_from_data(data, symbol):
#     for coin in data:
#         if coin['product_id'] == symbol:
#             price = coin.get('price', '')
#             if price:  # Check if price is not an empty string
#                 return float(price)
#             else:
#                 print(f"Warning: Price for {symbol} is missing or empty.")
#                 return None
#     return None

def calculate_changes_for_assets_old(directory, symbol):
    # Get all JSON files in the directory, sorted by creation time
    files = sorted(glob.glob(os.path.join(directory, '*.json')), key=os.path.getctime)

    if len(files) < 2:
        print("Not enough data files to calculate price changes.")
        return None

    # Load the oldest and newest data files
    with open(files[0], 'r') as old_file:
        old_data = json.load(old_file)
    with open(files[-1], 'r') as new_file:
        new_data = json.load(new_file)

    # Get prices for the specified symbol
    old_price = get_price_from_data(old_data, symbol)
    new_price = get_price_from_data(new_data, symbol)

    if old_price is None or new_price is None:
        print(f"Price data for {symbol} not found in the files.")
        return None

    # Calculate the price change percentage
    price_change_percentage = calculate_price_change_percentage(old_price, new_price)
    return price_change_percentage



def get_price_from_data(data, symbol):
    for coin in data:
        if coin['product_id'] == symbol:
            price = coin.get('price', '')
            if price:
                return float(price)
    return None

def calculate_coin_price_changes(directory, symbol):
    # Calculate the current time and get the list of all files
    current_time = time.time()
    file_paths = sorted(glob.glob(os.path.join(directory, '*.json')), key=os.path.getctime)

    # We will check for the files that correspond to the requested timeframes
    timeframes = [5*60, 15*60, 30*60, 60*60]  # Timeframes in seconds
    price_changes = {}

    for timeframe in timeframes:
        # Find the index of the file closest to the timeframe
        for i, file_path in enumerate(file_paths):
            file_time = os.path.getctime(file_path)
            if current_time - file_time <= timeframe:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                old_price = get_price_from_data(data, symbol)
                if old_price is not None:
                    break
        else:
            print(f"Not enough data for {timeframe//60} minutes timeframe.")
            old_price = None

        if old_price is None:
            continue

        # Always get the latest price
        with open(file_paths[-1], 'r') as file:
            new_data = json.load(file)
        new_price = get_price_from_data(new_data, symbol)

        if new_price is None:
            print(f"Price data for {symbol} not found in the latest file.")
            continue

        # Calculate the price change percentage
        price_change_percentage = ((new_price - old_price) / old_price) * 100
        price_changes[timeframe // 60] = price_change_percentage

    return price_changes

    # Usage example: calculate_coin_price_changes('/path/to/directory', 'BTC-USD')




#
#
#
#
# main logic loop
#

def iterate_assets(interval_minutes, interval_seconds, data_points_for_x_minutes):
    while True:

        print_local_time();

        #
        # ERROR TRACKING
        global LAST_EXCEPTION_ERROR
        global LAST_EXCEPTION_ERROR_COUNT

        current_listed_coins_dictionary = {}

        #
        # ALERT NEW COIN LISTINGS
        enable_new_listings_alert = True
        if enable_new_listings_alert:
            file_path = 'coinbase_listed_coins.json'
            current_listed_coins = coinbase_client.get_products()['products']
            current_listed_coins_dictionary = convert_products_to_dicts(current_listed_coins)
            new_coins = check_for_new_coinbase_listings(file_path, current_listed_coins_dictionary)
            if new_coins:
                print("New coins added:")
                for coin in new_coins:
                    coin_symbol = coin['product_id']
                    print(coin_symbol)
                    current_price = get_asset_price(coinbase_client, coin_symbol)
                    print(f"current_price: {current_price}")
                    time.sleep(2)
                    send_email_notification(
                        subject=f"New Coinbase listing: {coin_symbol}",
                        text_content=f"Coinbase just listed {coin_symbol}",
                        html_content=f"Coinbase just listed {coin_symbol}"
                    )
                    send_email_notification(
                        subject=f"New Coinbase listing: {coin_symbol}",
                        text_content=f"Coinbase just listed {coin_symbol}",
                        html_content=f"Coinbase just listed {coin_symbol}",
                        custom_recipient=mailjet_to_email_2,
                    )
            else:
                print("No new coins added.\n")
            save_listed_coins_to_file(current_listed_coins, file_path)

        #
        #
        # Save coinbase asset data and analyze

        coinbase_price_history_data = 'coinbase-data'
        if is_most_recent_file_older_than_x_minutes(coinbase_price_history_data, minutes=2):
            save_new_coinbase_data(current_listed_coins_dictionary, coinbase_price_history_data)
        delete_files_older_than_x_hours(coinbase_price_history_data, hours=2)

        files_in_folder = count_files_in_directory(coinbase_price_history_data)
        for coin in current_listed_coins_dictionary:
            if files_in_folder > 1:
                if coin['product_id'] == 'YFI-BTC' or "USDC" not in coin['product_id']:
                    continue


                #
                #
                #
                # DETECT TRENDING COINS

                # calculate price change over different timeframes minutes (5, 15, 30, 60)
                price_change_percentages = calculate_coin_price_changes(coinbase_price_history_data, coin['product_id'])
                print(price_change_percentages)

                #
                #
                #

                price_change_percentage_range = calculate_changes_for_assets_old(coinbase_price_history_data, coin['product_id'])
                # print('price_change_percentage_range', price_change_percentage_range)
                try:
                    volume_24h = float(coin['volume_24h'])
                    # print('volume_24h', volume_24h)
                    volume_percentage_change_24h = float(coin['volume_percentage_change_24h'])
                    # print('volume_percentage_change_24h', volume_percentage_change_24h)
                    price_percentage_change_24h = float(coin['price_percentage_change_24h'])
                    # print('price_percentage_change_24h', price_percentage_change_24h)
                except ValueError as e:
                    print(f"Error: Could not convert volume or price change data for {coin['product_id']}. Error: {e}")
                    continue

                volume_signal = generate_volume_signal(volume_24h, volume_percentage_change_24h, price_percentage_change_24h)
                volatility = volume_volatility_indicator(volume_24h, volume_percentage_change_24h, price_change_percentage_range)
                coinbase_data_signal = 'hold'
                if volume_signal == 1 and volatility < 2:
                    coinbase_data_signal = 'buy'
                elif volume_signal == -1 and volatility > 3:
                    coinbase_data_signal = 'sell'

                # storing in local arrays
                if coin['product_id'] not in COINBASE_DATA_RECOMMENDATIONS:
                    COINBASE_DATA_RECOMMENDATIONS[coin['product_id']] = 0
                if coin['product_id'] not in CMC_DATA_RECOMMENDATIONS:
                    CMC_DATA_RECOMMENDATIONS[coin['product_id']] = 0
                if coin['product_id'] not in TREND_NOTIFICATIONS:
                    TREND_NOTIFICATIONS[coin['product_id']] = 0

                if coinbase_data_signal != 'hold':

                    time.sleep(4) # helps with rate limiting

                    cmc_volume_data = fetch_coinmarketcap_volume_data(coin['product_id'])
                    cmc_data_signal = volume_based_strategy_recommendation(cmc_volume_data) # ('buy', 'sell', 'hold')

                    # tracking changes recommendations
                    cb_string = ''
                    cmc_string = ''
                    if COINBASE_DATA_RECOMMENDATIONS[coin['product_id']] != coinbase_data_signal:
                        cb_string = f"coinbase: {str(COINBASE_DATA_RECOMMENDATIONS[coin['product_id']]).upper()} --> {str(coinbase_data_signal).upper()}"
                        COINBASE_DATA_RECOMMENDATIONS[coin['product_id']] = coinbase_data_signal
                    if CMC_DATA_RECOMMENDATIONS[coin['product_id']] != cmc_data_signal:
                        cmc_string = f"cmc: {str(CMC_DATA_RECOMMENDATIONS[coin['product_id']]).upper()} --> {str(cmc_data_signal).upper()}"
                        CMC_DATA_RECOMMENDATIONS[coin['product_id']] = cmc_data_signal

                    # print(cmc_volume_data)

                    changed_recommendation = False
                    if cb_string != '' or cmc_string != '':
                        changed_recommendation = True

                    if changed_recommendation == True:
                        print(f"{coin['product_id']} ({price_change_percentage_range:.2f}%)")
                        print(f"1h: {cmc_volume_data['quote']['USD']['percent_change_1h']}%")
                        print(f"24h: {coin['price_percentage_change_24h']}%")
                        if cb_string != '':
                            print(cb_string)
                        if cmc_string != '':
                            print(cmc_string)
                        # print('coinbase signal: ', coinbase_data_signal.upper())
                        # print('cmc signal: ', cmc_data_signal.upper())

                    efficiency_ratio, description = calculate_market_cap_efficiency(cmc_volume_data['quote']['USD']['market_cap'], cmc_volume_data['quote']['USD']['fully_diluted_market_cap'])
                    print(coin['product_id'])
                    print(f"Market Cap Efficiency Ratio: {efficiency_ratio:.4f} - {description}")

                    if cmc_volume_data['quote']['USD']['percent_change_1h'] > 1.5:
                        alert_count = 0
                        if TREND_NOTIFICATIONS[coin['product_id']] == 0:
                            alert_count = '1'
                        else:
                            alert_count = '2+'

                        efficiency_ratio, description = calculate_market_cap_efficiency(cmc_volume_data['quote']['USD']['market_cap'], cmc_volume_data['quote']['USD']['fully_diluted_market_cap'])
                        print(coin['product_id'])
                        print(f"Market Cap Efficiency Ratio: {efficiency_ratio:.4f} - {description}")

                        if cmc_volume_data['quote']['USD']['percent_change_1h'] > TREND_NOTIFICATIONS[coin['product_id']]:
                            print(f"({alert_count}) uptrend: {coin['product_id']} - (1h change %: {cmc_volume_data['quote']['USD']['percent_change_1h']})\n")
                            send_email_notification(
                                subject=f"~ uptrend ({alert_count}): {coin['product_id']} - (1h change %: {cmc_volume_data['quote']['USD']['percent_change_1h']}, effeciency_ratio: {efficiency_ratio:.4f} - {description})",
                                text_content=f"uptrend ({alert_count}): {coin['product_id']}",
                                html_content=f"uptrend ({alert_count}): {coin['product_id']}"
                            )
                    TREND_NOTIFICATIONS[coin['product_id']] = cmc_volume_data['quote']['USD']['percent_change_1h']

                    if coinbase_data_signal == cmc_data_signal:
                        if changed_recommendation == True:
                            if coinbase_data_signal == 'buy':
                                print('BUY BUY')

                                efficiency_ratio, description = calculate_market_cap_efficiency(cmc_volume_data['quote']['USD']['market_cap'], cmc_volume_data['quote']['USD']['fully_diluted_market_cap'])
                                print(coin['product_id'])
                                print(f"Market Cap Efficiency Ratio: {efficiency_ratio:.4f} - {description}")

                                send_email_notification(
                                    subject=f"~ buy: {coin['product_id']}, efficiency_ratio: {efficiency_ratio:.4f} - {description})",
                                    text_content=f"Both signals indicated BUY: {coin['product_id']}",
                                    html_content=f"Both signals indicated BUY: {coin['product_id']}"
                                )
                            else:
                                print('SELL SELL')
                                # send_email_notification(
                                #     subject=f"~ buy: {coin['product_id']}",
                                #     text_content=f"Both signals indicated BUY: {coin['product_id']}",
                                #     html_content=f"Both signals indicated BUY: {coin['product_id']}"
                                # )

                    print('\n')

            else:
                print('waiting for more data to do calculations')

        #
        #
        # iterate through config assets
        #

        config = load_config('config.json')

        for asset in config['assets']:
            enabled = asset['enabled']
            symbol = asset['symbol']
            # fees
            stable_pair = asset['stable_pair']
            stable_buy_at_price = asset['stable_buy_at_price']
            stable_sell_at_price = asset['stable_sell_at_price']
            maker_f = coinbase_spot_maker_fee if stable_pair == False else coinbase_stable_pair_spot_maker_fee
            taker_f = coinbase_spot_taker_fee if stable_pair == False else coinbase_stable_pair_spot_taker_fee
            # trading flags
            READY_TO_TRADE = asset['ready_to_trade']
            TARGET_PROFIT_PERCENTAGE = asset['target_profit_percentage']
            BUY_AT_PRICE_POSITION_PERCENTAGE = asset['buy_at_price_position_percentage']
            ALERT_DOWNWARD_DIVERGENCE = asset['alert_downward_divergence']
            BUY_AT_DOWNWARD_DIVERGENCE_COUNT = asset['buy_at_downward_divergence_count']
            SHARES_TO_ACQUIRE = asset['shares_to_acquire']

            # indicators
            SUPPORT_RESISTANCE_WINDOW_SIZE = asset['support_resistance_window_size']
            TREND_1_TIMEFRAME_PERCENT = asset['trend_1_timeframe_percent']
            TREND_1_DISPLAY = asset['trend_1_display']
            TREND_2_TIMEFRAME_PERCENT = asset['trend_2_timeframe_percent']
            TREND_2_DISPLAY = asset['trend_2_display']

            ENABLE_GRAPH_DISPLAY = asset['enable_graph_display']
            ENABLE_GRAPH_SCREENSHOT = asset['enable_graph_screenshot']

            ENABLE_TEST_FAILURE = asset['enable_test_failure']

            PRICE_ALERT_ENABLED = asset['price_alert_enabled']
            PRICE_ALERT_BUY = asset['price_alert_buy']
            PRICE_ALERT_SELL = asset['price_alert_sell']

            if enabled:
                print(symbol)

                #
                #
                # Initialize price data storage if not already done
                #

                if symbol not in APP_START_TIME_DATA:
                    APP_START_TIME_DATA[symbol] = time.time()

                if symbol not in LOCAL_PRICE_DATA:
                    LOCAL_PRICE_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)

                current_price = get_asset_price(coinbase_client, symbol)

                if current_price is not None:
                    LOCAL_PRICE_DATA[symbol].append(current_price)

                if symbol not in LOCAL_TREND_1_DATA:
                    LOCAL_TREND_1_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_TREND_1_DATA[symbol] = TEST_TREND_1_DATA[symbol]

                if symbol not in LOCAL_UPWARD_TREND_DIVERGENCE_DATA:
                    LOCAL_UPWARD_TREND_DIVERGENCE_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_UPWARD_TREND_DIVERGENCE_DATA[symbol] = TEST_UPWARD_TREND_DIVERGENCE_DATA[symbol]

                if symbol not in LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA:
                    LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA[symbol] = TEST_DOWNWARD_TREND_DIVERGENCE_DATA[symbol]

                trend_1 = determine_trend_1(LOCAL_PRICE_DATA[symbol], data_points_for_x_minutes, TREND_1_TIMEFRAME_PERCENT)
                trend_1_offset_price = calculate_offset_price(current_price, trend_1, TREND_1_PRICE_OFFSET_PERCENT)
                LOCAL_TREND_1_DATA[symbol].append(trend_1_offset_price)

                # trend #2

                if symbol not in LOCAL_TREND_2_DATA:
                    LOCAL_TREND_2_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_TREND_2_DATA[symbol] = TEST_TREND_2_DATA[symbol]

                trend_2 = determine_trend_2(LOCAL_PRICE_DATA[symbol], TREND_2_TIMEFRAME_PERCENT)
                trend_2_offset_price = calculate_offset_price(current_price, trend_2, TREND_2_PRICE_OFFSET_PERCENT)
                LOCAL_TREND_2_DATA[symbol].append(trend_2_offset_price)

                # Only proceed if we have enough data
                if len(LOCAL_PRICE_DATA[symbol]) < data_points_for_x_minutes:
                    print(f"Waiting for more data... ({len(LOCAL_PRICE_DATA[symbol])}/{data_points_for_x_minutes})\n")
                    continue

                #
                #
                #
                #

                if PRICE_ALERT_ENABLED:
                    if current_price < PRICE_ALERT_BUY:
                        send_email_notification(
                            subject=f"{symbol} - BUY opportunity",
                            text_content="buy it and wait",
                            html_content="buy it and wait"
                        )
                    elif current_price > PRICE_ALERT_SELL:
                        send_email_notification(
                            subject=f"{symbol} - SELL opportunity",
                            text_content="sell it and take profit",
                            html_content="sell it and take profit"
                        )
                    else:
                        print('waiting for price movement')

                #
                #
                #
                #

                if ENABLE_TEST_FAILURE == True:
                    raise Exception('~~ test failure ~~')

                #
                #
                # Indicators
                #

                print('trend_1: ', trend_1)
                print('trend_2: ', trend_2)

                # divergence visualizations
                upward_divergence = trend_1 == 'upward' and trend_2 == 'bearish'
                if upward_divergence == True:
                    LOCAL_UPWARD_TREND_DIVERGENCE_DATA[symbol].append(current_price)
                    print('divergence: UP')
                downward_divergence = trend_1 == 'downward' and trend_2 == 'bullish'
                if downward_divergence == True:
                    LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA[symbol].append(current_price)
                    print('divergence: DOWN')

                # get trade recommendation based on volume
                if symbol not in VOLUME_BASED_RECOMMENDATIONS:
                    VOLUME_BASED_RECOMMENDATIONS[symbol] = 0

                volume_data = fetch_coinmarketcap_volume_data(symbol)
                # print(volume_data)
                volume_based_strategy = volume_based_strategy_recommendation(volume_data) # ('buy', 'sell', 'hold')
                print('volume_based_strategy: ', volume_based_strategy)
                # detect change in recommendation
                if VOLUME_BASED_RECOMMENDATIONS[symbol] != volume_based_strategy:
                    print(f"strategy change: {str(VOLUME_BASED_RECOMMENDATIONS[symbol]).upper()} --> {str(volume_based_strategy).upper()}")
                    if IS_TEST_MODE == False:
                        send_email_notification(
                            subject=f"{symbol} - strategy change: {str(VOLUME_BASED_RECOMMENDATIONS[symbol]).upper()} --> {str(volume_based_strategy).upper()}",
                            text_content=f"{str(VOLUME_BASED_RECOMMENDATIONS[symbol]).upper()} --> {str(volume_based_strategy).upper()}",
                            html_content="strategy changed"
                        )
                    # overwrite stored swing trade recommendation
                    VOLUME_BASED_RECOMMENDATIONS[symbol] = volume_based_strategy

                # Initialize last calculated support+resistance price if not set
                if symbol not in last_calculated_support_resistance_pivot_prices:
                    last_calculated_support_resistance_pivot_prices[symbol] = current_price

                # Get stored support and resistance levels
                levels = SUPPORT_RESISTANCE_LEVELS.get(symbol, {})

                # Calculate and set levels if empty not set yet
                if levels == {}:
                    pivot, support, resistance = calculate_support_resistance_1(LOCAL_PRICE_DATA[symbol], SUPPORT_RESISTANCE_WINDOW_SIZE)
                    # pivot, support, resistance = calculate_support_resistance_2(LOCAL_PRICE_DATA[symbol])
                    last_calculated_support_resistance_pivot_prices[symbol] = current_price  # Update the last calculated price

                    # Store the calculated support and resistance levels
                    SUPPORT_RESISTANCE_LEVELS[symbol] = {
                        'pivot': pivot,
                        'support': support,
                        'resistance': resistance
                    }

                # print(levels)
                pivot = levels.get('pivot', 0)
                support = levels.get('support', 0)
                resistance = levels.get('resistance', 0)

                # Check if we should recalculate support and resistance levels
                if should_recalculate_support_resistance_1(LOCAL_PRICE_DATA[symbol], last_calculated_support_resistance_pivot_prices[symbol]):
                    # recalculate support
                    pivot, support, resistance = calculate_support_resistance_1(LOCAL_PRICE_DATA[symbol], SUPPORT_RESISTANCE_WINDOW_SIZE)
                    # pivot, support, resistance = calculate_support_resistance_2(LOCAL_PRICE_DATA[symbol])
                    # Update the last calculated price
                    last_calculated_support_resistance_pivot_prices[symbol] = current_price

                    # Update stored support and resistance levels
                    SUPPORT_RESISTANCE_LEVELS[symbol] = {
                        'pivot': pivot,
                        'support': support,
                        'resistance': resistance
                    }
                    print(f"Recalculated support and resistance for {symbol}")


                print(f"current_price: {current_price}")
                print(f"support: {support}")
                print(f"resistance: {resistance}")

                # trade range
                minimum_price_in_chart = min(LOCAL_PRICE_DATA[symbol])
                maximum_price_in_chart = max(LOCAL_PRICE_DATA[symbol])

                trading_range_percentage = calculate_trading_range_percentage(minimum_price_in_chart, maximum_price_in_chart)
                print(f"trading_range_percentage: {trading_range_percentage}%")

                current_price_position_within_trading_range = calculate_current_price_position_within_trading_range(current_price, minimum_price_in_chart, maximum_price_in_chart)
                print(f"current_price_position_within_trading_range: {current_price_position_within_trading_range}%")
                print('buy_at_price_position_percentage: ', BUY_AT_PRICE_POSITION_PERCENTAGE)

                sma = calculate_sma(LOCAL_PRICE_DATA[symbol], period=20)
                # print(f"SMA: {sma}")
                macd_line, signal_line = calculate_macd(LOCAL_PRICE_DATA[symbol])
                # print(f"MACD Line: {macd_line}, Signal Line: {signal_line}")
                upper_bollinger_band, lower_bollinger_band, _ = calculate_bollinger_bands(LOCAL_PRICE_DATA[symbol])
                # print(f"Bollinger Bands - Upper: {upper_bollinger_band}, Lower: {lower_bollinger_band}")

                # Calculate Fibonacci levels
                fibonacci_levels = calculate_fibonacci_levels(LOCAL_PRICE_DATA[symbol])
                print(f"Fibonacci Levels: {fibonacci_levels}")

                #
                #
                # Manage order data (order types, order info, etc.) in local ledger
                #

                entry_price = 0

                last_order = get_last_order_from_local_json_ledger(symbol)
                last_order_type = detect_stored_coinbase_order_type(last_order)

                if READY_TO_TRADE == True:
                    print('ready_to_trade: ', READY_TO_TRADE)

                event_type = 'interval'

                #
                # Handle unverified BUY / SELL order
                if last_order_type == 'placeholder':
                    fulfilled_order_data = get_coinbase_order_by_order_id(coinbase_client, last_order['order_id'])
                    if fulfilled_order_data:
                        full_order_dict = fulfilled_order_data['order'] if isinstance(fulfilled_order_data, dict) else fulfilled_order_data.to_dict()
                        save_order_data_to_local_json_ledger(symbol, full_order_dict)
                        print('Updated ledger with full order data \n')
                    else:
                        print('still waiting to pull full order data info \n')

                #
                # BUY logic
                elif last_order_type == 'none' or last_order_type == 'sell':
                    print('looking to BUY')

                    #
                    # stable pairs
                    #
                    if stable_pair == True:
                        if current_price <= stable_buy_at_price:
                            if READY_TO_TRADE == True:
                                print(symbol)
                                print(SHARES_TO_ACQUIRE)
                                place_market_buy_order(coinbase_client, symbol, SHARES_TO_ACQUIRE)
                                event_type = 'buy'
                            else:
                                print('trading disabled')
                    #
                    # regular asset
                    #
                    else:

                        if float(trading_range_percentage) < float(TARGET_PROFIT_PERCENTAGE):
                            print('trading range smaller than target_profit_percentage')
                            continue

                        #
                        # Strategy #1
                        #
                        if current_price < pivot and current_price < lower_bollinger_band:
                            if current_price < fibonacci_levels['level_61.8']:
                                if current_price_position_within_trading_range < BUY_AT_PRICE_POSITION_PERCENTAGE:
                                    # if downward_divergence == True:
                                    # Determine the lower of the pivot and lower Bollinger band
                                    lower_threshold = min(pivot, lower_bollinger_band)
                                    # Track number of divergences
                                    downward_divergence_below_threshold_count = 0
                                    # Iterate backward through the price data from current price
                                    recent_prices = list(LOCAL_PRICE_DATA[symbol])
                                    for price in reversed(recent_prices):
                                        if price > lower_threshold:
                                            # print(' price broke threshold: ', price)
                                            break
                                        if price in LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA[symbol]:
                                            downward_divergence_below_threshold_count += 1

                                    print(f"downward_divergence_below_threshold_count: {downward_divergence_below_threshold_count}")
                                    print('ALERT_DOWNWARD_DIVERGENCE: ', ALERT_DOWNWARD_DIVERGENCE)
                                    print('BUY_AT_DOWNWARD_DIVERGENCE_COUNT: ', BUY_AT_DOWNWARD_DIVERGENCE_COUNT)
                                    if downward_divergence_below_threshold_count > 2 and ALERT_DOWNWARD_DIVERGENCE == True:
                                        send_email_notification(
                                            subject=f"downward divergence count: {downward_divergence_below_threshold_count}",
                                            text_content=f"downward dv - {downward_divergence_below_threshold_count}",
                                            html_content=f"downward dv - {downward_divergence_below_threshold_count}"
                                        )

                                    if downward_divergence_below_threshold_count >= BUY_AT_DOWNWARD_DIVERGENCE_COUNT:
                                        print('~ BUY OPPORTUNITY (current price < pivot, current_price < lower_bollinger_band, downward divergence, position is good)~')
                                        if READY_TO_TRADE == True:
                                            place_market_buy_order(coinbase_client, symbol, SHARES_TO_ACQUIRE)
                                            event_type = 'buy'
                                        else:
                                            print('trading disabled')


                #
                # SELL logic
                elif last_order_type == 'buy': # and volume_based_strategy == 'sell':
                    print('looking to SELL')

                    entry_price = float(last_order['order']['average_filled_price'])
                    print(f"entry_price: {entry_price}")

                    entry_position_value_after_fees = float(last_order['order']['total_value_after_fees'])
                    print(f"entry_position_value_after_fees: {entry_position_value_after_fees}")

                    number_of_shares = float(last_order['order']['filled_size'])
                    print('number_of_shares: ', number_of_shares)

                    # calculate profits if we were going to sell now
                    pre_tax_profit = (current_price - entry_price) * number_of_shares

                    sell_now_exchange_fee = calculate_exchange_fee(current_price, number_of_shares, taker_f)
                    print(f"sell_now_exchange_fee: {sell_now_exchange_fee}")

                    sell_now_tax_owed = (federal_tax_rate / 100) * pre_tax_profit
                    print(f"sell_now_taxes_owed: {sell_now_tax_owed}")

                    potential_profit = (current_price * number_of_shares) - entry_position_value_after_fees - sell_now_exchange_fee - sell_now_tax_owed
                    print(f"potential_profit_USD: {potential_profit}")

                    potential_profit_percentage = (potential_profit / entry_position_value_after_fees) * 100
                    print(f"potential_profit_percentage: {potential_profit_percentage:.4f}%")

                    #
                    # stable pairs
                    #
                    if stable_pair == True:
                        if current_price >= stable_sell_at_price or potential_profit_percentage >= TARGET_PROFIT_PERCENTAGE:
                            if READY_TO_TRADE == True:
                                place_market_sell_order(coinbase_client, symbol, number_of_shares, potential_profit, potential_profit_percentage)
                                event_type = 'sell'
                            else:
                                print('trading disabled')
                    #
                    # regular asset
                    #
                    else:

                        if potential_profit_percentage >= TARGET_PROFIT_PERCENTAGE:

                            # take profits at x percent
                            print('~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~')
                            if READY_TO_TRADE == True:
                                place_market_sell_order(coinbase_client, symbol, number_of_shares, potential_profit, potential_profit_percentage)
                                event_type = 'sell'
                            else:
                                print('trading disabled')


                            # fib level indicator
                            if fibonacci_levels['level_61.8'] > current_price:
                                print('~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~')
                                if READY_TO_TRADE == True:
                                    place_market_sell_order(coinbase_client, symbol, number_of_shares, potential_profit, potential_profit_percentage)
                                    event_type = 'sell'
                                else:
                                    print('trading disabled')


                current_time = time.time() # call it once instead of twice for the following functions

                # Clear errors if they're non-consecutive
                LAST_EXCEPTION_ERROR = None
                LAST_EXCEPTION_ERROR_COUNT = 0
                print('\n')

        time.sleep(interval_seconds)

if __name__ == "__main__":
    while True:
        try:
            iterate_assets(INTERVAL_MINUTES, INTERVAL_SECONDS, DATA_POINTS_FOR_X_MINUTES)
        except Exception as e:
            current_exception_error = str(e)
            print(f"An error occurred: {current_exception_error}. Restarting the program...")
            if current_exception_error != LAST_EXCEPTION_ERROR:
                send_email_notification(
                    subject="App crashed - restarting - scalp-scripts",
                    text_content=f"An error occurred: {current_exception_error}. Restarting the program...",
                    html_content=f"An error occurred: {current_exception_error}. Restarting the program..."
                )
                LAST_EXCEPTION_ERROR = current_exception_error
            else:
                LAST_EXCEPTION_ERROR_COUNT += 1
                if LAST_EXCEPTION_ERROR_COUNT == MAX_LAST_EXCEPTION_ERROR_COUNT:
                    print(F"Quitting program: {MAX_LAST_EXCEPTION_ERROR_COUNT}+ instances of same error ({current_exception_error})")
                    send_email_notification(
                        subject="Quitting program - scalp-scripts",
                        text_content=f"An error occurred: {current_exception_error}. QUITTING the program...",
                        html_content=f"An error occurred: {current_exception_error}. QUITTING the program..."
                    )
                    quit()
                else:
                    print(f"Same error as last time ({LAST_EXCEPTION_ERROR_COUNT}/{MAX_LAST_EXCEPTION_ERROR_COUNT})")
                    print('\n')

            time.sleep(10)
