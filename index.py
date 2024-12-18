import os
from dotenv import load_dotenv
from json import dumps, load
import json
import math
import time
from pprint import pprint
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import requests # supports CoinMarketCap
# coinbase api
from coinbase.rest import RESTClient
from mailjet_rest import Client
# parse CLI args
import argparse

load_dotenv()

def load_config(file_path):
    with open(file_path, 'r') as file:
        return load(file)


# Define the GRAPH_SCREENSHOT_FOLDER for saving screenshots
GRAPH_SCREENSHOT_FOLDER = 'screenshots'
if not os.path.exists(GRAPH_SCREENSHOT_FOLDER):
    os.makedirs(GRAPH_SCREENSHOT_FOLDER)

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
# Initialize a dictionary to store support and resistance levels for each asset
#

SUPPORT_RESISTANCE_LEVELS = {}
last_calculated_support_resistance_pivot_prices = {}  # Store the last calculated price for each asset

#
#
# Initialize a dictionary to store trend data for each asset
#

LOCAL_TREND_1_DATA = {}
TREND_1_PRICE_OFFSET_PERCENT = 0.05
LOCAL_TREND_2_DATA = {}
TREND_2_PRICE_OFFSET_PERCENT = 0.1
# for mapping the divergent outcomes between these 2 ^
LOCAL_UPWARD_TREND_DIVERGENCE_DATA = {}
LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA = {}

#
#
# Define time intervals
#

# INTERVAL_SECONDS = 1
# INTERVAL_MINUTES = 0.25
INTERVAL_SECONDS = 5
INTERVAL_MINUTES = 180
# INTERVAL_SECONDS = 15
# INTERVAL_MINUTES = 240 # 4 hour

DATA_POINTS_FOR_X_MINUTES = int((60 / INTERVAL_SECONDS) * INTERVAL_MINUTES)

#
#
# Coinbase API and taxes
#

coinbase_api_key = os.environ.get('COINBASE_API_KEY')
coinbase_api_secret = os.environ.get('COINBASE_API_SECRET')

coinbase_spot_maker_fee = float(os.environ.get('COINBASE_SPOT_MAKER_FEE'))
coinbase_spot_taker_fee = float(os.environ.get('COINBASE_SPOT_TAKER_FEE'))
federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))

client = RESTClient(api_key=coinbase_api_key, api_secret=coinbase_api_secret)

#
#
# Get current price
#

def get_asset_price(symbol):
    try:
        product = client.get_product(symbol)
        price = float(product["price"])
        return price
    except Exception as e:
        print(f"Error fetching product price for {symbol}: {e}")
        return None

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

    # Debugging: Print key variables
    # print(f"Current Price: {current_price}")
    # print(f"Recent High: {recent_high}")
    # print(f"Recent Low: {recent_low}")
    # print(f"ATR: {atr.iloc[-1]}")
    # print(f"Trend: {trend[-1]}")
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


def calculate_fibonacci_levels(prices):
    """
    Calculate Fibonacci retracement levels for a given set of stock prices.

    :param prices: deque of stock prices
    :return: dictionary containing Fibonacci levels
    """
    if not prices or len(prices) < 2:
        raise ValueError("Prices deque must contain at least two elements.")

    high = max(prices)
    low = min(prices)

    # Calculate Fibonacci levels
    diff = high - low
    levels = {
        'level_0': high,
        'level_23.6': high - 0.236 * diff,
        'level_38.2': high - 0.382 * diff,
        'level_50': high - 0.5 * diff,
        'level_61.8': high - 0.618 * diff,
        'level_100': low
    }

    return levels

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
# Used for visualizing on the chart
#

def calculate_offset_price(price, trend, percentage):
    """
    Calculate the offset price based on the trend and percentage.

    :param price: The original price.
    :param trend: The trend direction ('upward', 'downward', 'bullish', 'bearish').
    :param percentage: The percentage to offset the price.
    :return: The offset price.
    """
    if trend in ['upward', 'bullish']:
        price_trend_offset = price * (percentage / 100)
        return price + price_trend_offset
    elif trend in ['downward', 'bearish']:
        price_trend_offset = price * (-percentage / 100)
        return price + price_trend_offset
    return price

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

            start_price = get_asset_price(symbol)

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
            TEST_PRICE_DATA = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)
            TEST_TREND_1_DATA = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)
            TEST_TREND_2_DATA = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)
            TEST_UPWARD_TREND_DIVERGENCE_DATA = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)
            TEST_DOWNWARD_TREND_DIVERGENCE_DATA = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)

            raw_test_data = generate_test_price_data(start_price, DATA_POINTS_FOR_X_MINUTES, TEST_DATA_TREND_RATE, TEST_DATA_VOLATILITY_RATE)

            # generate indicator visualizations
            for price in raw_test_data:
                TEST_PRICE_DATA.append(price)
                #
                #
                # TREND #1
                #
                trend_1 = determine_trend_1(TEST_PRICE_DATA, DATA_POINTS_FOR_X_MINUTES, TREND_1_TIMEFRAME_PERCENT)
                trend_1_offset_price = calculate_offset_price(price, trend_1, TREND_1_PRICE_OFFSET_PERCENT)
                TEST_TREND_1_DATA.append(trend_1_offset_price)
                #
                #
                # TREND #2›
                #
                trend_2 = determine_trend_2(TEST_PRICE_DATA, 20)
                trend_2_offset_price = calculate_offset_price(price, trend_2, TREND_2_PRICE_OFFSET_PERCENT)
                TEST_TREND_2_DATA.append(trend_2_offset_price)
                #
                #
                # visualize indicator divergences
                #
                upward_divergence = trend_1 == 'upward' and trend_2 == 'bearish'
                if upward_divergence == True:
                    TEST_UPWARD_TREND_DIVERGENCE_DATA.append(price)
                downward_divergence = trend_1 == 'downward' and trend_2 == 'bullish'
                if downward_divergence == True:
                    TEST_DOWNWARD_TREND_DIVERGENCE_DATA.append(price)

            print('upward trend divergence(s): ', f"{len(TEST_UPWARD_TREND_DIVERGENCE_DATA)}/{len(TEST_PRICE_DATA)}")
            print('downward trend divergence(s): ', f"{len(TEST_DOWNWARD_TREND_DIVERGENCE_DATA)}/{len(TEST_PRICE_DATA)}")
else:
    print(f"Running in {mode} mode")

#
#
# Mailjet configuration
#

mailjet_api_key = os.environ.get('MAILJET_API_KEY')
mailjet_secret_key = os.environ.get('MAILJET_SECRET_KEY')
mailjet_from_email = os.environ.get('MAILJET_FROM_EMAIL')
mailjet_from_name = os.environ.get('MAILJET_FROM_NAME')
mailjet_to_email = os.environ.get('MAILJET_TO_EMAIL')
mailjet_to_name = os.environ.get('MAILJET_TO_NAME')

mailjet = Client(auth=(mailjet_api_key, mailjet_secret_key), version='v3.1')

def send_email_notification(subject, text_content, html_content):
    data = {
        'Messages': [
            {
                "From": {
                    "Email": mailjet_from_email,
                    "Name": mailjet_from_name
                },
                "To": [
                    {
                        "Email": mailjet_to_email,
                        "Name": mailjet_to_name
                    }
                ],
                "Subject": subject,
                "TextPart": text_content,
                "HTMLPart": html_content
            }
        ]
    }
    result = mailjet.send.create(data=data)
    if result.status_code == 200:
        print("Email sent successfully.")
    else:
        print(f"Failed to send email. Status code: {result.status_code}, Error: {result.json()}")


# send_email_notification(
#     subject="Test email subject",
#     text_content="testing text content",
#     html_content="testing content"
# )

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
# Get your current position
#

def get_current_asset_holdings(symbol, accounts):
    try:
        modified_symbol = symbol.split('-')[0] # DOUBLECHECK THIS WORKS FOR ALL ACCOUNTS

        for account in accounts['accounts']:
            if account['currency'] == modified_symbol:
                balance = account['balance']
                available_balance = float((account['available_balance']['value'])) # note this could be a number like 0.0000000564
                # print(account)

                return {
                    'currency': modified_symbol,
                    'balance': balance,
                    'available_balance': available_balance
                }

        print(f"No holdings found for asset: {symbol}.")
        return None
    except Exception as e:
        print(f"Error fetching position for asset {symbol}: {e}")
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

def get_coinbase_order_by_order_id(order_id):
    try:
        order = client.get_order(order_id=order_id)
        if order:
            return order
        else:
            print(f"No order found with ID: {order_id}.")
            return None
    except Exception as e:
        print(f"Error fetching order with ID {order_id}: {e}")
        return None

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
# Create a market BUY order
#

def place_market_buy_order(symbol, base_size):
    try:
        order = client.market_order_buy(
            client_order_id=generate_client_order_id(symbol, 'buy'), # id must be unique
            product_id=symbol,
            base_size=str(base_size)  # Convert base_size to string
        )

        if 'order_id' in order['response']:
            order_id = order['response']['order_id']
            print(f"BUY ORDER placed successfully. Order ID: {order_id}")

            # Convert the order object to a dictionary if necessary
            # Save the placeholder order data until we can lookup the completeed transaction
            order_data = order.response if hasattr(order, 'response') else order
            save_order_data_to_local_json_ledger(symbol, order_data)

            send_email_notification(
                subject="Buy Order Placed",
                text_content=f"BUY ORDER placed successfully for {symbol}. Order ID: {order_id}",
                html_content=f"<h3>BUY ORDER placed successfully for {symbol}. Order ID: {order_id}</h3>"
            )
        else:
            print(f"Unexpected response: {dumps(order)}")
    except Exception as e:
        print(f"Error placing BUY order for {symbol}: {e}")

#
#
# Create a market SELL order
#

def place_market_sell_order(symbol, base_size):
    try:
        order = client.market_order_sell(
            client_order_id=generate_client_order_id(symbol, 'sell'), # id must be unique
            product_id=symbol,
            base_size=str(base_size)  # Convert base_size to string
        )

        if 'order_id' in order['response']:
            order_id = order['response']['order_id']
            print(f"SELL ORDER placed successfully. Order ID: {order_id}")

            # Clear out existing ledger since there is no need to wait and confirm a sell transaction as long as we got programmatic confirmation
            reset_json_ledger_file(symbol)

            send_email_notification(
                subject="Sell Order Placed",
                text_content=f"SELL ORDER placed successfully for {symbol}. Order ID: {order_id}",
                html_content=f"<h3>SELL ORDER placed successfully for {symbol}. Order ID: {order_id}</h3>"
            )
        else:
            print(f"Unexpected response: {dumps(order)}")
    except Exception as e:
        print(f"Error placing SELL order for {symbol}: {e}")

#
#
# Generate (arbitrary) custom order id
#

def generate_client_order_id(symbol, action):
    symbol = symbol.lower()
    action = action.lower()
    custom_id = f"{symbol}_{action}_{time.time()}"
    return custom_id

#
#
# Trade info utils
#

def calculate_exchange_fee(price, number_of_shares, fee_type):
    fee_rate = coinbase_spot_maker_fee if fee_type.lower() == 'maker' else coinbase_spot_taker_fee
    fee = (fee_rate / 100) * price * number_of_shares
    return fee

def calculate_trade_profit(entry_price, sell_price, number_of_shares, fee_type):
    profit = (sell_price - entry_price) * number_of_shares
    exchange_fee = calculate_exchange_fee(sell_price, number_of_shares, fee_type)
    tax_owed = (federal_tax_rate / 100) * profit
    gross_profit = profit - exchange_fee - tax_owed
    investment = entry_price * number_of_shares
    gross_profit_percentage = (gross_profit / investment) * 100

    return {
        'sellPrice': sell_price,
        'profit': profit,
        'exchange_fee': exchange_fee,
        'tax_owed': tax_owed,
        'gross_profit': gross_profit,
        'gross_profit_percentage': gross_profit_percentage
    }

def calculate_transaction_cost(entry_price, number_of_shares, fee_type):
    if number_of_shares == 0:
        return 0
    base_cost = entry_price * number_of_shares
    exchange_fee = calculate_exchange_fee(entry_price, number_of_shares, fee_type)
    cost = base_cost + exchange_fee
    return cost

#
#
# Determine support and resistance levels
#

def should_recalculate_support_resistance(prices, last_calculated_price, price_change_threshold=1.0):
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
# Determine support and resistance levels
#

def calculate_support_resistance(prices):
    """
    Calculate support and resistance levels using pivot points for a given set of stock prices.

    :param prices: deque of stock prices
    :return: tuple containing pivot, support, and resistance levels
    """
    if not prices or len(prices) < 3:
        raise ValueError("Prices deque must contain at least three elements.")

    # Calculate high, low, and close prices
    high = max(prices)
    low = min(prices)
    close = prices[-1]  # Assuming the last price is the closing price

    # Calculate pivot point
    pivot = (high + low + close) / 3

    # Calculate support and resistance levels
    resistance = (2 * pivot) - low
    support = (2 * pivot) - high

    return pivot, support, resistance

#
#
#
#

def calculate_trading_range_percentage(num1, num2):
    if num1 == 0 and num2 == 0:
        return "0.00"
    difference = abs(num1 - num2)
    average = (num1 + num2) / 2
    percentage_difference = (difference / average) * 100
    return f"{percentage_difference:.2f}"

#
#
#
#

def calculate_current_price_position_within_trading_range(current_price, support, resistance):
    """
    Calculate the position of the current price within the trading range.

    :param current_price: The current price of the asset
    :param support: The support level price
    :param resistance: The resistance level price
    :return: The position of the current price within the trading range as a percentage
    """
    if resistance == support:
        return 0.0  # Avoid division by zero

    trading_range = resistance - support
    position_within_range = ((current_price - support) / trading_range) * 100

    return round(position_within_range, 2)


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



# Looks for trade recommendations based on volume

# def get_volume_based_recommendation_for_tradeable_assets(file_path):
#     """
#     Reads the tradeable assets from a JSON file and processes each asset with a 1-second delay.
#     """
#     LATEST_PRICE_API_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
#     headers = {
#         'X-CMC_PRO_API_KEY': coinmarketcap_api_key,
#         'Accept': 'application/json',
#     }
#     try:
#         with open(file_path, 'r') as file:
#             assets = json.load(file)
#
#         for asset in assets:
#             currency = asset.get('currency')
#             if currency:
#                 print(f"Processing asset: {currency}")
#                 params = {
#                     'symbol': currency,
#                 }
#                 # Here you can add the logic to process each asset
#                 try:
#                     response = requests.get(LATEST_PRICE_API_URL, headers=headers, params=params)
#                     response.raise_for_status()  # Raise an exception for HTTP errors
#                     data = response.json()
#                     d = volume_based_strategy_recommendation(data['data'][currency])
#                     print(f"recommendation for {currency}", d)
#                     # CMC_VOLUME_DATA_CACHE[symbol] = data['data'][CMC_SYMBOL]
#                     # CMC_VOLUME_DATA_TIMESTAMP[symbol] = current_time
#                     # return CMC_VOLUME_DATA_CACHE[symbol]
#                 except requests.exceptions.RequestException as e:
#                     print(f"Error fetching latest data for {currency}: {e}")
#                     return None
#
#                 # Wait for 1 second before processing the next asset
#                 time.sleep(1)
#             else:
#                 print("Invalid asset data found, skipping...")
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#     except json.JSONDecodeError:
#         print(f"Error decoding JSON from file: {file_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")


# Example usage
# get_volume_based_recommendation_for_tradeable_assets('tradeable-coinbase-assets.json')
# quit();



#
#
# Create chart
#

def plot_graph(
    enable_display, enable_screenshot,
    timeframe_minutes, symbol, price_data,
    pivot, support, resistance,
    trading_range_percentage, current_price_position_within_trading_range, entry_price, min_price, max_price,
    trend_1_data, trend_1_display,
    trend_2_data, trend_2_display,
    up_diverg, down_diverg,
    lower_bollinger_band, upper_bollinger_band
):
    if enable_display == False and enable_screenshot == False:
        return

    # init graph
    plt.figure(figsize=(12, 8))  # Set the figure size to 12x8 inches

    # entry price (if it exists)
    if entry_price > 0:
        plt.axhline(y=entry_price, color='m', linewidth=1.2, linestyle='-', label='entry price')

    # price data markers
    plt.plot(list(price_data), marker=',', label='price', c='black')

    # trend 1 data markers
    if trend_1_display == True:
        plt.plot(list(trend_1_data), marker=',', label='trend 1 (+/-)', c='orange', linewidth=0.5)

    # trend 2 data markers
    if trend_2_display == True:
        plt.plot(list(trend_2_data), marker=',', label='trend 2 (+/-)', c='blue', linewidth=0.5)

    # Plot upward divergence markers
    up_diverg_indices = [i for i, x in enumerate(price_data) if x in up_diverg]
    plt.scatter(up_diverg_indices, [price_data[i] for i in up_diverg_indices], color='cyan', label='up divergence', marker=2)

    # Plot downward divergence markers
    down_diverg_indices = [i for i, x in enumerate(price_data) if x in down_diverg]
    plt.scatter(down_diverg_indices, [price_data[i] for i in down_diverg_indices], color='red', label='down divergence', marker=3)

    # support, resistance, pivot levels
    plt.axhline(y=resistance, color='black', linewidth=1.4, linestyle='--', label='resistance')
    plt.axhline(y=support, color='black', linewidth=1.4, linestyle='--', label='support')
    plt.axhline(y=pivot, color='magenta', linewidth=1.3, linestyle=':', label='pivot')

    plt.axhline(y=min_price, color='black', linewidth=1.6, linestyle=':', label=f"min price ({min_price:.4f})")
    plt.axhline(y=max_price, color='black', linewidth=1.6, linestyle=':', label=f"max price ({max_price:.4f})")

    # bollinger bands
    plt.axhline(y=lower_bollinger_band, color='cyan', linewidth=1.4, linestyle=':', label=f"low bollinger ({lower_bollinger_band:.4f})")

    plt.title(f"{symbol}")
    plt.xlabel(f"time range ({timeframe_minutes} minutes)")
    plt.ylabel("price")
    plt.legend(loc='lower left', fontsize='small')  # Make the legend smaller

    # Set y-axis minimum and maximum to ensure support and resistance are visible
    min_displayed_price = min(min(price_data), support, resistance)
    max_displayed_price = max(max(price_data), support, resistance)

    # Calculate a dynamic buffer based on the price range
    price_range = max_displayed_price - min_displayed_price
    buffer = price_range * 0.05  # 2% buffer

    # Set y-axis limits with the dynamic buffer
    plt.gca().set_ylim(min_displayed_price - buffer, max_displayed_price + buffer)

    # Set x-axis to show time points
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Format y-axis to show values to the 4th decimal place
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

    plt.grid(True)
    plt.figtext(0.5, 0.01, f"trade range %: {trading_range_percentage}, current position %: {current_price_position_within_trading_range}", ha="center", fontsize=8)

    #
    #
    if enable_screenshot == True:
        filename = os.path.join(GRAPH_SCREENSHOT_FOLDER, f"{symbol}_chart.png")
        if os.path.exists(filename):
            os.remove(filename) # Overwrite existing screenshot and save new one
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved as {filename}")
    elif enable_display == True:
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

#
#
# main logic loop
#

def iterate_assets(interval_minutes, interval_seconds, data_points_for_x_minutes):
    while True:
        client_accounts = client.get_accounts()
        config = load_config('config.json')

        for asset in config['assets']:
            enabled = asset['enabled']
            symbol = asset['symbol']
            SHARES_TO_ACQUIRE = asset['shares_to_acquire']
            TARGET_PROFIT_PERCENTAGE = asset['target_profit_percentage']
            TREND_1_TIMEFRAME_PERCENT = asset['trend_1_timeframe_percent']
            TREND_1_DISPLAY = asset['trend_1_display']
            TREND_2_TIMEFRAME_PERCENT = asset['trend_2_timeframe_percent']
            TREND_2_DISPLAY = asset['trend_2_display']
            READY_TO_TRADE = asset['ready_to_trade']
            ENABLE_GRAPH_DISPLAY = asset['enable_graph_display']
            ENABLE_GRAPH_SCREENSHOT = asset['enable_graph_screenshot']

            if enabled:
                print(symbol)

                # if LOCAL_PRICE_DATA and LOCAL_PRICE_DATA[symbol]:
                #     print(LOCAL_PRICE_DATA[symbol])

                #
                #
                # Initialize price data storage if not already done
                #

                if symbol not in LOCAL_PRICE_DATA:
                    LOCAL_PRICE_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_PRICE_DATA[symbol] = TEST_PRICE_DATA

                current_price = get_asset_price(symbol)

                if current_price is not None:
                    LOCAL_PRICE_DATA[symbol].append(current_price)

                if symbol not in LOCAL_TREND_1_DATA:
                    LOCAL_TREND_1_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_TREND_1_DATA[symbol] = TEST_TREND_1_DATA

                if symbol not in LOCAL_UPWARD_TREND_DIVERGENCE_DATA:
                    LOCAL_UPWARD_TREND_DIVERGENCE_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_UPWARD_TREND_DIVERGENCE_DATA[symbol] = TEST_UPWARD_TREND_DIVERGENCE_DATA

                if symbol not in LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA:
                    LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA[symbol] = TEST_DOWNWARD_TREND_DIVERGENCE_DATA

                trend_1 = determine_trend_1(LOCAL_PRICE_DATA[symbol], data_points_for_x_minutes, TREND_1_TIMEFRAME_PERCENT)
                trend_1_offset_price = calculate_offset_price(current_price, trend_1, TREND_1_PRICE_OFFSET_PERCENT)
                LOCAL_TREND_1_DATA[symbol].append(trend_1_offset_price)

                # trend #2

                if symbol not in LOCAL_TREND_2_DATA:
                    LOCAL_TREND_2_DATA[symbol] = deque(maxlen=data_points_for_x_minutes)
                    if IS_TEST_MODE == True:
                        LOCAL_TREND_2_DATA[symbol] = TEST_TREND_2_DATA

                trend_2 = determine_trend_2(LOCAL_PRICE_DATA[symbol], TREND_2_TIMEFRAME_PERCENT)
                trend_2_offset_price = calculate_offset_price(current_price, trend_2, TREND_2_PRICE_OFFSET_PERCENT)
                LOCAL_TREND_2_DATA[symbol].append(trend_2_offset_price)

                # Only proceed if we have enough data
                if len(LOCAL_PRICE_DATA[symbol]) < data_points_for_x_minutes:
                    print(f"Waiting for more data... ({len(LOCAL_PRICE_DATA[symbol])}/{data_points_for_x_minutes})\n")
                    continue

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
                downward_divergence = trend_1 == 'downward' and trend_2 == 'bullish'
                if downward_divergence == True:
                    LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA[symbol].append(current_price)

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
                            subject=f"strategy change: {str(VOLUME_BASED_RECOMMENDATIONS[symbol]).upper()} --> {str(volume_based_strategy).upper()}",
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
                    pivot, support, resistance = calculate_support_resistance(LOCAL_PRICE_DATA[symbol])
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
                if should_recalculate_support_resistance(LOCAL_PRICE_DATA[symbol], last_calculated_support_resistance_pivot_prices[symbol]):
                    last_order = get_last_order_from_local_json_ledger(symbol)
                    last_order_type = detect_stored_coinbase_order_type(last_order)
                    # SELL-OFF
                    if last_order_type == 'buy':
                        if trend_1 == 'downward':
                            print('~ SELL OPPORTUNITY (downward trend detected) ~')
                            shares = last_order['order']['filled_size']
                            place_market_sell_order(symbol, shares)

                    # recalculate support
                    pivot, support, resistance = calculate_support_resistance(LOCAL_PRICE_DATA[symbol])
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

                trading_range_percentage = calculate_trading_range_percentage(min(LOCAL_PRICE_DATA[symbol]), max(LOCAL_PRICE_DATA[symbol]))
                print(f"trading_range_percentage: {trading_range_percentage}%")

                current_price_position_within_trading_range = calculate_current_price_position_within_trading_range(current_price, support, resistance)
                print(f"current_price_position_within_trading_range: {current_price_position_within_trading_range}%")

                sma = calculate_sma(LOCAL_PRICE_DATA[symbol], period=20)
                # print(f"SMA: {sma}")
                macd_line, signal_line = calculate_macd(LOCAL_PRICE_DATA[symbol])
                # print(f"MACD Line: {macd_line}, Signal Line: {signal_line}")
                upper_bollinger_band, lower_bollinger_band, _ = calculate_bollinger_bands(LOCAL_PRICE_DATA[symbol])
                # print(f"Bollinger Bands - Upper: {upper_bollinger_band}, Lower: {lower_bollinger_band}")

                minimum_price_in_chart = min(LOCAL_PRICE_DATA[symbol])
                maximum_price_in_chart = max(LOCAL_PRICE_DATA[symbol])

                # Calculate Fibonacci levels
                fibonacci_levels = calculate_fibonacci_levels(LOCAL_PRICE_DATA[symbol])
                print(f"Fibonacci Levels: {fibonacci_levels}")

                #
                #
                # current holdings
                #

                asset_holdings = get_current_asset_holdings(symbol, client_accounts)
                # print('asset_holdings: ', asset_holdings)
                owned_shares = asset_holdings['available_balance'] if asset_holdings else 0
                if owned_shares > 0:
                    print('owned_shares: ', owned_shares)

                #
                #
                # Manage order data/types in local ledger
                #

                entry_price = 0

                last_order = get_last_order_from_local_json_ledger(symbol)
                last_order_type = detect_stored_coinbase_order_type(last_order)

                if READY_TO_TRADE == False:
                    print('ready_to_trade: ', READY_TO_TRADE)

                #
                # Handle unverified BUY / SELL order
                if last_order_type == 'placeholder':
                    fulfilled_order_data = get_coinbase_order_by_order_id(last_order['order_id'])
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

                    if float(trading_range_percentage) < float(TARGET_PROFIT_PERCENTAGE):
                        print('trading range smaller than target_profit_percentage')
                        continue

                    # Calculate a buffer zone below the resistance
                    buffer_zone = (resistance - support) * 0.04  # 5% below resistance
                    anticipated_sell_price = resistance - buffer_zone

                    # Calculate expected profit and profit percentage
                    expected_profit = (anticipated_sell_price - current_price) * SHARES_TO_ACQUIRE
                    exchange_fee = calculate_exchange_fee(anticipated_sell_price, SHARES_TO_ACQUIRE, 'taker')
                    tax_owed = (federal_tax_rate / 100) * expected_profit
                    print(f"anticipated_tax_owed: {tax_owed}")

                    net_expected_profit = expected_profit - exchange_fee - tax_owed

                    expected_profit_percentage = (net_expected_profit / SHARES_TO_ACQUIRE) * 100

                    print(f"anticipated_sell_price: {anticipated_sell_price}")
                    print(f"expected_profit: {expected_profit}")
                    print(f"net_expected_profit: {net_expected_profit}")
                    print(f"expected_profit_percentage: {expected_profit_percentage:.2f}%")

                    # if trend_1 == 'upward' and current_price < pivot: # volume_based_strategy == 'buy':
                    #     print('~ BUY OPPORTUNITY (trend_1 == upward and current_price < pivot)~')
                    #     place_market_buy_order(symbol, SHARES_TO_ACQUIRE)
                    if current_price < pivot:
                        if current_price < lower_bollinger_band:
                            if current_price_position_within_trading_range < 50:
                            # if upward_divergence == True: # (need to add this)
                                print('~ BUY OPPORTUNITY (current_price < lower_bollinger_band)~')
                                if READY_TO_TRADE == True:
                                    place_market_buy_order(symbol, SHARES_TO_ACQUIRE)
                                else:
                                    print('trading disabled')


                        # Buy looking to current price crosses above SMA
                        # if sma is not None and macd_line is not None and signal_line is not None:
                        #     if current_price > sma and macd_line > signal_line:
                        #         print('~ BUY OPPORTUNITY (current_price > sma, MACD crossover)~')
                        #         place_market_buy_order(symbol, SHARES_TO_ACQUIRE)
                            # elif current_price < lower_bollinger_band:
                            #     print('~ BUY OPPORTUNITY (price below lower Bollinger Band)~')
                            #     place_market_buy_order(symbol, SHARES_TO_ACQUIRE)
                            # elif current_price_position_within_trading_range < 6:
                            #     print('~ BUY OPPORTUNITY (current_price_position_within_trading_range < 6)~')
                            #     place_market_buy_order(symbol, SHARES_TO_ACQUIRE)


                #
                # SELL logic
                elif last_order_type == 'buy': # and volume_based_strategy == 'sell':
                    print('looking to SELL')

                    if owned_shares == 0:
                        print('something went wrong with local buy/sell order data')
                        continue

                    entry_price = float(last_order['order']['average_filled_price'])
                    print(f"entry_price: {entry_price}")

                    entry_position_value_after_fees = float(last_order['order']['total_value_after_fees'])
                    print(f"entry_position_value_after_fees: {entry_position_value_after_fees}")

                    number_of_shares = float(last_order['order']['filled_size'])
                    print('number_of_shares: ', number_of_shares)

                    # calculate profits if we were going to sell now
                    pre_tax_profit = (current_price - entry_price) * number_of_shares

                    sell_now_exchange_fee = calculate_exchange_fee(current_price, number_of_shares, 'taker')
                    print(f"sell_now_exchange_fee: {sell_now_exchange_fee}")

                    sell_now_tax_owed = (federal_tax_rate / 100) * pre_tax_profit
                    print(f"sell_now_taxes_owed: {sell_now_tax_owed}")

                    potential_profit = (current_price * number_of_shares) - entry_position_value_after_fees - sell_now_exchange_fee - sell_now_tax_owed
                    print(f"potential_profit_USD: {potential_profit}")

                    potential_profit_percentage = (potential_profit / entry_position_value_after_fees) * 100
                    print(f"potential_profit_percentage: {potential_profit_percentage:.2f}%")

                    if potential_profit_percentage >= TARGET_PROFIT_PERCENTAGE:
                        if current_price >= resistance:
                            print('~ SELL OPPORTUNITY (price near resistance) ~')
                            if READY_TO_TRADE == True:
                                place_market_sell_order(symbol, number_of_shares)
                            else:
                                print('trading disabled')
                        elif sma is not None and current_price < sma:
                            print('~ SELL OPPORTUNITY (price < SMA) ~')
                            if READY_TO_TRADE == True:
                                place_market_sell_order(symbol, number_of_shares)
                            else:
                                print('trading disabled')
                        else:
                            print('~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~')
                            if READY_TO_TRADE == True:
                                place_market_sell_order(symbol, number_of_shares)
                            else:
                                print('trading disabled')

                # Indicators are passed into the plot graph
                plot_graph(
                    ENABLE_GRAPH_DISPLAY,
                    ENABLE_GRAPH_SCREENSHOT,
                    interval_minutes,
                    symbol,
                    LOCAL_PRICE_DATA[symbol],
                    pivot,
                    support, resistance,
                    trading_range_percentage,
                    current_price_position_within_trading_range,
                    entry_price,
                    minimum_price_in_chart,
                    maximum_price_in_chart,
                    LOCAL_TREND_1_DATA[symbol],
                    TREND_1_DISPLAY,
                    LOCAL_TREND_2_DATA[symbol],
                    TREND_2_DISPLAY,
                    LOCAL_UPWARD_TREND_DIVERGENCE_DATA[symbol],
                    LOCAL_DOWNWARD_TREND_DIVERGENCE_DATA[symbol],
                    lower_bollinger_band,
                    upper_bollinger_band
                )

                print('\n')

        time.sleep(interval_seconds)

if __name__ == "__main__":
    while True:
        try:
            iterate_assets(INTERVAL_MINUTES, INTERVAL_SECONDS, DATA_POINTS_FOR_X_MINUTES)
        except Exception as e:
            print(f"An error occurred: {e}. Restarting the program...")
            send_email_notification(
                subject="App crashed - restarting - scalp-scripts",
                text_content=f"An error occurred: {e}. Restarting the program...",
                html_content=f"An error occurred: {e}. Restarting the program..."
            )
            time.sleep(10)  # Wait 10 seconds before restarting
