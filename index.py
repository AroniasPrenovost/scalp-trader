# utils
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
import requests # supports CoinMarketCap
# coinbase api
from coinbase.rest import RESTClient
from mailjet_rest import Client

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
# Initialize a dictionary to store support and resistance levels for each asset
#

SUPPORT_RESISTANCE_LEVELS = {}
last_calculated_support_resistance_pivot_prices = {}  # Store the last calculated price for each asset

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

    # Check if we have cached data and if it's still valid (less than 5 minutes old)
    if symbol in CMC_VOLUME_DATA_CACHE and (current_time - CMC_VOLUME_DATA_TIMESTAMP[symbol] < 300):
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


def calculate_trading_range_percentage(num1, num2):
    if num1 == 0 and num2 == 0:
        return "0.00"
    difference = abs(num1 - num2)
    average = (num1 + num2) / 2
    percentage_difference = (difference / average) * 100
    return f"{percentage_difference:.2f}"


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
    upper_band = sma + (num_std_dev * std_dev)
    lower_band = sma - (num_std_dev * std_dev)
    return upper_band, lower_band, sma

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
# Create chart
#

def plot_graph(symbol, price_data, pivot, support, resistance, trading_range_percentage, current_price_position_within_trading_range, entry_price, min_price, max_price):
    if entry_price == 0:
        entry_price = pivot

    plt.figure()
    # price data
    plt.plot(list(price_data), marker='o', label='price')
    # support + resistance levels
    plt.axhline(y=resistance, color='b', linewidth=1.4, linestyle='--', label='resistance')
    plt.axhline(y=support, color='b', linewidth=1.4, linestyle='--', label='support')
    # etc...
    plt.axhline(y=pivot, color='r', linewidth=1, linestyle=':', label='pivot')

    entry_price_width = 1.2
    if entry_price == 0:
        entry_price_width = 0
    plt.axhline(y=entry_price, color='g', linewidth=entry_price_width, linestyle='-', label='entry price')

    plt.axhline(y=min_price, color='y', linewidth=1.2, linestyle='-', label='min price')
    plt.axhline(y=max_price, color='y', linewidth=1.2, linestyle='-', label='max price')

    plt.title(f"{symbol}")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.legend()

    # Set y-axis minimum and maximum to ensure support and resistance are visible
    min_displayed_price = min(min(price_data), support, resistance, entry_price)
    max_displayed_price = max(max(price_data), support, resistance, entry_price)

    # Set y-axis limits with a small buffer
    plt.gca().set_ylim(min_displayed_price - 0.002, max_displayed_price + 0.002)

    # Set x-axis to show time points
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.grid(True)
    plt.figtext(0.5, 0.01, f"trade range %: {trading_range_percentage}, current position %: {current_price_position_within_trading_range}", ha="center", fontsize=8)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

#
#
# main logic loop
#

def iterate_assets(interval_seconds, data_points_for_x_minutes):
    while True:
        client_accounts = client.get_accounts()
        config = load_config('config.json')

        for asset in config['assets']:
            enabled = asset['enabled']
            symbol = asset['symbol']
            SHARES_TO_ACQUIRE = asset['shares_to_acquire']
            TARGET_PROFIT_PERCENTAGE = asset['target_profit_percentage']

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

                current_price = get_asset_price(symbol)
                if current_price is not None:
                    LOCAL_PRICE_DATA[symbol].append(current_price)

                # Only proceed if we have enough data
                if len(LOCAL_PRICE_DATA[symbol]) < data_points_for_x_minutes:
                    print(f"Waiting for more data... ({len(LOCAL_PRICE_DATA[symbol])}/{data_points_for_x_minutes})\n")
                    continue

                #
                #
                # Indicators
                #

                # get trade recommendation based on volume
                volume_data = fetch_coinmarketcap_volume_data(symbol)
                # print(volume_data)
                volume_based_strategy = volume_based_strategy_recommendation(volume_data) # ('buy', 'sell', 'hold')
                print('volume_based_strategy: ', volume_based_strategy)

                # Initialize last calculated support+resistance price if not set
                if symbol not in last_calculated_support_resistance_pivot_prices:
                    last_calculated_support_resistance_pivot_prices[symbol] = current_price

                # Check if we should recalculate support and resistance levels
                if should_recalculate_support_resistance(LOCAL_PRICE_DATA[symbol], last_calculated_support_resistance_pivot_prices[symbol]):
                    last_order = get_last_order_from_local_json_ledger(symbol)
                    last_order_type = detect_stored_coinbase_order_type(last_order)
                    if last_order_type == 'buy':
                        # check if price has gone below support
                        if current_price < last_calculated_support_resistance_pivot_prices[symbol]:
                            # doublecheck volume-based strategy w/ most recent data
                            if symbol in CMC_VOLUME_DATA_CACHE:
                                del CMC_VOLUME_DATA_CACHE[symbol]
                            if symbol in CMC_VOLUME_DATA_TIMESTAMP:
                                del CMC_VOLUME_DATA_TIMESTAMP[symbol]
                            volume_data = fetch_coinmarketcap_volume_data(symbol)
                            volume_based_strategy = volume_based_strategy_recommendation(volume_data)
                            if volume_based_strategy == 'sell':
                                print('~ SELL OPPORTUNITY (cutting losses) ~')
                                shares = last_order['order']['filled_size']
                                place_market_sell_order(symbol, shares)

                    pivot, support, resistance = calculate_support_resistance(LOCAL_PRICE_DATA[symbol])
                    last_calculated_support_resistance_pivot_prices[symbol] = current_price  # Update the last calculated price

                    # Store the calculated support and resistance levels
                    SUPPORT_RESISTANCE_LEVELS[symbol] = {
                        'pivot': pivot,
                        'support': support,
                        'resistance': resistance
                    }
                    print(f"Recalculated support and resistance for {symbol}")

                # Set and retrieve the stored support and resistance levels
                levels = SUPPORT_RESISTANCE_LEVELS.get(symbol, {})
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
                upper_band, lower_band, _ = calculate_bollinger_bands(LOCAL_PRICE_DATA[symbol])
                # print(f"Bollinger Bands - Upper: {upper_band}, Lower: {lower_band}")

                minimum_price_in_chart = min(LOCAL_PRICE_DATA[symbol])
                maximum_price_in_chart = max(LOCAL_PRICE_DATA[symbol])
                #
                #
                # current holdings
                #

                asset_holdings = get_current_asset_holdings(symbol, client_accounts)
                # print('asset_holdings: ', asset_holdings)
                owned_shares = asset_holdings['available_balance'] if asset_holdings else 0
                print('owned_shares: ', owned_shares)

                #
                #
                # Manage order data/types in local ledger
                #

                entry_price = 0

                last_order = get_last_order_from_local_json_ledger(symbol)
                last_order_type = detect_stored_coinbase_order_type(last_order)

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
                    if volume_based_strategy == 'buy':
                        print('looking to BUY')

                        if trading_range_percentage < TARGET_PROFIT_PERCENTAGE:
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

                        # Buy signal: current price crosses above SMA
                        if sma is not None and macd_line is not None and signal_line is not None:
                            if current_price > sma and macd_line > signal_line:
                                print('~ BUY OPPORTUNITY (current_price > sma, MACD crossover)~')
                                place_market_buy_order(symbol, SHARES_TO_ACQUIRE)
                            elif current_price < lower_band:
                                print('~ BUY OPPORTUNITY (price below lower Bollinger Band)~')
                                place_market_buy_order(symbol, SHARES_TO_ACQUIRE)
                            elif current_price_position_within_trading_range < 6:
                                print('~ BUY OPPORTUNITY (current_price_position_within_trading_range < 6)~')
                                place_market_buy_order(symbol, SHARES_TO_ACQUIRE)
                            elif current_price < minimum_price_in_chart:
                                print('~ BUY OPPORTUNITY (current_price < minimum_price_in_chart)~')
                                place_market_buy_order(symbol, SHARES_TO_ACQUIRE)


                #
                # SELL logic
                elif last_order_type == 'buy' and volume_based_strategy == 'sell':
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
                        print('~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~')
                        if current_price >= resistance:
                            print('~ SELL OPPORTUNITY (price near resistance) ~')
                            place_market_sell_order(symbol, number_of_shares)
                        elif sma is not None and current_price < sma:
                            print('~ SELL OPPORTUNITY (price < SMA) ~')
                            place_market_sell_order(symbol, number_of_shares)

                # Indicators are passed into the plot graph
                plot_graph(symbol, LOCAL_PRICE_DATA[symbol], pivot, support, resistance, trading_range_percentage, current_price_position_within_trading_range, entry_price, minimum_price_in_chart, maximum_price_in_chart)
                print('\n')

        time.sleep(interval_seconds)

if __name__ == "__main__":
    while True:
        try:
            # Define time intervals
            INTERVAL_SECONDS = 1
            INTERVAL_MINUTES = 0.25 # 4 hour
            # 1440 # 1 day
            DATA_POINTS_FOR_X_MINUTES = int((60 / INTERVAL_SECONDS) * INTERVAL_MINUTES)
            iterate_assets(INTERVAL_SECONDS, DATA_POINTS_FOR_X_MINUTES)
        except Exception as e:
            print(f"An error occurred: {e}. Restarting the program...")
            # send_email_notification(
            #     subject="App crashed - restarting - scalp-scripts",
            #     text_content=f"An error occurred: {e}. Restarting the program...",
            #     html_content=f"An error occurred: {e}. Restarting the program..."
            # )
            time.sleep(10)  # Wait 10 seconds before restarting
