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
from utils.file_helpers import save_obj_dict_to_file, count_files_in_directory, delete_files_older_than_x_hours, is_most_recent_file_older_than_x_minutes, append_to_json_array, calculate_price_change, remove_old_entries
from utils.price_helpers import calculate_trading_range_percentage, calculate_current_price_position_within_trading_range, calculate_offset_price, calculate_price_change_percentage
from utils.technical_indicators import calculate_market_cap_efficiency, calculate_fibonacci_levels
from utils.time_helpers import print_local_time
# coinbase api
from utils.coinbase import get_coinbase_client, get_coinbase_order_by_order_id, place_market_buy_order, place_market_sell_order, get_asset_price, calculate_exchange_fee, save_order_data_to_local_json_ledger, get_last_order_from_local_json_ledger, reset_json_ledger_file, detect_stored_coinbase_order_type
coinbase_client = get_coinbase_client()
# custom coinbase listings check
from utils.new_coinbase_listings import check_for_new_coinbase_listings
# plotting data
from utils.matplotlib import plot_graph
from utils.volume_trends import volume_based_strategy_recommendation

# load .env
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
TREND_NOTIFICATIONS = {}
UPTREND_NOTIFICATIONS = {}
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
INTERVAL_SECONDS = 10
INTERVAL_MINUTES = 3
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
federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))

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
        price_changes[timeframe // 60] = round(price_change_percentage, 2)

    return price_changes


def calculate_price_momentum(price_changes):
    """
    Calculate the momentum of price changes to catch market pumps.

    :param price_changes: A dictionary with keys as time intervals (in minutes) and values as price changes.
    :return: A string indicating the momentum.
    """
    # Define weights for each time interval (minutes)
    weights = {5: 0.4, 15: 0.3, 30: 0.2, 60: 0.1}

    # Calculate weighted sum of price changes
    weighted_sum = sum(price_changes[time] * weights[time] for time in price_changes)

    str = 'neutral'
    if weighted_sum > 0:
        str = 'upward'
    elif weighted_sum < 0:
        str = 'downward'

    # Determine the momentum based on the weighted sum
    momentum_info = {
        'weighted_sum': weighted_sum,
        'signal': str,
    }

    return momentum_info




    # Usage example: calculate_coin_price_changes('/path/to/directory', 'BTC-USD')

#
#


def detect_volume_spike(current_volume_24h, volume_change_24h):
    """
    Analyzes the volume data of a given coin to detect potential volume spikes.

    Parameters:
    - coin_data: A dictionary containing volume and price change information for a coin.

    Returns:
    - A dictionary containing information about the volume spike.
    """

    try:
        # Assuming the "volume_change_24h" indicates the percentage change in volume, calculate expected volume
        if volume_change_24h is not None:
            previous_volume_24h = current_volume_24h / (1 + (volume_change_24h / 100))
        else:
            print("Volume change data not available.")
            return None

        # Determine if a spike has occurred by checking if the volume has increased significantly above previous volume
        threshold = 1.5  # Example threshold for what we consider a spike (50% more than the previously expected volume)
        spike_occurred = current_volume_24h > previous_volume_24h * threshold

        spike_info = {
            'current_volume_24h': current_volume_24h,
            'previous_volume_24h': previous_volume_24h,
            'volume_change_24h': volume_change_24h,
            'volume_spike_detected': spike_occurred
        }

        return spike_info

    except KeyError as e:
        print(f"Key error: Missing expected data field - {str(e)}")
        return None



def has_four_hours_passed(start_time):
    """
    Check if one hour has passed since the application started.

    :param start_time: The start time of the application in seconds since the epoch.
    :return: A string indicating whether one hour has passed.
    """
    # Calculate the elapsed time in seconds
    elapsed_time = time.time() - start_time

    # Check if one hour (3600 seconds) has passed
    if elapsed_time >= 14400:
        return True
    else:
        return False

#
#
#
# main logic loop
#

def iterate_assets(interval_minutes, interval_seconds, data_points_for_x_minutes):
    while True:


        if 'start_time' not in APP_START_TIME_DATA:
            APP_START_TIME_DATA['start_time'] = time.time()

        print_local_time();

        #
        # ERROR TRACKING
        global LAST_EXCEPTION_ERROR
        global LAST_EXCEPTION_ERROR_COUNT

        #
        # COINBASE ASSETS
        current_listed_coins = coinbase_client.get_products()['products']
        current_listed_coins_dictionary = {}
        current_listed_coins_dictionary = convert_products_to_dicts(current_listed_coins)

        #
        # ALERT NEW COIN LISTINGS
        enable_new_listings_alert = False
        if enable_new_listings_alert:
            coinbase_listed_coins_path = 'coinbase-listings/listed_coins.json'
            new_coins = check_for_new_coinbase_listings(coinbase_listed_coins_path, current_listed_coins_dictionary)
            if new_coins:
                for coin in new_coins:
                    print(f"NEW LISTING: {coin['product_id']}")
                    send_email_notification(
                        subject=f"New Coinbase listing: {coin['product_id']}",
                        text_content=f"Coinbase just listed {coin['product_id']}",
                        html_content=f"Coinbase just listed {coin['product_id']}"
                    )
                    time.sleep(2)
            save_obj_dict_to_file(coinbase_listed_coins_path, current_listed_coins)

        #
        #
        # Save coinbase assets data
        enable_all_coin_scanning = True
        if enable_all_coin_scanning:
            coinbase_price_history_directory = 'coinbase-data'
            if is_most_recent_file_older_than_x_minutes(coinbase_price_history_directory, minutes=15):
                save_new_coinbase_data(current_listed_coins_dictionary, coinbase_price_history_directory)
            delete_files_older_than_x_hours(coinbase_price_history_directory, hours=5)

            files_in_folder = count_files_in_directory(coinbase_price_history_directory)

            # if has_four_hours_passed(APP_START_TIME_DATA['start_time']) == False:
            #     print('Waiting to collect 4 full hours of data')
            #     continue;

            if files_in_folder < 2:
                print('waiting for more data to do calculations')
            else:
                for coin in current_listed_coins_dictionary:

                    # Define a list of top 20 cryptocurrency product_ids
                    top_20_cryptos = [
                        # 'BTC-USD', 'ETH-USD', 'BNB-USD', 'USDT-USD', 'USDC-USD',
                        # 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'SOL-USD', 'DOT-USD',
                        # 'MATIC-USD', 'LTC-USD', 'SHIB-USD', 'AVAX-USD', 'UNI-USD',
                        # 'WBTC-USD', 'LINK-USD', 'ATOM-USD', 'XMR-USD', 'BCH-USD'
                        'MATIC-USD'
                    ]

                    # Updated logic to check if the coin's product_id is in the top 20 list
                    if coin['product_id'] in top_20_cryptos:

                        # plot_graph(
                        #     True, # enabled
                        #     time.time(),
                        #     coin['product_id'],
                        #     [0, 1, 2, 3, 4],
                        #     0
                        # )

                        # calculate price change over different timeframes minutes (5, 15, 30, 60)
                        price_change_percentages = calculate_coin_price_changes(coinbase_price_history_directory, coin['product_id'])
                        momentum_info = calculate_price_momentum(price_change_percentages)

                        volume_24h = 0
                        volume_percentage_change_24h = 0
                        price_percentage_change_24h = 0
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


                        coin_obj = {
                            'symbol': coin['product_id'],
                            'weighted_sum': round(momentum_info['weighted_sum'], 2),
                            'price_change_intervals': price_change_percentages,
                            # 'spike_detected': spike_detected_info['volume_spike_detected'],
                            # 'volume_%_change_24h': round(spike_detected_info['volume_change_24h'], 2),
                            'price_%_change_24h': round(price_percentage_change_24h, 2),
                            'price': coin['price'],
                            'timestamp': time.time(),
                        }

                        print(coin_obj)

                        append_to_json_array('uptrend-data/data.json', coin_obj)
                        remove_old_entries('uptrend-data/data.json', 6)
                        continue

                        #
                        #
                        #


        #
        #
        # iterate through config assets
        #

        config = load_config('config.json')

        for asset in config['assets']:
            enabled = asset['enabled']
            symbol = asset['symbol']
            # fees
            maker_f = coinbase_spot_maker_fee
            taker_f = coinbase_spot_taker_fee
            # trading flags
            READY_TO_TRADE = asset['ready_to_trade']
            TARGET_PROFIT_PERCENTAGE = asset['target_profit_percentage']
            BUY_AT_PRICE = asset['buy_at_price']
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
                print(' ')
                print(symbol)

                current_price = get_asset_price(coinbase_client, symbol)
                print(f"current_price: {current_price}")


                #
                #
                # Manage order data (order types, order info, etc.) in local ledger

                entry_price = 0

                last_order = get_last_order_from_local_json_ledger(symbol)
                last_order_type = detect_stored_coinbase_order_type(last_order)

                if READY_TO_TRADE == True:
                    print('ready_to_trade: ', READY_TO_TRADE)

                #
                # Handle unverified BUY / SELL order
                if last_order_type == 'placeholder':
                    last_order_id = ''
                    if symbol == 'MATIC-USD':
                        last_order_id = last_order['response']['order_id']
                    else:
                        last_order_id = last_order['order_id']

                    fulfilled_order_data = get_coinbase_order_by_order_id(coinbase_client, last_order_id)
                    print('who')
                    print(fulfilled_order_data);

                    if fulfilled_order_data:
                        full_order_dict = fulfilled_order_data['order'] if isinstance(fulfilled_order_data, dict) else fulfilled_order_data.to_dict()
                        save_order_data_to_local_json_ledger(symbol, full_order_dict)
                        print('Updated ledger with full order data \n')
                    else:
                        print('still waiting to pull full order data info \n')

                #
                #
                # BUY logic
                elif last_order_type == 'none' or last_order_type == 'sell':
                    print('looking to BUY at $', BUY_AT_PRICE)

                    if float(trading_range_percentage) < float(TARGET_PROFIT_PERCENTAGE):
                        print('trading range smaller than target_profit_percentage')
                        continue

                    if current_price < BUY_AT_PRICE:
                        if READY_TO_TRADE == True:
                            place_market_buy_order(coinbase_client, symbol, SHARES_TO_ACQUIRE)

                #
                #
                # SELL logic
                elif last_order_type == 'buy':
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

                    if potential_profit_percentage >= TARGET_PROFIT_PERCENTAGE:
                        print('~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~')
                        if READY_TO_TRADE == True:
                            place_market_sell_order(coinbase_client, symbol, number_of_shares, potential_profit, potential_profit_percentage)
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
