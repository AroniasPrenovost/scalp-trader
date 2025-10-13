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
from utils.file_helpers import save_obj_dict_to_file, count_files_in_directory, delete_files_older_than_x_hours, is_most_recent_file_older_than_x_minutes, append_to_json_array, calculate_price_change, remove_old_entries, get_property_values_from_files
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
# LLM analysis (OpenAI)
# from utils.llm_analysis import analyze_trading_opportunity, analyze_position_management

# load .env
load_dotenv()

def load_config(file_path):
    with open(file_path, 'r') as file:
        return load(file)


#
#
# Define time intervals
#

INTERVAL_SECONDS = 600 # (10 minutes) # takes into account the 3 (dependent on number of assets) sleep(2)'s (minus 6 seconds)
INTERVAL_SAVE_DATA_MINUTES=30 # how often it saves stock data
DELETE_FILES_OLDER_THAN_X_HOURS=120 # 4 days

#
#
# Store the last error and manage number of errors before killing program
#

LAST_EXCEPTION_ERROR = None
LAST_EXCEPTION_ERROR_COUNT = 0
MAX_LAST_EXCEPTION_ERROR_COUNT = 8


#
#
#

coinbase_spot_maker_fee = float(os.environ.get('COINBASE_SPOT_MAKER_FEE'))
coinbase_spot_taker_fee = float(os.environ.get('COINBASE_SPOT_TAKER_FEE'))
federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))

#
#
#

# Function to save strcutured data to a timestamped file
def save_dictionary_data_to_local_file(data, directory, file_name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_name = f"{file_name}_{timestamp}.json"
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to '{file_name}_{file_path}'")

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
# main logic loop
#

def iterate_assets(interval_seconds):
    while True:
        print_local_time()

        #
        #
        # ERROR TRACKING
        global LAST_EXCEPTION_ERROR
        global LAST_EXCEPTION_ERROR_COUNT

        # load config
        config = load_config('config.json')
        enabled_cryptos = [asset['symbol'] for asset in config['assets'] if asset['enabled']]



        #
        #
        #
        # get data from coinbase
        coinbase_data = coinbase_client.get_products()['products']
        coinbase_data_dictionary = {}
        coinbase_data_dictionary = convert_products_to_dicts(coinbase_data)
        # to reduce file sizes, filter out all crypto data except for those defined in enabled_cryptos
        coinbase_data_dictionary = [coin for coin in coinbase_data_dictionary if coin['product_id'] in enabled_cryptos]


        #
        #
        #
        # ALERT NEW COIN LISTINGS
        enable_new_listings_alert = False
        if enable_new_listings_alert:
            coinbase_listed_coins_path = 'coinbase-listings/listed_coins.json'
            new_coins = check_for_new_coinbase_listings(coinbase_listed_coins_path, coinbase_data_dictionary)
            if new_coins:
                for coin in new_coins:
                    print(f"NEW LISTING: {coin['product_id']}")
                    send_email_notification(
                        subject=f"New Coinbase listing: {coin['product_id']}",
                        text_content=f"Coinbase just listed {coin['product_id']}",
                        html_content=f"Coinbase just listed {coin['product_id']}"
                    )
                    time.sleep(2)
            save_obj_dict_to_file(coinbase_listed_coins_path, coinbase_data)

        #
        #
        # Save coinbase asset data and analyze
        enable_all_coin_scanning = True
        if enable_all_coin_scanning:
            coinbase_price_history_directory = 'coinbase-data'
            if is_most_recent_file_older_than_x_minutes(coinbase_price_history_directory, minutes=INTERVAL_SAVE_DATA_MINUTES):
                save_dictionary_data_to_local_file(coinbase_data_dictionary, coinbase_price_history_directory, 'listed_coins')
            delete_files_older_than_x_hours(coinbase_price_history_directory, hours=DELETE_FILES_OLDER_THAN_X_HOURS)

            if count_files_in_directory(coinbase_price_history_directory) < 1:
                print('waiting for more data to do calculations')
            else:
                for coin in coinbase_data_dictionary:
                    print('\n______________\n')
                    time.sleep(2) # stop system from overheating
                    # print(coin['product_id'])

                    current_price = float(coin['price'])
                    current_price_percentage_change_24h = float(coin['price_percentage_change_24h'])
                    current_volume_24h = float(coin['volume_24h'])
                    current_volume_percentage_change_24h = float(coin['volume_percentage_change_24h'])

                    # Convert each list to a list of floats
                    # append current data to account for the gap in incrementally stored data
                    coin_prices_LIST = get_property_values_from_files(coinbase_price_history_directory, coin['product_id'], 'price')
                    coin_prices_LIST = [float(price) for price in coin_prices_LIST] # Convert to list of floats
                    coin_prices_LIST.append(current_price)

                    coin_price_percentage_change_24h_LIST = get_property_values_from_files(coinbase_price_history_directory, coin['product_id'], 'price_percentage_change_24h')
                    coin_price_percentage_change_24h_LIST = [float(price_percentage_change_24h) for price_percentage_change_24h in coin_price_percentage_change_24h_LIST]
                    coin_price_percentage_change_24h_LIST.append(current_price_percentage_change_24h)

                    coin_volume_24h_LIST = get_property_values_from_files(coinbase_price_history_directory, coin['product_id'], 'volume_24h')
                    coin_volume_24h_LIST = [float(volume_24h) for volume_24h in coin_volume_24h_LIST]
                    coin_volume_24h_LIST.append(current_volume_24h)

                    coin_volume_percentage_change_24h_LIST = get_property_values_from_files(coinbase_price_history_directory, coin['product_id'], 'volume_percentage_change_24h')
                    coin_volume_percentage_change_24h_LIST = [float(volume_percentage_change_24h) for volume_percentage_change_24h in coin_volume_percentage_change_24h_LIST]
                    coin_volume_percentage_change_24h_LIST.append(current_volume_percentage_change_24h)

                    #
                    min_price = min(coin_prices_LIST)
                    max_price = max(coin_prices_LIST)

                    trade_range_percentage = calculate_trading_range_percentage(min_price, max_price)
                    price_position_within_trade_range = calculate_current_price_position_within_trading_range(current_price, min_price, max_price)
                    # print('current price position: ', f"{price_position_within_trade_range}%")

                    # volume_based_strategy = volume_based_strategy_recommendation() (outdated?)

                    coin_obj = {
                        'symbol': coin['product_id'],
                        'price': current_price,
                        'price_percentage_change_24h': current_price_percentage_change_24h,
                        'volume_24h': current_volume_24h,
                        'volume_percentage_change_24h': current_volume_percentage_change_24h,
                        'timestamp': time.time(),
                        'prices_list': coin_prices_LIST,
                        'time_interval_minutes': INTERVAL_SAVE_DATA_MINUTES,
                    }

                    print(coin_obj)

                    # if (price_position_within_trade_range < 4):
                    #
                    #     plot_graph(
                    #         True, # enabled
                    #         time.time(),
                    #         coin['product_id'],
                    #         coin_prices_LIST,
                    #         min_price,
                    #         max_price,
                    #         trade_range_percentage,
                    #         0
                    #     )
                    #
                    #     coin_obj = {
                    #         'symbol': coin['product_id'],
                    #         'price': current_price,
                    #         'price_percentage_change_24h': current_price_percentage_change_24h,
                    #         'volume_24h': current_volume_24h,
                    #         'volume_percentage_change_24h': current_volume_percentage_change_24h,
                    #         'timestamp': time.time(),
                    #     }
                    #
                    #     append_to_json_array('uptrend-data/data.json', coin_obj)
                    #     remove_old_entries('uptrend-data/data.json', 6)
                    #
                    #     price_change_data = calculate_price_change('uptrend-data/data.json', coin['product_id'], current_price)
                    #     time_since_signal = price_change_data[0]
                    #     change = round(price_change_data[1], 2)
                    #
                    #     print(f"time_since: {time_since_signal}   ({change}%)")

                        # exit()

                        # plot_graph(
                        #     True, # enabled
                        #     time.time(),
                        #     coin['product_id'],
                        #     coin_price_percentage_change_24h_LIST,
                        #     0
                        # )

                        # plot_graph(
                        #     True, # enabled
                        #     time.time(),
                        #     coin['product_id'],
                        #     coin_volume_24h_LIST,
                        #     0
                        # )

                        # plot_graph(
                        #     True, # enabled
                        #     time.time(),
                        #     coin['product_id'],
                        #     coin_volume_percentage_change_24h_LIST,
                        #     0
                        # )


                    continue

                    #
                    #
                    #


        #
        #
        # iterate through config assets
        #

        print('Running analysis on assets enabled in "config.json"...')

        for asset in config['assets']:
            if asset['symbol'] == 'SYSTEM':
                continue

            # continue

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

                    # if float(trading_range_percentage) < float(TARGET_PROFIT_PERCENTAGE):
                    #     print('trading range smaller than target_profit_percentage')
                    #     continue

                    # if current_price < BUY_AT_PRICE:
                    if READY_TO_TRADE == True:
                        place_market_buy_order(coinbase_client, symbol, SHARES_TO_ACQUIRE)
                    else:
                        print('not ready to trade')

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

                #
                #
                # Clear errors if they're non-consecutive
                LAST_EXCEPTION_ERROR = None
                LAST_EXCEPTION_ERROR_COUNT = 0
                print('\n')

        #
        #
        # End of iteration function
        time.sleep(interval_seconds)

if __name__ == "__main__":
    while True:
        try:
            iterate_assets(INTERVAL_SECONDS)
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
