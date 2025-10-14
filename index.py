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

#
# custom imports
from utils.email import send_email_notification
from utils.file_helpers import save_obj_dict_to_file, count_files_in_directory, append_crypto_data_to_file, get_property_values_from_crypto_file, cleanup_old_crypto_data
from utils.price_helpers import calculate_trading_range_percentage, calculate_current_price_position_within_trading_range, calculate_offset_price, calculate_price_change_percentage
from utils.technical_indicators import calculate_market_cap_efficiency, calculate_fibonacci_levels
from utils.time_helpers import print_local_time
# Coinbase helpers and define client
from utils.coinbase import get_coinbase_client, get_coinbase_order_by_order_id, place_market_buy_order, place_market_sell_order, get_asset_price, calculate_exchange_fee, save_order_data_to_local_json_ledger, get_last_order_from_local_json_ledger, reset_json_ledger_file, detect_stored_coinbase_order_type, save_transaction_record
coinbase_client = get_coinbase_client()
# custom coinbase listings check
from utils.new_coinbase_listings import check_for_new_coinbase_listings
# plotting data
from utils.matplotlib import plot_graph
# openai analysis
from utils.openai_analysis import analyze_market_with_openai, save_analysis_to_file, load_analysis_from_file, should_refresh_analysis, delete_analysis_file

#
# end imports
#

#
#
# load .env file
load_dotenv()
#
#
# load config.json file
def load_config(file_path):
    with open(file_path, 'r') as file:
        return load(file)

#
#
# Define time intervals
#

INTERVAL_SECONDS = 300 # 900 # (15 minutes) # takes into account the 3 (dependent on number of assets) sleep(2)'s (minus 6 seconds)
INTERVAL_SAVE_DATA_EVERY_X_MINUTES = 5 # 15
DATA_RETENTION_HOURS = 120 # 4 days - rolling window for analysis and storage

#
#
# Store the last error and manage number of errors before exiting program
#

LAST_EXCEPTION_ERROR = None
LAST_EXCEPTION_ERROR_COUNT = 0
MAX_LAST_EXCEPTION_ERROR_COUNT = 5


#
#
# Set exchange fees and tax rates
#

# coinbase_spot_maker_fee = float(os.environ.get('COINBASE_SPOT_MAKER_FEE')) # unused
coinbase_spot_taker_fee   = float(os.environ.get('COINBASE_SPOT_TAKER_FEE'))
federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))

#
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
    print(f"Data saved to '{file_path}'")

#
#
#
#

# Function to convert Product objects to dictionaries
def convert_products_to_dicts(products):
    return [product.to_dict() if hasattr(product, 'to_dict') else product for product in products]


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
        enabled_assets = [asset['symbol'] for asset in config['assets'] if asset['enabled']]

        #
        #
        # get crypto price data from coinbase
        coinbase_data = coinbase_client.get_products()['products']
        coinbase_data_dictionary = {}
        coinbase_data_dictionary = convert_products_to_dicts(coinbase_data)
        # REDUCE FILE SIZE
        # filter out all crypto records except for those defined in enabled_assets
        coinbase_data_dictionary = [coin for coin in coinbase_data_dictionary if coin['product_id'] in enabled_assets]
        # strip unnecessary fields
        fields_to_remove = [
        'base_increment', 'quote_increment', 'quote_min_size', 'quote_max_size', 'base_min_size', 'base_max_size',
        'alias_to',
        'quote_name', 'base_name', 'watched', 'is_disabled', 'new', 'status', 'cancel_only',
        'limit_only', 'post_only', 'trading_disabled', 'auction_mode', 'product_type', 'quote_currency_id',
        'base_currency_id', 'fcm_trading_session_details', 'mid_market_price', 'alias',
            'base_display_symbol', 'quote_display_symbol', 'view_only', 'price_increment',
            'display_name', 'product_venue', 'approximate_quote_24h_volume', 'new_at', 'market_cap',
            'base_cbrn', 'quote_cbrn', 'product_cbrn'
        ]
        for coin in coinbase_data_dictionary:
            for field in fields_to_remove:
                coin.pop(field, None)

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
        # STORE COINBASE DATA AND ANALYZE
        enable_all_coin_scanning = True
        if enable_all_coin_scanning:
            coinbase_data_directory = 'coinbase-data'

            # NEW STORAGE APPROACH: Append each crypto's data to its own file
            # Store data for each enabled crypto
            for coin in coinbase_data_dictionary:
                product_id = coin['product_id']
                # Create entry with timestamp
                data_entry = {
                    'timestamp': time.time(),
                    **coin  # Include all coin data (price, volume, etc.)
                }
                append_crypto_data_to_file(coinbase_data_directory, product_id, data_entry)
            print(f"Appended data for {len(coinbase_data_dictionary)} cryptos\n")

            if count_files_in_directory(coinbase_data_directory) < 1:
                print('waiting for more data...\n')
            else:
                for coin in coinbase_data_dictionary:
                    # set data from coinbase data
                    symbol = coin['product_id']
                    print(f"[ {symbol} ]")

                    # set config.json data
                    READY_TO_TRADE = False
                    BUY_AMOUNT_USD = 0
                    ENABLE_SNAPSHOT = False
                    for asset in config['assets']:
                        if symbol == asset['symbol']:
                            READY_TO_TRADE = asset['ready_to_trade']
                            BUY_AMOUNT_USD = asset['buy_amount_usd']
                            ENABLE_SNAPSHOT = asset['enable_snapshot']

                    # Get current price and append to data to account for the gap in incrementally stored data
                    current_price = get_asset_price(coinbase_client, symbol) # current_price = float(coin['price'])

                    # NEW RETRIEVAL: Read from individual crypto file (only data from last X hours)
                    coin_prices_LIST = get_property_values_from_crypto_file(coinbase_data_directory, symbol, 'price', max_age_hours=DATA_RETENTION_HOURS)
                    coin_prices_LIST = [float(price) for price in coin_prices_LIST if price is not None] # Convert to list of floats
                    coin_prices_LIST.append(current_price) # append most recent API call result to data to account for the gap in stored data locally

                    coin_volume_24h_LIST = get_property_values_from_crypto_file(coinbase_data_directory, symbol, 'volume_24h', max_age_hours=DATA_RETENTION_HOURS)
                    coin_volume_24h_LIST = [float(volume_24h) for volume_24h in coin_volume_24h_LIST if volume_24h is not None]
                    current_volume_24h = float(coin['volume_24h']) # append most recent API call result
                    coin_volume_24h_LIST.append(current_volume_24h)

                    coin_price_percentage_change_24h_LIST = get_property_values_from_crypto_file(coinbase_data_directory, symbol, 'price_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS)
                    coin_price_percentage_change_24h_LIST = [float(price_percentage_change_24h) for price_percentage_change_24h in coin_price_percentage_change_24h_LIST if price_percentage_change_24h is not None]
                    current_price_percentage_change_24h = float(coin['price_percentage_change_24h'])
                    coin_price_percentage_change_24h_LIST.append(current_price_percentage_change_24h)

                    coin_volume_percentage_change_24h_LIST = get_property_values_from_crypto_file(coinbase_data_directory, symbol, 'volume_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS)
                    coin_volume_percentage_change_24h_LIST = [float(volume_percentage_change_24h) for volume_percentage_change_24h in coin_volume_percentage_change_24h_LIST if volume_percentage_change_24h is not None]
                    current_volume_percentage_change_24h = float(coin['volume_percentage_change_24h'])
                    coin_volume_percentage_change_24h_LIST.append(current_volume_percentage_change_24h)

                    # Periodically cleanup old data from crypto files (runs once per iteration, for each coin)
                    cleanup_old_crypto_data(coinbase_data_directory, symbol, DATA_RETENTION_HOURS)

                    min_price = min(coin_prices_LIST)
                    max_price = max(coin_prices_LIST)
                    trade_range_percentage = calculate_trading_range_percentage(min_price, max_price)
                    price_position_within_trade_range = calculate_current_price_position_within_trading_range(current_price, min_price, max_price)

                    coin_data = {
                        'symbol': symbol,
                        'timestamp': time.time(),
                        'time_interval_minutes': INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                        #
                        'current_price': current_price,
                        'current_volume_24h': current_volume_24h,
                        'current_volume_percentage_change_24h': current_volume_percentage_change_24h,
                        #
                        'coin_prices_list': coin_prices_LIST,
                        'coin_volume_24h_LIST': coin_volume_24h_LIST,
                        'coin_price_percentage_change_24h_LIST': coin_price_percentage_change_24h_LIST,
                        'coin_volume_percentage_change_24h_LIST': coin_volume_percentage_change_24h_LIST,
                    }
                    # print(coin_data)

                    #
                    #
                    #
                    # Manage order data (order types, order info, etc.) in local ledger files
                    entry_price = 0
                    last_order = get_last_order_from_local_json_ledger(symbol)
                    last_order_type = detect_stored_coinbase_order_type(last_order)

                    #
                    #
                    #
                    # Get or create AI analysis for trading parameters
                    actual_coin_prices_list_length = len(coin_prices_LIST) - 1 # account for offset
                    analysis = load_analysis_from_file(symbol)
                    if should_refresh_analysis(symbol, last_order_type):
                        print(f"Generating new AI analysis for {symbol}...")
                        # Check if we have enough data points
                        if actual_coin_prices_list_length < 15:
                            print(f"Insufficient price data for analysis ({actual_coin_prices_list_length}/15 points). Waiting for more data...")
                            analysis = None
                        else:
                            graph_path = None
                            # Generate graph path similar to plot_graph function
                            timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                            graph_path = f"screenshots/{symbol}_{timestamp_str}.png"

                            analysis = analyze_market_with_openai(
                                symbol,
                                coin_data,
                                taker_fee_percentage=coinbase_spot_taker_fee,
                                tax_rate_percentage=federal_tax_rate,
                                graph_image_path=graph_path
                            )
                            if analysis:
                                save_analysis_to_file(symbol, analysis)
                            else:
                                print(f"Warning: Failed to generate analysis for {symbol}")
                    else:
                        print("Found existing AI analysis")

                    # Only proceed with trading if we have a valid analysis
                    if not analysis:
                        print(f"No market analysis available for {symbol}. Skipping trading logic.")
                        print('\n')
                        continue

                    # Set trading parameters from analysis
                    BUY_AT_PRICE = analysis.get('buy_in_price')
                    PROFIT_PERCENTAGE = analysis.get('profit_target_percentage')
                    print(f"AI Strategy: Buy at ${BUY_AT_PRICE}, Target profit {PROFIT_PERCENTAGE}%")
                    print(f"Support: ${analysis.get('major_support', 'N/A')} | Resistance: ${analysis.get('major_resistance', 'N/A')}")
                    print(f"Market Trend: {analysis.get('market_trend', 'N/A')} | Confidence: {analysis.get('confidence_level', 'N/A')}")



                    #
                    #
                    # Pending BUY / SELL order
                    if last_order_type == 'placeholder':
                        print('STATUS: Processing pending order, please standby...')
                        last_order_id = ''
                        last_order_id = last_order['order_id']
                        # if symbol == 'MATIC-USD':
                        #     last_order_id = last_order['response']['order_id']

                        fulfilled_order_data = get_coinbase_order_by_order_id(coinbase_client, last_order_id)
                        print(fulfilled_order_data);

                        if fulfilled_order_data:
                            full_order_dict = fulfilled_order_data['order'] if isinstance(fulfilled_order_data, dict) else fulfilled_order_data.to_dict()
                            save_order_data_to_local_json_ledger(symbol, full_order_dict)
                            print('STATUS: Updated ledger with processed order data')
                        else:
                            print('STATUS: Still processing pending order')

                    #
                    #
                    # BUY logic
                    elif last_order_type == 'none' or last_order_type == 'sell':
                        print(f"STATUS: Looking to BUY at ${BUY_AT_PRICE}")
                        if current_price <= BUY_AT_PRICE:
                            if READY_TO_TRADE == True:
                                shares_to_buy = math.floor(BUY_AMOUNT_USD / current_price) # Calculate whole shares (rounded down)
                                print(f"Calculated shares to buy: {shares_to_buy} (${BUY_AMOUNT_USD} / ${current_price})")
                                if shares_to_buy > 0:
                                    place_market_buy_order(coinbase_client, symbol, shares_to_buy)
                                else:
                                    print(f"STATUS: Buy amount ${BUY_AMOUNT_USD} is too small to buy whole shares at ${current_price}")
                            else:
                                print('STATUS: Trading disabled')
                        else:
                            print(f"Current price ${current_price} is above buy target ${BUY_AT_PRICE}")

                    #
                    #
                    # SELL logic
                    elif last_order_type == 'buy':
                        print('STATUS: Looking to SELL')

                        entry_price = float(last_order['order']['average_filled_price'])
                        print(f"entry_price: {entry_price}")

                        entry_position_value_after_fees = float(last_order['order']['total_value_after_fees'])
                        print(f"entry_position_value_after_fees: {entry_position_value_after_fees}")

                        number_of_shares = float(last_order['order']['filled_size'])
                        print('number_of_shares: ', number_of_shares)

                        # calculate profits if we were going to sell now
                        pre_tax_profit = (current_price - entry_price) * number_of_shares

                        sell_now_exchange_fee = calculate_exchange_fee(current_price, number_of_shares, coinbase_spot_taker_fee)
                        print(f"sell_now_exchange_fee: {sell_now_exchange_fee}")

                        sell_now_tax_owed = (federal_tax_rate / 100) * pre_tax_profit
                        print(f"sell_now_taxes_owed: {sell_now_tax_owed}")

                        potential_profit = (current_price * number_of_shares) - entry_position_value_after_fees - sell_now_exchange_fee - sell_now_tax_owed
                        print(f"potential_profit_USD: {potential_profit}")

                        potential_profit_percentage = (potential_profit / entry_position_value_after_fees) * 100
                        print(f"potential_profit_percentage: {potential_profit_percentage:.4f}%")

                        if potential_profit_percentage >= PROFIT_PERCENTAGE:
                            print('~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~')
                            if READY_TO_TRADE == True:
                                place_market_sell_order(coinbase_client, symbol, number_of_shares, potential_profit, potential_profit_percentage)

                                # Save transaction record
                                buy_timestamp = last_order['order'].get('created_time')
                                save_transaction_record(
                                    symbol=symbol,
                                    buy_price=entry_price,
                                    sell_price=current_price,
                                    potential_profit_percentage=potential_profit_percentage,
                                    gross_profit=pre_tax_profit,
                                    taxes=sell_now_tax_owed,
                                    exchange_fees=sell_now_exchange_fee,
                                    total_profit=potential_profit,
                                    buy_timestamp=buy_timestamp
                                )

                                delete_analysis_file(symbol)
                            else:
                                print('STATUS: Trading disabled')


                    if ENABLE_SNAPSHOT == True:
                        plot_graph(
                            time.time(),
                            INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                            symbol,
                            coin_prices_LIST,
                            min_price,
                            max_price,
                            trade_range_percentage,
                            entry_price,
                            volume_data=coin_volume_24h_LIST,
                            analysis=analysis
                        )

                    print('\n')


                #
                #
                # ERROR TRACKING: reset error count if they're non-consecutive
                LAST_EXCEPTION_ERROR = None
                LAST_EXCEPTION_ERROR_COUNT = 0

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

            time.sleep(5)
