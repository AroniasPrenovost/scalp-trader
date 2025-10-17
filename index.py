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
from coinbase.rest import RESTClient # coinbase api
# from mailjet_rest import Client
import argparse # parse CLI args
import glob # related to price change % logic

# custom imports
from utils.email import send_email_notification
from utils.file_helpers import save_obj_dict_to_file, count_files_in_directory, append_crypto_data_to_file, get_property_values_from_crypto_file, cleanup_old_crypto_data
from utils.price_helpers import calculate_percentage_from_min, calculate_offset_price, calculate_price_change_percentage
from utils.time_helpers import print_local_time

# Coinbase-related
# Coinbase helpers and define client
from utils.coinbase import get_coinbase_client, get_coinbase_order_by_order_id, place_market_buy_order, place_market_sell_order, get_asset_price, calculate_exchange_fee, save_order_data_to_local_json_ledger, get_last_order_from_local_json_ledger, reset_json_ledger_file, detect_stored_coinbase_order_type, save_transaction_record
coinbase_client = get_coinbase_client()
# custom coinbase listings check
from utils.new_coinbase_listings import check_for_new_coinbase_listings

# plotting data
from utils.matplotlib import plot_graph, plot_simple_snapshot, plot_multi_timeframe_charts

# LLM-related
# openai analysis
from utils.openai_analysis import analyze_market_with_openai, save_analysis_to_file, load_analysis_from_file, should_refresh_analysis, delete_analysis_file
# trading context for LLM learning
from utils.trade_context import build_trading_context, calculate_wallet_metrics

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

# load config
config = load_config('config.json')

INTERVAL_SECONDS = config['data_retention']['interval_seconds'] # 900 # 15 minutes
INTERVAL_SAVE_DATA_EVERY_X_MINUTES = (INTERVAL_SECONDS / 60)
DATA_RETENTION_HOURS = config['data_retention']['max_hours'] # 730 # 1 month #

EXPECTED_DATA_POINTS = int((DATA_RETENTION_HOURS * 60) / INTERVAL_SAVE_DATA_EVERY_X_MINUTES)

#
#
# Store the last error and manage number of errors before exiting program
#

LAST_EXCEPTION_ERROR = None
LAST_EXCEPTION_ERROR_COUNT = 0
MAX_LAST_EXCEPTION_ERROR_COUNT = 5


#
#
# Set exchange fees and tax rate
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


def get_hours_since_last_sell(symbol):
    """
    Get the number of hours since the last sell for this asset.
    Returns None if no previous sells found.
    """
    from datetime import datetime
    from utils.trade_context import load_transaction_history

    try:
        transactions = load_transaction_history(symbol)

        if not transactions or len(transactions) == 0:
            return None

        # Get most recent transaction
        last_transaction = transactions[0]
        last_sell_timestamp_str = last_transaction.get('timestamp')

        if not last_sell_timestamp_str:
            return None

        # Parse timestamp and calculate hours elapsed
        last_sell_time = datetime.fromisoformat(last_sell_timestamp_str)
        current_time = datetime.now()
        time_elapsed = current_time - last_sell_time

        return time_elapsed.total_seconds() / 3600

    except Exception as e:
        print(f"ERROR: Failed to get hours since last sell: {e}")
        return None


#
#
# main logic loop
#

print_local_time()

def iterate_wallets(interval_seconds):
    while True:

        #
        #
        # ERROR TRACKING
        global LAST_EXCEPTION_ERROR
        global LAST_EXCEPTION_ERROR_COUNT

        # load config
        config = load_config('config.json')
        enabled_wallets = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]
        min_profit_target_percentage = config.get('min_profit_target_percentage', 3.0)
        no_trade_refresh_hours = config.get('no_trade_refresh_hours', 1.0)
        cooldown_hours_after_sell = config.get('cooldown_hours_after_sell', 0)
        low_confidence_wait_hours = config.get('low_confidence_wait_hours', 2.0)
        medium_confidence_wait_hours = config.get('medium_confidence_wait_hours', 1.0)

        # Volatility filter configuration
        volatility_config = config.get('volatility_filters', {})
        enable_volatility_checks = volatility_config.get('enable_volatility_checks', True)
        min_range_percentage = volatility_config.get('min_range_percentage', 5)
        max_range_percentage = volatility_config.get('max_range_percentage', 100)

        # LLM Learning configuration
        llm_learning_config = config.get('llm_learning', {})
        llm_learning_enabled = llm_learning_config.get('enabled', True)
        max_historical_trades = llm_learning_config.get('max_historical_trades', 10)
        include_screenshots = llm_learning_config.get('include_screenshots', True)
        prune_old_trades_after = llm_learning_config.get('prune_old_trades_after', 50)

        #
        #
        # get crypto price data from coinbase
        coinbase_data = coinbase_client.get_products()['products']
        coinbase_data_dictionary = {}
        coinbase_data_dictionary = convert_products_to_dicts(coinbase_data)
        # filter out all crypto records except for those defined in enabled_wallets
        coinbase_data_dictionary = [coin for coin in coinbase_data_dictionary if coin['product_id'] in enabled_wallets]

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
                # Create entry with only the 4 required properties
                data_entry = {
                    'timestamp': time.time(),
                    'product_id': product_id,
                    'price': coin.get('price'),
                    'volume_24h': coin.get('volume_24h')
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
                    ENABLE_CHART_SNAPSHOT = False
                    ENABLE_AI_ANALYSIS = False
                    STARTING_CAPITAL_USD = 0
                    for wallet in config['wallets']:
                        if symbol == wallet['symbol']:
                            READY_TO_TRADE = wallet['ready_to_trade']
                            ENABLE_CHART_SNAPSHOT = wallet['enable_chart_snapshot']
                            ENABLE_AI_ANALYSIS = wallet['enable_ai_analysis']
                            STARTING_CAPITAL_USD = wallet['starting_capital_usd']

                    wallet_metrics = calculate_wallet_metrics(symbol, STARTING_CAPITAL_USD)
                    pprint(wallet_metrics)

                    # Check cooldown period after sell
                    if cooldown_hours_after_sell > 0:
                        hours_since_last_sell = get_hours_since_last_sell(symbol)
                        if hours_since_last_sell is not None and hours_since_last_sell < cooldown_hours_after_sell:
                            hours_remaining = cooldown_hours_after_sell - hours_since_last_sell
                            print(f"STATUS: In cooldown period - {hours_remaining:.2f} hours remaining until analysis resumes")
                            print()
                            continue


                    # Get current price and append to data to account for the gap in incrementally stored data
                    current_price = get_asset_price(coinbase_client, symbol) # current_price = float(coin['price'])

                    # NEW RETRIEVAL: Read from individual crypto file (only data from last X hours)
                    # Note: get_property_values_from_crypto_file already converts prices to float
                    coin_prices_LIST = get_property_values_from_crypto_file(coinbase_data_directory, symbol, 'price', max_age_hours=DATA_RETENTION_HOURS)

                    # NOTE: coin_volume_24h represents Coinbase's rolling 24-hour volume at each data point.
                    # This is useful for 24h charts but misleading for longer timeframes (7d, 90d) since
                    # each historical point shows the 24h rolling volume at that moment, not the actual
                    # volume during that specific interval. See matplotlib.py:199-203 for conditional usage.
                    # Note: get_property_values_from_crypto_file already converts volumes to float
                    coin_volume_24h_LIST = get_property_values_from_crypto_file(coinbase_data_directory, symbol, 'volume_24h', max_age_hours=DATA_RETENTION_HOURS)
                    current_volume_24h = float(coin['volume_24h'])

                    # Periodically cleanup old data from crypto files (runs once per iteration, for each coin)
                    cleanup_old_crypto_data(coinbase_data_directory, symbol, DATA_RETENTION_HOURS)

                    # Validate price data before using min/max
                    if not coin_prices_LIST or len(coin_prices_LIST) == 0:
                        print(f"No price data available for {symbol} - skipping this iteration")
                        print()
                        continue

                    # Ensure all values are floats (safety check)
                    coin_prices_LIST = [float(p) for p in coin_prices_LIST]

                    min_price = min(coin_prices_LIST)
                    max_price = max(coin_prices_LIST)
                    range_percentage_from_min = calculate_percentage_from_min(min_price, max_price)

                    # Volatility check - skip trading if outside acceptable range
                    if enable_volatility_checks:
                        if range_percentage_from_min < min_range_percentage:
                            print(f"STATUS: Volatility too low ({range_percentage_from_min:.2f}% < {min_range_percentage}%) - skipping trade analysis")
                            print(f"Market is too flat for profitable trading (not enough price movement)")
                            print()
                            continue
                        elif range_percentage_from_min > max_range_percentage:
                            print(f"STATUS: Volatility too high ({range_percentage_from_min:.2f}% > {max_range_percentage}%) - skipping trade analysis")
                            print(f"Market is too volatile (excessive risk of whipsaw)")
                            print()
                            continue
                        else:
                            print(f"Volatility: {range_percentage_from_min:.2f}% (within acceptable range {min_range_percentage}-{max_range_percentage}%)")

                    coin_data = {
                        'current_price': current_price,
                        'current_volume_24h': current_volume_24h,
                        'coin_prices_list': coin_prices_LIST,
                        'coin_volume_24h_LIST': coin_volume_24h_LIST,
                    }

                    if ENABLE_CHART_SNAPSHOT:
                        # Filter data to only show last 14 days (336 hours) for snapshot chart
                        snapshot_hours = 336  # 14 days
                        snapshot_data_points = int((snapshot_hours * 60) / INTERVAL_SAVE_DATA_EVERY_X_MINUTES)
                        snapshot_prices = coin_prices_LIST[-snapshot_data_points:] if len(coin_prices_LIST) > snapshot_data_points else coin_prices_LIST
                        snapshot_min_price = min(snapshot_prices)
                        snapshot_max_price = max(snapshot_prices)
                        snapshot_range_percentage = calculate_percentage_from_min(snapshot_min_price, snapshot_max_price)

                        plot_simple_snapshot(
                            time.time(),
                            INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                            symbol,
                            snapshot_prices,
                            snapshot_min_price,
                            snapshot_max_price,
                            snapshot_range_percentage
                        )

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

                    # Check if we need to generate new analysis
                    should_refresh = should_refresh_analysis(symbol, last_order_type, no_trade_refresh_hours, low_confidence_wait_hours, medium_confidence_wait_hours)

                    if should_refresh and not ENABLE_AI_ANALYSIS:
                        print(f"AI analysis is disabled for {symbol} - skipping analysis generation")
                        analysis = None
                    elif should_refresh and ENABLE_AI_ANALYSIS:
                        print(f"Generating new AI analysis for {symbol}...")
                        # Check if we have enough data points
                        if actual_coin_prices_list_length < EXPECTED_DATA_POINTS:
                            print(f"Insufficient price data for analysis ({actual_coin_prices_list_length}/{EXPECTED_DATA_POINTS} points). Waiting for more data...")
                            analysis = None
                        else:
                            # Generate multi-timeframe charts for LLM analysis
                            print(f"Generating multi-timeframe charts for {symbol}...")
                            chart_paths = plot_multi_timeframe_charts(
                                current_timestamp=time.time(),
                                interval=INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                symbol=symbol,
                                price_data=coin_prices_LIST,
                                volume_data=coin_volume_24h_LIST,
                                analysis=None  # No existing analysis yet
                            )
                            print(f"✓ Generated {len(chart_paths)} timeframe charts: {', '.join(chart_paths.keys())}")

                            # Build historical trading context for LLM learning
                            trading_context = None
                            if llm_learning_enabled:
                                print(f"Building historical trading context for {symbol}...")
                                trading_context = build_trading_context(
                                    symbol,
                                    max_trades=max_historical_trades,
                                    include_screenshots=include_screenshots,
                                    starting_capital_usd=STARTING_CAPITAL_USD
                                )
                                if trading_context and trading_context.get('total_trades', 0) > 0:
                                    print(f"✓ Loaded {trading_context['trades_included']} historical trades for context")
                                    # Optionally prune old trades if we have too many
                                    if trading_context.get('total_trades', 0) > prune_old_trades_after:
                                        from utils.trade_context import prune_old_transactions
                                        prune_old_transactions(symbol, keep_count=prune_old_trades_after)
                                else:
                                    print("No historical trades found - this will be the first trade")
                            else:
                                print("LLM learning disabled in config")

                            analysis = analyze_market_with_openai(
                                symbol,
                                coin_data,
                                taker_fee_percentage=coinbase_spot_taker_fee,
                                tax_rate_percentage=federal_tax_rate,
                                min_profit_target_percentage=min_profit_target_percentage,
                                chart_paths=chart_paths,
                                trading_context=trading_context,
                                range_percentage_from_min=range_percentage_from_min
                            )
                            if analysis:
                                save_analysis_to_file(symbol, analysis)
                            else:
                                print(f"Warning: Failed to generate analysis for {symbol}")
                    elif analysis:
                        print(f"Using existing AI analysis (generated at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(analysis.get('analyzed_at', 0)))})")
                    else:
                        print(f"No existing AI analysis found for {symbol}")

                    # Only proceed with trading if we have a valid analysis
                    if not analysis:
                        print(f"No market analysis available for {symbol}. Skipping trading logic.")
                        print('\n')
                        continue

                    # Set trading parameters from analysis
                    BUY_AT_PRICE = analysis.get('buy_in_price')
                    PROFIT_PERCENTAGE = analysis.get('profit_target_percentage')
                    TRADE_RECOMMENDATION = analysis.get('trade_recommendation', 'buy')
                    CONFIDENCE_LEVEL = analysis.get('confidence_level', 'low')
                    STOP_LOSS_PRICE = analysis.get('stop_loss')
                    print(f"AI Strategy: Buy at ${BUY_AT_PRICE}, Target profit {PROFIT_PERCENTAGE}%")
                    print(f"Support: ${analysis.get('major_support', 'N/A')} | Resistance: ${analysis.get('major_resistance', 'N/A')}")
                    print(f"Market Trend: {analysis.get('market_trend', 'N/A')} | Confidence: {CONFIDENCE_LEVEL}")
                    print(f"Stop Loss: ${STOP_LOSS_PRICE if STOP_LOSS_PRICE else 'N/A'}")
                    print(f"Trade Recommendation: {TRADE_RECOMMENDATION}")

                    #
                    #
                    # Pending BUY / SELL order
                    if last_order_type == 'placeholder':
                        print('STATUS: Processing pending order, please standby...')
                        last_order_id = ''
                        last_order_id = last_order['order_id']

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
                        MARKET_TREND = analysis.get('market_trend', 'N/A')
                        if TRADE_RECOMMENDATION != 'buy':
                            print(f"STATUS: AI recommends '{TRADE_RECOMMENDATION}' - only executing buy orders when recommendation is 'buy'")
                        elif CONFIDENCE_LEVEL != 'high':
                            print(f"STATUS: AI confidence level is '{CONFIDENCE_LEVEL}' - only trading with HIGH confidence")
                        elif MARKET_TREND == 'bearish':
                            print(f"STATUS: Market trend is BEARISH - not executing buy orders in bearish markets")
                        else:
                            print(f"STATUS: Looking to BUY at ${BUY_AT_PRICE} (Confidence: {CONFIDENCE_LEVEL})")
                            if current_price <= BUY_AT_PRICE:
                                buy_screenshot_path = plot_graph(
                                    time.time(),
                                    INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                    symbol,
                                    coin_prices_LIST,
                                    min_price,
                                    max_price,
                                    range_percentage_from_min,
                                    entry_price,
                                    analysis=analysis,
                                    buy_event=True
                                )
                                if READY_TO_TRADE:
                                    # Get buy amount from LLM analysis - required
                                    if analysis and 'buy_amount_usd' in analysis:
                                        buy_amount = analysis.get('buy_amount_usd')
                                        print(f"Using buy amount: ${buy_amount} (from LLM analysis)")

                                        shares_to_buy = math.floor(buy_amount / current_price) # Calculate whole shares (rounded down)
                                        print(f"Calculated shares to buy: {shares_to_buy} (${buy_amount} / ${current_price})")
                                        if shares_to_buy > 0:
                                            place_market_buy_order(coinbase_client, symbol, shares_to_buy)
                                            # Store screenshot path for later use in transaction record
                                            # This will be retrieved from the ledger when we sell
                                            last_order = get_last_order_from_local_json_ledger(symbol)
                                            if last_order:
                                                last_order['buy_screenshot_path'] = buy_screenshot_path
                                                # Re-save the ledger with the screenshot path
                                                import json
                                                file_name = f"{symbol}_orders.json"
                                                with open(file_name, 'w') as file:
                                                    json.dump([last_order], file, indent=4)

                                        else:
                                            print(f"STATUS: Buy amount ${buy_amount} is too small to buy whole shares at ${current_price}")
                                    else:
                                        print("STATUS: No buy_amount_usd in analysis - skipping trade")
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

                        # Use the maximum of AI's target and configured minimum
                        effective_profit_target = max(PROFIT_PERCENTAGE, min_profit_target_percentage)
                        print(f"Effective profit target: {effective_profit_target}% (AI: {PROFIT_PERCENTAGE}%, Min: {min_profit_target_percentage}%)")

                        # Check for stop loss trigger
                        if STOP_LOSS_PRICE and current_price <= STOP_LOSS_PRICE:
                            print('~ STOP LOSS TRIGGERED - Selling to limit losses ~')
                            plot_graph(
                                time.time(),
                                INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                symbol,
                                coin_prices_LIST,
                                min_price,
                                max_price,
                                range_percentage_from_min,
                                entry_price,
                                analysis=analysis,
                                buy_event=False
                            )

                            if READY_TO_TRADE:
                                place_market_sell_order(coinbase_client, symbol, number_of_shares, potential_profit, potential_profit_percentage)
                                # Save transaction record
                                buy_timestamp = last_order['order'].get('created_time')
                                buy_screenshot_path = last_order.get('buy_screenshot_path')  # Get screenshot path from ledger
                                save_transaction_record(
                                    symbol=symbol,
                                    buy_price=entry_price,
                                    sell_price=current_price,
                                    potential_profit_percentage=potential_profit_percentage,
                                    gross_profit=pre_tax_profit,
                                    taxes=sell_now_tax_owed,
                                    exchange_fees=sell_now_exchange_fee,
                                    total_profit=potential_profit,
                                    buy_timestamp=buy_timestamp,
                                    buy_screenshot_path=buy_screenshot_path
                                )

                                delete_analysis_file(symbol)
                            else:
                                print('STATUS: Trading disabled')

                        # Check for profit target
                        elif potential_profit_percentage >= effective_profit_target:
                            print('~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~')
                            plot_graph(
                                time.time(),
                                INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                symbol,
                                coin_prices_LIST,
                                min_price,
                                max_price,
                                range_percentage_from_min,
                                entry_price,
                                analysis=analysis,
                                buy_event=False
                            )

                            if READY_TO_TRADE:
                                place_market_sell_order(coinbase_client, symbol, number_of_shares, potential_profit, potential_profit_percentage)
                                # Save transaction record
                                buy_timestamp = last_order['order'].get('created_time')
                                buy_screenshot_path = last_order.get('buy_screenshot_path')  # Get screenshot path from ledger
                                save_transaction_record(
                                    symbol=symbol,
                                    buy_price=entry_price,
                                    sell_price=current_price,
                                    potential_profit_percentage=potential_profit_percentage,
                                    gross_profit=pre_tax_profit,
                                    taxes=sell_now_tax_owed,
                                    exchange_fees=sell_now_exchange_fee,
                                    total_profit=potential_profit,
                                    buy_timestamp=buy_timestamp,
                                    buy_screenshot_path=buy_screenshot_path
                                )

                                delete_analysis_file(symbol)
                            else:
                                print('STATUS: Trading disabled')

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
            iterate_wallets(INTERVAL_SECONDS)
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

            time.sleep(3)
