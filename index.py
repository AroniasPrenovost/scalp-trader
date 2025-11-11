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
from utils.coingecko_live import fetch_coingecko_current_data, should_update_coingecko_data, get_last_coingecko_update_time

# Coinbase-related
# Coinbase helpers and define client
from utils.coinbase import get_coinbase_client, get_coinbase_order_by_order_id, place_market_buy_order, place_market_sell_order, place_limit_buy_order, place_limit_sell_order, get_asset_price, calculate_exchange_fee, save_order_data_to_local_json_ledger, get_last_order_from_local_json_ledger, reset_json_ledger_file, detect_stored_coinbase_order_type, save_transaction_record, get_current_fee_rates, cancel_order, clear_order_ledger
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

# daily summary email
from utils.daily_summary import should_send_daily_summary, send_daily_summary_email

# Terminal colors for output formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def format_wallet_metrics(symbol, metrics):
    """Format wallet metrics dictionary in a human-readable way with colors"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*50}")
    print(f"    {symbol}")
    print(f"{'='*50}{Colors.ENDC}\n")

    # Format currency values
    print(f"{Colors.BOLD}Starting Capital:{Colors.ENDC}    {Colors.BLUE}${metrics['starting_capital_usd']:,.2f}{Colors.ENDC}")
    print(f"{Colors.BOLD}Current Value:{Colors.ENDC}       {Colors.BLUE}${metrics['current_usd']:,.2f}{Colors.ENDC}")

    # Profit metrics with color based on positive/negative
    profit_color = Colors.GREEN if metrics['gross_profit'] >= 0 else Colors.RED
    print(f"{Colors.BOLD}Gross Profit:{Colors.ENDC}        {profit_color}${metrics['gross_profit']:,.2f}{Colors.ENDC}")

    # Percentage gain with color
    pct_color = Colors.GREEN if metrics['percentage_gain'] >= 0 else Colors.RED
    print(f"{Colors.BOLD}Percentage Gain:{Colors.ENDC}     {pct_color}{metrics['percentage_gain']:.2f}%{Colors.ENDC}")

    # Fees and taxes
    print(f"{Colors.BOLD}Exchange Fees:{Colors.ENDC}       {Colors.YELLOW}${metrics['exchange_fees']:,.2f}{Colors.ENDC}")
    print(f"{Colors.BOLD}Taxes:{Colors.ENDC}               {Colors.YELLOW}${metrics['taxes']:,.2f}{Colors.ENDC}")

    # Total profit
    total_color = Colors.GREEN if metrics['total_profit'] >= 0 else Colors.RED
    print(f"{Colors.BOLD}Net Profit:{Colors.ENDC}          {total_color}${metrics['total_profit']:,.2f}{Colors.ENDC}")

    print(f"\n{Colors.CYAN}{'='*50}{Colors.ENDC}\n")

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

config = load_config('config.json')

INTERVAL_SECONDS = config['data_retention']['interval_seconds'] # 3600 1 hour
INTERVAL_SAVE_DATA_EVERY_X_MINUTES = (INTERVAL_SECONDS / 60)
DATA_RETENTION_HOURS = config['data_retention']['max_hours'] # 730 # 1 month #

EXPECTED_DATA_POINTS = int((DATA_RETENTION_HOURS * 60) / INTERVAL_SAVE_DATA_EVERY_X_MINUTES)

#
#
#
# Store the last error and manage number of errors before exiting program

LAST_EXCEPTION_ERROR = None
LAST_EXCEPTION_ERROR_COUNT = 0
MAX_LAST_EXCEPTION_ERROR_COUNT = 5

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
    from datetime import datetime, timezone
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
        current_time = datetime.now(timezone.utc)
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

        # send_email_notification(
        #     subject="ello moto",
        #     text_content=f"An error occurred: T(ESTINGG)",
        #     html_content=f"An error occurred: (TESTING)."
        # )

        #
        #
        # ERROR TRACKING
        global LAST_EXCEPTION_ERROR
        global LAST_EXCEPTION_ERROR_COUNT

        # Get taxes and Coinbase fees
        federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))
        fee_rates = get_current_fee_rates(coinbase_client)
        # NOTE: Using TAKER fees because we place MARKET orders (not limit orders)
        # Maker = adds liquidity to order book (lower fee, e.g., 0.4%) - used for limit orders
        # Taker = takes liquidity from order book (higher fee, e.g., 1.2%) - used for market orders
        coinbase_spot_taker_fee = fee_rates['taker_fee'] if fee_rates else 1.2 # Tier: 'Intro 1' taker fee
        coinbase_spot_maker_fee = fee_rates['maker_fee'] if fee_rates else 0.6 # Tier: 'Intro 1' maker fee (not used)

        # load config
        config = load_config('config.json')
        enabled_wallets = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]
        min_profit_target_percentage = config.get('min_profit_target_percentage', 3.0)
        no_trade_refresh_hours = config.get('no_trade_refresh_hours', 1.0)
        cooldown_hours_after_sell = config.get('cooldown_hours_after_sell', 0)
        low_confidence_wait_hours = config.get('low_confidence_wait_hours', 2.0)
        medium_confidence_wait_hours = config.get('medium_confidence_wait_hours', 1.0)
        limit_order_timeout_minutes = config.get('limit_order_timeout_minutes', 60)

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

            #
            # COLLECT GLOBAL VOLUME DATA FROM COINGECKO
            # This maintains consistency with backfilled historical data
            # CoinGecko returns global volume across all exchanges (not just Coinbase)
            # Uses the same interval as Coinbase data collection (from data_retention.interval_seconds)
            #
            coingecko_config = config.get('coingecko', {})
            enable_coingecko_live_collection = coingecko_config.get('enable_live_collection', True)
            coingecko_update_interval_seconds = INTERVAL_SECONDS  # Use same interval as data_retention setting

            if enable_coingecko_live_collection:
                global_volume_directory = 'coingecko-global-volume'

                for wallet in config['wallets']:
                    if not wallet.get('enabled', False):
                        continue

                    symbol = wallet['symbol']
                    coingecko_id = wallet.get('coingecko_id')

                    if not coingecko_id:
                        continue

                    # Check if it's time to update (uses data_retention.interval_seconds)
                    data_file = f"{global_volume_directory}/{symbol}.json"
                    last_update = get_last_coingecko_update_time(data_file)

                    if should_update_coingecko_data(last_update, coingecko_update_interval_seconds):
                        print(f"Fetching global volume data from CoinGecko for {symbol}...")

                        coingecko_data = fetch_coingecko_current_data(coingecko_id, 'usd')

                        if coingecko_data:
                            # Create data entry matching our format
                            global_volume_entry = {
                                'timestamp': time.time(),
                                'product_id': symbol,
                                'price': coingecko_data['price'],
                                'volume_24h': coingecko_data['volume_24h']  # Global volume in BTC
                            }

                            append_crypto_data_to_file(global_volume_directory, symbol, global_volume_entry)
                            print(f"  ✓ Appended global volume: {float(coingecko_data['volume_24h']):,.0f} BTC (${float(coingecko_data['volume_24h_usd']):,.0f} USD)")
                        else:
                            print(f"  ✗ Failed to fetch global volume for {symbol}")
                    else:
                        time_until_next = coingecko_update_interval_seconds - (time.time() - last_update)
                        interval_minutes = coingecko_update_interval_seconds / 60
                        print(f"Skipping CoinGecko update for {symbol} (next update in {time_until_next/60:.1f} of {interval_minutes:.0f} minutes)")

                print()  # Blank line for readability

            if count_files_in_directory(coinbase_data_directory) < 1:
                print('waiting for more data...\n')
            else:
                for coin in coinbase_data_dictionary:
                    # set data from coinbase data
                    symbol = coin['product_id'] # 'BTC-USD', 'ETH-USD', etc..

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
                    format_wallet_metrics(symbol, wallet_metrics)

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

                    # RETRIEVAL: Read from individual crypto file (only data from last X hours)
                    # Note: get_property_values_from_crypto_file already converts prices to float
                    coin_prices_LIST = get_property_values_from_crypto_file(coinbase_data_directory, symbol, 'price', max_age_hours=DATA_RETENTION_HOURS)

                    # NOTE: Using global volume data from CoinGecko for consistency with backfilled historical data.
                    # CoinGecko provides global volume across all exchanges (not just Coinbase).
                    # This ensures volume charts don't have a nosedive between backfilled and live data.
                    # Note: get_property_values_from_crypto_file already converts volumes to float
                    global_volume_directory = 'coingecko-global-volume'
                    coin_volume_24h_LIST = get_property_values_from_crypto_file(global_volume_directory, symbol, 'volume_24h', max_age_hours=DATA_RETENTION_HOURS)
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

                    # Calculate volatility using only last 24 hours of data (not full retention period)
                    # Each data point is 1 hour apart, so last 24 points = last 24 hours
                    volatility_window_hours = 24
                    volatility_data_points = int(volatility_window_hours / (INTERVAL_SECONDS / 3600))
                    recent_prices = coin_prices_LIST[-volatility_data_points:] if len(coin_prices_LIST) >= volatility_data_points else coin_prices_LIST

                    min_price = min(recent_prices)
                    max_price = max(recent_prices)
                    range_percentage_from_min = calculate_percentage_from_min(min_price, max_price)

                    #
                    #
                    #
                    #
                    if ENABLE_CHART_SNAPSHOT:
                        # Load analysis for snapshot if available (analysis will be loaded later in the code flow)
                        snapshot_analysis = load_analysis_from_file(symbol)

                        # Generate all timeframe charts (5 price + 1 volume) for comprehensive market view
                        print(f"Generating snapshot charts for {symbol}...")
                        chart_paths = plot_multi_timeframe_charts(
                            current_timestamp=time.time(),
                            interval=INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                            symbol=symbol,
                            price_data=coin_prices_LIST,
                            volume_data=coin_volume_24h_LIST,
                            analysis=snapshot_analysis
                        )
                        if chart_paths:
                            print(f"✓ Generated {len(chart_paths)} snapshot charts: {', '.join(chart_paths.keys())}")
                        else:
                            print(f"Warning: No snapshot charts generated (insufficient data)")

                    # Check for open or pending positions BEFORE applying volatility filter
                    # This ensures we can sell existing positions even when volatility is low
                    last_order = get_last_order_from_local_json_ledger(symbol)
                    last_order_type = detect_stored_coinbase_order_type(last_order)
                    has_open_position = last_order_type in ['placeholder', 'buy']

                    # Volatility check - skip trading if outside acceptable range (but NOT if we have an open position)
                    if enable_volatility_checks and not has_open_position:
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
                    elif enable_volatility_checks and has_open_position:
                        print(f"Volatility: {range_percentage_from_min:.2f}% (outside range, but allowing trade management for open position)")

                    coin_data = {
                        'current_price': current_price,
                        'current_volume_24h': current_volume_24h,
                        'coin_prices_list': coin_prices_LIST,
                        'coin_volume_24h_LIST': coin_volume_24h_LIST,
                    }

                    #
                    #
                    #
                    # Manage order data (order types, order info, etc.) in local ledger files
                    # Note: last_order and last_order_type already retrieved above (before volatility check)
                    entry_price = 0
                    # print(f"[MAIN] Last order type detected: '{last_order_type}'")

                    #
                    #
                    #
                    # Get or create AI analysis for trading parameters
                    actual_coin_prices_list_length = len(coin_prices_LIST) - 1 # account for offset
                    analysis = load_analysis_from_file(symbol)

                    # Check if we need to generate new analysis
                    should_refresh = should_refresh_analysis(
                        symbol,
                        last_order_type,
                        no_trade_refresh_hours,
                        low_confidence_wait_hours,
                        medium_confidence_wait_hours,
                        coin_data=coin_data,
                        config=config
                    )

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
                                exchange_fee_percentage=coinbase_spot_taker_fee,  # Using taker fee for market orders
                                tax_rate_percentage=federal_tax_rate,
                                min_profit_target_percentage=min_profit_target_percentage,
                                chart_paths=chart_paths,
                                trading_context=trading_context,
                                range_percentage_from_min=range_percentage_from_min,
                                config=config
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

                    # If we have a pending order or open position, use the ORIGINAL analysis from ledger
                    # This ensures the recommendation stays locked while in a trade
                    if last_order_type in ['placeholder', 'buy']:
                        original_analysis = last_order.get('original_analysis')
                        if original_analysis:
                            print(f"✓ Using LOCKED original analysis from buy decision (generated at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(original_analysis.get('analyzed_at', 0)))})")
                            analysis = original_analysis
                        else:
                            print(f"⚠️  No original analysis found in ledger - using current analysis file")

                    # Set trading parameters from analysis
                    BUY_AT_PRICE = analysis.get('buy_in_price')
                    PROFIT_PERCENTAGE = analysis.get('profit_target_percentage')
                    TRADE_RECOMMENDATION = analysis.get('trade_recommendation', 'buy')
                    CONFIDENCE_LEVEL = analysis.get('confidence_level', 'low')
                    STOP_LOSS_PRICE = analysis.get('stop_loss')
                    print('--- AI STRATEGY ---')
                    print(f"buy_at: ${BUY_AT_PRICE}, stop_loss: ${STOP_LOSS_PRICE if STOP_LOSS_PRICE else 'N/A'}, target_profit_%: {PROFIT_PERCENTAGE}%")
                    print(f"current_price: ${current_price}, support: ${analysis.get('major_support', 'N/A')}, resistance: ${analysis.get('major_resistance', 'N/A')}")
                    print(f"market_trend: {analysis.get('market_trend', 'N/A')}, confidence: {CONFIDENCE_LEVEL}")

                    #
                    #
                    # Pending BUY / SELL order
                    if last_order_type == 'placeholder':
                        print('STATUS: Processing pending order, please standby...')
                        # Extract order_id from different possible locations
                        last_order_id = None
                        if 'order_id' in last_order:
                            last_order_id = last_order['order_id']
                        elif 'success_response' in last_order and 'order_id' in last_order['success_response']:
                            last_order_id = last_order['success_response']['order_id']
                        elif 'response' in last_order and 'order_id' in last_order['response']:
                            last_order_id = last_order['response']['order_id']

                        if not last_order_id:
                            print('ERROR: Could not find order_id in pending order')
                            print('\n')
                            continue

                        fulfilled_order_data = get_coinbase_order_by_order_id(coinbase_client, last_order_id)

                        if fulfilled_order_data:
                            # Convert to dict if it's an object
                            if isinstance(fulfilled_order_data, dict):
                                full_order_dict = fulfilled_order_data
                            else:
                                full_order_dict = fulfilled_order_data.to_dict()

                            # Now check if we need to extract nested 'order' key (this is common with Coinbase responses)
                            if 'order' in full_order_dict and isinstance(full_order_dict['order'], dict):
                                full_order_dict = full_order_dict['order']

                            # If there are not many fields, print the full structure (redacted)
                            if len(full_order_dict.keys()) <= 20:
                                import json

                            order_status = full_order_dict.get('status', 'UNKNOWN')
                            print(f"Order status: {order_status}")

                            # Check if order is filled
                            if order_status == 'FILLED':
                                # Preserve the original analysis and screenshot from the placeholder before replacing
                                # These need to persist until the sell transaction is recorded
                                if 'original_analysis' in last_order:
                                    full_order_dict['original_analysis'] = last_order['original_analysis']
                                    print('✓ Preserved original analysis from placeholder order')
                                if 'buy_screenshot_path' in last_order:
                                    full_order_dict['buy_screenshot_path'] = last_order['buy_screenshot_path']
                                    print('✓ Preserved buy screenshot path from placeholder order')

                                # POST-FILL ADJUSTMENT: Check if actual fill price differs significantly from AI recommendation
                                if 'original_analysis' in full_order_dict:
                                    original_analysis = full_order_dict['original_analysis']
                                    ai_recommended_price = original_analysis.get('buy_in_price')
                                    actual_fill_price = float(full_order_dict.get('average_filled_price', 0))

                                    if ai_recommended_price and actual_fill_price > 0:
                                        fill_delta_pct = abs((actual_fill_price - ai_recommended_price) / ai_recommended_price) * 100

                                        print(f"Fill price check: AI recommended ${ai_recommended_price:.2f}, filled at ${actual_fill_price:.2f} (delta: {fill_delta_pct:.2f}%)")

                                        # Threshold: 3% delta triggers AI re-analysis
                                        if fill_delta_pct >= 3.0:
                                            print(f"⚠️  Significant fill delta detected ({fill_delta_pct:.2f}%) - triggering AI post-fill adjustment...")

                                            # Generate fresh charts for AI analysis
                                            chart_paths = plot_multi_timeframe_charts(
                                                current_timestamp=time.time(),
                                                interval=INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                                symbol=symbol,
                                                price_data=coin_prices_LIST,
                                                volume_data=coin_volume_24h_LIST,
                                                analysis=original_analysis
                                            )

                                            # Call AI to adjust analysis based on actual fill
                                            from utils.openai_analysis import adjust_analysis_for_actual_fill
                                            adjusted_analysis = adjust_analysis_for_actual_fill(
                                                symbol=symbol,
                                                original_analysis=original_analysis,
                                                actual_fill_price=actual_fill_price,
                                                current_price=current_price,
                                                chart_paths=chart_paths,
                                                exchange_fee_percentage=coinbase_spot_taker_fee,  # Using taker fee for market orders
                                                tax_rate_percentage=federal_tax_rate
                                            )

                                            if adjusted_analysis:
                                                full_order_dict['original_analysis'] = adjusted_analysis
                                                print('✓ Updated analysis with AI-adjusted targets based on actual fill price')
                                            else:
                                                print('⚠️  AI adjustment failed - keeping original analysis')

                                        elif fill_delta_pct >= 0.5:  # Small delta: use percentage-based adjustment
                                            print(f"Small fill delta ({fill_delta_pct:.2f}%) - applying percentage-based adjustment...")

                                            # Calculate original risk/reward percentages
                                            original_stop_loss = original_analysis.get('stop_loss')
                                            original_sell_price = original_analysis.get('sell_price')

                                            if original_stop_loss and original_sell_price:
                                                # Calculate percentage distances from AI's recommended entry
                                                stop_loss_pct = ((ai_recommended_price - original_stop_loss) / ai_recommended_price)
                                                profit_target_pct = ((original_sell_price - ai_recommended_price) / ai_recommended_price)

                                                # Apply same percentages to actual fill price
                                                adjusted_stop_loss = actual_fill_price * (1 - stop_loss_pct)
                                                adjusted_sell_price = actual_fill_price * (1 + profit_target_pct)

                                                # Update analysis with adjusted values
                                                full_order_dict['original_analysis']['buy_in_price'] = actual_fill_price
                                                full_order_dict['original_analysis']['stop_loss'] = adjusted_stop_loss
                                                full_order_dict['original_analysis']['sell_price'] = adjusted_sell_price

                                                print(f"  Stop loss: ${original_stop_loss:.2f} → ${adjusted_stop_loss:.2f}")
                                                print(f"  Sell price: ${original_sell_price:.2f} → ${adjusted_sell_price:.2f}")
                                                print('✓ Applied percentage-based adjustment (maintained R/R ratio)')

                                # Now replace the entire ledger with the filled order (including preserved data)
                                # This prevents the ledger from accumulating multiple entries
                                import json
                                file_name = f"{symbol}_orders.json"
                                with open(file_name, 'w') as file:
                                    json.dump([full_order_dict], file, indent=4)
                                print('STATUS: Updated ledger with filled order data (analysis preserved until sell)')
                            # Check if order has expired or is still pending
                            elif order_status in ['OPEN', 'PENDING', 'QUEUED']:
                                # Check order age against timeout
                                from datetime import datetime, timezone
                                order_created_time = full_order_dict.get('created_time')
                                if order_created_time:
                                    try:
                                        # Parse ISO format timestamp
                                        order_time = datetime.fromisoformat(order_created_time.replace('Z', '+00:00'))
                                        current_time = datetime.now(timezone.utc)
                                        age_minutes = (current_time - order_time).total_seconds() / 60

                                        print(f"Order age: {age_minutes:.1f} minutes (timeout: {limit_order_timeout_minutes} minutes)")

                                        if age_minutes >= limit_order_timeout_minutes:
                                            print(f"⏰ LIMIT ORDER EXPIRED - Order has been pending for {age_minutes:.1f} minutes")
                                            print(f"   Cancelling order {last_order_id} and restarting with fresh analysis...")

                                            # Cancel the order
                                            cancel_success = cancel_order(coinbase_client, last_order_id)

                                            if cancel_success:
                                                # Clear the ledger to restart trading
                                                clear_order_ledger(symbol)

                                                # Delete analysis file to force new analysis
                                                delete_analysis_file(symbol)

                                                print(f"✓ Order cancelled and ledger cleared. Will generate new analysis on next iteration.")
                                            else:
                                                print(f"⚠️  Failed to cancel order - will retry on next iteration")
                                        else:
                                            print(f'STATUS: Order still pending ({age_minutes:.1f}/{limit_order_timeout_minutes} min)')
                                    except Exception as e:
                                        print(f"Error parsing order timestamp: {e}")
                                        print('STATUS: Still processing pending order')
                                else:
                                    print('STATUS: Still processing pending order (no timestamp found)')
                            # Order was cancelled or failed
                            elif order_status in ['CANCELLED', 'EXPIRED', 'FAILED', 'REJECTED']:
                                print(f"⚠️  Order status: {order_status}")
                                print("   Clearing ledger and restarting with fresh analysis...")
                                clear_order_ledger(symbol)
                                delete_analysis_file(symbol)
                                print("✓ Ledger cleared. Will generate new analysis on next iteration.")
                            else:
                                print(f"⚠️  Unknown order status: {order_status}")
                                print(f"   Available order fields: {list(full_order_dict.keys())}")
                                if 'order' in full_order_dict:
                                    print(f"   Nested 'order' detected - fields inside: {list(full_order_dict['order'].keys()) if isinstance(full_order_dict['order'], dict) else 'Not a dict'}")
                                    # If there's a nested order structure, try to extract status from it
                                    nested_order = full_order_dict.get('order', {})
                                    if isinstance(nested_order, dict) and 'status' in nested_order:
                                        nested_status = nested_order.get('status')
                                        print(f"   Found nested status: {nested_status}")
                                print("   Will retry on next iteration...")
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
                            print(f"STATUS: Executing BUY at market price ~${current_price} (AI target was ${BUY_AT_PRICE}, Confidence: {CONFIDENCE_LEVEL})")
                            # Filter data to match snapshot chart (3 months = 2160 hours)
                            buy_chart_hours = 2160  # 90 days
                            buy_chart_data_points = int((buy_chart_hours * 60) / INTERVAL_SAVE_DATA_EVERY_X_MINUTES)
                            buy_chart_prices = coin_prices_LIST[-buy_chart_data_points:] if len(coin_prices_LIST) > buy_chart_data_points else coin_prices_LIST
                            buy_chart_min = min(buy_chart_prices)
                            buy_chart_max = max(buy_chart_prices)
                            buy_chart_range_pct = calculate_percentage_from_min(buy_chart_min, buy_chart_max)

                            buy_screenshot_path = plot_graph(
                                time.time(),
                                INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                symbol,
                                buy_chart_prices,
                                buy_chart_min,
                                buy_chart_max,
                                buy_chart_range_pct,
                                entry_price,
                                analysis=analysis,
                                buy_event=True
                            )
                            if READY_TO_TRADE:
                                # Get buy amount from LLM analysis - required
                                if analysis and 'buy_amount_usd' in analysis:
                                    buy_amount = analysis.get('buy_amount_usd')
                                    print(f"Using buy amount: ${buy_amount} (from LLM analysis)")

                                    # Calculate shares: use whole shares if we can afford at least 1, otherwise use fractional
                                    shares_calculation = buy_amount / current_price
                                    if shares_calculation >= 1:
                                        shares_to_buy = math.floor(shares_calculation)  # Round down to whole shares
                                        print(f"Calculated shares to buy: {shares_to_buy} whole shares (${buy_amount} / ${current_price})")
                                    else:
                                        # Round fractional shares to 8 decimal places (satoshi precision)
                                        shares_to_buy = round(shares_calculation, 8)
                                        print(f"Calculated shares to buy: {shares_to_buy} fractional shares (${buy_amount} / ${current_price})")

                                    if shares_to_buy > 0:
                                        # Use market order for guaranteed execution
                                        place_market_buy_order(coinbase_client, symbol, shares_to_buy)

                                        # Store screenshot path AND original analysis for later use in transaction record
                                        # This will be retrieved from the ledger when we sell
                                        # IMPORTANT: Store the original analysis NOW to prevent it from being overwritten
                                        last_order = get_last_order_from_local_json_ledger(symbol)
                                        if last_order:
                                            last_order['buy_screenshot_path'] = buy_screenshot_path
                                            last_order['original_analysis'] = analysis.copy()  # Store the analysis that drove this buy decision
                                            # Re-save the ledger with both the screenshot path and analysis
                                            import json
                                            file_name = f"{symbol}_orders.json"
                                            with open(file_name, 'w') as file:
                                                json.dump([last_order], file, indent=4)
                                            print(f"✓ Stored buy screenshot and original AI analysis in ledger (to preserve buy reasoning)")
                                    else:
                                        print(f"STATUS: Buy amount ${buy_amount} must be greater than 0")
                                else:
                                    print("STATUS: No buy_amount_usd in analysis - skipping trade")
                            else:
                                print('STATUS: Trading disabled')

                    #
                    #
                    # SELL logic
                    elif last_order_type == 'buy':
                        print('--- OPEN POSITION ---')

                        # Handle both possible order structures: last_order['order']['field'] or last_order['field']
                        order_data = last_order.get('order', last_order)

                        # Check if this order has been filled and has the necessary data
                        order_status = order_data.get('status', 'UNKNOWN')
                        if order_status not in ['FILLED', 'UNKNOWN']:
                            print(f'WARNING: Order status is {order_status}, not FILLED. Skipping sell logic.')
                            print('\n')
                            continue

                        # Safely extract order fields with fallbacks
                        if 'average_filled_price' not in order_data:
                            print('ERROR: Order data missing required fields for sell logic')
                            print(f'Available fields: {list(order_data.keys())}')
                            print('This may indicate the order has not been fully filled/updated yet')
                            print('\n')
                            continue

                        entry_price = float(order_data['average_filled_price'])
                        print(f"entry_price: ${entry_price}")

                        # Try multiple field names for total value
                        entry_position_value_after_fees = None
                        if 'total_value_after_fees' in order_data:
                            entry_position_value_after_fees = float(order_data['total_value_after_fees'])
                        elif 'filled_value' in order_data and 'total_fees' in order_data:
                            entry_position_value_after_fees = float(order_data['filled_value']) + float(order_data['total_fees'])
                        elif 'filled_value' in order_data:
                            entry_position_value_after_fees = float(order_data['filled_value'])
                        else:
                            print('ERROR: Cannot find total value field in order data')
                            print(f'Available fields: {list(order_data.keys())}')
                            print('\n')
                            continue
                        print(f"entry_position_value_after_fees: ${entry_position_value_after_fees}")

                        # Note: Original analysis is already loaded earlier (at line 629-637) for both 'buy' and 'placeholder'
                        # So we're guaranteed to be using the locked analysis that drove the buy decision

                        number_of_shares = float(order_data['filled_size'])
                        print('number_of_shares: ', number_of_shares)

                        # ============================================================
                        # PROFIT CALCULATION BREAKDOWN (Step-by-Step)
                        # ============================================================
                        # This calculates your actual take-home profit if you sold right now.
                        # All costs (entry fees, exit fees, taxes) are accounted for.

                        # STEP 1: Calculate current market value of your position
                        current_position_value_usd = current_price * number_of_shares
                        print(f"current_market_value: ${current_position_value_usd:.2f}")
                        print(f"  ({number_of_shares:.8f} shares × ${current_price:.2f}/share)")

                        # STEP 2: Calculate what you originally paid (including entry fees)
                        total_cost_basis_usd = entry_position_value_after_fees
                        print(f"total_cost_basis: ${total_cost_basis_usd:.2f}")
                        print(f"  (Original purchase price + entry fees)")

                        # STEP 3: Calculate gross profit (before exit fees and taxes)
                        gross_profit_before_exit_costs = current_position_value_usd - total_cost_basis_usd
                        print(f"gross_profit (before exit costs): ${gross_profit_before_exit_costs:.2f}")
                        print(f"  (${current_position_value_usd:.2f} - ${total_cost_basis_usd:.2f})")

                        # STEP 4: Calculate exit/sell exchange fee (using taker fee for market orders)
                        exit_exchange_fee_usd = calculate_exchange_fee(current_price, number_of_shares, coinbase_spot_taker_fee)
                        print(f"exit_exchange_fee: ${exit_exchange_fee_usd:.2f}")
                        print(f"  ({coinbase_spot_taker_fee}% taker fee on ${current_position_value_usd:.2f})")

                        # STEP 5: Calculate capital gain (for tax purposes)
                        # Capital gain = current value - total cost basis (including entry fees)
                        # This is the same as gross profit before exit costs
                        unrealized_gain_usd = current_position_value_usd - total_cost_basis_usd
                        print(f"unrealized_capital_gain: ${unrealized_gain_usd:.2f}")
                        print(f"  (${current_position_value_usd:.2f} - ${total_cost_basis_usd:.2f})")

                        # STEP 6: Calculate taxes owed on capital gains
                        capital_gains_tax_usd = (federal_tax_rate / 100) * unrealized_gain_usd
                        print(f"capital_gains_tax_owed: ${capital_gains_tax_usd:.2f}")
                        print(f"  ({federal_tax_rate}% tax rate on ${unrealized_gain_usd:.2f} gain)")

                        # STEP 7: Calculate NET PROFIT (your actual take-home after ALL costs)
                        net_profit_after_all_costs_usd = current_position_value_usd - total_cost_basis_usd - exit_exchange_fee_usd - capital_gains_tax_usd
                        print(f"NET_PROFIT (take-home): ${net_profit_after_all_costs_usd:.2f}")
                        print(f"  Formula: Current Value - Cost Basis - Exit Fee - Taxes")
                        print(f"  ${current_position_value_usd:.2f} - ${total_cost_basis_usd:.2f} - ${exit_exchange_fee_usd:.2f} - ${capital_gains_tax_usd:.2f}")

                        # STEP 8: Calculate percentage return on investment
                        net_profit_percentage = (net_profit_after_all_costs_usd / total_cost_basis_usd) * 100
                        print(f"NET_PROFIT %: {net_profit_percentage:.4f}%")
                        print(f"  (${net_profit_after_all_costs_usd:.2f} ÷ ${total_cost_basis_usd:.2f} × 100)")
                        print(f"  net_profit_usd: ${net_profit_after_all_costs_usd:.2f}")

                        # Use the maximum of AI's target and configured minimum
                        effective_profit_target = max(PROFIT_PERCENTAGE, min_profit_target_percentage)
                        print(f"effective_profit_target: {effective_profit_target}% (AI: {PROFIT_PERCENTAGE}%, Min: {min_profit_target_percentage}%)")

                        # Check for stop loss trigger
                        if STOP_LOSS_PRICE and current_price <= STOP_LOSS_PRICE:
                            print('~ STOP LOSS TRIGGERED - Selling to limit losses ~')
                            # Filter data to match snapshot chart (3 months = 2160 hours)
                            sell_chart_hours = 2160  # 90 days
                            sell_chart_data_points = int((sell_chart_hours * 60) / INTERVAL_SAVE_DATA_EVERY_X_MINUTES)
                            sell_chart_prices = coin_prices_LIST[-sell_chart_data_points:] if len(coin_prices_LIST) > sell_chart_data_points else coin_prices_LIST
                            sell_chart_min = min(sell_chart_prices)
                            sell_chart_max = max(sell_chart_prices)
                            sell_chart_range_pct = calculate_percentage_from_min(sell_chart_min, sell_chart_max)

                            plot_graph(
                                time.time(),
                                INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                symbol,
                                sell_chart_prices,
                                sell_chart_min,
                                sell_chart_max,
                                sell_chart_range_pct,
                                entry_price,
                                analysis=analysis,
                                buy_event=False
                            )

                            if READY_TO_TRADE:
                                # Use market order for guaranteed execution on stop loss
                                place_market_sell_order(coinbase_client, symbol, number_of_shares, net_profit_after_all_costs_usd, net_profit_percentage)
                                # Save transaction record
                                buy_timestamp = order_data.get('created_time')
                                buy_screenshot_path = last_order.get('buy_screenshot_path')  # Get screenshot path from ledger

                                # Build market context at entry
                                entry_market_conditions = {
                                    "volatility_range_pct": range_percentage_from_min,
                                    "current_trend": analysis.get('market_trend') if analysis else None,
                                    "confidence_level": analysis.get('confidence_level') if analysis else None,
                                    "entry_reasoning": analysis.get('reasoning') if analysis else None,
                                }

                                # Build position sizing data
                                position_sizing_data = {
                                    "buy_amount_usd": analysis.get('buy_amount_usd') if analysis else None,
                                    "actual_shares": number_of_shares,
                                    "entry_position_value": entry_position_value_after_fees,
                                    "starting_capital": STARTING_CAPITAL_USD,
                                    "wallet_allocation_pct": (entry_position_value_after_fees / STARTING_CAPITAL_USD * 100) if STARTING_CAPITAL_USD > 0 else None,
                                }

                                save_transaction_record(
                                    symbol=symbol,
                                    buy_price=entry_price,
                                    sell_price=current_price,
                                    potential_profit_percentage=net_profit_percentage,
                                    gross_profit=unrealized_gain_usd,
                                    taxes=capital_gains_tax_usd,
                                    exchange_fees=exit_exchange_fee_usd,
                                    total_profit=net_profit_after_all_costs_usd,
                                    buy_timestamp=buy_timestamp,
                                    buy_screenshot_path=buy_screenshot_path,
                                    analysis=analysis,
                                    entry_market_conditions=entry_market_conditions,
                                    exit_trigger='stop_loss',
                                    position_sizing_data=position_sizing_data
                                )

                                delete_analysis_file(symbol)
                            else:
                                print('STATUS: Trading disabled')

                        # Check for profit target
                        elif net_profit_percentage >= effective_profit_target:
                            print('~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~')
                            # Filter data to match snapshot chart (3 months = 2160 hours)
                            sell_chart_hours = 2160  # 90 days
                            sell_chart_data_points = int((sell_chart_hours * 60) / INTERVAL_SAVE_DATA_EVERY_X_MINUTES)
                            sell_chart_prices = coin_prices_LIST[-sell_chart_data_points:] if len(coin_prices_LIST) > sell_chart_data_points else coin_prices_LIST
                            sell_chart_min = min(sell_chart_prices)
                            sell_chart_max = max(sell_chart_prices)
                            sell_chart_range_pct = calculate_percentage_from_min(sell_chart_min, sell_chart_max)

                            plot_graph(
                                time.time(),
                                INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                symbol,
                                sell_chart_prices,
                                sell_chart_min,
                                sell_chart_max,
                                sell_chart_range_pct,
                                entry_price,
                                analysis=analysis,
                                buy_event=False
                            )

                            if READY_TO_TRADE:
                                # Use market order for guaranteed execution on profit target
                                place_market_sell_order(coinbase_client, symbol, number_of_shares, net_profit_after_all_costs_usd, net_profit_percentage)
                                # Save transaction record
                                buy_timestamp = order_data.get('created_time')
                                buy_screenshot_path = last_order.get('buy_screenshot_path')  # Get screenshot path from ledger

                                # Build market context at entry
                                entry_market_conditions = {
                                    "volatility_range_pct": range_percentage_from_min,
                                    "current_trend": analysis.get('market_trend') if analysis else None,
                                    "confidence_level": analysis.get('confidence_level') if analysis else None,
                                    "entry_reasoning": analysis.get('reasoning') if analysis else None,
                                }

                                # Build position sizing data
                                position_sizing_data = {
                                    "buy_amount_usd": analysis.get('buy_amount_usd') if analysis else None,
                                    "actual_shares": number_of_shares,
                                    "entry_position_value": entry_position_value_after_fees,
                                    "starting_capital": STARTING_CAPITAL_USD,
                                    "wallet_allocation_pct": (entry_position_value_after_fees / STARTING_CAPITAL_USD * 100) if STARTING_CAPITAL_USD > 0 else None,
                                }

                                save_transaction_record(
                                    symbol=symbol,
                                    buy_price=entry_price,
                                    sell_price=current_price,
                                    potential_profit_percentage=net_profit_percentage,
                                    gross_profit=unrealized_gain_usd,
                                    taxes=capital_gains_tax_usd,
                                    exchange_fees=exit_exchange_fee_usd,
                                    total_profit=net_profit_after_all_costs_usd,
                                    buy_timestamp=buy_timestamp,
                                    buy_screenshot_path=buy_screenshot_path,
                                    analysis=analysis,
                                    entry_market_conditions=entry_market_conditions,
                                    exit_trigger='profit_target',
                                    position_sizing_data=position_sizing_data
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
        # Daily Summary Email: Check if it's time to send daily summary
        #
        try:
            daily_summary_enabled = config.get('daily_summary', {}).get('enabled', True)
            daily_summary_hour = config.get('daily_summary', {}).get('send_hour', 8)

            if daily_summary_enabled and should_send_daily_summary(target_hour=daily_summary_hour):
                print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
                print(f"  📧 Sending Daily Summary Email")
                print(f"{'='*60}{Colors.ENDC}\n")
                send_daily_summary_email(config['wallets'])
        except Exception as e:
            print(f"{Colors.RED}Error sending daily summary email: {e}{Colors.ENDC}")

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
