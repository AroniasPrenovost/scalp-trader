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
from utils.file_helpers import save_obj_dict_to_file, count_files_in_directory, append_crypto_data_to_file, get_property_values_from_crypto_file, cleanup_old_crypto_data, cleanup_old_screenshots
from utils.price_helpers import calculate_percentage_from_min, calculate_offset_price, calculate_price_change_percentage
from utils.time_helpers import print_local_time
from utils.coingecko_live import fetch_coingecko_current_data, should_update_coingecko_data, get_last_coingecko_update_time
from utils.range_support_strategy import check_range_support_buy_signal

# Coinbase-related
# Coinbase helpers and define client
from utils.coinbase import get_coinbase_client, get_coinbase_order_by_order_id, place_market_buy_order, place_market_sell_order, place_limit_buy_order, place_limit_sell_order, get_asset_price, get_asset_balance, calculate_exchange_fee, save_order_data_to_local_json_ledger, get_last_order_from_local_json_ledger, reset_json_ledger_file, detect_stored_coinbase_order_type, save_transaction_record, get_current_fee_rates, cancel_order, clear_order_ledger
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
# core learnings system for persistent pattern avoidance
from utils.core_learnings import (
    load_core_learnings, save_core_learnings, evaluate_hard_rules,
    check_pattern_blacklist, apply_calibrations, get_position_size_multiplier,
    update_learnings_from_trade, format_learnings_for_display
)

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
CHECK_INTERVAL_SECONDS = config['data_retention'].get('check_interval_seconds', 300) # 300 = 5 minutes
INTERVAL_SAVE_DATA_EVERY_X_MINUTES = (INTERVAL_SECONDS / 60)
DATA_RETENTION_HOURS = config['data_retention']['max_hours'] # 730 # 1 month #

EXPECTED_DATA_POINTS = int((DATA_RETENTION_HOURS * 60) / INTERVAL_SAVE_DATA_EVERY_X_MINUTES)
# Allow 99% of expected data points to account for minor gaps (e.g., script restarts, network issues)
MINIMUM_DATA_POINTS = int(EXPECTED_DATA_POINTS * 0.99)

#
#
#
# Store the last error and manage number of errors before exiting program

LAST_EXCEPTION_ERROR = None
LAST_EXCEPTION_ERROR_COUNT = 0
MAX_LAST_EXCEPTION_ERROR_COUNT = 5

# Track when hourly operations were last run (will be loaded from file below)
LAST_HOURLY_OPERATION_TIME = 0

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
# Hourly operation timestamp persistence
#
HOURLY_TIMESTAMP_FILE = 'last_hourly_operation.json'

def save_last_hourly_operation_time(timestamp):
    """Save the last hourly operation timestamp to a file"""
    try:
        with open(HOURLY_TIMESTAMP_FILE, 'w') as file:
            json.dump({'last_hourly_operation': timestamp}, file, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save hourly operation timestamp: {e}")

def load_last_hourly_operation_time():
    """Load the last hourly operation timestamp from file, return 0 if file doesn't exist"""
    try:
        if os.path.exists(HOURLY_TIMESTAMP_FILE):
            with open(HOURLY_TIMESTAMP_FILE, 'r') as file:
                data = json.load(file)
                return data.get('last_hourly_operation', 0)
        else:
            return 0
    except Exception as e:
        print(f"Warning: Failed to load hourly operation timestamp: {e}")
        return 0

#
# Screenshot cleanup timestamp persistence
#
SCREENSHOT_CLEANUP_TIMESTAMP_FILE = 'last_screenshot_cleanup.json'

def save_last_screenshot_cleanup_time(timestamp):
    """Save the last screenshot cleanup timestamp to a file"""
    try:
        with open(SCREENSHOT_CLEANUP_TIMESTAMP_FILE, 'w') as file:
            json.dump({'last_screenshot_cleanup': timestamp}, file, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save screenshot cleanup timestamp: {e}")

def load_last_screenshot_cleanup_time():
    """Load the last screenshot cleanup timestamp from file, return 0 if file doesn't exist"""
    try:
        if os.path.exists(SCREENSHOT_CLEANUP_TIMESTAMP_FILE):
            with open(SCREENSHOT_CLEANUP_TIMESTAMP_FILE, 'r') as file:
                data = json.load(file)
                return data.get('last_screenshot_cleanup', 0)
        else:
            return 0
    except Exception as e:
        print(f"Warning: Failed to load screenshot cleanup timestamp: {e}")
        return 0

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

# Load last hourly operation time from file (for crash recovery)
LAST_HOURLY_OPERATION_TIME = load_last_hourly_operation_time()
if LAST_HOURLY_OPERATION_TIME > 0:
    hours_since = (time.time() - LAST_HOURLY_OPERATION_TIME) / 3600
    print(f"Loaded last hourly operation timestamp: {hours_since:.2f} hours ago\n")

# Load last screenshot cleanup time from file (for crash recovery)
LAST_SCREENSHOT_CLEANUP_TIME = load_last_screenshot_cleanup_time()
if LAST_SCREENSHOT_CLEANUP_TIME > 0:
    hours_since = (time.time() - LAST_SCREENSHOT_CLEANUP_TIME) / 3600
    print(f"Loaded last screenshot cleanup timestamp: {hours_since:.2f} hours ago\n")

def iterate_wallets(check_interval_seconds, hourly_interval_seconds):
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
        global LAST_HOURLY_OPERATION_TIME
        global LAST_SCREENSHOT_CLEANUP_TIME

        # Check if it's time to run hourly operations (data collection, CoinGecko, etc.)
        current_time = time.time()
        time_since_last_hourly = current_time - LAST_HOURLY_OPERATION_TIME
        should_run_hourly_operations = time_since_last_hourly >= hourly_interval_seconds

        if should_run_hourly_operations:
            print(f"{Colors.BOLD}{Colors.CYAN}⏰ Running hourly operations - APPENDING NEW PRICE DATA (last run: {time_since_last_hourly/3600:.2f} hours ago){Colors.ENDC}\n")

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
        coinbase_data_dictionary = convert_products_to_dicts(coinbase_data)
        # Store the original full dictionary before filtering (needed for new listings alert and spike scanner)
        coinbase_data_dictionary_all = coinbase_data_dictionary
        # filter out all crypto records except for those defined in enabled_wallets
        coinbase_data_dictionary = [coin for coin in coinbase_data_dictionary if coin['product_id'] in enabled_wallets]

        #
        #
        # ALERT NEW COIN LISTINGS
        enable_new_listings_alert = False
        if enable_new_listings_alert:
            coinbase_listed_coins_path = 'coinbase-listings/listed_coins.json'
            new_coins = check_for_new_coinbase_listings(coinbase_listed_coins_path, coinbase_data_dictionary_all)
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
        # HOURLY OPERATIONS: Store price/volume data and collect CoinGecko data
        # This runs every hour to build historical dataset
        if should_run_hourly_operations:
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
                print(f"✓ Appended 1 new data point for {len(coinbase_data_dictionary)} cryptos (next append in 1 hour)\n")

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

                    print()  # Blank line for readability

            # Update the last hourly operation timestamp and save to file
            LAST_HOURLY_OPERATION_TIME = time.time()
            save_last_hourly_operation_time(LAST_HOURLY_OPERATION_TIME)
            print(f"{Colors.BOLD}{Colors.GREEN}✓ Hourly operations completed{Colors.ENDC}\n")
        else:
            time_until_next_hourly = hourly_interval_seconds - time_since_last_hourly
            print(f"{Colors.CYAN}⏭  Skipping data collection - NOT appending (last collection: {time_since_last_hourly/60:.1f} min ago, next in {time_until_next_hourly/60:.1f} min){Colors.ENDC}\n")

        #
        # SCREENSHOT CLEANUP: Run periodically based on config
        #
        screenshot_retention_config = config.get('screenshot_retention', {})
        screenshot_cleanup_enabled = screenshot_retention_config.get('enabled', True)
        screenshot_cleanup_interval_hours = screenshot_retention_config.get('cleanup_interval_hours', 6)

        if screenshot_cleanup_enabled:
            time_since_last_cleanup = current_time - LAST_SCREENSHOT_CLEANUP_TIME
            should_run_screenshot_cleanup = time_since_last_cleanup >= (screenshot_cleanup_interval_hours * 3600)

            if should_run_screenshot_cleanup:
                print(f"{Colors.BOLD}{Colors.CYAN}🧹 Running screenshot cleanup (last run: {time_since_last_cleanup/3600:.2f} hours ago){Colors.ENDC}\n")
                try:
                    screenshots_dir = 'screenshots'
                    transactions_dir = 'transactions'
                    cleanup_stats = cleanup_old_screenshots(screenshots_dir, transactions_dir, config)

                    # Update the last cleanup timestamp
                    LAST_SCREENSHOT_CLEANUP_TIME = time.time()
                    save_last_screenshot_cleanup_time(LAST_SCREENSHOT_CLEANUP_TIME)

                    if cleanup_stats['deleted'] > 0:
                        print(f"{Colors.GREEN}✓ Screenshot cleanup completed: Deleted {cleanup_stats['deleted']} files ({cleanup_stats['size_freed_mb']:.2f} MB freed){Colors.ENDC}\n")
                    else:
                        print(f"{Colors.GREEN}✓ Screenshot cleanup completed: No files to delete{Colors.ENDC}\n")
                except Exception as e:
                    print(f"{Colors.RED}Error during screenshot cleanup: {e}{Colors.ENDC}\n")

        #
        #
        # 5-MINUTE OPERATIONS: Check prices and execute trades
        # This runs every 5 minutes to catch market movements
        enable_all_coin_scanning = True
        if enable_all_coin_scanning:
            coinbase_data_directory = 'coinbase-data'

            if count_files_in_directory(coinbase_data_directory) < 1:
                print('waiting for more data...\n')
            else:
                # MARKET ROTATION: Find the single best opportunity across all enabled assets
                market_rotation_config = config.get('market_rotation', {})
                market_rotation_enabled = market_rotation_config.get('enabled', False)

                best_opportunity_symbol = None  # Will hold the symbol we should trade
                active_position_symbols = []    # Track which assets have active positions

                # First, check for any existing active positions
                for symbol in enabled_wallets:
                    last_order = get_last_order_from_local_json_ledger(symbol)
                    last_order_type = detect_stored_coinbase_order_type(last_order)
                    if last_order_type in ['placeholder', 'buy']:
                        active_position_symbols.append(symbol)

                if market_rotation_enabled:
                    from utils.opportunity_scorer import find_best_opportunity, print_opportunity_report, score_opportunity

                    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*100}")
                    print(f"🔍 MARKET ROTATION: Scanning {len(enabled_wallets)} assets for best opportunity...")
                    if active_position_symbols:
                        print(f"📊 ACTIVE POSITION(S): {Colors.GREEN}{', '.join(active_position_symbols)}{Colors.CYAN} (managing these)")
                        print(f"🔎 MONITORING: {', '.join([s for s in enabled_wallets if s not in active_position_symbols])} (scanning for next opportunity)")
                    else:
                        print(f"💰 NO ACTIVE POSITIONS - Capital ready to deploy")
                    print(f"{'='*100}{Colors.ENDC}\n")

                    # Find the best opportunity across all enabled assets
                    best_opportunity = find_best_opportunity(
                        config=config,
                        coinbase_client=coinbase_client,
                        enabled_symbols=enabled_wallets,
                        interval_seconds=INTERVAL_SECONDS,
                        data_retention_hours=DATA_RETENTION_HOURS
                    )

                    # Optionally print detailed report
                    if market_rotation_config.get('print_opportunity_report', True):
                        # Score all opportunities for the report
                        all_opportunities = []
                        for symbol in enabled_wallets:
                            try:
                                current_price = get_asset_price(coinbase_client, symbol)
                                coin_prices_list = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'price', max_age_hours=DATA_RETENTION_HOURS
                                )
                                if coin_prices_list and len(coin_prices_list) > 0:
                                    volatility_window_hours = 24
                                    volatility_data_points = int(volatility_window_hours / (INTERVAL_SECONDS / 3600))
                                    recent_prices = coin_prices_list[-volatility_data_points:] if len(coin_prices_list) >= volatility_data_points else coin_prices_list
                                    min_price = min(recent_prices)
                                    max_price = max(recent_prices)
                                    range_pct = calculate_percentage_from_min(min_price, max_price)

                                    opp = score_opportunity(
                                        symbol=symbol,
                                        config=config,
                                        coinbase_client=coinbase_client,
                                        coin_prices_list=coin_prices_list,
                                        current_price=current_price,
                                        range_percentage_from_min=range_pct
                                    )
                                    all_opportunities.append(opp)
                            except Exception as e:
                                print(f"  Error scoring {symbol}: {e}")
                                continue

                        print_opportunity_report(all_opportunities, best_opportunity)

                    if best_opportunity:
                        best_opportunity_symbol = best_opportunity['symbol']
                        min_score = market_rotation_config.get('min_opportunity_score', 50)

                        if best_opportunity['score'] >= min_score:
                            if active_position_symbols:
                                print(f"{Colors.BOLD}{Colors.GREEN}🎯 BEST NEXT OPPORTUNITY: {best_opportunity_symbol}{Colors.ENDC}")
                                print(f"   Score: {best_opportunity['score']:.1f}/100 | Strategy: {best_opportunity['strategy'].replace('_', ' ').title()}")
                                print(f"   {Colors.YELLOW}⏸  Waiting for active position(s) to close: {', '.join(active_position_symbols)}{Colors.ENDC}")
                                print(f"   {Colors.CYAN}→ Will trade {best_opportunity_symbol} immediately after exit{Colors.ENDC}\n")
                                best_opportunity_symbol = None  # Don't enter new trade while position open
                            else:
                                print(f"{Colors.BOLD}{Colors.GREEN}✅ TRADING NOW: {best_opportunity_symbol}{Colors.ENDC}")
                                print(f"   Score: {best_opportunity['score']:.1f}/100 | Strategy: {best_opportunity['strategy'].replace('_', ' ').title()}")
                                print(f"   Entry: ${best_opportunity['entry_price']:.4f} | Stop: ${best_opportunity['stop_loss']:.4f} | Target: ${best_opportunity['profit_target']:.4f}")
                                if best_opportunity['risk_reward_ratio']:
                                    print(f"   Risk/Reward: 1:{best_opportunity['risk_reward_ratio']:.2f}")
                                print()
                        else:
                            print(f"{Colors.YELLOW}⚠️  Best opportunity {best_opportunity_symbol} has score {best_opportunity['score']:.1f} below minimum {min_score}")
                            print(f"   Skipping all NEW trades this iteration - waiting for better setups")
                            if active_position_symbols:
                                print(f"   {Colors.CYAN}✓ Continuing to manage active position(s): {', '.join(active_position_symbols)}{Colors.ENDC}")
                            print()
                            best_opportunity_symbol = None  # Don't trade anything
                    else:
                        print(f"{Colors.YELLOW}⚠️  No tradeable opportunities found - all assets have open positions or no valid setups")
                        if active_position_symbols:
                            print(f"   {Colors.CYAN}✓ Continuing to manage active position(s): {', '.join(active_position_symbols)}{Colors.ENDC}")
                        else:
                            print(f"   Waiting for market conditions to improve")
                        print()

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

                    # Check if this asset has an active position FIRST (before formatting wallet metrics)
                    # Note: we check verbosity after we know if it's an open position
                    last_order = get_last_order_from_local_json_ledger(symbol, verbose=False)
                    last_order_type = detect_stored_coinbase_order_type(last_order)
                    has_open_position = last_order_type in ['placeholder', 'buy']

                    # ENHANCED LOGGING: Show clearly if this is an active trade or just monitoring
                    # Only show detailed logging for open positions or selected opportunities
                    show_detailed_logs = has_open_position or (market_rotation_enabled and symbol == best_opportunity_symbol)

                    if has_open_position:
                        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*100}")
                        print(f"  🔥 ACTIVE TRADE: {symbol} - Managing Open Position")
                        print(f"{'='*100}{Colors.ENDC}")
                        format_wallet_metrics(symbol, wallet_metrics)
                        print(f"{Colors.CYAN}📊 Monitoring other assets in background for next opportunity after exit{Colors.ENDC}\n")
                    elif market_rotation_enabled and symbol == best_opportunity_symbol:
                        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*100}")
                        print(f"  🎯 SELECTED OPPORTUNITY: {symbol} - Evaluating Entry")
                        print(f"{'='*100}{Colors.ENDC}")
                        format_wallet_metrics(symbol, wallet_metrics)
                    # Else: silent monitoring - no output for non-selected opportunities

                    # MARKET ROTATION: Only ENTER trades on the best opportunity
                    # But we still analyze every wallet to:
                    # 1. Manage existing open positions (sell logic)
                    # 2. Keep scoring updated for next opportunity selection
                    # 3. Provide visibility into all market conditions
                    should_allow_new_entry = True
                    if market_rotation_enabled and best_opportunity_symbol:
                        if symbol != best_opportunity_symbol and not has_open_position:
                            should_allow_new_entry = False
                            if show_detailed_logs:
                                print(f"{Colors.CYAN}💡 Analyzing {symbol} (best opportunity: {best_opportunity_symbol} - will only enter {best_opportunity_symbol} if conditions met){Colors.ENDC}\n")

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
                    cleanup_old_crypto_data(coinbase_data_directory, symbol, DATA_RETENTION_HOURS, verbose=show_detailed_logs)

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
                    # Note: last_order already retrieved above without verbose output
                    # last_order = get_last_order_from_local_json_ledger(symbol)
                    # last_order_type = detect_stored_coinbase_order_type(last_order)
                    # has_open_position = last_order_type in ['placeholder', 'buy']

                    # Volatility check - skip trading if outside acceptable range (but NOT if we have an open position)
                    if enable_volatility_checks and not has_open_position:
                        if range_percentage_from_min < min_range_percentage:
                            if show_detailed_logs:
                                print(f"STATUS: Volatility too low ({range_percentage_from_min:.2f}% < {min_range_percentage}%) - skipping trade analysis")
                                print(f"Market is too flat for profitable trading (not enough price movement)")
                                print()
                            continue
                        elif range_percentage_from_min > max_range_percentage:
                            if show_detailed_logs:
                                print(f"STATUS: Volatility too high ({range_percentage_from_min:.2f}% > {max_range_percentage}%) - skipping trade analysis")
                                print(f"Market is too volatile (excessive risk of whipsaw)")
                                print()
                            continue
                        else:
                            if show_detailed_logs:
                                print(f"Volatility: {range_percentage_from_min:.2f}% (within acceptable range {min_range_percentage}-{max_range_percentage}%)")
                    elif enable_volatility_checks and has_open_position:
                        if show_detailed_logs:
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
                    actual_coin_prices_list_length = len(coin_prices_LIST)
                    analysis = load_analysis_from_file(symbol)

                    # Load core learnings early so it's always available
                    core_learnings = load_core_learnings(symbol)

                    # Build historical trading context for LLM learning (always, even for cached analysis)
                    # This is needed for position sizing calculations
                    trading_context = None
                    if llm_learning_enabled:
                        trading_context = build_trading_context(
                            symbol,
                            max_trades=max_historical_trades,
                            include_screenshots=include_screenshots,
                            starting_capital_usd=STARTING_CAPITAL_USD
                        )
                        if trading_context and trading_context.get('total_trades', 0) > 0:
                            # Optionally prune old trades if we have too many
                            if trading_context.get('total_trades', 0) > prune_old_trades_after:
                                from utils.trade_context import prune_old_transactions
                                prune_old_transactions(symbol, keep_count=prune_old_trades_after)

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
                        # Check if we have enough data points (with 99% tolerance for minor gaps)
                        if actual_coin_prices_list_length < MINIMUM_DATA_POINTS:
                            print(f"Insufficient price data for analysis ({actual_coin_prices_list_length}/{EXPECTED_DATA_POINTS} points, minimum: {MINIMUM_DATA_POINTS}). Waiting for more data...")
                            analysis = None
                        else:
                            # Show warning if not at full expected data but still proceeding
                            if actual_coin_prices_list_length < EXPECTED_DATA_POINTS:
                                print(f"⚠️  Using {actual_coin_prices_list_length}/{EXPECTED_DATA_POINTS} data points (minor gaps detected, but sufficient for analysis)")
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

                            # Display context information
                            if trading_context and trading_context.get('total_trades', 0) > 0:
                                print(f"✓ Loaded {trading_context['trades_included']} historical trades for context")
                            else:
                                print("No historical trades found - this will be the first trade")

                            # Display core learnings if present (already loaded earlier)
                            if core_learnings and (
                                core_learnings.get('hard_rules') or
                                core_learnings.get('pattern_blacklist') or
                                core_learnings.get('loss_streaks', {}).get('current_streak', 0) > 0
                            ):
                                print(format_learnings_for_display(core_learnings))

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
                        if show_detailed_logs:
                            print(f"Using existing AI analysis (generated at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(analysis.get('analyzed_at', 0)))})")

                        # CRITICAL: Apply position sizing to cached analysis if buy_amount_usd is missing or 0
                        # This fixes the issue where old cached analysis had buy_amount_usd set to 0
                        if analysis.get('buy_amount_usd', 0) == 0 and trading_context and config and range_percentage_from_min is not None:
                            from utils.dynamic_refresh import calculate_volatility_adjusted_position_size

                            wallet_metrics = trading_context.get('wallet_metrics', {})
                            current_usd_value = wallet_metrics.get('current_usd', 0)
                            starting_capital = wallet_metrics.get('starting_capital_usd', 0)
                            confidence_level = analysis.get('confidence_level', 'low')

                            if current_usd_value > 0:
                                adjusted_position = calculate_volatility_adjusted_position_size(
                                    range_percentage_from_min=range_percentage_from_min,
                                    starting_capital_usd=starting_capital,
                                    current_usd_value=current_usd_value,
                                    confidence_level=confidence_level,
                                    config=config
                                )

                                analysis['buy_amount_usd'] = adjusted_position
                                analysis['position_sizing_method'] = 'volatility_adjusted'
                                if show_detailed_logs:
                                    print(f"✓ Applied position sizing to cached analysis: ${adjusted_position:.2f}")

                                # Save the updated analysis back to file
                                save_analysis_to_file(symbol, analysis)
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

                    # Apply core learnings guardrails (only for new trades, not locked positions)
                    if last_order_type not in ['placeholder', 'buy']:
                        # Apply calibrations (just adjustments, doesn't block)
                        market_trend = analysis.get('market_trend', 'unknown')
                        analysis = apply_calibrations(core_learnings, analysis, market_trend)

                        # Check hard rules (only blocks if explicit rules exist from repeated failures)
                        from datetime import datetime as dt
                        current_conditions = {
                            'market_trend': market_trend,
                            'confidence_level': analysis.get('confidence_level', 'low')
                        }
                        should_trade, block_reason = evaluate_hard_rules(core_learnings, current_conditions)
                        if not should_trade:
                            print(block_reason)
                            print('\n')
                            continue

                        # Check pattern blacklist (only 0% win rate patterns after 3+ attempts)
                        pattern_description = f"{market_trend} {analysis.get('confidence_level', 'low')}"
                        should_trade, block_reason = check_pattern_blacklist(
                            core_learnings,
                            pattern_description,
                            market_trend,
                            analysis.get('confidence_level', 'low')
                        )
                        if not should_trade:
                            print(block_reason)
                            print('\n')
                            continue

                        # Reduce position size if on loss streak (3+ losses)
                        position_multiplier = get_position_size_multiplier(core_learnings)
                        if position_multiplier < 1.0 and analysis.get('buy_amount_usd', 0) > 0:
                            original_buy_amount = analysis['buy_amount_usd']
                            analysis['buy_amount_usd'] = original_buy_amount * position_multiplier
                            print(f"📉 Position reduced to {position_multiplier:.0%} due to loss streak (${original_buy_amount:.2f} → ${analysis['buy_amount_usd']:.2f})")

                    # Set trading parameters from analysis
                    BUY_AT_PRICE = analysis.get('buy_in_price')
                    PROFIT_PERCENTAGE = analysis.get('profit_target_percentage')
                    TRADE_RECOMMENDATION = analysis.get('trade_recommendation', 'buy')
                    CONFIDENCE_LEVEL = analysis.get('confidence_level', 'low')
                    STOP_LOSS_PRICE = analysis.get('stop_loss')
                    if show_detailed_logs:
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
                        # SAFETY CHECK: Verify we don't have an open position (double-check in case of ledger corruption)
                        if has_open_position:
                            print(f"{Colors.RED}⚠️  SAFETY CHECK FAILED: Detected open position but last_order_type='{last_order_type}'{Colors.ENDC}")
                            print(f"{Colors.RED}   This indicates a ledger inconsistency. Skipping buy logic to prevent double-position.{Colors.ENDC}")
                            print(f"{Colors.YELLOW}   Please review the ledger file: {symbol}_orders.json{Colors.ENDC}\n")
                            continue

                        MARKET_TREND = analysis.get('market_trend', 'N/A')

                        # Load range support strategy configuration
                        range_strategy_config = config.get('range_support_strategy', {})
                        range_strategy_enabled = range_strategy_config.get('enabled', True)

                        # Check range support strategy if enabled
                        range_signal = None
                        if range_strategy_enabled:
                            print(f"\n{Colors.BOLD}{Colors.CYAN}--- RANGE SUPPORT STRATEGY CHECK ---{Colors.ENDC}")
                            range_signal = check_range_support_buy_signal(
                                prices=coin_prices_LIST,
                                current_price=current_price,
                                min_touches=range_strategy_config.get('min_touches', 2),
                                zone_tolerance_percentage=range_strategy_config.get('zone_tolerance_percentage', 3.0),
                                entry_tolerance_percentage=range_strategy_config.get('entry_tolerance_percentage', 1.5),
                                extrema_order=range_strategy_config.get('extrema_order', 5),
                                lookback_window=range_strategy_config.get('lookback_window_hours', 336)
                            )

                            if range_signal['signal'] == 'buy':
                                zone = range_signal['zone']
                                print(f"{Colors.GREEN}✓ RANGE SIGNAL: BUY{Colors.ENDC}")
                                print(f"  Support zone: ${zone['zone_price_min']:.2f} - ${zone['zone_price_max']:.2f} (avg: ${zone['zone_price_avg']:.2f})")
                                print(f"  Zone strength: {zone['touches']} touches")
                                print(f"  Current price: ${current_price:.2f}")
                                print(f"  Distance from zone avg: {range_signal['distance_from_zone_avg']:+.2f}%")
                                print(f"  {range_signal['reasoning']}")
                            else:
                                print(f"{Colors.YELLOW}✗ RANGE SIGNAL: NO BUY{Colors.ENDC}")
                                print(f"  {range_signal['reasoning']}")
                                if range_signal['all_zones']:
                                    strongest_zone = range_signal['all_zones'][0]
                                    print(f"  Nearest support zone: ${strongest_zone['zone_price_avg']:.2f} ({range_signal['distance_from_zone_avg']:+.2f}% away)")

                        # Check all buy conditions
                        # Note: If market rotation is enabled and this is the selected best opportunity,
                        # we trust the opportunity scorer's strategy validation
                        # Otherwise, fall back to the traditional checks (AI + range strategy)

                        is_selected_opportunity = market_rotation_enabled and symbol == best_opportunity_symbol
                        should_execute_buy = False  # Track if we should execute the buy

                        if not should_allow_new_entry:
                            if show_detailed_logs:
                                print(f"\n{Colors.YELLOW}STATUS: {symbol} is not the best opportunity - skipping NEW entry (best: {best_opportunity_symbol}){Colors.ENDC}")
                            # When market rotation is enabled, ONLY the best opportunity can trigger new buys
                            # Do not fall through to traditional buy logic
                        elif is_selected_opportunity:
                            # This is the best opportunity selected by the scorer - it already validated the strategy
                            # Just verify it's actually a quality trade with a valid signal and meets minimum score
                            if best_opportunity and best_opportunity.get('signal') == 'buy':
                                min_score = market_rotation_config.get('min_opportunity_score', 50)
                                if best_opportunity.get('score', 0) >= min_score:
                                    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}")
                                    print(f"  ✓ BEST OPPORTUNITY - EXECUTING BUY")
                                    print(f"{'='*60}{Colors.ENDC}")
                                    print(f"{Colors.BOLD}{Colors.CYAN}Strategy: {best_opportunity['strategy'].replace('_', ' ').title()}{Colors.ENDC}")
                                    print(f"Score: {best_opportunity['score']:.1f}/100 | Confidence: {best_opportunity['confidence'].upper()}")
                                    print(f"Trend: {best_opportunity.get('trend', 'unknown').title()}")
                                    should_execute_buy = True
                                else:
                                    print(f"\n{Colors.YELLOW}STATUS: Selected opportunity {symbol} score {best_opportunity['score']:.1f} below minimum {min_score} - skipping{Colors.ENDC}")
                            else:
                                print(f"\n{Colors.YELLOW}STATUS: Selected opportunity {symbol} no longer has valid buy signal - skipping{Colors.ENDC}")
                        elif not market_rotation_enabled:
                            # Traditional path only allowed when market rotation is disabled
                            if range_strategy_enabled and range_signal and range_signal['signal'] != 'buy':
                                print(f"\n{Colors.YELLOW}STATUS: Not in support zone - waiting for price to reach support{Colors.ENDC}")
                            elif TRADE_RECOMMENDATION != 'buy':
                                print(f"\n{Colors.YELLOW}STATUS: AI recommends '{TRADE_RECOMMENDATION}' - only executing buy orders when recommendation is 'buy'{Colors.ENDC}")
                            elif CONFIDENCE_LEVEL != 'high':
                                print(f"\n{Colors.YELLOW}STATUS: AI confidence level is '{CONFIDENCE_LEVEL}' - only trading with HIGH confidence{Colors.ENDC}")
                            else:
                                # Traditional path: all individual checks passed
                                print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}")
                                print(f"  ✓ ALL BUY CONDITIONS MET - EXECUTING BUY")
                                print(f"{'='*60}{Colors.ENDC}")
                                should_execute_buy = True

                        # Execute buy if conditions met
                        if should_execute_buy:
                            # Show which strategy triggered the buy
                            if is_selected_opportunity and best_opportunity:
                                # Market rotation path - show the strategy that scored highest
                                strategy_name = best_opportunity['strategy'].replace('_', ' ').title()
                                print(f"{Colors.GREEN}✓ {strategy_name} Strategy Selected (Score: {best_opportunity['score']:.1f}/100){Colors.ENDC}")
                                if best_opportunity.get('reasoning'):
                                    print(f"{Colors.CYAN}Reasoning: {best_opportunity['reasoning']}{Colors.ENDC}")
                            else:
                                # Traditional path - show individual checks
                                if range_strategy_enabled and range_signal and range_signal['signal'] == 'buy':
                                    zone = range_signal['zone']
                                    print(f"{Colors.GREEN}Range Strategy: ✓ In support zone (${zone['zone_price_avg']:.2f}, {zone['touches']} touches){Colors.ENDC}")
                                print(f"{Colors.GREEN}AI Analysis: ✓ BUY recommendation with HIGH confidence{Colors.ENDC}")

                            print(f"{Colors.GREEN}Market Price: ${current_price:.2f} (AI target: ${BUY_AT_PRICE:.2f}){Colors.ENDC}\n")
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
                                        # Always use limit orders for guaranteed price and lower fees (maker vs taker)
                                        # Get optional price buffer from config (default 0% = exact AI price)
                                        order_execution_config = config.get('order_execution', {})
                                        limit_price_buffer_pct = order_execution_config.get('limit_order_price_buffer_pct', 0.0)

                                        # Calculate limit price with optional buffer
                                        # Buffer of 0% means we get EXACTLY the AI recommended price or better
                                        # Small buffer (0.3-0.5%) increases fill probability in fast markets
                                        limit_price = BUY_AT_PRICE * (1 + limit_price_buffer_pct / 100)
                                        print(f"Placing LIMIT buy order at ${limit_price:.4f} (AI target: ${BUY_AT_PRICE:.4f}, buffer: {limit_price_buffer_pct}%)")
                                        place_limit_buy_order(coinbase_client, symbol, shares_to_buy, limit_price)

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
                        print(f"{Colors.CYAN}NET_PROFIT (take-home): ${net_profit_after_all_costs_usd:.2f}{Colors.ENDC}")
                        print(f"  Formula: Current Value - Cost Basis - Exit Fee - Taxes")
                        print(f"  ${current_position_value_usd:.2f} - ${total_cost_basis_usd:.2f} - ${exit_exchange_fee_usd:.2f} - ${capital_gains_tax_usd:.2f}")

                        # STEP 8: Calculate percentage return on investment
                        net_profit_percentage = (net_profit_after_all_costs_usd / total_cost_basis_usd) * 100
                        print(f"{Colors.CYAN}NET_PROFIT %: {net_profit_percentage:.4f}%{Colors.ENDC}")
                        print(f"  (${net_profit_after_all_costs_usd:.2f} ÷ ${total_cost_basis_usd:.2f} × 100)")
                        print(f"  net_profit_usd: ${net_profit_after_all_costs_usd:.2f}")

                        # Use the maximum of AI's target and configured minimum
                        effective_profit_target = max(PROFIT_PERCENTAGE, min_profit_target_percentage)
                        print(f"effective_profit_target: {effective_profit_target}% (AI: {PROFIT_PERCENTAGE}%, Min: {min_profit_target_percentage}%)")

                        print(f"--- POSITION STATUS ---")
                        print(f"Entry price: ${entry_price:.2f}")
                        print(f"Current price: ${current_price:.2f}")
                        print(f"Current profit: ${net_profit_after_all_costs_usd:.2f} ({net_profit_percentage:.4f}%)")
                        print(f"Stop loss: ${STOP_LOSS_PRICE:.2f} | Profit target: {effective_profit_target:.2f}%")

                        # INTELLIGENT ROTATION: Check if we should exit a profitable position for a better opportunity
                        should_rotate_position = False
                        rotation_reason = None

                        intelligent_rotation_config = market_rotation_config.get('intelligent_rotation', {})
                        intelligent_rotation_enabled = intelligent_rotation_config.get('enabled', False)

                        if market_rotation_enabled and intelligent_rotation_enabled and best_opportunity:
                            # Only consider rotation if we're currently profitable
                            min_profit_for_rotation = intelligent_rotation_config.get('min_profit_to_consider_rotation', 0.5)

                            if net_profit_percentage >= min_profit_for_rotation:
                                # We're in profit - check if there's a significantly better opportunity
                                current_symbol_score = 0

                                # Try to find the current position's opportunity score
                                for symbol_check in enabled_wallets:
                                    if symbol_check == symbol:
                                        try:
                                            # Score the current position
                                            current_opp = score_opportunity(
                                                symbol=symbol,
                                                config=config,
                                                coinbase_client=coinbase_client,
                                                coin_prices_list=coin_prices_LIST,
                                                current_price=current_price,
                                                range_percentage_from_min=range_percentage_from_min
                                            )
                                            current_symbol_score = current_opp.get('score', 0)
                                        except:
                                            current_symbol_score = 0
                                        break

                                best_opp_score = best_opportunity.get('score', 0)
                                best_opp_symbol = best_opportunity.get('symbol')
                                min_score_advantage = intelligent_rotation_config.get('min_score_advantage', 15)

                                print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 INTELLIGENT ROTATION CHECK{Colors.ENDC}")
                                print(f"  Current position ({symbol}): Score {current_symbol_score:.1f}/100, Profit: {net_profit_percentage:.2f}%")
                                print(f"  Best opportunity ({best_opp_symbol}): Score {best_opp_score:.1f}/100")
                                print(f"  Score advantage required: {min_score_advantage}+")

                                # Check if the best opportunity is significantly better
                                score_difference = best_opp_score - current_symbol_score

                                if best_opp_symbol != symbol and score_difference >= min_score_advantage:
                                    # New opportunity is significantly better by score
                                    print(f"  {Colors.GREEN}✓ Score advantage: {score_difference:.1f} (meets threshold){Colors.ENDC}")

                                    # Optional: Also check if the new opportunity has better profit target potential
                                    require_profit_advantage = intelligent_rotation_config.get('require_profit_target_advantage', True)

                                    if require_profit_advantage:
                                        # Calculate remaining upside in current position
                                        current_remaining_upside = effective_profit_target - net_profit_percentage

                                        # Get new opportunity's expected profit target
                                        new_opp_entry = best_opportunity.get('entry_price', current_price)
                                        new_opp_target = best_opportunity.get('profit_target', 0)

                                        if new_opp_entry > 0 and new_opp_target > 0:
                                            new_opp_upside = ((new_opp_target - new_opp_entry) / new_opp_entry) * 100
                                            min_profit_advantage = intelligent_rotation_config.get('min_profit_target_advantage_percentage', 1.0)

                                            print(f"  Current remaining upside: {current_remaining_upside:.2f}%")
                                            print(f"  New opportunity upside: {new_opp_upside:.2f}%")
                                            print(f"  Profit advantage required: {min_profit_advantage}%+")

                                            profit_advantage = new_opp_upside - current_remaining_upside

                                            if profit_advantage >= min_profit_advantage:
                                                print(f"  {Colors.GREEN}✓ Profit advantage: {profit_advantage:.2f}% (meets threshold){Colors.ENDC}")
                                                should_rotate_position = True
                                                rotation_reason = f"Rotating to {best_opp_symbol}: {score_difference:.1f} better score, {profit_advantage:.2f}% more upside potential"
                                            else:
                                                print(f"  {Colors.YELLOW}✗ Profit advantage: {profit_advantage:.2f}% (below {min_profit_advantage}%){Colors.ENDC}")
                                                print(f"  {Colors.YELLOW}Staying in current position - better profit potential here{Colors.ENDC}")
                                        else:
                                            # Can't calculate profit advantage - use score only
                                            should_rotate_position = True
                                            rotation_reason = f"Rotating to {best_opp_symbol}: {score_difference:.1f} better score"
                                    else:
                                        # Score advantage alone is enough
                                        should_rotate_position = True
                                        rotation_reason = f"Rotating to {best_opp_symbol}: {score_difference:.1f} better score"
                                else:
                                    if best_opp_symbol == symbol:
                                        print(f"  {Colors.CYAN}✓ Current position IS the best opportunity - holding{Colors.ENDC}")
                                    else:
                                        print(f"  {Colors.YELLOW}✗ Score advantage: {score_difference:.1f} (below {min_score_advantage}){Colors.ENDC}")
                                        print(f"  {Colors.YELLOW}Staying in current position - advantage not significant enough{Colors.ENDC}")

                                print()  # Blank line for readability
                            else:
                                print(f"\n{Colors.CYAN}ℹ️  Profit {net_profit_percentage:.2f}% below rotation threshold ({min_profit_for_rotation}%) - not considering rotation{Colors.ENDC}\n")

                        # Execute intelligent rotation if triggered
                        if should_rotate_position:
                            print(f'{Colors.BOLD}{Colors.CYAN}🔄 INTELLIGENT ROTATION TRIGGERED{Colors.ENDC}')
                            print(f'{Colors.CYAN}Reason: {rotation_reason}{Colors.ENDC}')
                            print(f'{Colors.GREEN}Exiting {symbol} with {net_profit_percentage:.2f}% profit to enter {best_opportunity["symbol"]}{Colors.ENDC}')

                            # Generate sell chart
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
                                # Execute the rotation sell
                                place_market_sell_order(coinbase_client, symbol, number_of_shares, net_profit_after_all_costs_usd, net_profit_percentage)

                                # Save transaction record
                                buy_timestamp = order_data.get('created_time')
                                buy_screenshot_path = last_order.get('buy_screenshot_path')

                                entry_market_conditions = {
                                    "volatility_range_pct": range_percentage_from_min,
                                    "current_trend": analysis.get('market_trend') if analysis else None,
                                    "confidence_level": analysis.get('confidence_level') if analysis else None,
                                    "entry_reasoning": analysis.get('reasoning') if analysis else None,
                                }

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
                                    exit_trigger='intelligent_rotation',
                                    position_sizing_data=position_sizing_data
                                )

                                # Update core learnings
                                from utils.trade_context import load_transaction_history
                                trade_outcome = {
                                    'profit': net_profit_after_all_costs_usd,
                                    'exit_trigger': 'intelligent_rotation',
                                    'confidence_level': analysis.get('confidence_level', 'unknown'),
                                    'market_trend': analysis.get('market_trend', 'unknown')
                                }
                                transactions = load_transaction_history(symbol)
                                update_learnings_from_trade(symbol, trade_outcome, transactions)

                                delete_analysis_file(symbol)

                                # IMMEDIATE ROTATION: Recalculate best opportunity after exit
                                print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 CAPITAL FREED - Recalculating best opportunity...{Colors.ENDC}\n")
                                best_opportunity = find_best_opportunity(
                                    config=config,
                                    coinbase_client=coinbase_client,
                                    enabled_symbols=enabled_wallets,
                                    interval_seconds=INTERVAL_SECONDS,
                                    data_retention_hours=DATA_RETENTION_HOURS
                                )
                                if best_opportunity:
                                    min_score = market_rotation_config.get('min_opportunity_score', 50)
                                    if best_opportunity['score'] >= min_score:
                                        best_opportunity_symbol = best_opportunity['symbol']
                                        print(f"{Colors.GREEN}✅ NEW BEST OPPORTUNITY: {best_opportunity_symbol} (score: {best_opportunity['score']:.1f}){Colors.ENDC}")
                                        print(f"   Will enter {best_opportunity_symbol} when we reach it in this iteration\n")
                                    else:
                                        best_opportunity_symbol = None
                                        print(f"{Colors.YELLOW}⚠️  Best opportunity score {best_opportunity['score']:.1f} below minimum {min_score} - waiting{Colors.ENDC}\n")
                                else:
                                    best_opportunity_symbol = None
                                    print(f"{Colors.YELLOW}⚠️  No tradeable opportunities found - capital will remain idle{Colors.ENDC}\n")

                                # Skip the rest of sell logic since we already sold
                                print('\n')
                                continue
                            else:
                                print('STATUS: Trading disabled')

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

                                # Update core learnings based on trade outcome
                                from utils.trade_context import load_transaction_history
                                trade_outcome = {
                                    'profit': net_profit_after_all_costs_usd,
                                    'exit_trigger': 'stop_loss',
                                    'confidence_level': analysis.get('confidence_level', 'unknown'),
                                    'market_trend': analysis.get('market_trend', 'unknown')
                                }
                                transactions = load_transaction_history(symbol)
                                update_learnings_from_trade(symbol, trade_outcome, transactions)

                                delete_analysis_file(symbol)

                                # IMMEDIATE ROTATION: Recalculate best opportunity after exit
                                # This allows us to enter the next-best trade in the same iteration
                                if market_rotation_enabled:
                                    print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 CAPITAL FREED - Recalculating best opportunity...{Colors.ENDC}\n")
                                    best_opportunity = find_best_opportunity(
                                        config=config,
                                        coinbase_client=coinbase_client,
                                        enabled_symbols=enabled_wallets,
                                        interval_seconds=INTERVAL_SECONDS,
                                        data_retention_hours=DATA_RETENTION_HOURS
                                    )
                                    if best_opportunity:
                                        min_score = market_rotation_config.get('min_opportunity_score', 50)
                                        if best_opportunity['score'] >= min_score:
                                            best_opportunity_symbol = best_opportunity['symbol']
                                            print(f"{Colors.GREEN}✅ NEW BEST OPPORTUNITY: {best_opportunity_symbol} (score: {best_opportunity['score']:.1f}){Colors.ENDC}")
                                            print(f"   Will enter {best_opportunity_symbol} when we reach it in this iteration\n")
                                        else:
                                            best_opportunity_symbol = None
                                            print(f"{Colors.YELLOW}⚠️  Best opportunity score {best_opportunity['score']:.1f} below minimum {min_score} - waiting{Colors.ENDC}\n")
                                    else:
                                        best_opportunity_symbol = None
                                        print(f"{Colors.YELLOW}⚠️  No tradeable opportunities found - capital will remain idle{Colors.ENDC}\n")
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

                                # Update core learnings based on trade outcome
                                from utils.trade_context import load_transaction_history
                                trade_outcome = {
                                    'profit': net_profit_after_all_costs_usd,
                                    'exit_trigger': 'profit_target',
                                    'confidence_level': analysis.get('confidence_level', 'unknown'),
                                    'market_trend': analysis.get('market_trend', 'unknown')
                                }
                                transactions = load_transaction_history(symbol)
                                update_learnings_from_trade(symbol, trade_outcome, transactions)

                                delete_analysis_file(symbol)

                                # IMMEDIATE ROTATION: Recalculate best opportunity after exit
                                # This allows us to enter the next-best trade in the same iteration
                                if market_rotation_enabled:
                                    print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 CAPITAL FREED - Recalculating best opportunity...{Colors.ENDC}\n")
                                    best_opportunity = find_best_opportunity(
                                        config=config,
                                        coinbase_client=coinbase_client,
                                        enabled_symbols=enabled_wallets,
                                        interval_seconds=INTERVAL_SECONDS,
                                        data_retention_hours=DATA_RETENTION_HOURS
                                    )
                                    if best_opportunity:
                                        min_score = market_rotation_config.get('min_opportunity_score', 50)
                                        if best_opportunity['score'] >= min_score:
                                            best_opportunity_symbol = best_opportunity['symbol']
                                            print(f"{Colors.GREEN}✅ NEW BEST OPPORTUNITY: {best_opportunity_symbol} (score: {best_opportunity['score']:.1f}){Colors.ENDC}")
                                            print(f"   Will enter {best_opportunity_symbol} when we reach it in this iteration\n")
                                        else:
                                            best_opportunity_symbol = None
                                            print(f"{Colors.YELLOW}⚠️  Best opportunity score {best_opportunity['score']:.1f} below minimum {min_score} - waiting{Colors.ENDC}\n")
                                    else:
                                        best_opportunity_symbol = None
                                        print(f"{Colors.YELLOW}⚠️  No tradeable opportunities found - capital will remain idle{Colors.ENDC}\n")
                            else:
                                print('STATUS: Trading disabled')

                    print('\n')


                #
                #
                # ASSET PERFORMANCE SUMMARY: Show all assets ranked by P&L
                # Collect data for all assets
                asset_performance = []
                for summary_symbol in enabled_wallets:
                    summary_order = get_last_order_from_local_json_ledger(summary_symbol)
                    summary_order_type = detect_stored_coinbase_order_type(summary_order)
                    summary_has_position = summary_order_type in ['placeholder', 'buy']

                    summary_price = get_asset_price(coinbase_client, summary_symbol)

                    # Skip assets where price fetch failed
                    if summary_price is None:
                        continue

                    # Get wallet metrics to calculate cumulative P&L from transaction history
                    wallet_config = next((w for w in config.get('wallets', []) if w['symbol'] == summary_symbol), None)
                    if wallet_config:
                        summary_starting_capital = wallet_config['starting_capital_usd']
                    else:
                        summary_starting_capital = 3250.0  # Default fallback

                    summary_wallet_metrics = calculate_wallet_metrics(summary_symbol, summary_starting_capital)
                    pnl_pct = summary_wallet_metrics.get('percentage_gain', 0.0)
                    total_profit_usd = summary_wallet_metrics.get('total_profit', 0.0)
                    total_trades = 0
                    wins = 0

                    # Count total trades and wins from transaction history
                    from utils.trade_context import load_transaction_history
                    summary_transactions = load_transaction_history(summary_symbol)
                    total_trades = len(summary_transactions)
                    wins = len([t for t in summary_transactions if t.get('total_profit', 0) > 0])

                    # Determine status
                    status = "NO POSITION"
                    if summary_has_position:
                        if summary_order_type == 'placeholder':
                            status = "PENDING ORDER"
                        elif summary_order_type == 'buy':
                            order_data = summary_order.get('order', summary_order)
                            if 'average_filled_price' in order_data:
                                entry_price = float(order_data['average_filled_price'])
                                unrealized_pct = ((summary_price - entry_price) / entry_price) * 100
                                status = f"OPEN ({unrealized_pct:+.1f}%)"

                    asset_performance.append({
                        'symbol': summary_symbol,
                        'price': summary_price,
                        'pnl_pct': pnl_pct,
                        'total_profit_usd': total_profit_usd,
                        'status': status,
                        'has_position': summary_has_position,
                        'total_trades': total_trades,
                        'wins': wins
                    })

                # Sort by P&L percentage (highest to lowest), then by total profit USD as tie-breaker
                asset_performance.sort(key=lambda x: (x['pnl_pct'], x['total_profit_usd']), reverse=True)

                # Display header with actual count
                print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*100}")
                print(f"  📊 ASSET PERFORMANCE SUMMARY - {len(asset_performance)} ASSETS")
                print(f"{'='*100}{Colors.ENDC}")

                # Display ranked summary (one line per asset)
                for idx, asset in enumerate(asset_performance, 1):
                    pnl_color = Colors.GREEN if asset['pnl_pct'] > 0 else (Colors.RED if asset['pnl_pct'] < 0 else Colors.YELLOW)
                    position_indicator = "🔥" if asset['has_position'] else "  "

                    # Calculate win rate
                    if asset['total_trades'] > 0:
                        win_pct = (asset['wins'] / asset['total_trades']) * 100
                        win_rate_str = f" | {asset['wins']}/{asset['total_trades']} ({win_pct:.0f}%)"
                    else:
                        win_rate_str = f" | 0/0 (0%)"

                    profit_color = Colors.GREEN if asset['total_profit_usd'] > 0 else (Colors.RED if asset['total_profit_usd'] < 0 else Colors.YELLOW)
                    print(f"  {position_indicator} {idx:2d}. {asset['symbol']:10s} | {profit_color}${asset['total_profit_usd']:>+8.2f}{Colors.ENDC} | {pnl_color}P&L: {asset['pnl_pct']:>+6.2f}%{Colors.ENDC}{win_rate_str} | {asset['status']}")

                # Calculate and display lifetime total
                total_lifetime_profit = sum(asset['total_profit_usd'] for asset in asset_performance)
                lifetime_color = Colors.GREEN if total_lifetime_profit > 0 else (Colors.RED if total_lifetime_profit < 0 else Colors.YELLOW)
                print(f"{Colors.CYAN}{'-'*100}{Colors.ENDC}")
                print(f"  {Colors.BOLD}LIFETIME TOTAL:      | {lifetime_color}${total_lifetime_profit:>+8.2f}{Colors.ENDC}{Colors.BOLD}{Colors.ENDC}")
                print(f"{Colors.CYAN}{'='*100}{Colors.ENDC}\n")

                #
                #
                # ITERATION SUMMARY: Show status of all positions and next action
                if market_rotation_enabled:
                    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*100}")
                    print(f"  📋 ITERATION SUMMARY")
                    print(f"{'='*100}{Colors.ENDC}")

                    # Count active positions
                    current_active_positions = []
                    for symbol in enabled_wallets:
                        last_order = get_last_order_from_local_json_ledger(symbol)
                        last_order_type = detect_stored_coinbase_order_type(last_order)
                        if last_order_type in ['placeholder', 'buy']:
                            current_active_positions.append(symbol)

                    if current_active_positions:
                        print(f"  {Colors.GREEN}🔥 ACTIVE POSITION(S): {', '.join(current_active_positions)}{Colors.ENDC}")
                        print(f"  {Colors.CYAN}💰 Capital Status: DEPLOYED (managing position){Colors.ENDC}")
                        if best_opportunity and best_opportunity['score'] >= market_rotation_config.get('min_opportunity_score', 50):
                            print(f"  {Colors.YELLOW}🎯 Next Opportunity Queued: {best_opportunity['symbol']} (score: {best_opportunity['score']:.1f}){Colors.ENDC}")
                            print(f"  {Colors.YELLOW}⏭  Will trade immediately after current position exits{Colors.ENDC}")
                        else:
                            print(f"  {Colors.YELLOW}🔎 Monitoring {len(enabled_wallets)} assets for next opportunity{Colors.ENDC}")
                    else:
                        if best_opportunity_symbol:
                            print(f"  {Colors.GREEN}✅ Ready to Trade: {best_opportunity_symbol}{Colors.ENDC}")
                            print(f"  {Colors.CYAN}💰 Capital Status: READY ($3,000 available){Colors.ENDC}")
                        else:
                            print(f"  {Colors.YELLOW}⏸  No Strong Opportunities Currently{Colors.ENDC}")
                            print(f"  {Colors.CYAN}💰 Capital Status: IDLE (waiting for quality setup){Colors.ENDC}")
                            print(f"  {Colors.CYAN}🔎 Monitoring {len(enabled_wallets)} assets: {', '.join(enabled_wallets)}{Colors.ENDC}")

                    print(f"  {Colors.CYAN}⏰ Next scan in {check_interval_seconds/60:.0f} minutes{Colors.ENDC}")
                    print(f"{Colors.CYAN}{'='*100}{Colors.ENDC}\n")

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
        time.sleep(check_interval_seconds)

if __name__ == "__main__":
    while True:
        try:
            iterate_wallets(CHECK_INTERVAL_SECONDS, INTERVAL_SECONDS)
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
