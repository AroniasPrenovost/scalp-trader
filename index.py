import os
from dotenv import load_dotenv
from json import load
import json
import math
import time

# custom imports
from utils.email import send_email_notification
from utils.file_helpers import count_files_in_directory, append_crypto_data_to_file, get_property_values_from_crypto_file, cleanup_old_crypto_data, cleanup_old_screenshots, convert_products_to_dicts
from utils.price_helpers import calculate_percentage_from_min
from utils.time_helpers import print_local_time

# Coinbase helpers and define client
from utils.coinbase import get_coinbase_client, get_coinbase_order_by_order_id, place_market_buy_order, place_market_sell_order, get_asset_price, calculate_exchange_fee, save_order_data_to_local_json_ledger, get_last_order_from_local_json_ledger, reset_json_ledger_file, detect_stored_coinbase_order_type, save_transaction_record, get_current_fee_rates, cancel_order, clear_order_ledger
coinbase_client = get_coinbase_client()
# profit calculator for standardized profitability calculations
from utils.profit_calculator import calculate_net_profit_from_price_move

# Wallet metrics helpers
from utils.wallet_helpers import calculate_wallet_metrics

# Matplotlib for charting
from utils.matplotlib import plot_graph, plot_coinbase_metrics_chart

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
    DIM = '\033[2m'

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
# Allow 99% of expected data points to account for minor gaps (e.g., script restarts, network issues)
MINIMUM_DATA_POINTS = int(EXPECTED_DATA_POINTS * 0.99)

#
#
#
# Store the last error and manage number of errors before exiting program

LAST_EXCEPTION_ERROR = None
LAST_EXCEPTION_ERROR_COUNT = 0
MAX_LAST_EXCEPTION_ERROR_COUNT = 5

# Track when data collection operations were last run (will be loaded from file below)
LAST_DATA_COLLECTION_TIME = 0

#
# Data collection timestamp persistence
#
DATA_COLLECTION_TIMESTAMP_FILE = 'local-state/last_data_collection.json'

def save_last_data_collection_time(timestamp):
    """Save the last data collection timestamp to a file"""
    try:
        with open(DATA_COLLECTION_TIMESTAMP_FILE, 'w') as file:
            json.dump({'last_data_collection': timestamp}, file, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save data collection timestamp: {e}")

def load_last_data_collection_time():
    """Load the last data collection timestamp from file, return 0 if file doesn't exist"""
    try:
        if os.path.exists(DATA_COLLECTION_TIMESTAMP_FILE):
            with open(DATA_COLLECTION_TIMESTAMP_FILE, 'r') as file:
                data = json.load(file)
                return data.get('last_data_collection', 0)
        else:
            return 0
    except Exception as e:
        print(f"Warning: Failed to load data collection timestamp: {e}")
        return 0

#
# Screenshot cleanup timestamp persistence
#
SCREENSHOT_CLEANUP_TIMESTAMP_FILE = 'local-state/last_screenshot_cleanup.json'

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

def get_hours_since_last_sell(symbol):
    """
    Get the number of hours since the last sell for this asset.
    Returns None if no previous sells found.
    """
    from datetime import datetime, timezone
    from utils.wallet_helpers import load_transaction_history

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

# Load last data collection time from file (for crash recovery)
LAST_DATA_COLLECTION_TIME = load_last_data_collection_time()
if LAST_DATA_COLLECTION_TIME > 0:
    hours_since = (time.time() - LAST_DATA_COLLECTION_TIME) / 3600
    print(f"Loaded last data collection timestamp: {hours_since:.2f} hours ago\n")

# Load last screenshot cleanup time from file (for crash recovery)
LAST_SCREENSHOT_CLEANUP_TIME = load_last_screenshot_cleanup_time()
if LAST_SCREENSHOT_CLEANUP_TIME > 0:
    hours_since = (time.time() - LAST_SCREENSHOT_CLEANUP_TIME) / 3600
    print(f"Loaded last screenshot cleanup timestamp: {hours_since:.2f} hours ago\n")

def iterate_wallets(data_collection_interval_seconds):
    # Ensure screenshots directory exists
    os.makedirs('screenshots', exist_ok=True)

    while True:
        # Track iteration start time for precise interval timing
        iteration_start_time = time.time()

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
        global LAST_DATA_COLLECTION_TIME
        global LAST_SCREENSHOT_CLEANUP_TIME

        # Load config at the start of each iteration (fast operation)
        config = load_config('config.json')

        # Check if it's time to run data collection operations
        current_time = time.time()
        time_since_last_collection = current_time - LAST_DATA_COLLECTION_TIME
        should_run_data_collection = time_since_last_collection >= data_collection_interval_seconds

        if should_run_data_collection:
            print(f"{Colors.BOLD}{Colors.CYAN}⏰ Running data collection - APPENDING NEW PRICE DATA (last run: {time_since_last_collection/3600:.2f} hours ago){Colors.ENDC}\n")

            # Get taxes and Coinbase fees
            federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))
            fee_rates = get_current_fee_rates(coinbase_client)
            # NOTE: Using TAKER fees because we place MARKET orders (not limit orders)
            # Maker = adds liquidity to order book (lower fee, e.g., 0.4%) - used for limit orders
            # Taker = takes liquidity from order book (higher fee, e.g., 1.2%) - used for market orders
            coinbase_spot_taker_fee = fee_rates['taker_fee'] if fee_rates else 1.2 # Tier: 'Intro 1' taker fee
            coinbase_spot_maker_fee = fee_rates['maker_fee'] if fee_rates else 0.6 # Tier: 'Intro 1' maker fee (not used)

            # Extract wallet and trading settings from config
            enabled_wallets = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]
            min_profit_target_percentage = config.get('min_profit_target_percentage', 3.0)
            no_trade_refresh_hours = config.get('no_trade_refresh_hours', 1.0)
            cooldown_hours_after_sell = config.get('cooldown_hours_after_sell', 0)
            low_confidence_wait_hours = config.get('low_confidence_wait_hours', 1.0)
            medium_confidence_wait_hours = config.get('medium_confidence_wait_hours', 1.0)
            high_confidence_max_age_hours = config.get('high_confidence_max_age_hours', 2.0)

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
            # DATA COLLECTION OPERATIONS: Store price/volume data
            # This runs at the configured interval (see config.json interval_seconds)
            enable_all_coin_scanning = True
            if enable_all_coin_scanning:
                coinbase_data_directory = 'coinbase-data'

                # NEW STORAGE APPROACH: Append each crypto's data to its own file
                # Store data for each enabled crypto
                for coin in coinbase_data_dictionary:
                    product_id = coin['product_id']
                    # Create entry with core properties + momentum/volume indicators
                    data_entry = {
                        'timestamp': time.time(),
                        'product_id': product_id,
                        'price': coin.get('price'),
                        'volume_24h': coin.get('volume_24h'),
                        # NEW: Built-in momentum indicators from Coinbase
                        'price_percentage_change_24h': coin.get('price_percentage_change_24h'),
                        'volume_percentage_change_24h': coin.get('volume_percentage_change_24h'),
                        # NEW: Trading status flags (filter out disabled markets)
                        'trading_disabled': coin.get('trading_disabled', False),
                        'cancel_only': coin.get('cancel_only', False),
                        'post_only': coin.get('post_only', False),
                        'is_disabled': coin.get('is_disabled', False)
                    }
                    append_crypto_data_to_file(coinbase_data_directory, product_id, data_entry)
                print(f"✓ Appended 1 new data point for {len(coinbase_data_dictionary)} cryptos (next append in {INTERVAL_SAVE_DATA_EVERY_X_MINUTES:.0f} min)\n")

            # Update the last data collection timestamp and save to file
            LAST_DATA_COLLECTION_TIME = time.time()
            save_last_data_collection_time(LAST_DATA_COLLECTION_TIME)

        #
        # SCREENSHOT CLEANUP: Run periodically based on config
        #
        screenshot_config = config.get('screenshot', {})
        screenshot_cleanup_enabled = screenshot_config.get('enabled', True)
        screenshot_cleanup_interval_hours = 6  # Run cleanup every 6 hours

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
        # TRADING OPERATIONS: Check prices and execute trades
        # This runs at the same interval as data collection
        if should_run_data_collection:
            enable_all_coin_scanning = True
            if enable_all_coin_scanning:
                coinbase_data_directory = 'coinbase-data'

                if count_files_in_directory(coinbase_data_directory) < 1:
                    print('waiting for more data...\n')
                else:
                    # MARKET ROTATION: Find the single best opportunity across all enabled assets
                    market_rotation_config = config.get('market_rotation', {})
                    market_rotation_enabled = market_rotation_config.get('enabled', False)

                    best_opportunity_symbol = None  # Will hold the symbol we should trade (single best mode)
                    racing_opportunities = []       # Will hold multiple opportunities (order racing mode)
                    active_position_symbols = []    # Track which assets have active positions
                    pending_order_symbols = []      # Track which assets have pending orders (for racing mode)

                    # First, check for any existing active positions and pending orders
                    for symbol in enabled_wallets:
                        last_order = get_last_order_from_local_json_ledger(symbol)
                        last_order_type = detect_stored_coinbase_order_type(last_order)
                        if last_order_type in ['placeholder', 'buy']:
                            active_position_symbols.append(symbol)
                            # Check if it's a pending order (placeholder) or filled buy
                            if last_order_type == 'placeholder':
                                pending_order_symbols.append(symbol)
    
                    if market_rotation_enabled:
                        from utils.opportunity_scorer import find_best_opportunity, print_opportunity_report, score_opportunity
    
                        rotation_mode = market_rotation_config.get('mode', 'single_best_opportunity')
                        max_concurrent_orders = market_rotation_config.get('max_concurrent_orders', 5)
    
                        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*100}")
                        if rotation_mode == 'order_racing':
                            print(f"🏁 ORDER RACING MODE: Monitoring {len(enabled_wallets)} setups, entering best position and rotating to others")
                        else:
                            print(f"🔍 MARKET ROTATION: Scanning {len(enabled_wallets)} assets for best opportunity...")
    
                        if active_position_symbols:
                            if pending_order_symbols:
                                print(f"📊 PENDING ORDERS: {Colors.YELLOW}{', '.join(pending_order_symbols)}{Colors.CYAN} (waiting for fill)")
                            filled_positions = [s for s in active_position_symbols if s not in pending_order_symbols]
                            if filled_positions:
                                print(f"📊 FILLED POSITIONS: {Colors.GREEN}{', '.join(filled_positions)}{Colors.CYAN} (managing these)")
                            print(f"🔎 MONITORING: {', '.join([s for s in enabled_wallets if s not in active_position_symbols])} (scanning for next opportunity)")
                        else:
                            print(f"💰 NO ACTIVE POSITIONS - Capital ready to deploy")
                        print(f"{'='*100}{Colors.ENDC}\n")
    
                        # Find opportunities based on mode
                        min_score = market_rotation_config.get('min_score_for_entry', 50)
    
                        if rotation_mode == 'order_racing' and not active_position_symbols:
                            # Order racing mode: get multiple opportunities
                            racing_opportunities = find_best_opportunity(
                                config=config,
                                coinbase_client=coinbase_client,
                                enabled_symbols=enabled_wallets,
                                interval_seconds=INTERVAL_SECONDS,
                                data_retention_hours=DATA_RETENTION_HOURS,
                                min_score=min_score,
                                return_multiple=True,
                                max_opportunities=max_concurrent_orders,
                                entry_fee_pct=coinbase_spot_taker_fee,
                                exit_fee_pct=coinbase_spot_taker_fee,
                                tax_rate_pct=federal_tax_rate
                            )
                            best_opportunity = racing_opportunities[0] if racing_opportunities else None
                        else:
                            # Single best mode or we already have positions
                            best_opportunity = find_best_opportunity(
                                config=config,
                                coinbase_client=coinbase_client,
                                enabled_symbols=enabled_wallets,
                                interval_seconds=INTERVAL_SECONDS,
                                data_retention_hours=DATA_RETENTION_HOURS,
                                min_score=min_score,
                                entry_fee_pct=coinbase_spot_taker_fee,
                                exit_fee_pct=coinbase_spot_taker_fee,
                                tax_rate_pct=federal_tax_rate
                            )
    
                        # Optionally print detailed report
                        if market_rotation_config.get('print_opportunity_report', True):
                            # Score all opportunities for the report
                            all_opportunities = []
                            current_prices = {}
                            for symbol in enabled_wallets:
                                try:
                                    current_price = get_asset_price(coinbase_client, symbol)
                                    current_prices[symbol] = current_price
                                    coin_prices_list = get_property_values_from_crypto_file(
                                        coinbase_data_directory, symbol, 'price', max_age_hours=DATA_RETENTION_HOURS
                                    )
    
                                    # Calculate volatility metrics (will be 0 if no data)
                                    range_pct = 0
                                    if coin_prices_list and len(coin_prices_list) > 0:
                                        volatility_window_hours = 24
                                        volatility_data_points = int(volatility_window_hours / (INTERVAL_SECONDS / 3600))
                                        recent_prices = coin_prices_list[-volatility_data_points:] if len(coin_prices_list) >= volatility_data_points else coin_prices_list
                                        min_price = min(recent_prices)
                                        max_price = max(recent_prices)
                                        range_pct = calculate_percentage_from_min(min_price, max_price)
    
                                    # Score ALL symbols, even with insufficient data
                                    # score_opportunity handles the no-data case gracefully
                                    opp = score_opportunity(
                                        symbol=symbol,
                                        config=config,
                                        coinbase_client=coinbase_client,
                                        coin_prices_list=coin_prices_list or [],
                                        current_price=current_price,
                                        range_percentage_from_min=range_pct
                                    )
                                    all_opportunities.append(opp)
                                except Exception as e:
                                    print(f"  Error scoring {symbol}: {e}")
                                    continue
    
                            trading_capital = market_rotation_config.get('total_trading_capital_usd', 100)
                            print_opportunity_report(all_opportunities, best_opportunity, racing_opportunities, current_prices, coinbase_spot_taker_fee, federal_tax_rate, trading_capital)
    
                        if best_opportunity:
                            best_opportunity_symbol = best_opportunity['symbol']
                            min_score = market_rotation_config.get('min_score_for_entry', 50)
    
                            if best_opportunity['score'] >= min_score:
                                if active_position_symbols:
                                    # Get current price for distance calculation
                                    current_coin = next((c for c in coinbase_data_dictionary if c['product_id'] == best_opportunity_symbol), None)
                                    current_price = float(current_coin['price']) if current_coin else None
    
                                    # Calculate distance from entry
                                    distance_str = ""
                                    if current_price and best_opportunity['entry_price']:
                                        distance_pct = ((current_price - best_opportunity['entry_price']) / best_opportunity['entry_price']) * 100
                                        color = Colors.GREEN if distance_pct <= 0 else Colors.YELLOW
                                        distance_str = f" | Current: ${current_price:.4f} ({color}{distance_pct:+.2f}%{Colors.ENDC} from entry)"
    
                                    print(f"{Colors.BOLD}{Colors.GREEN}🎯 BEST NEXT OPPORTUNITY: {best_opportunity_symbol}{Colors.ENDC}")
                                    print(f"   Score: {best_opportunity['score']:.1f}/100 | Strategy: {best_opportunity['strategy'].replace('_', ' ').title()}")
                                    print(f"   Entry: ${best_opportunity['entry_price']:.4f} | Stop: ${best_opportunity['stop_loss']:.4f} | Target: ${best_opportunity['profit_target']:.4f}{distance_str}")
                                    print(f"   {Colors.YELLOW}⏸  Waiting for active position(s) to close: {', '.join(active_position_symbols)}{Colors.ENDC}")
                                    print(f"   {Colors.CYAN}→ Will trade {best_opportunity_symbol} immediately after exit{Colors.ENDC}\n")
                                    best_opportunity_symbol = None  # Don't enter new trade while position open
                                    racing_opportunities = []  # Clear racing opportunities
                                else:
                                    # Show appropriate message based on mode
                                    if rotation_mode == 'order_racing' and len(racing_opportunities) > 1:
                                        print(f"{Colors.BOLD}{Colors.GREEN}🏁 ORDER RACING: Placing orders on {len(racing_opportunities)} opportunities{Colors.ENDC}")
                                        for i, opp in enumerate(racing_opportunities, 1):
                                            # Get current price for this opportunity
                                            current_coin = next((c for c in coinbase_data_dictionary if c['product_id'] == opp['symbol']), None)
                                            current_price = float(current_coin['price']) if current_coin else None
    
                                            # Calculate distance from entry
                                            distance_str = ""
                                            if current_price and opp['entry_price']:
                                                distance_pct = ((current_price - opp['entry_price']) / opp['entry_price']) * 100
                                                color = Colors.GREEN if distance_pct <= 0 else Colors.YELLOW
                                                distance_str = f" | Current: ${current_price:.4f} ({color}{distance_pct:+.2f}%{Colors.ENDC})"
    
                                            print(f"   #{i}. {opp['symbol']} - Score: {opp['score']:.1f}/100 | Entry: ${opp['entry_price']:.4f} | Target: ${opp['profit_target']:.4f}{distance_str}")
                                        print(f"   {Colors.YELLOW}⚡ First order to fill wins - others will be auto-cancelled{Colors.ENDC}")
                                        print()
                                    else:
                                        # Get current price for distance calculation
                                        current_coin = next((c for c in coinbase_data_dictionary if c['product_id'] == best_opportunity_symbol), None)
                                        current_price = float(current_coin['price']) if current_coin else None
    
                                        # Calculate distance from entry
                                        distance_str = ""
                                        if current_price and best_opportunity['entry_price']:
                                            distance_pct = ((current_price - best_opportunity['entry_price']) / best_opportunity['entry_price']) * 100
                                            color = Colors.GREEN if distance_pct <= 0 else Colors.YELLOW
                                            distance_str = f" | Current: ${current_price:.4f} ({color}{distance_pct:+.2f}%{Colors.ENDC} from entry)"
    
                                        print(f"{Colors.BOLD}{Colors.GREEN}✅ TRADING NOW: {best_opportunity_symbol}{Colors.ENDC}")
                                        print(f"   Score: {best_opportunity['score']:.1f}/100 | Strategy: {best_opportunity['strategy'].replace('_', ' ').title()}")
                                        print(f"   Entry: ${best_opportunity['entry_price']:.4f} | Stop: ${best_opportunity['stop_loss']:.4f} | Target: ${best_opportunity['profit_target']:.4f}{distance_str}")
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
                                racing_opportunities = []  # Clear racing opportunities
                        else:
                            # No tradeable opportunities - message already printed by print_opportunity_report()
                            # print(f"{Colors.YELLOW}⚠️  No tradeable opportunities found - all assets have open positions or no valid setups")
                            # if active_position_symbols:
                            #     print(f"   {Colors.CYAN}✓ Continuing to manage active position(s): {', '.join(active_position_symbols)}{Colors.ENDC}")
                            # else:
                            #     print(f"   Waiting for market conditions to improve")
                            # print()
                            racing_opportunities = []  # Clear racing opportunities
                    else:
                        pass
    
                    for coin in coinbase_data_dictionary:
                        # set data from coinbase data
                        symbol = coin['product_id'] # 'BTC-USD', 'ETH-USD', etc..
    
                        # set config.json data
                        READY_TO_TRADE = False
                        STARTING_CAPITAL_USD = 0
                        for wallet in config['wallets']:
                            if symbol == wallet['symbol']:
                                READY_TO_TRADE = wallet['ready_to_trade']
                                STARTING_CAPITAL_USD = wallet['starting_capital_usd']
    
                        wallet_metrics = calculate_wallet_metrics(symbol, STARTING_CAPITAL_USD)
    
                        # Check if this asset has an active position FIRST (before formatting wallet metrics)
                        # Note: we check verbosity after we know if it's an open position
                        last_order = get_last_order_from_local_json_ledger(symbol, verbose=False)
                        last_order_type = detect_stored_coinbase_order_type(last_order)
                        has_open_position = last_order_type in ['placeholder', 'buy']
    
                        # ENHANCED LOGGING: Show clearly if this is an active trade or just monitoring
                        # Only show detailed logging for open positions or selected opportunities
                        is_racing_opportunity = any(opp['symbol'] == symbol for opp in racing_opportunities)
                        show_detailed_logs = has_open_position or (market_rotation_enabled and symbol == best_opportunity_symbol) or is_racing_opportunity
    
                        if has_open_position:
                            print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*100}")
                            print(f"  🔥 ACTIVE TRADE: {symbol} - Managing Open Position")
                            print(f"{'='*100}{Colors.ENDC}")
                            format_wallet_metrics(symbol, wallet_metrics)
                            print(f"{Colors.CYAN}📊 Monitoring other assets in background for next opportunity after exit{Colors.ENDC}\n")
                        # Silent monitoring for all opportunities - no verbose output
    
                        # MARKET ROTATION: Only ENTER trades on the best opportunity or racing opportunities
                        # But we still analyze every wallet to:
                        # 1. Manage existing open positions (sell logic)
                        # 2. Keep scoring updated for next opportunity selection
                        # 3. Provide visibility into all market conditions
                        should_allow_new_entry = True
                        if market_rotation_enabled:
                            # Allow entry if: symbol is best opportunity OR symbol is in racing opportunities
                            is_selected = (symbol == best_opportunity_symbol) or is_racing_opportunity
                            if not is_selected and not has_open_position:
                                should_allow_new_entry = False
                                if show_detailed_logs:
                                    if best_opportunity_symbol:
                                        print(f"{Colors.CYAN}💡 Analyzing {symbol} (best opportunity: {best_opportunity_symbol} - will only enter selected opportunities){Colors.ENDC}\n")
                                    elif racing_opportunities:
                                        racing_symbols = ', '.join([opp['symbol'] for opp in racing_opportunities])
                                        print(f"{Colors.CYAN}💡 Analyzing {symbol} (racing opportunities: {racing_symbols}){Colors.ENDC}\n")
    
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
    
                        # Manage order data (order types, order info, etc.) in local ledger files
                        # Note: last_order and last_order_type already retrieved above (before volatility check)
                        entry_price = 0
                        # print(f"[MAIN] Last order type detected: '{last_order_type}'")
    
                        # ITERATION SCREENSHOTS: Take screenshot if enabled (moved before analysis check)
                        screenshot_config = config.get('screenshot', {})
                        if screenshot_config.get('enabled', False):
                            # Use ALL available data points for iteration screenshots
                            if coin_prices_LIST and len(coin_prices_LIST) > 0:
                                try:
                                    iteration_chart_min = min(coin_prices_LIST)
                                    iteration_chart_max = max(coin_prices_LIST)
                                    iteration_chart_range_pct = calculate_percentage_from_min(iteration_chart_min, iteration_chart_max)
                                    # print(f"DEBUG: Calculated chart params - min: {iteration_chart_min}, max: {iteration_chart_max}, range: {iteration_chart_range_pct}%")
    
                                    # Determine entry price for chart (if in position)
                                    chart_entry_price = entry_price if last_order_type in ['placeholder', 'buy'] else 0
    
                                    # Calculate timeframe label based on total data span
                                    total_hours = (len(coin_prices_LIST) * INTERVAL_SAVE_DATA_EVERY_X_MINUTES) / 60
                                    if total_hours >= 8760:  # 1 year or more
                                        timeframe_label = f"{int(total_hours / 8760)}y"
                                    elif total_hours >= 720:  # 1 month or more
                                        timeframe_label = f"{int(total_hours / 720)}mo"
                                    elif total_hours >= 168:  # 1 week or more
                                        timeframe_label = f"{int(total_hours / 168)}w"
                                    elif total_hours >= 24:  # 1 day or more
                                        timeframe_label = f"{int(total_hours / 24)}d"
                                    else:
                                        timeframe_label = f"{int(total_hours)}h"
    
                                    # print(f"DEBUG: Calling plot_graph for {symbol} with {len(coin_prices_LIST)} data points, timeframe: {timeframe_label}")
                                    iteration_screenshot_path = plot_graph(
                                        time.time(),
                                        INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                        symbol,
                                        coin_prices_LIST,
                                        iteration_chart_min,
                                        iteration_chart_max,
                                        iteration_chart_range_pct,
                                        chart_entry_price,
                                        analysis=None,  # Don't include analysis on iteration screenshots
                                        event_type=None,
                                        screenshot_type='iteration',
                                        timeframe_label=timeframe_label
                                    )
    
                                    print(f"DEBUG: plot_graph returned: {iteration_screenshot_path}")
                                    if iteration_screenshot_path:
                                        print(f"📸 Iteration screenshot saved: {iteration_screenshot_path}")
                                    else:
                                        pass
                                        # print(f"DEBUG: plot_graph returned None for {symbol}")
    
                                    # COINBASE METRICS SCREENSHOT: Generate correlation chart
                                    # Get all Coinbase metrics data for correlation analysis
                                    coin_volume_24h_LIST = get_property_values_from_crypto_file(
                                        coinbase_data_directory, symbol, 'volume_24h', max_age_hours=DATA_RETENTION_HOURS
                                    )
                                    coin_price_pct_change_24h_LIST = get_property_values_from_crypto_file(
                                        coinbase_data_directory, symbol, 'price_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                    )
                                    coin_volume_pct_change_24h_LIST = get_property_values_from_crypto_file(
                                        coinbase_data_directory, symbol, 'volume_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                    )
    
                                    # Generate metrics correlation chart if we have data
                                    print(f"DEBUG: Metrics data lengths - volume_24h: {len(coin_volume_24h_LIST) if coin_volume_24h_LIST else 0}, price_pct: {len(coin_price_pct_change_24h_LIST) if coin_price_pct_change_24h_LIST else 0}, volume_pct: {len(coin_volume_pct_change_24h_LIST) if coin_volume_pct_change_24h_LIST else 0}")
                                    if (coin_volume_24h_LIST and coin_price_pct_change_24h_LIST and
                                        coin_volume_pct_change_24h_LIST and len(coin_prices_LIST) > 0):
    
                                        print(f"DEBUG: Calling plot_coinbase_metrics_chart for {symbol}")
                                        metrics_screenshot_path = plot_coinbase_metrics_chart(
                                            time.time(),
                                            INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                            symbol,
                                            coin_prices_LIST,
                                            coin_volume_24h_LIST,
                                            coin_price_pct_change_24h_LIST,
                                            coin_volume_pct_change_24h_LIST,
                                            event_type=None,
                                            screenshot_type='iteration',
                                            timeframe_label=timeframe_label
                                        )
    
                                        print(f"DEBUG: plot_coinbase_metrics_chart returned: {metrics_screenshot_path}")
                                        if metrics_screenshot_path:
                                            print(f"📊 Metrics correlation screenshot saved: {metrics_screenshot_path}")
                                        else:
                                            print(f"DEBUG: plot_coinbase_metrics_chart returned None for {symbol}")
                                    else:
                                        print(f"DEBUG: Skipping metrics screenshot - missing data for {symbol}")
                                except Exception as e:
                                    print(f"{Colors.RED}Error generating screenshots for {symbol}: {e}{Colors.ENDC}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print(f"DEBUG: Skipping screenshot for {symbol} - no price data available")
    
                        #
                        #
                        #
                        # Analysis for trading parameters
                        # In market rotation mode, analysis comes from opportunity scorer or order ledger
                        analysis = None
    
                        # Only proceed with trading if we have a valid analysis
                        # EXCEPTION: In market rotation mode, selected opportunities use strategy-based analysis
                        # EXCEPTION: Symbols with open positions ALWAYS need to be processed (for sell logic and placeholder fills)
                        is_selected_for_rotation = market_rotation_enabled and (symbol == best_opportunity_symbol or is_racing_opportunity)
    
                        # CRITICAL: If we have an open position (placeholder or buy), load analysis from ledger NOW
                        # This must happen BEFORE the skip check below, or pending orders will never be processed!
                        if not analysis and last_order_type in ['placeholder', 'buy']:
                            original_analysis = last_order.get('original_analysis')
                            if original_analysis:
                                if show_detailed_logs:
                                    print(f"✓ Loading LOCKED analysis from ledger for position management (generated at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(original_analysis.get('analyzed_at', 0)))})")
                                analysis = original_analysis
                            else:
                                if show_detailed_logs:
                                    print(f"⚠️  Warning: Open position but no original_analysis in ledger - will try to process anyway")
    
                        if not analysis and not is_selected_for_rotation and not has_open_position:
                            # Only print individual skip messages if opportunity report is disabled
                            # (otherwise the report already shows all symbols and their status)
                            if not market_rotation_config.get('print_opportunity_report', True):
                                print(f"No market analysis available for {symbol}. Skipping trading logic.")
                                print('\n')
                            continue
    
                        # If we don't have analysis but we ARE selected for rotation, create a placeholder
                        # with the required fields populated from the opportunity data
                        if not analysis and is_selected_for_rotation:
                            # Get the opportunity data (either best_opportunity or from racing list)
                            opp_data = best_opportunity if symbol == best_opportunity_symbol else next((opp for opp in racing_opportunities if opp['symbol'] == symbol), None)
    
                            if opp_data:
                                analysis = {
                                    'analyzed_at': time.time(),
                                    'source': 'market_rotation',
                                    'strategy': opp_data.get('strategy', 'momentum_scalping'),
                                    'strategy_type': opp_data.get('strategy_type', 'unknown'),
                                    'entry_price': opp_data.get('entry_price'),  # For scalping mode
                                    'stop_loss': opp_data.get('stop_loss'),       # For scalping mode
                                    'profit_target': opp_data.get('profit_target'),  # For scalping mode
                                    'confidence_level': opp_data.get('confidence', 'high'),
                                    'recommendation': 'buy',
                                    'reasoning': opp_data.get('reasoning', '')
                                }
                            else:
                                # Fallback if opportunity data is missing
                                analysis = {
                                    'analyzed_at': time.time(),
                                    'source': 'market_rotation',
                                    'note': 'Selected by momentum scalping strategy'
                                }
    
                        # Note: Analysis loading for open positions now happens earlier (before skip check)
                        # to ensure placeholder orders and open positions are always processed
                        # This block kept for safety in case analysis wasn't loaded above
                        if last_order_type in ['placeholder', 'buy'] and not analysis:
                            original_analysis = last_order.get('original_analysis')
                            if original_analysis:
                                print(f"✓ Loading LOCKED original analysis (fallback - should have loaded earlier)")
                                analysis = original_analysis
                            else:
                                print(f"⚠️  Critical: No original analysis found in ledger for open position!")
    
                        # Set trading parameters - SCALPING MODE ONLY
                        # All parameters come from momentum scalping strategy
                        if analysis and 'entry_price' in analysis:
                            BUY_AT_PRICE = analysis.get('entry_price')
                            STOP_LOSS_PRICE = analysis.get('stop_loss')
                            # Calculate profit percentage from profit_target price
                            profit_target_price = analysis.get('profit_target')
                            if profit_target_price and BUY_AT_PRICE:
                                PROFIT_PERCENTAGE = ((profit_target_price - BUY_AT_PRICE) / BUY_AT_PRICE) * 100
                            else:
                                PROFIT_PERCENTAGE = 0.8  # Default scalping target
                            TRADE_RECOMMENDATION = 'buy'
                            CONFIDENCE_LEVEL = analysis.get('confidence_level', 'high')
                            if show_detailed_logs:
                                print('--- SCALPING STRATEGY (ALGO) ---')
                                print(f"entry: ${BUY_AT_PRICE:.4f}, stop_loss: ${STOP_LOSS_PRICE:.4f}, target: {PROFIT_PERCENTAGE:.2f}%")
                                print(f"strategy: {analysis.get('strategy_type', 'unknown')}, confidence: {CONFIDENCE_LEVEL}")
                        else:
                            BUY_AT_PRICE = None
                            STOP_LOSS_PRICE = None
                            PROFIT_PERCENTAGE = None
                            TRADE_RECOMMENDATION = 'hold'
                            CONFIDENCE_LEVEL = 'low'
    
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
                                    print(f"{Colors.GREEN}✓ ORDER FILLED!{Colors.ENDC}")
    
                                    # ORDER RACING: If this was a racing order, cancel all other pending orders
                                    if rotation_mode == 'order_racing' and pending_order_symbols and len(pending_order_symbols) > 1:
                                        print(f"\n{Colors.BOLD}{Colors.YELLOW}🏁 RACING ORDER FILLED - Cancelling other pending orders...{Colors.ENDC}")
                                        for racing_symbol in pending_order_symbols:
                                            if racing_symbol != symbol:  # Don't try to cancel the order that just filled
                                                try:
                                                    racing_last_order = get_last_order_from_local_json_ledger(racing_symbol)
                                                    if racing_last_order:
                                                        # Extract order ID
                                                        racing_order_id = None
                                                        if 'order_id' in racing_last_order:
                                                            racing_order_id = racing_last_order['order_id']
                                                        elif 'success_response' in racing_last_order and 'order_id' in racing_last_order['success_response']:
                                                            racing_order_id = racing_last_order['success_response']['order_id']
                                                        elif 'response' in racing_last_order and 'order_id' in racing_last_order['response']:
                                                            racing_order_id = racing_last_order['response']['order_id']
    
                                                        if racing_order_id:
                                                            print(f"  Cancelling {racing_symbol} order {racing_order_id}...")
                                                            cancel_result = cancel_order(coinbase_client, racing_order_id)
                                                            if cancel_result:
                                                                # Clear the ledger for this symbol since order was cancelled
                                                                clear_order_ledger(racing_symbol)
                                                                print(f"  {Colors.GREEN}✓ Cancelled {racing_symbol} racing order{Colors.ENDC}")
                                                            else:
                                                                print(f"  {Colors.YELLOW}⚠️  Failed to cancel {racing_symbol} order - may have already filled{Colors.ENDC}")
                                                except Exception as e:
                                                    print(f"  {Colors.RED}Error cancelling {racing_symbol} order: {e}{Colors.ENDC}")
                                        print(f"{Colors.GREEN}✓ Racing order cleanup complete - {symbol} won the race!{Colors.ENDC}\n")
    
                                    # Preserve the original analysis and screenshot from the placeholder before replacing
                                    # These need to persist until the sell transaction is recorded
                                    if 'original_analysis' in last_order:
                                        full_order_dict['original_analysis'] = last_order['original_analysis']
                                        print('✓ Preserved original analysis from placeholder order')
                                    if 'buy_screenshot_path' in last_order:
                                        full_order_dict['buy_screenshot_path'] = last_order['buy_screenshot_path']
                                        print('✓ Preserved buy screenshot path from placeholder order')
    
                                    # POST-FILL ADJUSTMENT: Check if actual fill price differs significantly from recommended price
                                    if 'original_analysis' in full_order_dict:
                                        original_analysis = full_order_dict['original_analysis']
                                        recommended_entry_price = original_analysis.get('buy_in_price')
                                        actual_fill_price = float(full_order_dict.get('average_filled_price', 0))
    
                                        if recommended_entry_price and actual_fill_price > 0:
                                            fill_delta_pct = abs((actual_fill_price - recommended_entry_price) / recommended_entry_price) * 100
    
                                            print(f"Fill price check: Recommended ${recommended_entry_price:.2f}, filled at ${actual_fill_price:.2f} (delta: {fill_delta_pct:.2f}%)")
    
                                            # If fill price differs, apply percentage-based adjustment to maintain risk/reward ratio
                                            if fill_delta_pct >= 0.5:
                                                print(f"Fill delta ({fill_delta_pct:.2f}%) - applying percentage-based adjustment...")
    
                                                # Calculate original risk/reward percentages
                                                original_stop_loss = original_analysis.get('stop_loss')
                                                original_sell_price = original_analysis.get('sell_price')
    
                                                if original_stop_loss and original_sell_price:
                                                    # Calculate percentage distances from recommended entry
                                                    stop_loss_pct = ((recommended_entry_price - original_stop_loss) / recommended_entry_price)
                                                    profit_target_pct = ((original_sell_price - recommended_entry_price) / recommended_entry_price)
    
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
                                    file_name = f"coinbase-orders/{symbol}_orders.json"
                                    with open(file_name, 'w') as file:
                                        json.dump([full_order_dict], file, indent=4)
                                    print('STATUS: Updated ledger with filled order data (analysis preserved until sell)')
                                # Check if order has expired or is still pending
                                elif order_status in ['OPEN', 'PENDING', 'QUEUED']:
                                    # Market orders execute immediately or fail - if still pending, just wait
                                    print('STATUS: Still processing pending order')
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
                                print(f"{Colors.YELLOW}   Please review the ledger file: coinbase-orders/{symbol}_orders.json{Colors.ENDC}\n")
                                continue
    
                            # Check all buy conditions
                            # Note: If market rotation is enabled and this is the selected best opportunity or racing opportunity,
                            # we trust the opportunity scorer's strategy validation
    
                            is_selected_opportunity = market_rotation_enabled and symbol == best_opportunity_symbol
                            should_execute_buy = False  # Track if we should execute the buy
                            current_opportunity = None  # Will hold the opportunity data for this symbol
    
                            if not should_allow_new_entry:
                                if show_detailed_logs:
                                    print(f"\n{Colors.YELLOW}STATUS: {symbol} is not a selected opportunity - skipping NEW entry{Colors.ENDC}")
                                # When market rotation is enabled, ONLY selected opportunities can trigger new buys
                            elif is_racing_opportunity:
                                # This is a racing opportunity - find its data
                                current_opportunity = next((opp for opp in racing_opportunities if opp['symbol'] == symbol), None)
                                if current_opportunity and current_opportunity.get('signal') == 'buy':
                                    min_score = market_rotation_config.get('min_score_for_entry', 50)
                                    if current_opportunity.get('score', 0) >= min_score:
                                        print(f"{Colors.BOLD}{Colors.CYAN}Strategy: {current_opportunity['strategy'].replace('_', ' ').title()}{Colors.ENDC}")
                                        print(f"Score: {current_opportunity['score']:.1f}/100 | Confidence: {current_opportunity['confidence'].upper()}")
                                        should_execute_buy = True
                                    else:
                                        print(f"\n{Colors.YELLOW}STATUS: Racing opportunity {symbol} score {current_opportunity['score']:.1f} below minimum {min_score} - skipping{Colors.ENDC}")
                                else:
                                    print(f"\n{Colors.YELLOW}STATUS: Racing opportunity {symbol} no longer has valid buy signal - skipping{Colors.ENDC}")
                            elif is_selected_opportunity:
                                # This is the best opportunity selected by the scorer - it already validated the strategy
                                # Just verify it's actually a quality trade with a valid signal and meets minimum score
                                current_opportunity = best_opportunity
                                if best_opportunity and best_opportunity.get('signal') == 'buy':
                                    min_score = market_rotation_config.get('min_score_for_entry', 50)
                                    if best_opportunity.get('score', 0) >= min_score:
                                        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}")
                                        print(f"  ✓ BEST OPPORTUNITY - EXECUTING BUY")
                                        print(f"{'='*60}{Colors.ENDC}")
                                        print(f"{Colors.BOLD}{Colors.CYAN}Strategy: {best_opportunity['strategy'].replace('_', ' ').title()}{Colors.ENDC}")
                                        print(f"Score: {best_opportunity['score']:.1f}/100 | Confidence: {best_opportunity['confidence'].upper()}")
                                        should_execute_buy = True
                                    else:
                                        print(f"\n{Colors.YELLOW}STATUS: Selected opportunity {symbol} score {best_opportunity['score']:.1f} below minimum {min_score} - skipping{Colors.ENDC}")
                                else:
                                    print(f"\n{Colors.YELLOW}STATUS: Selected opportunity {symbol} no longer has valid buy signal - skipping{Colors.ENDC}")
    
                            # Execute buy if conditions met
                            if should_execute_buy:
                                # Show which strategy triggered the buy
                                if is_selected_opportunity and best_opportunity:
                                    # Market rotation path - show the strategy that scored highest
                                    strategy_name = best_opportunity['strategy'].replace('_', ' ').title()
                                    print(f"{Colors.GREEN}✓ {strategy_name} Strategy Selected (Score: {best_opportunity['score']:.1f}/100){Colors.ENDC}")
                                    if best_opportunity.get('reasoning'):
                                        print(f"{Colors.CYAN}Reasoning: {best_opportunity['reasoning']}{Colors.ENDC}")
    
                                print(f"{Colors.GREEN}Current Market Price: ${current_price:.2f}{Colors.ENDC}\n")
    
                                if READY_TO_TRADE:
                                    # Determine buy amount and target price based on strategy mode
                                    buy_amount = None
                                    target_price = None
    
                                    # Scalping strategy: use opportunity data and rotation capital
                                    if current_opportunity and is_selected_opportunity:
                                        buy_amount = market_rotation_config.get('total_trading_capital_usd', STARTING_CAPITAL_USD)
                                        target_price = current_opportunity.get('entry_price', current_price)
                                        print(f"Using buy amount: ${buy_amount} (from scalping capital)")
                                        print(f"Target entry price: ${target_price:.4f} (from opportunity scorer)")
    
                                    if buy_amount and target_price:
                                        # Calculate shares accounting for exchange fees
                                        # We want: (subtotal + fee) = buy_amount
                                        # Where: fee = (fee_rate/100) * subtotal
                                        # So: subtotal * (1 + fee_rate/100) = buy_amount
                                        # Therefore: subtotal = buy_amount / (1 + fee_rate/100)
    
                                        fee_multiplier = 1 + (coinbase_spot_taker_fee / 100)
                                        subtotal_amount = buy_amount / fee_multiplier
                                        shares_calculation = subtotal_amount / current_price
    
                                        if shares_calculation >= 1:
                                            shares_to_buy = math.floor(shares_calculation)  # Round down to whole shares
                                            estimated_subtotal = shares_to_buy * current_price
                                            estimated_fee = calculate_exchange_fee(current_price, shares_to_buy, coinbase_spot_taker_fee)
                                            estimated_total = estimated_subtotal + estimated_fee
                                            print(f"Calculated shares to buy: {shares_to_buy} whole shares")
                                            print(f"  Subtotal: ${estimated_subtotal:.2f} + Fee: ${estimated_fee:.2f} = Total: ${estimated_total:.2f} (target: ${buy_amount:.2f})")
                                        else:
                                            # Round fractional shares to 8 decimal places (satoshi precision)
                                            shares_to_buy = round(shares_calculation, 8)
                                            estimated_subtotal = shares_to_buy * current_price
                                            estimated_fee = calculate_exchange_fee(current_price, shares_to_buy, coinbase_spot_taker_fee)
                                            estimated_total = estimated_subtotal + estimated_fee
                                            print(f"Calculated shares to buy: {shares_to_buy} fractional shares")
                                            print(f"  Subtotal: ${estimated_subtotal:.2f} + Fee: ${estimated_fee:.2f} = Total: ${estimated_total:.2f} (target: ${buy_amount:.2f})")
    
                                        if shares_to_buy > 0:
                                            # Determine entry tolerance based on strategy type
                                            # Different strategies have different entry requirements:
                                            strategy_type = current_opportunity.get('strategy_type') if current_opportunity else None
    
                                            # CONSERVATIVE slippage limits to avoid overpaying
                                            if strategy_type == 'breakout':
                                                # Breakouts move up - allow small slippage (0.3% max)
                                                entry_tolerance_pct = 0.3
                                            elif strategy_type == 'consolidation_break':
                                                # Consolidation breaks can move fast - allow small slippage (0.3% max)
                                                entry_tolerance_pct = 0.3
                                            elif strategy_type == 'support_bounce':
                                                # Support bounces - STRICT entry (only at or below target)
                                                entry_tolerance_pct = 0.0
                                            else:
                                                # Default: very conservative
                                                entry_tolerance_pct = 0.1
    
                                            max_entry_price = target_price * (1 + entry_tolerance_pct / 100)
    
                                            # Execute market order when price is within tolerance
                                            if current_price <= max_entry_price:
                                                if current_price <= target_price:
                                                    print(f"{Colors.GREEN}✓ Price at/below target! Current: ${current_price:.4f} <= Target: ${target_price:.4f}{Colors.ENDC}")
                                                else:
                                                    slippage_pct = ((current_price - target_price) / target_price) * 100
                                                    print(f"{Colors.GREEN}✓ Executing with {slippage_pct:.2f}% slippage (within {entry_tolerance_pct}% tolerance for {strategy_type}){Colors.ENDC}")
                                                    print(f"   Current: ${current_price:.4f} | Target: ${target_price:.4f} | Max: ${max_entry_price:.4f}{Colors.ENDC}")
    
                                                # Generate chart snapshot for trade documentation (only when buy is executed)
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
                                                    target_price,  # Use target price as entry price for chart
                                                    analysis=analysis if analysis else None,
                                                    event_type='buy'
                                                )
    
                                                # Generate Coinbase metrics correlation chart for buy event
                                                buy_coin_volume_24h = get_property_values_from_crypto_file(
                                                    coinbase_data_directory, symbol, 'volume_24h', max_age_hours=DATA_RETENTION_HOURS
                                                )
                                                buy_coin_price_pct_change_24h = get_property_values_from_crypto_file(
                                                    coinbase_data_directory, symbol, 'price_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                                )
                                                buy_coin_volume_pct_change_24h = get_property_values_from_crypto_file(
                                                    coinbase_data_directory, symbol, 'volume_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                                )
    
                                                # Align metrics data with price data window
                                                if (buy_coin_volume_24h and buy_coin_price_pct_change_24h and
                                                    buy_coin_volume_pct_change_24h):
                                                    buy_metrics_volume = buy_coin_volume_24h[-buy_chart_data_points:] if len(buy_coin_volume_24h) > buy_chart_data_points else buy_coin_volume_24h
                                                    buy_metrics_price_pct = buy_coin_price_pct_change_24h[-buy_chart_data_points:] if len(buy_coin_price_pct_change_24h) > buy_chart_data_points else buy_coin_price_pct_change_24h
                                                    buy_metrics_volume_pct = buy_coin_volume_pct_change_24h[-buy_chart_data_points:] if len(buy_coin_volume_pct_change_24h) > buy_chart_data_points else buy_coin_volume_pct_change_24h
    
                                                    buy_metrics_screenshot_path = plot_coinbase_metrics_chart(
                                                        time.time(),
                                                        INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                                        symbol,
                                                        buy_chart_prices,
                                                        buy_metrics_volume,
                                                        buy_metrics_price_pct,
                                                        buy_metrics_volume_pct,
                                                        event_type='buy'
                                                    )
                                                    if buy_metrics_screenshot_path:
                                                        print(f"📊 Buy metrics chart: {buy_metrics_screenshot_path}")
    
                                                print(f"Placing MARKET buy order for {shares_to_buy} shares at ${current_price:.4f}")
                                                place_market_buy_order(coinbase_client, symbol, shares_to_buy)
                                            else:
                                                # Price is beyond acceptable slippage - wait for better entry
                                                price_diff = ((current_price - target_price) / target_price) * 100
                                                print(f"{Colors.YELLOW}⏳ Price beyond tolerance: Current ${current_price:.4f} is {price_diff:+.2f}% vs target ${target_price:.4f}{Colors.ENDC}")
                                                print(f"   Max acceptable: ${max_entry_price:.4f} (tolerance: {entry_tolerance_pct}% for {strategy_type})")
                                                print(f"   Waiting for price to come within range...")
                                                # Don't place any order yet - just continue monitoring
                                                print(f"   Continuing to monitor {symbol}...")
                                                print('\n')
                                                continue
    
                                            # Store screenshot path AND original analysis for later use in transaction record
                                            # This will be retrieved from the ledger when we sell
                                            # IMPORTANT: Store the original analysis NOW to prevent it from being overwritten
                                            last_order = get_last_order_from_local_json_ledger(symbol)
                                            if last_order:
                                                last_order['buy_screenshot_path'] = buy_screenshot_path
                                                # Store analysis if available, otherwise store opportunity data
                                                if analysis:
                                                    last_order['original_analysis'] = analysis.copy()  # Store the analysis that drove this buy decision
                                                elif current_opportunity:
                                                    # Store opportunity data as pseudo-analysis for market rotation mode
                                                    last_order['original_analysis'] = {
                                                        'strategy': current_opportunity.get('strategy'),
                                                        'strategy_type': current_opportunity.get('strategy_type'),
                                                        'score': current_opportunity.get('score'),
                                                        'confidence_level': current_opportunity.get('confidence'),
                                                        'reasoning': current_opportunity.get('reasoning'),
                                                        'entry_price': current_opportunity.get('entry_price'),
                                                        'stop_loss': current_opportunity.get('stop_loss'),
                                                        'profit_target': current_opportunity.get('profit_target'),
                                                        'risk_reward_ratio': current_opportunity.get('risk_reward_ratio'),
                                                        'buy_amount_usd': buy_amount  # Store the actual buy amount used
                                                    }
                                                # Re-save the ledger with both the screenshot path and analysis
                                                import json
                                                file_name = f"coinbase-orders/{symbol}_orders.json"
                                                with open(file_name, 'w') as file:
                                                    json.dump([last_order], file, indent=4)
                                                print(f"✓ Stored buy screenshot and trade data in ledger")
                                        else:
                                            print(f"STATUS: Buy amount ${buy_amount} must be greater than 0")
                                    else:
                                        print("STATUS: No buy amount or target price available - skipping trade")
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
    
                            # Use atomic profit calculator for accurate, standardized calculations
                            profit_calc = calculate_net_profit_from_price_move(
                                entry_price=entry_price,
                                exit_price=current_price,
                                shares=number_of_shares,
                                entry_fee_pct=coinbase_spot_taker_fee,
                                exit_fee_pct=coinbase_spot_taker_fee,
                                tax_rate_pct=federal_tax_rate,
                                cost_basis_usd=entry_position_value_after_fees  # Use actual cost basis from order
                            )
    
                            print(f"\n{Colors.BOLD}{'='*80}")
                            print(f"📊 PROFITABILITY BREAKDOWN - {symbol}")
                            print(f"{'='*80}{Colors.ENDC}\n")
    
                            # Position Details
                            print(f"{Colors.BOLD}POSITION DETAILS:{Colors.ENDC}")
                            print(f"  Shares: {number_of_shares:.8f}")
                            print(f"  Entry Price: ${entry_price:.4f}")
                            print(f"  Current Price: ${current_price:.4f}")
                            print(f"  Price Change: {profit_calc['price_change_pct']:+.2f}%")
                            print()
    
                            # Entry Costs
                            print(f"{Colors.BOLD}ENTRY COSTS:{Colors.ENDC}")
                            print(f"  Entry Value: ${profit_calc['entry_value_usd']:.2f}")
                            print(f"    ({number_of_shares:.8f} shares × ${entry_price:.4f})")
                            print(f"  Entry Fee ({coinbase_spot_taker_fee}%): ${profit_calc['entry_fee_usd']:.2f}")
                            print(f"  {Colors.YELLOW}Total Cost Basis: ${profit_calc['cost_basis_usd']:.2f}{Colors.ENDC}")
                            print()
    
                            # Current Value
                            print(f"{Colors.BOLD}CURRENT VALUE:{Colors.ENDC}")
                            print(f"  Market Value: ${profit_calc['exit_value_usd']:.2f}")
                            print(f"    ({number_of_shares:.8f} shares × ${current_price:.4f})")
                            print()
    
                            # Exit Costs (if sold now)
                            print(f"{Colors.BOLD}EXIT COSTS (if sold now):{Colors.ENDC}")
                            print(f"  Exit Fee ({coinbase_spot_taker_fee}%): ${profit_calc['exit_fee_usd']:.2f}")
                            print(f"  Proceeds After Fee: ${profit_calc['exit_proceeds_usd']:.2f}")
                            print()
    
                            # Profit Before Tax
                            gross_profit_color = Colors.GREEN if profit_calc['gross_profit_usd'] >= 0 else Colors.RED
                            print(f"{Colors.BOLD}PROFIT (before tax):{Colors.ENDC}")
                            print(f"  {gross_profit_color}Gross Profit: ${profit_calc['gross_profit_usd']:+.2f}{Colors.ENDC}")
                            print(f"    (${profit_calc['exit_proceeds_usd']:.2f} proceeds - ${profit_calc['cost_basis_usd']:.2f} cost)")
                            print()
    
                            # Taxes
                            print(f"{Colors.BOLD}TAXES:{Colors.ENDC}")
                            if profit_calc['capital_gain_usd'] > 0:
                                print(f"  Capital Gain: ${profit_calc['capital_gain_usd']:.2f}")
                                print(f"  Tax Rate: {federal_tax_rate}%")
                                print(f"  Tax Owed: ${profit_calc['tax_usd']:.2f}")
                            else:
                                print(f"  Capital Gain: ${profit_calc['capital_gain_usd']:.2f}")
                                print(f"  Tax Owed: $0.00 (no tax on losses)")
                            print()
    
                            # NET PROFIT (Final Take-Home)
                            net_profit_color = Colors.GREEN if profit_calc['net_profit_usd'] >= 0 else Colors.RED
                            print(f"{Colors.BOLD}{'='*80}")
                            print(f"{net_profit_color}💰 NET PROFIT (Your Take-Home): ${profit_calc['net_profit_usd']:+.2f} ({profit_calc['net_profit_pct']:+.2f}%){Colors.ENDC}")
                            print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")
                            print(f"\n{Colors.CYAN}Formula: Market Value - Cost Basis - Exit Fee - Taxes")
                            print(f"  ${profit_calc['exit_value_usd']:.2f} - ${profit_calc['cost_basis_usd']:.2f} - ${profit_calc['exit_fee_usd']:.2f} - ${profit_calc['tax_usd']:.2f} = ${profit_calc['net_profit_usd']:+.2f}{Colors.ENDC}\n")
    
                            # Store values for later use
                            current_position_value_usd = profit_calc['exit_value_usd']
                            total_cost_basis_usd = profit_calc['cost_basis_usd']
                            gross_profit_before_exit_costs = profit_calc['gross_profit_usd']
                            exit_exchange_fee_usd = profit_calc['exit_fee_usd']
                            unrealized_gain_usd = profit_calc['capital_gain_usd']
                            capital_gains_tax_usd = profit_calc['tax_usd']
                            net_profit_after_all_costs_usd = profit_calc['net_profit_usd']
                            net_profit_percentage = profit_calc['net_profit_pct']
    
                            # Use the maximum of calculated target and configured minimum
                            effective_profit_target = max(PROFIT_PERCENTAGE, min_profit_target_percentage)
                            print(f"effective_profit_target: {effective_profit_target}% (Calculated: {PROFIT_PERCENTAGE}%, Min: {min_profit_target_percentage}%)")
    
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
    
                            # Show rotation check status even when conditions aren't met
                            if market_rotation_enabled:
                                if not best_opportunity:
                                    print(f"\n{Colors.CYAN}🔄 ROTATION CHECK: Skipping - no alternative opportunities available{Colors.ENDC}\n")
                                elif not intelligent_rotation_enabled:
                                    print(f"\n{Colors.CYAN}🔄 ROTATION CHECK: Skipping - intelligent rotation disabled in config{Colors.ENDC}\n")
    
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
                                    # Profit is below rotation threshold
                                    print(f"\n{Colors.CYAN}🔄 ROTATION CHECK: Skipping - profit {net_profit_percentage:.2f}% below threshold ({min_profit_for_rotation}%){Colors.ENDC}\n")
    
                            # EARLY PROFIT ROTATION: Take profits early when good opportunities appear
                            # This prevents the scenario where position goes: negative → +$6 → negative
                            early_profit_config = market_rotation_config.get('early_profit_rotation', {})
                            early_profit_enabled = early_profit_config.get('enabled', False)
    
                            if market_rotation_enabled and early_profit_enabled and not should_rotate_position:
                                min_new_opp_score = early_profit_config.get('min_new_opportunity_score', 80)
                                ignore_profit_advantage = early_profit_config.get('ignore_profit_advantage_requirement', True)
    
                                # Peak-based downturn detection
                                require_downturn = early_profit_config.get('require_downturn_from_peak', False)
                                min_peak_profit_usd = early_profit_config.get('min_peak_profit_usd', 6.0)
                                downturn_threshold_usd = early_profit_config.get('downturn_threshold_usd', 3.0)
    
                                # Get opportunity details if available
                                best_opp_score = best_opportunity.get('score', 0) if best_opportunity else 0
                                best_opp_symbol = best_opportunity.get('symbol') if best_opportunity else None
    
                                # Check if we should consider early profit action
                                # Can exit without opportunity if downturn triggered, or rotate if good opportunity exists
                                has_valid_opportunity = best_opp_symbol and best_opp_symbol != symbol and best_opp_score >= min_new_opp_score
    
                                if has_valid_opportunity or require_downturn:
                                    print(f"\n{Colors.BOLD}{Colors.CYAN}💰 EARLY PROFIT ROTATION CHECK{Colors.ENDC}")
                                    print(f"  Current position ({symbol}): Profit ${net_profit_after_all_costs_usd:.2f} ({net_profit_percentage:.2f}%)")
                                    if best_opp_symbol:
                                        print(f"  New opportunity ({best_opp_symbol}): Score {best_opp_score:.1f}/100")
                                        print(f"  Minimum new opportunity score: {min_new_opp_score}")
                                    else:
                                        print(f"  No alternative opportunity available")
    
                                    # Check peak-based downturn if enabled
                                    downturn_triggered = False
                                    # NOTE: position_tracker module not implemented yet
                                    # if require_downturn:
                                    #     from utils.position_tracker import should_exit_on_downturn, get_peak_profit
                                    #
                                    #     should_exit, peak_info = should_exit_on_downturn(
                                    #         symbol=symbol,
                                    #         current_profit_usd=net_profit_after_all_costs_usd,
                                    #         current_profit_pct=net_profit_percentage,
                                    #         min_peak_profit_usd=min_peak_profit_usd,
                                    #         downturn_threshold_usd=downturn_threshold_usd
                                    #     )
                                    #
                                    #     if peak_info:
                                    #         peak_profit_usd = peak_info['peak_profit_usd']
                                    #         downturn_amount = peak_profit_usd - net_profit_after_all_costs_usd
                                    #
                                    #         print(f"\n  {Colors.CYAN}PEAK DOWNTURN ANALYSIS:{Colors.ENDC}")
                                    #         print(f"  Peak profit reached: ${peak_profit_usd:.2f}")
                                    #         print(f"  Current profit: ${net_profit_after_all_costs_usd:.2f}")
                                    #         print(f"  Downturn from peak: ${downturn_amount:.2f}")
                                    #         print(f"  Minimum peak to consider: ${min_peak_profit_usd:.2f}")
                                    #         print(f"  Downturn threshold: ${downturn_threshold_usd:.2f}")
                                    #
                                    #         if should_exit:
                                    #             print(f"  {Colors.GREEN}✓ Downturn trigger met - exiting to preserve gains{Colors.ENDC}")
                                    #             downturn_triggered = True
                                    #         else:
                                    #             if peak_profit_usd < min_peak_profit_usd:
                                    #                 print(f"  {Colors.YELLOW}⏳ Peak not high enough yet (${peak_profit_usd:.2f} < ${min_peak_profit_usd:.2f}){Colors.ENDC}")
                                    #             else:
                                    #                 print(f"  {Colors.YELLOW}⏳ Downturn not significant enough (${downturn_amount:.2f} < ${downturn_threshold_usd:.2f}){Colors.ENDC}")
                                    #             print(f"  {Colors.YELLOW}→ Holding position - waiting for larger downturn{Colors.ENDC}")
    
                                    # Decision logic
                                    if ignore_profit_advantage and (not require_downturn or downturn_triggered):
                                        # Simple mode: If we're profitable and new opportunity is good, rotate
                                        # (or downturn mode: exit if downturn triggered, rotate if good opportunity exists)
    
                                        # CRITICAL: Only act if we're actually profitable
                                        min_profit_pct = early_profit_config.get('min_profit_percentage', 0.45)
    
                                        # If downturn triggered, allow exit with any profit > $0 (bypass percentage check)
                                        # Otherwise, require min_profit_percentage
                                        if require_downturn and downturn_triggered:
                                            # Downturn mode: only require net profit > $0 to preserve gains
                                            if net_profit_after_all_costs_usd <= 0:
                                                print(f"  {Colors.RED}✗ Cannot exit - position at loss (${net_profit_after_all_costs_usd:.2f}){Colors.ENDC}")
                                                print(f"  {Colors.YELLOW}→ Downturn exit requires net profit > $0{Colors.ENDC}")
                                            else:
                                                # Allow exit even with small profit to preserve gains from downturn
                                                print(f"  {Colors.GREEN}✓ Downturn detected with profit ${net_profit_after_all_costs_usd:.2f} ({net_profit_percentage:.2f}%){Colors.ENDC}")
                                                if net_profit_percentage < min_profit_pct:
                                                    print(f"  {Colors.YELLOW}⚠ Profit below normal minimum ({min_profit_pct}%) but exiting to preserve gains from downturn{Colors.ENDC}")
                                        elif net_profit_percentage < min_profit_pct:
                                            print(f"  {Colors.RED}✗ Cannot exit - profit {net_profit_percentage:.2f}% below minimum ({min_profit_pct}%){Colors.ENDC}")
                                            print(f"  {Colors.YELLOW}→ Early profit exit requires net profit >= {min_profit_pct}%{Colors.ENDC}")
    
                                        # Proceed with exit logic if checks passed
                                        if (require_downturn and downturn_triggered and net_profit_after_all_costs_usd > 0) or (net_profit_percentage >= min_profit_pct):
                                            # If downturn triggered, exit even without opportunity
                                            if require_downturn and downturn_triggered:
                                                if has_valid_opportunity:
                                                    # Rotate to better opportunity
                                                    print(f"  {Colors.GREEN}✓ New opportunity quality: {best_opp_score:.1f} >= {min_new_opp_score}{Colors.ENDC}")
                                                    print(f"  {Colors.GREEN}✓ Downturn from peak detected - securing profit and rotating{Colors.ENDC}")
                                                    should_rotate_position = True
                                                    rotation_reason = f"Early profit rotation to {best_opp_symbol}: Secured ${net_profit_after_all_costs_usd:.2f} profit, rotating to fresh {best_opp_score:.1f} score setup"
                                                else:
                                                    # Exit to preserve profit even without opportunity
                                                    print(f"  {Colors.GREEN}✓ Downturn from peak detected - exiting to preserve profit{Colors.ENDC}")
                                                    print(f"  {Colors.YELLOW}⚠ No valid opportunity to rotate into - will exit position{Colors.ENDC}")
                                                    should_rotate_position = True
                                                    rotation_reason = f"Early profit exit (downturn): Secured ${net_profit_after_all_costs_usd:.2f} profit, exiting to prevent further decline"
                                            elif has_valid_opportunity:
                                                # Standard rotation to better opportunity
                                                print(f"  {Colors.GREEN}✓ New opportunity quality: {best_opp_score:.1f} >= {min_new_opp_score}{Colors.ENDC}")
                                                print(f"  {Colors.GREEN}✓ Taking profit now rather than risk giving it back{Colors.ENDC}")
                                                should_rotate_position = True
                                                rotation_reason = f"Early profit rotation to {best_opp_symbol}: Secured ${net_profit_after_all_costs_usd:.2f} profit, rotating to fresh {best_opp_score:.1f} score setup"
                                    else:
                                        # Check if we should exit due to downturn even without opportunity
                                        if require_downturn and downturn_triggered and not has_valid_opportunity:
                                            # Exit to preserve profit even without better opportunity
                                            min_profit_pct = early_profit_config.get('min_profit_percentage', 0.45)
    
                                            # For downturn exits, only require net profit > $0 (bypass percentage requirement)
                                            if net_profit_after_all_costs_usd > 0:
                                                print(f"  {Colors.GREEN}✓ Current profit: ${net_profit_after_all_costs_usd:.2f} ({net_profit_percentage:.2f}%){Colors.ENDC}")
                                                if net_profit_percentage < min_profit_pct:
                                                    print(f"  {Colors.YELLOW}⚠ Profit below normal minimum ({min_profit_pct}%) but exiting to preserve gains from downturn{Colors.ENDC}")
                                                print(f"  {Colors.GREEN}✓ Downturn from peak detected - exiting to preserve profit{Colors.ENDC}")
                                                print(f"  {Colors.YELLOW}⚠ No valid opportunity to rotate into - will exit position{Colors.ENDC}")
                                                should_rotate_position = True
                                                rotation_reason = f"Early profit exit (downturn): Secured ${net_profit_after_all_costs_usd:.2f} profit, exiting to prevent further decline"
                                            else:
                                                print(f"  {Colors.RED}✗ Cannot exit - position at loss (${net_profit_after_all_costs_usd:.2f}){Colors.ENDC}")
                                                print(f"  {Colors.YELLOW}→ Downturn exit requires net profit > $0{Colors.ENDC}")
                                        elif has_valid_opportunity:
                                            # Check profit advantage for rotation
                                            new_opp_entry = best_opportunity.get('entry_price', 0)
                                            new_opp_target = best_opportunity.get('profit_target', 0)
    
                                            if new_opp_entry > 0 and new_opp_target > 0:
                                                new_opp_upside = ((new_opp_target - new_opp_entry) / new_opp_entry) * 100
                                                current_remaining_upside = effective_profit_target - net_profit_percentage
    
                                                print(f"  Current remaining upside: {current_remaining_upside:.2f}%")
                                                print(f"  New opportunity upside: {new_opp_upside:.2f}%")
    
                                                # CRITICAL: Only rotate if we're in positive net profit
                                                if net_profit_after_all_costs_usd <= 0:
                                                    print(f"  {Colors.RED}✗ Cannot rotate - position at loss (${net_profit_after_all_costs_usd:.2f}){Colors.ENDC}")
                                                    print(f"  {Colors.YELLOW}→ Rotation only allowed when net profit > $0{Colors.ENDC}")
                                                elif new_opp_upside >= current_remaining_upside:
                                                    print(f"  {Colors.GREEN}✓ New opportunity has equal or better upside{Colors.ENDC}")
                                                    should_rotate_position = True
                                                    rotation_reason = f"Early profit rotation to {best_opp_symbol}: Secured ${net_profit_after_all_costs_usd:.2f}, rotating to {new_opp_upside:.2f}% upside opportunity"
                                                else:
                                                    print(f"  {Colors.YELLOW}✗ Current position has better remaining upside - holding{Colors.ENDC}")
    
                                    print()  # Blank line for readability
    
                            # Execute rotation if triggered (either intelligent or early profit)
                            if should_rotate_position:
                                # Determine rotation type for logging
                                rotation_type = 'early_profit_rotation' if 'Early profit' in rotation_reason else 'intelligent_rotation'
                                rotation_emoji = '💰' if rotation_type == 'early_profit_rotation' else '🔄'
    
                                print(f'{Colors.BOLD}{Colors.CYAN}{rotation_emoji} ROTATION TRIGGERED{Colors.ENDC}')
                                print(f'{Colors.CYAN}Reason: {rotation_reason}{Colors.ENDC}')
    
                                # Handle both rotation (with opportunity) and exit (without opportunity)
                                if best_opportunity and best_opportunity.get('symbol'):
                                    print(f'{Colors.GREEN}Exiting {symbol} with {net_profit_percentage:.2f}% profit to enter {best_opportunity["symbol"]}{Colors.ENDC}')
                                else:
                                    print(f'{Colors.GREEN}Exiting {symbol} with {net_profit_percentage:.2f}% profit (no rotation opportunity){Colors.ENDC}')
    
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
                                    event_type='sell'
                                )
    
                                # Generate Coinbase metrics correlation chart for sell event
                                sell_coin_volume_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'volume_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
                                sell_coin_price_pct_change_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'price_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
                                sell_coin_volume_pct_change_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'volume_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
    
                                if (sell_coin_volume_24h and sell_coin_price_pct_change_24h and
                                    sell_coin_volume_pct_change_24h):
                                    sell_metrics_volume = sell_coin_volume_24h[-sell_chart_data_points:] if len(sell_coin_volume_24h) > sell_chart_data_points else sell_coin_volume_24h
                                    sell_metrics_price_pct = sell_coin_price_pct_change_24h[-sell_chart_data_points:] if len(sell_coin_price_pct_change_24h) > sell_chart_data_points else sell_coin_price_pct_change_24h
                                    sell_metrics_volume_pct = sell_coin_volume_pct_change_24h[-sell_chart_data_points:] if len(sell_coin_volume_pct_change_24h) > sell_chart_data_points else sell_coin_volume_pct_change_24h
    
                                    sell_metrics_screenshot_path = plot_coinbase_metrics_chart(
                                        time.time(),
                                        INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                        symbol,
                                        sell_chart_prices,
                                        sell_metrics_volume,
                                        sell_metrics_price_pct,
                                        sell_metrics_volume_pct,
                                        event_type='sell'
                                    )
                                    if sell_metrics_screenshot_path:
                                        print(f"📊 Sell metrics chart: {sell_metrics_screenshot_path}")
    
                                if READY_TO_TRADE:
                                    # Execute the rotation sell
                                    place_market_sell_order(coinbase_client, symbol, number_of_shares, net_profit_after_all_costs_usd, net_profit_percentage)
    
                                    # Save transaction record
                                    buy_timestamp = order_data.get('created_time')
                                    buy_screenshot_path = last_order.get('buy_screenshot_path')
    
                                    entry_market_conditions = {
                                        "volatility_range_pct": range_percentage_from_min,
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
                                        exit_trigger=rotation_type,
                                        position_sizing_data=position_sizing_data
                                    )
    
                                    # Clear position state (peak profit tracking)
                                    # NOTE: position_tracker module not implemented yet
                                    # from utils.position_tracker import clear_position_state
                                    # clear_position_state(symbol)
    
                                    # IMMEDIATE ROTATION: Recalculate best opportunity after exit
                                    print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 CAPITAL FREED - Recalculating best opportunity...{Colors.ENDC}\n")
                                    min_score = market_rotation_config.get('min_score_for_entry', 50)
                                    best_opportunity = find_best_opportunity(
                                        config=config,
                                        coinbase_client=coinbase_client,
                                        enabled_symbols=enabled_wallets,
                                        interval_seconds=INTERVAL_SECONDS,
                                        data_retention_hours=DATA_RETENTION_HOURS,
                                        min_score=min_score,
                                        entry_fee_pct=coinbase_spot_taker_fee,
                                        exit_fee_pct=coinbase_spot_taker_fee,
                                        tax_rate_pct=federal_tax_rate
                                    )
                                    if best_opportunity:
                                        best_opportunity_symbol = best_opportunity['symbol']
                                        print(f"{Colors.GREEN}✅ NEW BEST OPPORTUNITY: {best_opportunity_symbol} (score: {best_opportunity['score']:.1f}){Colors.ENDC}")
                                        print(f"   Will enter {best_opportunity_symbol} when we reach it in this iteration\n")
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
                                    event_type='sell'
                                )
    
                                # Generate Coinbase metrics correlation chart for stop loss sell
                                sell_coin_volume_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'volume_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
                                sell_coin_price_pct_change_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'price_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
                                sell_coin_volume_pct_change_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'volume_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
    
                                if (sell_coin_volume_24h and sell_coin_price_pct_change_24h and
                                    sell_coin_volume_pct_change_24h):
                                    sell_metrics_volume = sell_coin_volume_24h[-sell_chart_data_points:] if len(sell_coin_volume_24h) > sell_chart_data_points else sell_coin_volume_24h
                                    sell_metrics_price_pct = sell_coin_price_pct_change_24h[-sell_chart_data_points:] if len(sell_coin_price_pct_change_24h) > sell_chart_data_points else sell_coin_price_pct_change_24h
                                    sell_metrics_volume_pct = sell_coin_volume_pct_change_24h[-sell_chart_data_points:] if len(sell_coin_volume_pct_change_24h) > sell_chart_data_points else sell_coin_volume_pct_change_24h
    
                                    sell_metrics_screenshot_path = plot_coinbase_metrics_chart(
                                        time.time(),
                                        INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                        symbol,
                                        sell_chart_prices,
                                        sell_metrics_volume,
                                        sell_metrics_price_pct,
                                        sell_metrics_volume_pct,
                                        event_type='sell'
                                    )
                                    if sell_metrics_screenshot_path:
                                        print(f"📊 Stop loss metrics chart: {sell_metrics_screenshot_path}")
    
                                if READY_TO_TRADE:
                                    # Use market order for guaranteed execution on stop loss
                                    place_market_sell_order(coinbase_client, symbol, number_of_shares, net_profit_after_all_costs_usd, net_profit_percentage)
                                    # Save transaction record
                                    buy_timestamp = order_data.get('created_time')
                                    buy_screenshot_path = last_order.get('buy_screenshot_path')  # Get screenshot path from ledger
    
                                    # Build market context at entry
                                    entry_market_conditions = {
                                        "volatility_range_pct": range_percentage_from_min,
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
    
                                    # Clear position state (peak profit tracking)
                                    # NOTE: position_tracker module not implemented yet
                                    # from utils.position_tracker import clear_position_state
                                    # clear_position_state(symbol)
    
                                    delete_analysis_file(symbol)
    
                                    # IMMEDIATE ROTATION: Recalculate best opportunity after exit
                                    # This allows us to enter the next-best trade in the same iteration
                                    if market_rotation_enabled:
                                        print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 CAPITAL FREED - Recalculating best opportunity...{Colors.ENDC}\n")
                                        min_score = market_rotation_config.get('min_score_for_entry', 50)
                                        best_opportunity = find_best_opportunity(
                                            config=config,
                                            coinbase_client=coinbase_client,
                                            enabled_symbols=enabled_wallets,
                                            interval_seconds=INTERVAL_SECONDS,
                                            data_retention_hours=DATA_RETENTION_HOURS,
                                            min_score=min_score
                                        )
                                        if best_opportunity:
                                            best_opportunity_symbol = best_opportunity['symbol']
                                            print(f"{Colors.GREEN}✅ NEW BEST OPPORTUNITY: {best_opportunity_symbol} (score: {best_opportunity['score']:.1f}){Colors.ENDC}")
                                            print(f"   Will enter {best_opportunity_symbol} when we reach it in this iteration\n")
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
                                    event_type='sell'
                                )
    
                                # Generate Coinbase metrics correlation chart for profit target sell
                                sell_coin_volume_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'volume_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
                                sell_coin_price_pct_change_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'price_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
                                sell_coin_volume_pct_change_24h = get_property_values_from_crypto_file(
                                    coinbase_data_directory, symbol, 'volume_percentage_change_24h', max_age_hours=DATA_RETENTION_HOURS
                                )
    
                                if (sell_coin_volume_24h and sell_coin_price_pct_change_24h and
                                    sell_coin_volume_pct_change_24h):
                                    sell_metrics_volume = sell_coin_volume_24h[-sell_chart_data_points:] if len(sell_coin_volume_24h) > sell_chart_data_points else sell_coin_volume_24h
                                    sell_metrics_price_pct = sell_coin_price_pct_change_24h[-sell_chart_data_points:] if len(sell_coin_price_pct_change_24h) > sell_chart_data_points else sell_coin_price_pct_change_24h
                                    sell_metrics_volume_pct = sell_coin_volume_pct_change_24h[-sell_chart_data_points:] if len(sell_coin_volume_pct_change_24h) > sell_chart_data_points else sell_coin_volume_pct_change_24h
    
                                    sell_metrics_screenshot_path = plot_coinbase_metrics_chart(
                                        time.time(),
                                        INTERVAL_SAVE_DATA_EVERY_X_MINUTES,
                                        symbol,
                                        sell_chart_prices,
                                        sell_metrics_volume,
                                        sell_metrics_price_pct,
                                        sell_metrics_volume_pct,
                                        event_type='sell'
                                    )
                                    if sell_metrics_screenshot_path:
                                        print(f"📊 Profit target metrics chart: {sell_metrics_screenshot_path}")
    
                                if READY_TO_TRADE:
                                    # Use market order for guaranteed execution on profit target
                                    place_market_sell_order(coinbase_client, symbol, number_of_shares, net_profit_after_all_costs_usd, net_profit_percentage)
                                    # Save transaction record
                                    buy_timestamp = order_data.get('created_time')
                                    buy_screenshot_path = last_order.get('buy_screenshot_path')  # Get screenshot path from ledger
    
                                    # Build market context at entry
                                    entry_market_conditions = {
                                        "volatility_range_pct": range_percentage_from_min,
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
    
                                    # Clear position state (peak profit tracking)
                                    # NOTE: position_tracker module not implemented yet
                                    # from utils.position_tracker import clear_position_state
                                    # clear_position_state(symbol)
    
                                    delete_analysis_file(symbol)
    
                                    # IMMEDIATE ROTATION: Recalculate best opportunity after exit
                                    # This allows us to enter the next-best trade in the same iteration
                                    if market_rotation_enabled:
                                        print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 CAPITAL FREED - Recalculating best opportunity...{Colors.ENDC}\n")
                                        min_score = market_rotation_config.get('min_score_for_entry', 50)
                                        best_opportunity = find_best_opportunity(
                                            config=config,
                                            coinbase_client=coinbase_client,
                                            enabled_symbols=enabled_wallets,
                                            interval_seconds=INTERVAL_SECONDS,
                                            data_retention_hours=DATA_RETENTION_HOURS,
                                            min_score=min_score
                                        )
                                        if best_opportunity:
                                            best_opportunity_symbol = best_opportunity['symbol']
                                            print(f"{Colors.GREEN}✅ NEW BEST OPPORTUNITY: {best_opportunity_symbol} (score: {best_opportunity['score']:.1f}){Colors.ENDC}")
                                            print(f"   Will enter {best_opportunity_symbol} when we reach it in this iteration\n")
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
                        gross_profit_usd = summary_wallet_metrics.get('gross_profit', 0.0)
                        total_trades = 0
                        wins = 0
    
                        # Count total trades and wins from transaction history
                        from utils.wallet_helpers import load_transaction_history
                        summary_transactions = load_transaction_history(summary_symbol)
                        total_trades = len(summary_transactions)
                        wins = len([t for t in summary_transactions if t.get('total_profit', 0) > 0])
    
                        # Calculate total volume traded in USD (sum of all buy orders)
                        total_volume_usd = 0
                        for tx in summary_transactions:
                            # Each transaction represents a completed buy/sell cycle
                            # Use buy_price and a standard position size to estimate volume
                            buy_price = tx.get('buy_price', 0)
                            # Assuming each trade uses the starting capital
                            if buy_price > 0:
                                total_volume_usd += summary_starting_capital
    
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
                            'gross_profit_usd': gross_profit_usd,
                            'status': status,
                            'has_position': summary_has_position,
                            'total_trades': total_trades,
                            'wins': wins,
                            'total_volume_usd': total_volume_usd
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
                            win_rate_str = f"{asset['wins']}/{asset['total_trades']} ({win_pct:>3.0f}%)"
                        else:
                            win_rate_str = f"0/0 (  0%)"
    
                        profit_color = Colors.GREEN if asset['total_profit_usd'] > 0 else (Colors.RED if asset['total_profit_usd'] < 0 else Colors.YELLOW)
                        gross_profit_color = Colors.GREEN if asset['gross_profit_usd'] > 0 else (Colors.RED if asset['gross_profit_usd'] < 0 else Colors.YELLOW)
    
                        # Format columns with consistent width
                        symbol_col = f"{asset['symbol']:<10s}"
                        profit_col = f"{profit_color}${asset['total_profit_usd']:>+8.2f}{Colors.ENDC}"
                        gross_profit_col = f"{gross_profit_color}${asset['gross_profit_usd']:>+8.2f}{Colors.ENDC}"
                        volume_col = f"${asset['total_volume_usd']:>8,.0f}"
                        pnl_col = f"{pnl_color}{asset['pnl_pct']:>+6.2f}%{Colors.ENDC}"
                        win_col = f"{win_rate_str:>12s}"
                        status_col = f"{asset['status']:<20s}"
    
                        print(f"  {position_indicator} {idx:2d}. {symbol_col} | Net: {profit_col} | Gross: {gross_profit_col} | Vol: {volume_col} | P&L: {pnl_col} | W/L: {win_col} | {status_col}")
    
                    # Calculate and display lifetime total
                    total_lifetime_profit = sum(asset['total_profit_usd'] for asset in asset_performance)
                    total_lifetime_gross_profit = sum(asset['gross_profit_usd'] for asset in asset_performance)
                    lifetime_color = Colors.GREEN if total_lifetime_profit > 0 else (Colors.RED if total_lifetime_profit < 0 else Colors.YELLOW)
                    lifetime_gross_color = Colors.GREEN if total_lifetime_gross_profit > 0 else (Colors.RED if total_lifetime_gross_profit < 0 else Colors.YELLOW)
                    print(f"{Colors.CYAN}{'-'*100}{Colors.ENDC}")
                    print(f"  {Colors.BOLD}LIFETIME TOTAL:      | Net: {lifetime_color}${total_lifetime_profit:>+8.2f}{Colors.ENDC}{Colors.BOLD} | Gross: {lifetime_gross_color}${total_lifetime_gross_profit:>+8.2f}{Colors.ENDC}{Colors.BOLD}{Colors.ENDC}")
                    print(f"{Colors.CYAN}{'='*100}{Colors.ENDC}\n")
    
                    #
                    #
                    # ERROR TRACKING: reset error count if they're non-consecutive
                    LAST_EXCEPTION_ERROR = None
                    LAST_EXCEPTION_ERROR_COUNT = 0
    
        #
        #
        # End of iteration function
        # Sleep 1 second between iterations to avoid busy-waiting
        # Data collection timing is controlled by the time-based gate (should_run_data_collection)
        time.sleep(1)

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
