# utils
import os
from dotenv import load_dotenv
from json import dumps, load
import math
import time
from pprint import pprint
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# coinbase api
from coinbase.rest import RESTClient

#
#
# Load environment variables from .env file and connect to coinbase api
#

load_dotenv()
coinbase_api_key = os.environ.get('COINBASE_API_KEY')
coinbase_api_secret = os.environ.get('COINBASE_API_SECRET')

coinbase_spot_maker_fee = float(os.environ.get('COINBASE_SPOT_MAKER_FEE'))
coinbase_spot_taker_fee = float(os.environ.get('COINBASE_SPOT_TAKER_FEE'))
federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))

client = RESTClient(api_key=coinbase_api_key, api_secret=coinbase_api_secret)

# Initialize a dictionary to store price data for each asset
LOCAL_PRICE_DATA = {}
TARGET_PROFIT_PERCENTAGE = 0.5

# Assuming buying 1 share
SHARES_TO_ACQUIRE = 100

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
                # round to nearest whole number to ignore 0.0000000564....
                available_balance = round(float((account['available_balance']['value'])))
                print(account)

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
# Get the last buy order for the asset
#

def get_most_recent_buy_order_for_asset(symbol):
    try:
        orders = client.list_orders(product_id=symbol, order_status="FILLED")
        if orders:
            for order in orders['orders']:
                if order['side'] == 'BUY':
                    return order
        print(f"No filled buy orders found for {symbol}.")
        return None
    except Exception as e:
        print(f"Error fetching filled orders for {symbol}: {e}")
        return None

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

# def calculate_support_avg_resistance(prices):
#     support = min(prices)
#     average = sum(prices) / len(prices)
#     resistance = max(prices)
#     return support, average, resistance

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
# Create chart
#

def plot_graph(symbol, price_data, pivot, support, resistance, trading_range_percentage, current_price_position_within_trading_range, entry_price):
    plt.figure()
    # price data
    plt.plot(list(price_data), marker='o', label='Price Data')
    # support + resistance levels
    plt.axhline(y=support, color='y', linewidth=1.5, linestyle='--', label='Support')
    plt.axhline(y=resistance, color='r', linewidth=1.5, linestyle='--', label='Resistance')
    # etc...
    plt.axhline(y=pivot, color='r', linewidth=1, linestyle=':', label='Pivot')

    if entry_price == 0:
        entry_price = pivot

    plt.axhline(y=entry_price, color='g', linewidth=1.2, linestyle='-', label='Entry Price')

    plt.title(f"Price Data for {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    # Set y-axis minimum
    min_displayed_price = min(price_data)
    if entry_price < min_displayed_price:
        min_displayed_price = entry_price
    # Set y-axis maximum
    max_displayed_price = max(price_data)
    if entry_price > max_displayed_price:
        max_displayed_price = entry_price

    # Set y-axis to show every increment
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.001))
    # plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().set_ylim(min_displayed_price - 0.002, max_displayed_price + 0.002)

    # Set x-axis to show time points
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.grid(True)
    plt.figtext(0.5, 0.01, f"Trading Range %: {trading_range_percentage}, Current Position %: {current_price_position_within_trading_range}", ha="center", fontsize=8)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

#
#
# main logic loop
#

def load_config(file_path):
    with open(file_path, 'r') as file:
        return load(file)

def iterate_assets(config, INTERVAL_SECONDS):
    while True:
        client_accounts = client.get_accounts()

        for asset in config['assets']:
            enabled = asset['enabled']
            symbol = asset['symbol']

            if enabled:

                print(symbol)

                # if LOCAL_PRICE_DATA and LOCAL_PRICE_DATA[symbol]:
                #     print(LOCAL_PRICE_DATA[symbol])

                # Initialize price data storage if not already done
                if symbol not in LOCAL_PRICE_DATA:
                    LOCAL_PRICE_DATA[symbol] = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)

                current_price = get_asset_price(symbol)
                print(f"current_price: {current_price}")
                if current_price is not None:
                    LOCAL_PRICE_DATA[symbol].append(current_price)

                # Only proceed if we have enough data
                if len(LOCAL_PRICE_DATA[symbol]) < DATA_POINTS_FOR_X_MINUTES:
                    print(f"Waiting for more data... ({len(LOCAL_PRICE_DATA[symbol])}/{DATA_POINTS_FOR_X_MINUTES})\n")
                    continue

                # pass all these into the graph
                pivot, support, resistance = calculate_support_resistance(LOCAL_PRICE_DATA[symbol])
                print(f"support: {support}")
                print(f"resistance: {resistance}")

                trading_range_percentage = calculate_trading_range_percentage(support, resistance)
                print(f"trading_range_percentage: {trading_range_percentage}%")

                current_price_position_within_trading_range = calculate_current_price_position_within_trading_range(current_price, support, resistance)
                print(f"current_price_position_within_trading_range: {current_price_position_within_trading_range}%")

                # Continue with existing business logic
                asset_holdings = get_current_asset_holdings(symbol, client_accounts)
                # print('asset_holdings: ', asset_holdings)
                owned_shares = asset_holdings['available_balance'] if asset_holdings else 0
                print('owned_shares: ', owned_shares)

                entry_price = 0

                if owned_shares == 0:

                    # Calculate a buffer zone below the resistance
                    buffer_zone = (resistance - support) * 0.05  # 5% below resistance
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

                    if expected_profit_percentage >= TARGET_PROFIT_PERCENTAGE and current_price_position_within_trading_range <= 45:
                        print('~ BUY OPPORTUNITY ~')
                        place_market_buy_order(symbol, SHARES_TO_ACQUIRE)

                elif owned_shares > 0:

                    corresponding_buy_order = get_most_recent_buy_order_for_asset(symbol)
                    if corresponding_buy_order:

                        entry_price = float(corresponding_buy_order['average_filled_price'])
                        print(f"entry_price: {entry_price}")

                        number_of_shares = float(corresponding_buy_order['filled_size'])
                        # print('number_of_shares: ', number_of_shares)

                        if number_of_shares != owned_shares:
                            print('Something went wrong. number_of_shares should match owned_shares')
                            quit()

                        position_value_at_purchase = entry_price * number_of_shares
                        print(f"purchase_position_value: {position_value_at_purchase}")

                        current_position_value = current_price * number_of_shares
                        print(f"current_position_value: {current_position_value}")

                        exchange_fee = calculate_exchange_fee(current_price, number_of_shares, 'taker')
                        print(f"sell_now_exchange_fee: {exchange_fee}")

                        profit = (current_price - entry_price) * number_of_shares
                        tax_owed = (federal_tax_rate / 100) * profit
                        print(f"sell_now_taxes_owed: {tax_owed}")

                        potential_profit = profit - exchange_fee - tax_owed
                        print(f"sell_now_post_tax_profit: {potential_profit}")

                        investment = entry_price * number_of_shares
                        potential_profit_percentage = (potential_profit / investment) * 100
                        print(f"sell_now_post_tax_profit_percentage: {potential_profit_percentage:.2f}%")

                        if potential_profit_percentage >= TARGET_PROFIT_PERCENTAGE:
                            print('~ SELL OPPORTUNITY ~')
                            place_market_sell_order(symbol, owned_shares)

                plot_graph(symbol, LOCAL_PRICE_DATA[symbol], pivot, support, resistance, trading_range_percentage, current_price_position_within_trading_range, entry_price)  # Plot the graph each time data is updated

                print('\n')

        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    config = load_config('config.json')
    # Define time intervals
    INTERVAL_SECONDS = 15
    MINUTES = 240 # 4 hours
    DATA_POINTS_FOR_X_MINUTES = int((60 / INTERVAL_SECONDS) * MINUTES)
    iterate_assets(config, INTERVAL_SECONDS)
