# utils
import os
from dotenv import load_dotenv
from json import dumps, load
import math
import time
from pprint import pprint
from collections import deque
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
price_data = {}

#
#
# Get the current price of an asset
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
#
#

def get_asset_position(symbol, accounts):
    try:
        modified_symbol = symbol.split('-')[0]

        for account in accounts['accounts']:
            if account['currency'] == modified_symbol:
                balance = account['balance']
                available = account['available']
                hold = account['hold']

                return {
                    'currency': modified_symbol,
                    'balance': balance,
                    'available': available,
                    'hold': hold
                }

        print(f"No holdings found for asset: {symbol}.")
        return None
    except Exception as e:
        print(f"Error fetching position for asset {symbol}: {e}")
        return None

#
#
# Create a market order
#

def place_market_order(symbol, base_size, action):
    try:
        client_order_id = generate_client_order_id(symbol, action)
        order = client.market_order_buy(
            client_order_id=client_order_id,
            product_id=symbol,
            base_size=base_size
        )

        if 'order_id' in order['response']:
            order_id = order['response']['order_id']
            print(f"Order placed successfully. Order ID: {order_id}")
        else:
            print(f"Unexpected response: {dumps(order)}")
    except Exception as e:
        print(f"Error placing order for {symbol}: {e}")

#
#
# Check current open orders for a given symbol
#

def get_open_order(symbol, action):
    try:
        orders = client.list_orders(product_id=symbol, order_status="OPEN")
        if orders:
            matching_orders = []
            for order in orders['orders']:
                # if action == 'sell': # TEMPORARY HACK UNTIL WE TEST IN PROD
                #     matching_orders.append(order) # TEMPORARY HACK UNTIL WE TEST IN PROD
                # match with custom order_id so we know it is the corresponding buy/sell order
                if order['order_id'] == generate_client_order_id(symbol, action):
                    matching_orders.append(order)
            if matching_orders:
                return matching_orders
            else:
                return []
        else:
            print(f"No open orders found for {symbol}.")
            return []
    except Exception as e:
        print(f"Error fetching orders for {symbol}: {e}")
        return []

#
#
# Get existing buy order (if it exists)
#

def get_corresponding_buy_order(symbol, action):
    try:
        orders = client.list_orders(product_id=symbol, order_status="FILLED")
        if orders:
            for order in orders['orders']:
                # return order # TEMPORARY HACK UNTIL WE TEST IN PROD
                if order['order_id'] == generate_client_order_id(symbol, action):
                    return order
        print(f"No filled orders found for {symbol}.")
        return None
    except Exception as e:
        print(f"Error fetching filled orders for {symbol}: {e}")
        return None

#
#
# Generate (arbitrary) custom order id
#

def generate_client_order_id(symbol, action):
    symbol = symbol.lower()
    action = action.lower()
    custom_id = f"custom_scalp_{action}_{symbol}"
    return custom_id

#
#
# Tax utils
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

def calculate_optimal_buy_price(support, resistance, trade_range_percentage, buffer_percentage=0.5):
    """
    Calculate the optimal buy price based on support, resistance, and trade range percentage.
    A buffer percentage is applied below the resistance to ensure a profitable trade within the range.

    :param support: The support level price
    :param resistance: The resistance level price
    :param trade_range_percentage: The percentage range between support and resistance
    :param buffer_percentage: The percentage buffer below resistance to set the buy price
    :return: A dictionary containing the optimal buy price, anticipated sell price, and price position within trade range
    """
    # Convert trade range percentage to a decimal
    trade_range_decimal = float(trade_range_percentage) / 100

    # Calculate the trade range
    trade_range = resistance - support

    # Calculate the buffer amount
    buffer_amount = trade_range * (buffer_percentage / 100)

    # Calculate the optimal buy price
    optimal_buy_price = support + buffer_amount

    # Ensure the buy price is within the lower x percent of the trade range
    if optimal_buy_price > support + trade_range * trade_range_decimal:
        optimal_buy_price = support + trade_range * trade_range_decimal

    # Calculate anticipated sell price
    anticipated_sell_price = optimal_buy_price + (trade_range * (1 - buffer_percentage / 100))

    # Calculate price position within trade range
    price_position_within_trade_range = ((optimal_buy_price - support) / trade_range) * 100

    return optimal_buy_price, anticipated_sell_price, price_position_within_trade_range

#
#
# Determine support and resistance levels
#

def determine_support_resistance(prices):
    support = min(prices)
    resistance = max(prices)
    return support, resistance


def calculate_trade_range_percentage(num1, num2):
    if num1 == 0 and num2 == 0:
        return "0.00"
    difference = abs(num1 - num2)
    average = (num1 + num2) / 2
    percentage_difference = (difference / average) * 100
    return f"{percentage_difference:.2f}"

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

                print(price_data)

                # Initialize price data storage if not already done
                if symbol not in price_data:
                    price_data[symbol] = deque(maxlen=DATA_POINTS_FOR_X_MINUTES)

                current_price = get_asset_price(symbol)
                print(f"{symbol} current_price: {current_price}")
                if current_price is not None:
                    price_data[symbol].append(current_price)

                # Only proceed if we have enough data
                if len(price_data[symbol]) < DATA_POINTS_FOR_X_MINUTES:
                    print(f"Waiting for more data...\n")
                    continue

                # Calculate support and resistance
                support, resistance = determine_support_resistance(price_data[symbol])
                print(f"Support: {support}")
                print(f"Resistance: {resistance}")
                trade_range_percentage = calculate_trade_range_percentage(support, resistance)
                print(f"trade_range_percentage: {trade_range_percentage}%")

                # Continue with existing business logic
                asset_position = get_asset_position(symbol, client_accounts)
                asset_shares = float(asset_position['hold']['value'])
                print('asset_shares: ', asset_shares)

                open_buy_order = get_open_order(symbol, 'buy')
                open_sell_order = get_open_order(symbol, 'sell')
                print('open_buy_order: ', len(open_buy_order) == 1)
                print('open_sell_order: ', len(open_sell_order) == 1)

                if asset_shares == 0:

                    optimal_buy_price, anticipated_sell_price, price_position_within_trade_range = calculate_optimal_buy_price(support, resistance, trade_range_percentage)
                    print(f"optimal_buy_price: {optimal_buy_price}")
                    print(f"anticipated_sell_price: {anticipated_sell_price}")
                    print(f"price_position_within_trade_range: {price_position_within_trade_range}")

                    #
                    number_of_shares = 1

                    exchange_fee = calculate_exchange_fee(anticipated_sell_price, number_of_shares, 'taker')
                    print(f"anticipated_sell_exchange_fee: {exchange_fee}")

                    profit = (anticipated_sell_price - optimal_buy_price) * number_of_shares
                    tax_owed = (federal_tax_rate / 100) * profit
                    print(f"anticipated_sell_taxes_owed: {tax_owed}")

                    potential_profit = profit - exchange_fee - tax_owed
                    print(f"anticipated_sell_post_tax_profit: {potential_profit}")

                    investment = optimal_buy_price * number_of_shares
                    potential_profit_percentage = (potential_profit / investment) * 100
                    print(f"anticipated_sell_post_tax_profit_percentage: {potential_profit_percentage:.2f}%")
                    #

                    if open_buy_order == []:
                        if potential_profit_percentage >= 0.2:
                        #  or current_price <= optimal_buy_price:
                            print('BUY OPPORTUNITY')
                            # place_market_order(symbol, 1, 'buy')

                elif asset_shares > 0:
                    corresponding_buy_order = get_corresponding_buy_order(symbol, 'buy')
                    if corresponding_buy_order:
                        entry_price = float(corresponding_buy_order['average_filled_price'])
                        print(f"entry_price: {entry_price}")

                        number_of_shares = float(corresponding_buy_order['filled_size'])
                        print('shares_purchased: ', number_of_shares)

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

                        if open_sell_order == [] and potential_profit_percentage >= 0.2:
                            print('SELL OPPORTUNITY')
                            # place_market_order(symbol, asset_shares, 'sell')

                print('\n')

        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    config = load_config('config.json')
    # Define the interval and calculate the number of data points needed for 5 minute interval
    INTERVAL_SECONDS = 10
    MINUTES = 5
    DATA_POINTS_FOR_X_MINUTES = int((60 / INTERVAL_SECONDS) * MINUTES)
    iterate_assets(config, INTERVAL_SECONDS)


# place_market_order(symbol, base_size, 'buy')




# try:
#     print('price - cro')
#     print(cro_usd_price)
#     order = client.market_order_buy(
#         client_order_id=generate_client_order_id("CRO-USD", "buy")
#         product_id="CRO-USD",
#         base_size="1"
#     )
#     # print(order);
#     # {'success': True, 'response': {'order_id': '64684d63-1108-464f-95ea-395ae1aab674', 'product_id': 'CRO-USD', 'side': 'BUY', 'client_order_id': '00000003', 'attached_order_id': ''}, 'order_configuration': {'market_market_ioc': {'base_size': '1', 'rfq_enabled': False, 'rfq_disabled': False}}}
#
#
#     if 'order_id' in order['response']:
#         order_id = order['response']['order_id']
#         print(f"Order placed successfully. Order ID: {order_id}")
#     else:
#         print(f"Unexpected response: {json.dumps(order)}")
# except Exception as e:
#     print(f"Error placing order: {e}")
