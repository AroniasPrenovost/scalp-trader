# utils
import os
from dotenv import load_dotenv
from json import dumps, load
import math
import time
from pprint import pprint
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

#
#
# Get the current price of an asset
#

def get_asset_price(symbol):
    try:
        product = client.get_product(symbol)
        price = float(product["price"])
        # print(f"Current {symbol} price: {price}")
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
        # accounts object
        # [{'uuid': 'some_uuid', 'name': 'ARB Wallet', 'currency': 'ARB',
        #     'available_balance': {'value': '0.0031831259076048', 'currency': 'ARB'}, 'default': True,
        #     'active': True, 'created_at': '2024-11-14T02:11:40.374Z', 'updated_at': '2024-11-14T02:12:34.556Z',
        #     'deleted_at': None, 'type': 'ACCOUNT_TYPE_CRYPTO', 'ready': True,
        #     'hold': {'value': '0', 'currency': 'ARB'}, 'retail_portfolio_id': 'some_id'
        # }, ...]

        #  pprint(accounts)
        modified_symbol = symbol.split('-')[0]

        # Iterate through accounts to find the specified asset
        for account in accounts['accounts']:
            if account['currency'] == modified_symbol:
                # print(account)
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
            client_order_id=client_order_id,  # must be unique
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

def get_open_scalp_order(symbol, action):
    try:
        orders = client.list_orders(product_id=symbol, order_status="OPEN")
        # pprint(orders)
        if orders:
            matching_orders = []
            for order in orders['orders']:
                # manually match order id so we know it is the corresponding buy/sell order
                if action == 'sell':
                    matching_orders.append(order) # TEMPORARY HACK UNTIL WE TEST IN PROD
                # if order['order_id'] == generate_client_order_id(symbol, action):
                #     matching_orders.append(order)
            if matching_orders:
                return matching_orders
            else:
                # print(f"No matching open {action.upper()} orders found for {symbol}.")
                return []
        else:
            print(f"No open orders found for {symbol}.")
            return []
    except Exception as e:
        print(f"Error fetching orders for {symbol}: {e}")
        return []

#
#
#
#

def get_corresponding_buy_order(symbol, action):
    try:
        orders = client.list_orders(product_id=symbol, order_status="FILLED")
        if orders:
            for order in orders['orders']:
                return order # TEMPORARY HACK UNTIL WE TEST IN PROD
                # if order['order_id'] == generate_client_order_id(symbol, action):
                    # return order
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
    # unix_timestamp = int(time.time())
    custom_id = f"custom_scalp_{action}_{symbol}" # "custom_scalp_buy_cro"
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

def calculate_trade_range_percentage(num1, num2):
    if num1 == 0 and num2 == 0:
        return "0.00%"
    difference = abs(num1 - num2)
    average = (num1 + num2) / 2
    percentage_difference = (difference / average) * 100
    return f"{percentage_difference:.2f}%"


#
#
# main logic loop
#


# Example usage
# symbol = "CRO"
# base_size = "1"
# client_order_id = generate_client_order_id("CRO-USD", "buy")
#
# current_asset_price = get_asset_price(symbol)
# print(current_asset_price)
# print(client_order_id)

# y = get_open_scalp_order(symbol)
# print(y)


#
#
#
#

def load_config(file_path):
    with open(file_path, 'r') as file:
        return load(file)

def iterate_assets(config, interval):
    while True:

        client_accounts = client.get_accounts()

        for asset in config['assets']:
            enabled = asset['enabled']
            symbol = asset['symbol']
            support = asset['support']
            resistance = asset['resistance']
            buy_limit = asset['buy_limit_1']
            sell_limit = asset['sell_limit_1']

            if enabled:
                print(f"\n____ {symbol}")

                # START BUSINESS LOGIC
                #
                #

                current_price = get_asset_price(symbol)

                asset_position = get_asset_position(symbol, client_accounts)
                asset_shares = float(asset_position['hold']['value'])
                # print('double_check_shares_are_same_as_below_will_eventually_remove: ', asset_shares)

                open_buy_order = get_open_scalp_order(symbol, 'buy')
                open_sell_order = get_open_scalp_order(symbol, 'sell')
                print('open_buy_order: ', len(open_buy_order) == 1)
                print('open_sell_order: ', len(open_sell_order) == 1)

                # Support + Resistance trend breaks
                if current_price <= support:
                    print('ALERT: price dropped below support, may need to adjust SUPPORT level')
                if current_price >= resistance:
                    print('ALERT: price hit or broke above support, may need to adjust RESISTANCE level')

                # **BUY** triggers
                if asset_shares == 0:
                    # create order
                    if open_buy_order == []:
                        if current_price <= support or current_price <= buy_limit:
                            print('price lower than support price or buy limit - time to buy')
                            #  place_market_order(symbol, 1, 'buy')

                # **SELL** triggers
                elif asset_shares > 0:

                    # get corresponding buy order
                    corresponding_buy_order = get_corresponding_buy_order(symbol, 'buy')
                    # print(corresponding_buy_order)
                    if corresponding_buy_order:
                        #
                        entry_price = float(corresponding_buy_order['average_filled_price'])
                        print(f"entry_price: {entry_price}")

                        #
                        number_of_shares = float(corresponding_buy_order['filled_size'])
                        print('shares_purchased: ',number_of_shares)

                        #
                        position_value_at_purchase = entry_price * number_of_shares
                        print(f"purchase_position_value: {position_value_at_purchase}")

                        #
                        print(f"current_price: {current_price}")

                        #
                        current_position_value = current_price * number_of_shares
                        print(f"current_position_value: {current_position_value}")

                        #
                        exchange_fee = calculate_exchange_fee(current_price, number_of_shares, 'taker') # Assuming taker fee for simplicity
                        print(f"sell_now_exchange_fee: {exchange_fee}")

                        #
                        federal_tax_rate_float = float(federal_tax_rate)
                        profit = (current_price - entry_price) * number_of_shares
                        tax_owed = (federal_tax_rate_float / 100) * profit
                        print(f"sell_now_taxes_owed: {tax_owed}")

                        #
                        potential_profit = profit - exchange_fee - tax_owed
                        print(f"sell_now_post_tax_profit: {potential_profit}")

                        #
                        investment = entry_price * number_of_shares
                        potential_profit_percentage = (potential_profit / investment) * 100
                        print(f"sell_now_post_tax_profit_percentage: {potential_profit_percentage:.2f}%")

                    print('\n')

                    # create order
                    if open_sell_order == []:
                        if current_price >= resistance or current_price >= sell_limit:
                            print('current price higher than resistance, might be good time to sell')
                            #  place_market_order(symbol, asset_shares, 'sell')


                #
                #
                # END BUSINESS LOGIC
        time.sleep(interval)

if __name__ == "__main__":
    config = load_config('config.json')
    interval = 60  # Set your desired interval in seconds
    iterate_assets(config, interval)


# quit()

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
