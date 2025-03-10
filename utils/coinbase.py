# boilerplate
import os
from dotenv import load_dotenv
load_dotenv()
# end boilerplate

from coinbase.rest import RESTClient

coinbase_api_key = os.environ.get('COINBASE_API_KEY')
coinbase_api_secret = os.environ.get('COINBASE_API_SECRET')

coinbase_spot_maker_fee = float(os.environ.get('COINBASE_SPOT_MAKER_FEE'))
coinbase_spot_taker_fee = float(os.environ.get('COINBASE_SPOT_TAKER_FEE'))
coinbase_stable_pair_spot_maker_fee = float(os.environ.get('COINBASE_STABLE_PAIR_SPOT_MAKER_FEE'))
coinbase_stable_pair_spot_taker_fee = float(os.environ.get('COINBASE_STABLE_PAIR_SPOT_TAKER_FEE'))
federal_tax_rate = float(os.environ.get('FEDERAL_TAX_RATE'))

def get_coinbase_client():
    return RESTClient(api_key=coinbase_api_key, api_secret=coinbase_api_secret)


def calculate_exchange_fee(price, number_of_shares, fee_rate):
    fee = (fee_rate / 100) * price * number_of_shares
    return fee


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
#
#

def get_asset_price(client, symbol):
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

def get_coinbase_order_by_order_id(client, order_id):
    try:
        order = client.get_order(order_id=order_id)
        if order:
            return order
        else:
            print(f"No order found with ID: {order_id}.")
            return None
    except Exception as e:
        print(f"Error fetching order with ID {order_id}: {e}")
        return None

#
#
#
#

def place_market_buy_order(client, symbol, base_size):
    try:
        order = client.market_order_buy(
            client_order_id=generate_client_order_id(symbol, 'buy'),  # id must be unique
            product_id=symbol,
            base_size=str(base_size)  # Convert base_size to string
        )

        if 'order_id' in order['response']:
            order_id = order['response']['order_id']
            print(f"BUY ORDER placed successfully. Order ID: {order_id}")

            # Convert the order object to a dictionary if necessary
            order_data = order.response if hasattr(order, 'response') else order
            order_data_dict = order_data.to_dict() if hasattr(order_data, 'to_dict') else order_data

            # Save the placeholder order data until we can lookup the completed transaction
            save_order_data_to_local_json_ledger(symbol, order_data_dict)

            send_email_notification(
                subject="Buy Order Placed",
                text_content=f"BUY ORDER placed successfully for {symbol}. Order ID: {order_id}",
                html_content=f"<h3>BUY ORDER placed successfully for {symbol}. Order ID: {order_id}</h3>"
            )
        else:
            print(f"Unexpected response: {dumps(order)}")
    except Exception as e:
        print(f"Error placing BUY order for {symbol}: {e}")



#
#
#
#



def place_market_sell_order(client, symbol, base_size, potential_profit, potential_profit_percentage):
    try:
        order = client.market_order_sell(
            client_order_id=generate_client_order_id(symbol, 'sell'), # id must be unique
            product_id=symbol,
            base_size=str(base_size)  # Convert base_size to string
        )

        if 'order_id' in order['response']:
            order_id = order['response']['order_id']
            print(f"SELL ORDER placed successfully. Order ID: {order_id}")

            # Clear out existing ledger since there is no need to wait and confirm a sell transaction as long as we got programmatic confirmation
            reset_json_ledger_file(symbol)

            send_email_notification(
                subject=f"Sell Order - {symbol}: ${potential_profit} (+{potential_profit_percentage}%)",
                text_content=f"SELL ORDER placed successfully for {symbol}. Order ID: {order_id}",
                html_content=f"<h3>SELL ORDER placed successfully for {symbol}. Order ID: {order_id}</h3>"
            )
        else:
            print(f"Unexpected response: {dumps(order)}")
    except Exception as e:
        print(f"Error placing SELL order for {symbol}: {e}")


#
#
#
