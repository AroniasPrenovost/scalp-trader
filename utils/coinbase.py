# boilerplate
import os
from dotenv import load_dotenv
load_dotenv()
# end boilerplate
import time
import json
from json import dumps, load

from utils.email import send_email_notification

# Function to convert Product objects to dictionaries
def convert_products_to_dicts(products):
    return [product.to_dict() if hasattr(product, 'to_dict') else product for product in products]

#
#
# Save order data to local json ledger
#

def save_order_data_to_local_json_ledger(symbol, order_data):
    """
    Save order data to a local file specific to the symbol.
    """
    file_name = f"{symbol}_orders.json"
    try:
        # Load existing data if the file exists and is not empty
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        # Append the new order data
        existing_data.append(order_data)

        # Save the updated data back to the file
        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

        print(f"Order data saved to {file_name}.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_name}. The file might be corrupted.")
        # Attempt to overwrite the file with the new order data
        with open(file_name, 'w') as file:
            json.dump([order_data], file, indent=4)
        print(f"Order data saved to {file_name} after resetting the file.")
    except Exception as e:
        print(f"Error saving order data for {symbol}: {e}")




#


def reset_json_ledger_file(symbol):
    # Construct the file name based on the symbol
    file_name = f"{symbol}_orders.json"

    # Define the relative path to the root directory
    file_path = f"../{file_name}"

    print(f"Resetting file: {file_path}")

    # Write an empty list to the file to reset it
    with open(file_path, 'w') as file:
        json.dump([], file)






def detect_stored_coinbase_order_type(last_order):
    if last_order is None:
        return 'none'
    if 'order' in last_order: # if 'order' key exists, it's a finalized order
        if 'side' in last_order['order']:
            type = last_order['order']['side']
            return type.lower()
    else:
        return 'placeholder'

#
#
#


from coinbase.rest import RESTClient

coinbase_api_key = os.environ.get('COINBASE_API_KEY')
coinbase_api_secret = os.environ.get('COINBASE_API_SECRET')

coinbase_spot_maker_fee = float(os.environ.get('COINBASE_SPOT_MAKER_FEE'))
coinbase_spot_taker_fee = float(os.environ.get('COINBASE_SPOT_TAKER_FEE'))
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

        # Convert the order object to a dictionary if necessary
        order_data_dict = convert_products_to_dicts([order])[0]

        if 'order_id' in order_data_dict['response']:
            order_id = order_data_dict['response']['order_id']
            print(f"BUY ORDER placed successfully. Order ID: {order_id}")

            # Save the placeholder order data until we can lookup the completed transaction
            save_order_data_to_local_json_ledger(symbol, order_data_dict)

            send_email_notification(
                subject="Buy Order Placed",
                text_content=f"BUY ORDER placed successfully for {symbol}. Order ID: {order_id}",
                html_content=f"<h3>BUY ORDER placed successfully for {symbol}. Order ID: {order_id}</h3>"
            )
        else:
            print(f"Unexpected response: {dumps(order_data_dict)}")
    except Exception as e:
        print(f"Error placing BUY order for {symbol}: {e}")


def place_market_sell_order(client, symbol, base_size, potential_profit, potential_profit_percentage):
    try:
        order = client.market_order_sell(
            client_order_id=generate_client_order_id(symbol, 'sell'), # id must be unique
            product_id=symbol,
            base_size=str(base_size)  # Convert base_size to string
        )

        # Convert the order object to a dictionary if necessary
        order_data_dict = convert_products_to_dicts([order])[0]

        if 'order_id' in order_data_dict['response']:
            order_id = order_data_dict['response']['order_id']
            print(f"SELL ORDER placed successfully. Order ID: {order_id}")

            # Clear out existing ledger since there is no need to wait and confirm a sell transaction as long as we got programmatic confirmation
            reset_json_ledger_file(symbol)

            send_email_notification(
                subject=f"Sell Order - {symbol}: ${potential_profit} (+{potential_profit_percentage}%)",
                text_content=f"SELL ORDER placed successfully for {symbol}. Order ID: {order_id}",
                html_content=f"<h3>SELL ORDER placed successfully for {symbol}. Order ID: {order_id}</h3>"
            )
        else:
            print(f"Unexpected response: {dumps(order_data_dict)}")
    except Exception as e:
        print(f"Error placing SELL order for {symbol}: {e}")


#
#
# get_last_order_from_local_json_ledger
#

def get_last_order_from_local_json_ledger(symbol):
    """
    Retrieve the most recent order from the local JSON ledger for the given symbol.
    """
    file_name = f"{symbol}_orders.json"
    try:
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, 'r') as file:
                orders = json.load(file)
                if orders:
                    # Return the most recent order
                    return orders[-1]
        print(f"No orders found in ledger for {symbol}.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_name}. The file might be corrupted.")
        return None
    except Exception as e:
        print(f"Error retrieving order from ledger for {symbol}: {e}")
        return None
