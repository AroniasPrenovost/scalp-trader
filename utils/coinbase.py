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
