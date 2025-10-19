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


def get_current_fee_rates(client):
    """
    Fetch current taker and maker fee rates from Coinbase Advanced Trade API.

    Returns:
        dict: {
            'taker_fee': float (percentage, e.g., 0.6 for 0.6%),
            'maker_fee': float (percentage, e.g., 0.4 for 0.4%),
            'tier': str (fee tier name, e.g., 'Advanced 1'),
            'pricing_tier': str (detailed tier info),
            'usd_from': str (volume range start),
            'usd_to': str (volume range end)
        }
        or None if there's an error
    """
    try:
        summary = client.get_transaction_summary()
        # Convert response object to dict if it has to_dict method
        if hasattr(summary, 'to_dict'):
            summary = summary.to_dict()
        fee_tier = summary.get('fee_tier', {})

        # Convert fee rates from decimal to percentage (e.g., 0.006 -> 0.6%)
        taker_fee_rate = float(fee_tier.get('taker_fee_rate', 0)) * 100
        maker_fee_rate = float(fee_tier.get('maker_fee_rate', 0)) * 100

        fee_info = {
            'taker_fee': taker_fee_rate,
            'maker_fee': maker_fee_rate,
            'tier': fee_tier.get('pricing_tier', 'Unknown'),
            'pricing_tier': fee_tier.get('pricing_tier', 'Unknown'),
            'usd_from': fee_tier.get('usd_from', '0'),
            'usd_to': fee_tier.get('usd_to', 'N/A')
        }

        # print(f"Current fee rates - Taker: {taker_fee_rate}%, Maker: {maker_fee_rate}% (Tier: {fee_info['tier']})")
        return fee_info

    except Exception as e:
        print(f"Error fetching current fee rates: {e}")
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


def save_trade_record(symbol, buy_price, sell_price, total_profit_percentage, taxes, exchange_fee, total_profit):
    """
    Append a completed trade record to the trade history JSON file.
    """
    file_name = "trade_history.json"
    try:
        # Load existing trade history if the file exists and is not empty
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, 'r') as file:
                trade_history = json.load(file)
        else:
            trade_history = []

        # Create the trade record
        trade_record = {
            "symbol": symbol,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "buy_price": buy_price,
            "sell_price": sell_price,
            "total_profit_percentage": total_profit_percentage,
            "taxes": taxes,
            "exchange_fee": exchange_fee,
            "total_profit": total_profit
        }

        # Append the new trade record
        trade_history.append(trade_record)

        # Save the updated trade history back to the file
        with open(file_name, 'w') as file:
            json.dump(trade_history, file, indent=4)

        print(f"Trade record saved to {file_name}.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_name}. The file might be corrupted.")
        # Attempt to overwrite the file with the new trade record
        with open(file_name, 'w') as file:
            json.dump([trade_record], file, indent=4)
        print(f"Trade record saved to {file_name} after resetting the file.")
    except Exception as e:
        print(f"Error saving trade record for {symbol}: {e}")


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
# save_transaction_record
#

def save_transaction_record(symbol, buy_price, sell_price, potential_profit_percentage, gross_profit, taxes, exchange_fees, total_profit, buy_timestamp, buy_screenshot_path=None, analysis=None, entry_market_conditions=None, exit_trigger=None, position_sizing_data=None):
    """
    Store/append successful transaction records to /transactions/data.json

    Args:
        symbol: Trading pair symbol
        buy_price: Entry price
        sell_price: Exit price
        potential_profit_percentage: Profit percentage
        gross_profit: Gross profit before fees/taxes
        taxes: Taxes owed
        exchange_fees: Exchange fees
        total_profit: Net profit after all costs
        buy_timestamp: Timestamp when position was opened
        buy_screenshot_path: Optional path to the buy event screenshot
        analysis: Optional dict containing AI analysis data (support, resistance, reasoning, etc.)
        entry_market_conditions: Optional dict with market context at entry (volatility, trend, etc.)
        exit_trigger: Optional string indicating what triggered the exit ('profit_target', 'stop_loss', 'manual')
        position_sizing_data: Optional dict with position sizing decisions (amount, allocation %, etc.)
    """
    import datetime

    # Calculate time held position
    sell_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Parse buy timestamp and calculate time held
    time_held_seconds = None
    if buy_timestamp:
        try:
            buy_time = datetime.datetime.fromisoformat(buy_timestamp.replace('Z', '+00:00'))
            sell_time = datetime.datetime.now(datetime.timezone.utc)
            time_held_seconds = (sell_time - buy_time).total_seconds()
            time_held_position = f"{time_held_seconds / 3600:.2f} hours"
        except Exception as e:
            print(f"Error calculating time held: {e}")
            time_held_position = "unknown"
    else:
        time_held_position = "unknown"

    # Create base transaction record
    transaction_record = {
        "symbol": symbol,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "potential_profit_percentage": potential_profit_percentage,
        "timestamp": sell_timestamp,
        "gross_profit": gross_profit,
        "taxes": taxes,
        "exchange_fees": exchange_fees,
        "total_profit": total_profit,
        "time_held_position": time_held_position,
        "time_held_seconds": time_held_seconds,
        "buy_timestamp": buy_timestamp,
        "buy_screenshot_path": buy_screenshot_path
    }

    # Add market context at entry if provided
    if entry_market_conditions:
        transaction_record["market_context_at_entry"] = entry_market_conditions

    # Add technical signals and AI analysis if provided
    if analysis:
        # Extract technical signals from analysis
        technical_signals = {
            "major_support": analysis.get("major_support"),
            "minor_support": analysis.get("minor_support"),
            "major_resistance": analysis.get("major_resistance"),
            "minor_resistance": analysis.get("minor_resistance"),
            "buy_in_price_target": analysis.get("buy_in_price"),
            "actual_entry_price": buy_price,
            "stop_loss_set": analysis.get("stop_loss"),
            "profit_target_percentage": analysis.get("profit_target_percentage"),
            "actual_profit_percentage": potential_profit_percentage,
            "risk_reward_ratio": analysis.get("risk_reward_ratio"),
        }

        # Calculate price position relative to support/resistance
        if analysis.get("major_support"):
            support_distance = ((buy_price - analysis["major_support"]) / analysis["major_support"]) * 100
            technical_signals["price_vs_support"] = f"{support_distance:.2f}% above major support"

        if analysis.get("major_resistance"):
            resistance_distance = ((analysis["major_resistance"] - buy_price) / buy_price) * 100
            technical_signals["price_vs_resistance"] = f"{resistance_distance:.2f}% below major resistance"

        transaction_record["technical_signals_at_entry"] = technical_signals

        # Store AI model metadata
        transaction_record["ai_model_data"] = {
            "model_used": analysis.get("model_used"),
            "analyzed_at": analysis.get("analyzed_at"),
            "confidence_level": analysis.get("confidence_level"),
            "trade_recommendation": analysis.get("trade_recommendation"),
            "market_trend": analysis.get("market_trend"),
            "volume_confirmation": analysis.get("volume_confirmation"),
            "reasoning": analysis.get("reasoning"),
            "trade_invalidation_price": analysis.get("trade_invalidation_price"),
        }

    # Add exit analysis
    exit_analysis = {
        "exit_trigger": exit_trigger or "unknown",
        "profit_target_percentage": analysis.get("profit_target_percentage") if analysis else None,
        "actual_profit_percentage": potential_profit_percentage,
    }
    transaction_record["exit_analysis"] = exit_analysis

    # Add position sizing data if provided
    if position_sizing_data:
        transaction_record["position_sizing"] = position_sizing_data

    # Create transactions directory if it doesn't exist
    transactions_dir = "transactions"
    if not os.path.exists(transactions_dir):
        os.makedirs(transactions_dir)

    # Path to data.json
    file_path = os.path.join(transactions_dir, "data.json")

    # Load existing data or create new list
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        else:
            data = []
    except Exception as e:
        print(f"Error reading transactions file: {e}")
        data = []

    # Append new transaction
    data.append(transaction_record)

    # Save back to file
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Transaction record saved to {file_path}")
    except Exception as e:
        print(f"Error saving transaction record: {e}")


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
