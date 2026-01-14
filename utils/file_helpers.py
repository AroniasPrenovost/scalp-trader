# boilerplate
import os
from dotenv import load_dotenv
load_dotenv()
# end boilerplate
import time
import json


# Function to convert Product objects to dictionaries
def convert_products_to_dicts(products):
    return [product.to_dict() if hasattr(product, 'to_dict') else product for product in products]


#
#
#


# Function to save the list of Coinbase products to a local file
def count_files_in_directory(directory):
    """
    Returns the number of files in the specified directory.

    :param directory: Path to the directory
    :return: Number of files in the directory
    """
    try:
        # List all entries in the directory
        entries = os.listdir(directory)

        # Filter out directories, only count files
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            if file_creation_time < cutoff_time:
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")





# Function to check if the most recent file is older than 30 minutes
def append_to_json_array(file_path, obj):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the existing data from the file
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            if not isinstance(data, list):
                raise ValueError("The JSON data is not an array.")
        except json.JSONDecodeError:
            raise ValueError("The file does not contain valid JSON data.")

    # Check for the existence of the symbol in the current data older than 20 mins
    current_time = time.time()
    twenty_minutes_ago = current_time - 1200  # 20 minutes in seconds
    if any(entry.get('symbol') == obj.get('symbol') and entry.get('timestamp', 0) > twenty_minutes_ago for entry in data):
        return

    # Append the new object to the array

    # Iterate through the data to find the entry with the matching symbol
    for entry in data:
        if entry.get('symbol') == symbol:
            # Calculate time since entry
            current_timestamp = time.time()
            seconds_since_entry = current_timestamp - entry['timestamp']
            time_since_signal = pretty_print_duration(seconds_since_entry)

            # Calculate price change percentage
            original_price = float(entry['price'])
            price_change_percentage = ((float(current_price) - original_price) / original_price) * 100

            return time_since_signal, price_change_percentage

    # If the symbol is not found, return default values
    return "0 hours 0 minutes", 0


#
#
#

    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    # Get a sorted list of JSON files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])

    # Iterate through each file
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Read the JSON data from the file
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError("The JSON data is not an array.")
            except json.JSONDecodeError:
                continue

        # Extract the specified property for the given product_id
        for entry in data:
            if entry.get('product_id') == product_id:
                property_value = entry.get(property_name)
                if property_value is not None:
                    property_values.append(property_value)

    return property_values


#
#
# NEW APPEND-BASED STORAGE FUNCTIONS
#
#

def append_crypto_data_to_file(directory, product_id, data_entry):
    """
    Appends a single crypto data entry to its dedicated JSON file.
    Each crypto gets its own file: {directory}/{product_id}.json
    The file contains a JSON array of all data snapshots.

    :param directory: Path to the directory containing crypto data files
    :param product_id: The product_id (e.g., 'BTC-USD')
    :param data_entry: Dictionary containing the data to append (must include 'timestamp')
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create file path: e.g., coinbase-data/BTC-USD.json
    file_name = f"{product_id}.json"
    file_path = os.path.join(directory, file_name)

    # Add timestamp if not present
    if 'timestamp' not in data_entry:
        data_entry['timestamp'] = time.time()

    # Read existing data or create new array
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new entry
    data.append(data_entry)

    # Write back to file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def get_crypto_data_from_file(directory, product_id, max_age_hours=None):
    """
    Reads all data entries for a specific crypto from its dedicated JSON file.

    :param directory: Path to the directory containing crypto data files
    :param product_id: The product_id (e.g., 'BTC-USD')
    :param max_age_hours: Optional filter to only return entries younger than X hours
    :return: List of data entries (dictionaries)
    """
    file_name = f"{product_id}.json"
    file_path = os.path.join(directory, file_name)

    # Return empty list if file doesn't exist
    if not os.path.exists(file_path):
        return []

    # Read the JSON array
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            if not isinstance(data, list):
                return []
        except json.JSONDecodeError:
            return []

    # Filter by age if specified
    if max_age_hours is not None:
        cutoff_time = time.time() - (max_age_hours * 3600)
        data = [entry for entry in data if entry.get('timestamp', 0) >= cutoff_time]

    return data


def get_property_values_from_crypto_file(directory, product_id, property_name, max_age_hours=None):
    """
    Extracts a specific property from all entries in a crypto's data file.

    :param directory: Path to the directory containing crypto data files
    :param product_id: The product_id (e.g., 'BTC-USD')
    :param property_name: The property to extract from each entry
    :param max_age_hours: Optional filter to only return entries younger than X hours
    :return: List of property values (numeric properties are converted to float)
    """
    entries = get_crypto_data_from_file(directory, product_id, max_age_hours)
    values = []

    for entry in entries:
        if property_name in entry:
            value = entry.get(property_name)

            # Convert numeric string values to float for price/volume properties
            if property_name in ['price', 'volume_24h']:
                # Skip None values
                if value is None:
                    continue
                try:
                    # Handle empty strings and convert to float
                    if isinstance(value, str):
                        value = value.strip()
                        if value == '':
                            continue  # Skip empty strings
                    # Always convert to float for price/volume properties
                    values.append(float(value))
                except (ValueError, TypeError) as e:
                    # Skip values that can't be converted to float
                    print(f"Warning: Skipping invalid {property_name} value: {value} (type: {type(value).__name__}, error: {e})")
                    continue
            else:
                values.append(value)

    return values


def cleanup_old_crypto_data(directory, product_id, max_age_hours, verbose=True):
    """
    Removes old entries from a crypto's data file, keeping only recent data.
    Rewrites the file with only entries younger than max_age_hours.

    :param directory: Path to the directory containing crypto data files
    :param product_id: The product_id (e.g., 'BTC-USD')
    :param max_age_hours: Maximum age in hours for entries to keep
    :param verbose: Whether to print cleanup message (default: True)
    """
    file_name = f"{product_id}.json"
    file_path = os.path.join(directory, file_name)

    if not os.path.exists(file_path):
        return

    # Get all entries that are still fresh
    fresh_entries = get_crypto_data_from_file(directory, product_id, max_age_hours)

    # Rewrite the file with only fresh entries
    with open(file_path, 'w') as file:
        json.dump(fresh_entries, file, indent=4)

    if verbose:
        print(f"Cleaned up old data for {product_id}: kept {len(fresh_entries)} entries")


def cleanup_old_screenshots(screenshots_dir, transactions_dir, config):
    """
    Cleans up old screenshots based on a tiered retention system:
    - Analysis screenshots (multi-timeframe, volume, snapshot): Keep for X hours (default 24)
    - Buy/Sell event screenshots: Keep for Y days (default 30) or until referenced in open positions

    :param screenshots_dir: Path to the screenshots directory
    :param transactions_dir: Path to the transactions directory
    :param config: Config object with screenshot_retention settings
    :return: Dictionary with cleanup statistics
    """
    if not os.path.exists(screenshots_dir):
        print(f"Screenshots directory {screenshots_dir} does not exist.")
        return {"deleted": 0, "kept": 0, "errors": 0}

    # Get retention settings from config
    retention_config = config.get('screenshot_retention', {})
    analysis_retention_hours = retention_config.get('analysis_hours', 24)
    event_retention_days = retention_config.get('event_days', 30)

    # Calculate cutoff times
    current_time = time.time()
    analysis_cutoff = current_time - (analysis_retention_hours * 3600)
    event_cutoff = current_time - (event_retention_days * 24 * 3600)

    # Get list of screenshots referenced in active transactions
    protected_screenshots = _get_protected_screenshots(transactions_dir)

    stats = {"deleted": 0, "kept": 0, "errors": 0, "size_freed_mb": 0}

    for filename in os.listdir(screenshots_dir):
        if not filename.endswith('.png'):
            continue

        file_path = os.path.join(screenshots_dir, filename)

        try:
            # Get file creation time and size
            file_creation_time = os.path.getctime(file_path)
            file_size = os.path.getsize(file_path)

            # Check if this screenshot is protected (referenced in open positions)
            if file_path in protected_screenshots or filename in protected_screenshots:
                stats["kept"] += 1
                continue

            # Determine screenshot type and apply appropriate retention policy
            should_delete = False

            if _is_analysis_screenshot(filename):
                # Analysis screenshots (multi-timeframe, volume, snapshot)
                if file_creation_time < analysis_cutoff:
                    should_delete = True
            elif _is_event_screenshot(filename):
                # Buy/sell event screenshots
                if file_creation_time < event_cutoff:
                    should_delete = True

            if should_delete:
                os.remove(file_path)
                stats["deleted"] += 1
                stats["size_freed_mb"] += file_size / (1024 * 1024)
            else:
                stats["kept"] += 1

        except Exception as e:
            print(f"Error processing screenshot {filename}: {e}")
            stats["errors"] += 1

    if stats["deleted"] > 0:
        print(f"Screenshot cleanup: Deleted {stats['deleted']} files, freed {stats['size_freed_mb']:.2f} MB, kept {stats['kept']} files")

    return stats


def _is_analysis_screenshot(filename):
    """
    Determines if a screenshot is an analysis screenshot (multi-timeframe, volume, or snapshot).

    Analysis screenshot patterns:
    - {symbol}_{timeframe}-window_{candle_label}-candles_{timestamp}.png
    - {symbol}_volume-24h-snapshots_{timestamp}.png
    - {symbol}_{candle_label}-candles_snapshot_{timestamp}.png

    :param filename: Screenshot filename
    :return: True if analysis screenshot
    """
    # Multi-timeframe analysis charts
    if '-window_' in filename:
        return True

    # Volume trend charts
    if 'volume-24h-snapshots' in filename:
        return True

    # Simple snapshot charts
    if '_snapshot_' in filename:
        return True

    return False


def _is_event_screenshot(filename):
    """
    Determines if a screenshot is a buy/sell event screenshot.

    Event screenshot patterns:
    - {symbol}_{candle_label}-candles_buy_{timestamp}.png
    - {symbol}_{candle_label}-candles_sell_{timestamp}.png

    :param filename: Screenshot filename
    :return: True if event screenshot
    """
    return '_buy_' in filename or '_sell_' in filename


def _get_protected_screenshots(transactions_dir):
    """
    Gets list of screenshot paths that should be protected from deletion.
    These are screenshots referenced in open positions (transactions without sell_time).

    :param transactions_dir: Path to the transactions directory
    :return: Set of protected screenshot paths/filenames
    """
    protected = set()

    if not os.path.exists(transactions_dir):
        return protected

    # Check all transaction files (symbol-specific JSON files)
    for filename in os.listdir(transactions_dir):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(transactions_dir, filename)

        try:
            with open(file_path, 'r') as file:
                transactions = json.load(file)

                if not isinstance(transactions, list):
                    continue

                # Find open positions (no sell_time)
                for transaction in transactions:
                    if 'sell_time' not in transaction or transaction.get('sell_time') is None:
                        # This is an open position, protect its screenshots
                        if 'buy_screenshot_path' in transaction:
                            screenshot_path = transaction['buy_screenshot_path']
                            protected.add(screenshot_path)
                            # Also add just the filename in case path formats differ
                            protected.add(os.path.basename(screenshot_path))

        except Exception as e:
            print(f"Error reading transaction file {filename}: {e}")

    return protected
