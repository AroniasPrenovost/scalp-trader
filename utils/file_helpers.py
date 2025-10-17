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
def save_obj_dict_to_file(file_path, data):
    # Convert products to dictionaries
    data_dicts = convert_products_to_dicts(data)
    # Save to file
    with open(file_path, 'w') as file:
        json.dump(data_dicts, file, indent=4)
    print(f"JSON object saved to {file_path}.\n")


#
#
#

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
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]

        return len(files)
    except Exception as e:
        print(f"Error counting files in directory {directory}: {e}")
        return None





# Function to delete files older than a specified number of hours
def delete_files_older_than_x_hours(directory, hours):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    cutoff_time = time.time() - (hours * 3600)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            if file_creation_time < cutoff_time:
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")





# Function to check if the most recent file is older than 30 minutes
def is_most_recent_file_older_than_x_minutes(directory, minutes):
    if not os.path.exists(directory):
        return True

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        return True

    most_recent_file = max(files, key=os.path.getctime)
    file_creation_time = os.path.getctime(most_recent_file)
    return (time.time() - file_creation_time) > (minutes * 60)




#
#
#





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
    data.append(obj)

    # Write the updated array back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage:
# append_to_json_array('data.json', {"new_key": "new_value"})




def pretty_print_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    hrs = ''
    if hours > 0:
        hrs = f"{hours} hrs"
    mins = ''
    if minutes > 0:
        mins = f"{minutes} mins"
    return f"{hrs} {mins}".strip()

def calculate_price_change(file_path, symbol, current_price):
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
            return "0 hours 0 minutes", 0

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

def remove_old_entries(file_path, max_hours_old):
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

    # Get the current timestamp
    current_timestamp = time.time()

    # Filter out entries that are older than the specified number of hours
    filtered_data = [
        entry for entry in data
        if (current_timestamp - entry['timestamp']) / 3600 <= max_hours_old
    ]

    # Write the filtered data back to the file
    with open(file_path, 'w') as file:
        json.dump(filtered_data, file, indent=4)

    # Example usage:
    # remove_old_entries('/path/to/your/data.json', 24)

#
#
#


def get_property_values_from_files(directory, product_id, property_name):
    """
    Iterates through JSON files in the specified directory and extracts the specified property
    for the given product_id.

    :param directory: Path to the directory containing JSON files
    :param product_id: The product_id to search for in the files
    :param property_name: The property to extract from the product data
    :return: List of property values for the specified product_id
    """
    property_values = []

    # Ensure the directory exists
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


def cleanup_old_crypto_data(directory, product_id, max_age_hours):
    """
    Removes old entries from a crypto's data file, keeping only recent data.
    Rewrites the file with only entries younger than max_age_hours.

    :param directory: Path to the directory containing crypto data files
    :param product_id: The product_id (e.g., 'BTC-USD')
    :param max_age_hours: Maximum age in hours for entries to keep
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

    print(f"Cleaned up old data for {product_id}: kept {len(fresh_entries)} entries")
