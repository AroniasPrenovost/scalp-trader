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
