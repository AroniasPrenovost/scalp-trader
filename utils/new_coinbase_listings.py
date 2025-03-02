# boilerplate
import os
# end boilerplate
import json

# Function to check if the file has changed and return new objects
def check_for_new_coinbase_listings(file_path, new_listed_coins):
    if not os.path.exists(file_path):
        return new_listed_coins  # If file doesn't exist, all coins are new

    with open(file_path, 'r') as file:
        old_listed_coins = json.load(file)

    # Find new coins
    old_coin_ids = {coin['product_id'] for coin in old_listed_coins}
    new_coins = [coin for coin in new_listed_coins if coin['product_id'] not in old_coin_ids]

    return new_coins
