# boilerplate
# import os
# from dotenv import load_dotenv
# load_dotenv()
# end boilerplate
import numpy as np
import pandas as pd


#
#
# Volume Analysis Functions
#

def volume_strength_index(volume_24h, volume_change_24h, price_change_24h):
    """
    Calculate a volume strength index based on 24-hour metrics.
    :param volume_24h: Total volume in 24 hours
    :param volume_change_24h: Percentage change in volume
    :param price_change_24h: Percentage change in price
    :return: Volume strength score (-1 to 1)
    """
    volume_momentum = volume_change_24h / 100
    price_volume_correlation = (price_change_24h / 100) * (abs(volume_change_24h) / 100) if volume_24h > 0 else 0
    strength_score = (volume_momentum + price_volume_correlation) / 2
    return max(min(strength_score, 1), -1)



def generate_volume_signal(volume_24h, volume_change_24h, price_change_24h, volume_threshold=5000000, volume_change_threshold=-20):
    """
    Generate trading signals based on volume characteristics.
    :param volume_24h: Total volume in 24 hours
    :param volume_change_24h: Percentage change in volume
    :param price_change_24h: Percentage change in price
    :param volume_threshold: Minimum volume for consideration
    :param volume_change_threshold: Minimum volume change threshold
    :return: Trading signal (-1: sell, 0: hold, 1: buy)
    """
    strength_score = volume_strength_index(volume_24h, volume_change_24h, price_change_24h)
    if volume_24h < volume_threshold:
        return 0  # Insufficient volume
    if volume_change_24h < volume_change_threshold:
        return -1 if price_change_24h > 0 else 0  # Potential bearish divergence
    if strength_score > 0.5:
        return 1  # Strong bullish signal
    elif strength_score < -0.5:
        return -1  # Strong bearish signal
    return 0  # Neutral signal



def volume_volatility_indicator(volume_24h, volume_change_24h, price_change_1h):
    """
    Calculate volume volatility.
    :param volume_24h: Total volume in 24 hours
    :param volume_change_24h: Percentage change in volume
    :param price_change_1h: Percentage change in 1 hour
    :return: Volatility score
    """
    volume_volatility = abs(volume_change_24h)
    price_momentum = abs(price_change_1h)
    volatility_score = (volume_volatility * price_momentum) / 100
    return volatility_score



def volume_based_strategy_recommendation(data):
    """
    Scalp trading strategy using volume analysis.
    :param data: Dictionary containing volume and price change data
    :return: Trading recommendation
    """
    volume_24h = data['quote']['USD']['volume_24h']
    volume_change_24h = data['quote']['USD']['volume_change_24h']
    price_change_24h = data['quote']['USD']['percent_change_24h']
    price_change_1h = data['quote']['USD']['percent_change_1h']

    volume_signal = generate_volume_signal(volume_24h, volume_change_24h, price_change_24h)
    volatility = volume_volatility_indicator(volume_24h, volume_change_24h, price_change_1h)

    if volume_signal == 1 and volatility < 2:
        return 'buy'
        # Strong volume support with low volatility"
    elif volume_signal == -1 and volatility > 3:
        return 'sell'
        # Weakening volume and high volatility"
    else:
        return 'hold'
        # Insufficient clear signals"



def detect_volume_spike(current_volume_24h, volume_change_24h):
    """
    Analyzes the volume data of a given coin to detect potential volume spikes.

    Parameters:
    - coin_data: A dictionary containing volume and price change information for a coin.

    Returns:
    - A dictionary containing information about the volume spike.
    """

    try:
        # Assuming the "volume_change_24h" indicates the percentage change in volume, calculate expected volume
        if volume_change_24h is not None:
            previous_volume_24h = current_volume_24h / (1 + (volume_change_24h / 100))
        else:
            print("Volume change data not available.")
            return None

        # Determine if a spike has occurred by checking if the volume has increased significantly above previous volume
        threshold = 1.5  # Example threshold for what we consider a spike (50% more than the previously expected volume)
        spike_occurred = current_volume_24h > previous_volume_24h * threshold

        spike_info = {
            'current_volume_24h': current_volume_24h,
            'previous_volume_24h': previous_volume_24h,
            'volume_change_24h': volume_change_24h,
            'volume_spike_detected': spike_occurred
        }

        return spike_info

    except KeyError as e:
        print(f"Key error: Missing expected data field - {str(e)}")
        return None
