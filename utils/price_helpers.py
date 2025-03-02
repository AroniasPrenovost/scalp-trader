#
#
#
#

def calculate_trading_range_percentage(num1, num2):
    if num1 == 0 and num2 == 0:
        return "0.00"
    difference = abs(num1 - num2)
    average = (num1 + num2) / 2
    percentage_difference = (difference / average) * 100
    return f"{percentage_difference:.2f}"

#
#
#
#

def calculate_current_price_position_within_trading_range(current_price, min, max):
    """
    Calculate the position of the current price within the trading range.

    :param current_price: The current price of the asset
    :param min: The min level price
    :param max: The max level price
    :return: The position of the current price within the trading range as a percentage
    """
    if max == min:
        return 0.0  # Avoid division by zero

    trading_range = max - min
    position_within_range = ((current_price - min) / trading_range) * 100

    return round(position_within_range, 2)

#
#
# Used for visualizing on the chart
#

def calculate_offset_price(price, trend, percentage):
    """
    Calculate the offset price based on the trend and percentage.

    :param price: The original price.
    :param trend: The trend direction ('upward', 'downward', 'bullish', 'bearish').
    :param percentage: The percentage to offset the price.
    :return: The offset price.
    """
    if trend in ['upward', 'bullish']:
        price_trend_offset = price * (percentage / 100)
        return price + price_trend_offset
    elif trend in ['downward', 'bearish']:
        price_trend_offset = price * (-percentage / 100)
        return price + price_trend_offset
    return price
