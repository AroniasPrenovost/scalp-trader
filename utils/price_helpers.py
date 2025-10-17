#
#
#

def calculate_price_change_percentage(old_price, new_price):
    if old_price == 0:
        return 0
    return ((new_price - old_price) / old_price) * 100

#
#
#
#

def calculate_percentage_from_min(min_price, max_price):
    """
    Calculate the percentage change from minimum to maximum price.

    :param min_price: The minimum price in the range
    :param max_price: The maximum price in the range
    :return: Percentage change from min formatted as string
    """
    if min_price == 0:
        return "0.00"
    percentage_change = ((max_price - min_price) / min_price) * 100
    return f"{percentage_change:.2f}"

#
#
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
