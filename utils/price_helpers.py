def calculate_trading_range_percentage(num1, num2):
    if num1 == 0 and num2 == 0:
        return "0.00"
    difference = abs(num1 - num2)
    average = (num1 + num2) / 2
    percentage_difference = (difference / average) * 100
    return f"{percentage_difference:.2f}"

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
