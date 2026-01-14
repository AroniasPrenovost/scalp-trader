def calculate_percentage_from_min(min_price, max_price):
    """
    Calculate the percentage change from minimum to maximum price.

    :param min_price: The minimum price in the range
    :param max_price: The maximum price in the range
    :return: Percentage change from min as float
    """
    if min_price == 0:
        return 0.0
    percentage_change = ((max_price - min_price) / min_price) * 100
    return percentage_change
