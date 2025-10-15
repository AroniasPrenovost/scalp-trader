# boilerplate
import os
from dotenv import load_dotenv
load_dotenv()
# end boilerplate


### Explanation:
# Market Cap Efficiency Ratio:
#   This ratio is calculated by dividing the current market cap by the fully diluted market cap. It gives an idea of how much of the potential market cap is currently realized.
# Usage: A ratio closer to 1 indicates that the current market cap is close to the fully diluted market cap, suggesting that most of the potential supply is already in circulation.
#   A lower ratio indicates that there is still a significant amount of potential supply that is not yet realized in the market cap.
def calculate_market_cap_efficiency(market_cap, fully_diluted_market_cap):
    """
    Calculate the Market Cap Efficiency Ratio and provide a descriptive assessment.

    :param market_cap: Current market capitalization
    :param fully_diluted_market_cap: Fully diluted market capitalization
    :return: Tuple containing the Market Cap Efficiency Ratio and a descriptive string
    """
    if fully_diluted_market_cap == 0:
        raise ValueError("Fully diluted market cap cannot be zero.")

    efficiency_ratio = market_cap / fully_diluted_market_cap

    if efficiency_ratio >= 0.8:
        description = "Almost full"
    elif 0.5 <= efficiency_ratio < 0.8:
        description = "Moderate"
    else:
        description = "Low"

    return efficiency_ratio, description

    # Example usage
    # market_cap = 1949346935.5773864
    # fully_diluted_market_cap = 2200864628.78
    #
    # efficiency_ratio, description = calculate_market_cap_efficiency(market_cap, fully_diluted_market_cap)
    # print(f"Market Cap Efficiency Ratio: {efficiency_ratio:.4f} - {description}")


#
#
#

def calculate_fibonacci_levels(prices):
    """
    Calculate Fibonacci retracement levels for a given set of stock prices.

    :param prices: deque of stock prices
    :return: dictionary containing Fibonacci levels
    """
    if not prices or len(prices) < 2:
        raise ValueError("Prices deque must contain at least two elements.")

    high = max(prices)
    low = min(prices)

    # Calculate Fibonacci levels
    diff = high - low
    levels = {
        'level_0': high,
        'level_23.6': high - 0.236 * diff,
        'level_38.2': high - 0.382 * diff,
        'level_50': high - 0.5 * diff,
        'level_61.8': high - 0.618 * diff,
        'level_100': low
    }

    return levels
