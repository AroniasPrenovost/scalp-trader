from typing import List, Optional


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


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).

    RSI < 30 = Oversold (good time to buy)
    RSI > 70 = Overbought (good time to sell)
    RSI ~50 = Neutral (mean)

    Args:
        prices: Historical prices
        period: RSI period (default 14)

    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None

    # Calculate price changes
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    # Use last 'period' changes
    recent_gains = gains[-(period):]
    recent_losses = losses[-(period):]

    # Calculate average gain and loss
    avg_gain = sum(recent_gains) / period
    avg_loss = sum(recent_losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
