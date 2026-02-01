"""
Technical Indicators for Trading Strategies

Implements common technical analysis indicators:
- Bollinger Bands (volatility squeeze detection)
- MACD (momentum and trend direction)
- ATR (Average True Range - volatility measurement)
- Moving Averages (trend identification)
"""

from typing import List, Optional, Dict, Tuple
import statistics


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        prices: Historical prices
        period: Number of periods for the moving average

    Returns:
        SMA value or None if insufficient data
    """
    if len(prices) < period:
        return None

    return statistics.mean(prices[-period:])


def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[Dict[str, float]]:
    """
    Calculate Bollinger Bands (upper band, lower band, middle band, and bandwidth).

    Bollinger Bands squeeze (low bandwidth) indicates consolidation and potential breakout.

    Args:
        prices: Historical prices
        period: Moving average period (default 20)
        std_dev: Number of standard deviations (default 2.0)

    Returns:
        Dictionary with 'upper', 'lower', 'middle', 'bandwidth' or None if insufficient data
    """
    if len(prices) < period:
        return None

    recent_prices = prices[-period:]
    middle_band = statistics.mean(recent_prices)
    std = statistics.stdev(recent_prices)

    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)

    # Bandwidth as percentage of middle band (measures squeeze intensity)
    bandwidth = ((upper_band - lower_band) / middle_band) * 100 if middle_band > 0 else 0

    return {
        'upper': upper_band,
        'lower': lower_band,
        'middle': middle_band,
        'bandwidth': bandwidth,
        'std': std
    }


def is_bollinger_squeeze(prices: List[float], period: int = 20, lookback: int = 20) -> bool:
    """
    Detect if Bollinger Bands are in a squeeze (abnormally low volatility).

    A squeeze occurs when the current bandwidth is near the minimum of recent periods,
    indicating energy buildup before a potential breakout.

    Args:
        prices: Historical prices
        period: Bollinger Band period
        lookback: How many periods to check for minimum bandwidth

    Returns:
        True if in squeeze, False otherwise
    """
    if len(prices) < period + lookback:
        return False

    # Calculate current bandwidth
    current_bb = calculate_bollinger_bands(prices, period)
    if not current_bb:
        return False

    current_bandwidth = current_bb['bandwidth']

    # Calculate historical bandwidths
    bandwidths = []
    for i in range(lookback):
        historical_prices = prices[-(period + lookback - i):-(lookback - i)] if i < lookback - 1 else prices[-period:]
        bb = calculate_bollinger_bands(historical_prices, period)
        if bb:
            bandwidths.append(bb['bandwidth'])

    if not bandwidths:
        return False

    # Squeeze detected if current bandwidth is in the lowest 20% of recent history
    min_bandwidth = min(bandwidths)
    threshold = min_bandwidth * 1.2  # Within 20% of minimum

    return current_bandwidth <= threshold


def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[Dict[str, float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD is a momentum indicator showing the relationship between two moving averages.

    Args:
        prices: Historical prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        Dictionary with 'macd', 'signal', 'histogram' or None if insufficient data
    """
    if len(prices) < slow_period + signal_period:
        return None

    # Calculate EMAs
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    if fast_ema is None or slow_ema is None:
        return None

    # MACD line = Fast EMA - Slow EMA
    macd_line = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD line)
    # For simplicity, we'll calculate MACD values for signal_period and then get their EMA
    macd_values = []
    for i in range(signal_period):
        idx = -(signal_period - i)
        if idx == 0:
            subset = prices
        else:
            subset = prices[:idx]

        if len(subset) >= slow_period:
            f_ema = calculate_ema(subset, fast_period)
            s_ema = calculate_ema(subset, slow_period)
            if f_ema is not None and s_ema is not None:
                macd_values.append(f_ema - s_ema)

    if len(macd_values) < signal_period:
        return None

    signal_line = statistics.mean(macd_values)  # Simplified signal calculation

    # Histogram = MACD - Signal
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent prices than SMA.

    Args:
        prices: Historical prices
        period: Number of periods

    Returns:
        EMA value or None if insufficient data
    """
    if len(prices) < period:
        return None

    multiplier = 2 / (period + 1)

    # Start with SMA as initial EMA
    ema = statistics.mean(prices[:period])

    # Calculate EMA for remaining prices
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return ema


def calculate_atr(candles: List[Dict], period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR) - volatility indicator.

    ATR measures market volatility by decomposing the entire range of price movement.
    Used for setting dynamic stop-losses.

    Args:
        candles: List of candle dicts with 'high', 'low', 'close' keys
        period: Number of periods (default 14)

    Returns:
        ATR value or None if insufficient data
    """
    if len(candles) < period + 1:
        return None

    true_ranges = []

    for i in range(1, len(candles)):
        high = float(candles[i]['high'])
        low = float(candles[i]['low'])
        prev_close = float(candles[i-1]['close'])

        # True Range = max of:
        # 1. Current High - Current Low
        # 2. abs(Current High - Previous Close)
        # 3. abs(Current Low - Previous Close)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    # Take the most recent 'period' true ranges
    recent_tr = true_ranges[-period:]

    return statistics.mean(recent_tr)


def calculate_volume_average(volumes: List[float], period: int = 10) -> Optional[float]:
    """
    Calculate average volume over a period.

    Args:
        volumes: Historical volume data
        period: Number of periods to average

    Returns:
        Average volume or None if insufficient data
    """
    if len(volumes) < period:
        return None

    return statistics.mean(volumes[-period:])


def is_volume_spike(volumes: List[float], multiplier: float = 2.0, period: int = 10) -> bool:
    """
    Detect if current volume is significantly higher than average (volume spike).

    High volume confirms breakout validity (prevents fakeouts).

    Args:
        volumes: Historical volume data
        multiplier: How many times average volume (default 2.0x)
        period: Period for average calculation

    Returns:
        True if volume spike detected, False otherwise
    """
    if len(volumes) < period + 1:
        return False

    avg_volume = calculate_volume_average(volumes[:-1], period)  # Average of previous periods
    current_volume = volumes[-1]

    if not avg_volume or avg_volume == 0:
        return False

    return current_volume >= (avg_volume * multiplier)


def aggregate_candles_to_timeframe(candles_5min: List[Dict], target_minutes: int) -> List[Dict]:
    """
    Aggregate 5-minute candles into a higher timeframe (e.g., 4-hour = 240 minutes).

    Args:
        candles_5min: List of 5-minute candle dicts with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        target_minutes: Target timeframe in minutes (e.g., 240 for 4H, 1440 for Daily)

    Returns:
        List of aggregated candles
    """
    if not candles_5min or len(candles_5min) == 0:
        return []

    candles_per_period = target_minutes // 5  # How many 5-min candles in target timeframe
    aggregated = []

    # Group candles by period
    for i in range(0, len(candles_5min), candles_per_period):
        period_candles = candles_5min[i:i + candles_per_period]

        if len(period_candles) == 0:
            continue

        # Aggregate OHLCV
        agg_candle = {
            'timestamp': period_candles[0]['timestamp'],  # Use first timestamp
            'open': period_candles[0]['open'],
            'high': max(float(c['high']) for c in period_candles),
            'low': min(float(c['low']) for c in period_candles),
            'close': period_candles[-1]['close'],
            'volume': sum(float(c.get('volume', 0)) for c in period_candles)
        }

        aggregated.append(agg_candle)

    return aggregated
