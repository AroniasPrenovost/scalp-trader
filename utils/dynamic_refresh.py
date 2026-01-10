import numpy as np
import pandas as pd


def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Args:
        prices: List of price values
        period: RSI period (default 14)

    Returns:
        RSI value (0-100), or None if insufficient data
    """
    if len(prices) < period + 1:
        return None

    # Clean prices to ensure all numeric values
    clean_prices = []
    for p in prices:
        try:
            clean_prices.append(float(p))
        except (ValueError, TypeError):
            continue

    if len(clean_prices) < period + 1:
        return None

    prices_array = np.array(clean_prices, dtype=float)
    deltas = np.diff(prices_array)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def should_trigger_dynamic_refresh(
    symbol,
    current_price,
    price_history,
    volume_history,
    current_volume,
    analysis,
    config
):
    """
    Determines if market conditions have changed significantly enough to warrant
    a new analysis, even if the normal time-based refresh hasn't been triggered.

    This checks for:
    1. Significant price movement since last analysis
    2. Volume spikes indicating new market activity
    3. RSI approaching oversold/overbought extremes
    4. Price approaching key support/resistance levels

    Args:
        symbol: Trading pair symbol
        current_price: Current market price
        price_history: List of recent prices
        volume_history: List of recent 24h volumes
        current_volume: Current 24h volume
        analysis: Existing analysis dictionary (or None)
        config: Config dictionary with dynamic_refresh settings

    Returns:
        Tuple of (should_refresh: bool, reason: str)
    """

    # If dynamic refresh is disabled, don't trigger
    dynamic_config = config.get('dynamic_refresh', {})
    if not dynamic_config.get('enabled', False):
        return False, None

    # If no existing analysis, let normal logic handle it
    if not analysis:
        return False, None

    # Get thresholds from config
    price_change_threshold = dynamic_config.get('price_change_threshold_percentage', 2.0)
    volume_spike_multiplier = dynamic_config.get('volume_spike_threshold_multiplier', 1.5)
    rsi_oversold = dynamic_config.get('rsi_oversold_threshold', 30)
    rsi_overbought = dynamic_config.get('rsi_overbought_threshold', 70)
    support_resistance_proximity = dynamic_config.get('support_resistance_proximity_percentage', 1.0)

    # Get price at last analysis
    analyzed_price = analysis.get('buy_in_price')
    if not analyzed_price:
        return False, None

    # Check 1: Significant price movement since analysis
    price_change_pct = abs((current_price - analyzed_price) / analyzed_price * 100)
    if price_change_pct >= price_change_threshold:
        return True, f"Price moved {price_change_pct:.2f}% since last analysis (threshold: {price_change_threshold}%)"

    # Check 2: Volume spike detection
    if len(volume_history) >= 5 and current_volume is not None:
        # Clean volume data to ensure all numeric
        clean_volumes = []
        for v in volume_history:
            try:
                clean_volumes.append(float(v))
            except (ValueError, TypeError):
                continue

        # Ensure current_volume is numeric
        try:
            current_volume_float = float(current_volume)
        except (ValueError, TypeError):
            current_volume_float = None

        if len(clean_volumes) >= 5 and current_volume_float is not None:
            recent_avg_volume = np.mean(clean_volumes[-24:]) if len(clean_volumes) >= 24 else np.mean(clean_volumes)
            if current_volume_float > recent_avg_volume * volume_spike_multiplier:
                volume_increase_pct = ((current_volume_float / recent_avg_volume) - 1) * 100
                return True, f"Volume spike detected: {volume_increase_pct:.1f}% above average (threshold: {(volume_spike_multiplier-1)*100:.0f}%)"

    # Check 3: RSI approaching extremes
    if len(price_history) >= 15:
        rsi = calculate_rsi(price_history, period=14)
        if rsi is not None:
            if rsi <= rsi_oversold:
                return True, f"RSI oversold at {rsi:.1f} (threshold: {rsi_oversold})"
            elif rsi >= rsi_overbought:
                return True, f"RSI overbought at {rsi:.1f} (threshold: {rsi_overbought})"

    # Check 4: Price approaching key support/resistance levels
    major_support = analysis.get('major_support')
    major_resistance = analysis.get('major_resistance')

    if major_support:
        distance_to_support_pct = abs((current_price - major_support) / current_price * 100)
        if distance_to_support_pct <= support_resistance_proximity:
            return True, f"Price within {distance_to_support_pct:.2f}% of major support ${major_support}"

    if major_resistance:
        distance_to_resistance_pct = abs((current_price - major_resistance) / current_price * 100)
        if distance_to_resistance_pct <= support_resistance_proximity:
            return True, f"Price within {distance_to_resistance_pct:.2f}% of major resistance ${major_resistance}"

    return False, None


def calculate_volatility_adjusted_position_size(
    range_percentage_from_min,
    starting_capital_usd,
    current_usd_value,
    confidence_level,
    config
):
    """
    Calculates position size based on volatility and confidence level.
    Higher volatility = smaller position to manage risk.

    Args:
        range_percentage_from_min: Volatility metric (price range min to max %)
        starting_capital_usd: Starting capital for the wallet
        current_usd_value: Current USD value available for trading
        confidence_level: HIGH, MEDIUM, or LOW
        config: Config dictionary with position_sizing settings

    Returns:
        Recommended position size in USD
    """

    position_config = config.get('position_sizing', {})

    # If volatility scaling is disabled, use simple approach
    if not position_config.get('volatility_scaling_enabled', False):
        if confidence_level == 'high':
            base_percentage = position_config.get('base_position_size_high_confidence', 75)
            base_position = starting_capital_usd * (base_percentage / 100)
            # Limit to current available capital
            return min(base_position, current_usd_value)
        else:
            # Medium/Low confidence should not trade
            return 0

    # Only trade on HIGH confidence
    if confidence_level != 'high':
        return 0

    # Get volatility thresholds
    low_vol_max = position_config.get('low_volatility_max_percentage', 15)
    moderate_vol_max = position_config.get('moderate_volatility_max_percentage', 30)
    high_vol_max = position_config.get('high_volatility_max_percentage', 50)

    # Get position multipliers
    low_vol_multiplier = position_config.get('low_volatility_position_multiplier', 1.0)
    moderate_vol_multiplier = position_config.get('moderate_volatility_position_multiplier', 0.85)
    high_vol_multiplier = position_config.get('high_volatility_position_multiplier', 0.65)
    extreme_vol_multiplier = position_config.get('extreme_volatility_position_multiplier', 0.5)

    # Base position size for HIGH confidence (use starting capital, not current balance)
    base_percentage = position_config.get('base_position_size_high_confidence', 75)
    base_position = starting_capital_usd * (base_percentage / 100)

    # Apply volatility multiplier
    if range_percentage_from_min < low_vol_max:
        # Low volatility - use full position
        adjusted_position = base_position * low_vol_multiplier
        vol_category = "LOW"
    elif range_percentage_from_min < moderate_vol_max:
        # Moderate volatility - slightly reduce position
        adjusted_position = base_position * moderate_vol_multiplier
        vol_category = "MODERATE"
    elif range_percentage_from_min < high_vol_max:
        # High volatility - significantly reduce position
        adjusted_position = base_position * high_vol_multiplier
        vol_category = "HIGH"
    else:
        # Extreme volatility - minimal position
        adjusted_position = base_position * extreme_vol_multiplier
        vol_category = "EXTREME"

    # Ensure we never exceed current available capital (can't trade with money we don't have)
    max_position = current_usd_value
    final_position = min(adjusted_position, max_position)

    # Apply minimum position size to ensure fee efficiency
    min_position = position_config.get('minimum_position_size_usd', 0)
    if min_position > 0 and final_position < min_position:
        print(f"Position Sizing: Volatility={range_percentage_from_min:.1f}% ({vol_category})")
        print(f"  Base position: ${base_position:.2f} ({base_percentage}% of starting capital ${starting_capital_usd:.2f})")
        print(f"  Volatility-adjusted: ${adjusted_position:.2f}")
        print(f"  Final position: ${final_position:.2f} (limited by current balance: ${current_usd_value:.2f})")
        print(f"  ⚠️  Position too small (< ${min_position}) - fees would eat profits. Skipping trade.")
        return 0  # Don't trade if position would be too small

    print(f"Position Sizing: Volatility={range_percentage_from_min:.1f}% ({vol_category})")
    print(f"  Base position: ${base_position:.2f} ({base_percentage}% of starting capital ${starting_capital_usd:.2f})")
    print(f"  Volatility-adjusted: ${adjusted_position:.2f}")
    if final_position < adjusted_position:
        print(f"  Final position: ${final_position:.2f} (limited by current balance: ${current_usd_value:.2f})")
    else:
        print(f"  Final position: ${final_position:.2f}")

    return final_position
