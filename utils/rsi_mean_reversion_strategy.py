"""
RSI Mean Reversion Strategy

Walk-forward validated RSI mean reversion strategy for crypto trading.
Uses close-only data (no synthetic OHLC) for honest signal generation.

Entry: RSI(14) drops below threshold (oversold condition)
Exit priority:
  1. Disaster stop (wide price-based stop, 5%+)
  2. Trailing stop (activates after small gain)
  3. RSI full recovery (RSI >= rsi_exit)
  4. RSI partial recovery + profitable (RSI >= rsi_partial AND profit > 0)
  5. Max hold time

Validated symbols:
  - ATOM-USD @ 15min: 35 trades, 69% WR, PF 4.81, +18.47% AT
  - LINK-USD @ 2H:    8 trades, 75% WR, PF 14.63, +10.92% AT
"""

from typing import Dict, Optional
from utils.file_helpers import get_property_values_from_crypto_file
from utils.price_helpers import calculate_rsi


def aggregate_closes_to_timeframe(closes: list, timestamps: list, target_minutes: int) -> list:
    """
    Aggregate 5-minute close prices to a higher timeframe using clock-aligned
    UTC boundaries (e.g., 00:00, 02:00, 04:00 for 2H candles).

    Args:
        closes: List of close prices (5-min intervals)
        timestamps: List of timestamps corresponding to closes
        target_minutes: Target timeframe in minutes (e.g., 15, 120)

    Returns:
        List of aggregated close prices
    """
    if not closes or not timestamps or len(closes) != len(timestamps):
        return []

    target_seconds = target_minutes * 60
    if target_seconds <= 300:
        return closes

    aggregated = []
    current_candle_start = (timestamps[0] // target_seconds) * target_seconds
    last_close = closes[0]

    for i in range(len(closes)):
        candle_boundary = (timestamps[i] // target_seconds) * target_seconds
        if candle_boundary != current_candle_start:
            aggregated.append(last_close)
            current_candle_start = candle_boundary
        last_close = closes[i]

    # Append last candle
    aggregated.append(last_close)

    return aggregated


def check_rsi_mean_reversion_signal(
    symbol: str,
    timeframe_minutes: int,
    config_params: Dict,
    data_directory: str = 'coinbase-data',
    current_price: float = None,
    max_age_hours: int = 5040
) -> Dict:
    """
    Check for RSI mean reversion entry signal.

    Args:
        symbol: Product ID (e.g., 'ATOM-USD')
        timeframe_minutes: Timeframe in minutes (15 for ATOM, 120 for LINK)
        config_params: Symbol-specific config dict with rsi_entry, rsi_exit, etc.
        data_directory: Directory containing price data files
        current_price: Current market price (optional)
        max_age_hours: Maximum data age in hours

    Returns:
        Signal dictionary:
        {
            'signal': 'buy' or 'no_signal',
            'strategy': 'rsi_mean_reversion',
            'confidence': 'high', 'medium', or 'low',
            'entry_price': float or None,
            'stop_loss': float or None,
            'profit_target': float or None,
            'reasoning': str,
            'metrics': dict
        }
    """
    rsi_period = config_params.get('rsi_period', 14)
    rsi_entry = config_params.get('rsi_entry', 20)
    rsi_exit = config_params.get('rsi_exit', 48)
    rsi_partial_exit = config_params.get('rsi_partial_exit', 35)
    disaster_stop_pct = config_params.get('disaster_stop_pct', 5.0)
    max_hold_bars = config_params.get('max_hold_bars', 24)
    trailing_activate_pct = config_params.get('trailing_activate_pct', 0.3)
    trailing_stop_pct = config_params.get('trailing_stop_pct', 0.2)

    no_signal_base = {
        'signal': 'no_signal',
        'strategy': 'rsi_mean_reversion',
        'confidence': 'low',
        'entry_price': None,
        'stop_loss': None,
        'profit_target': None,
    }

    # Load close prices and timestamps
    closes = get_property_values_from_crypto_file(
        data_directory, symbol, 'price', max_age_hours=max_age_hours
    )
    timestamps = get_property_values_from_crypto_file(
        data_directory, symbol, 'timestamp', max_age_hours=max_age_hours
    )

    if not closes or len(closes) < 100:
        return {
            **no_signal_base,
            'reasoning': f"Insufficient data: {len(closes) if closes else 0} data points (need 100+)",
            'metrics': {}
        }

    # Aggregate to target timeframe
    agg_closes = aggregate_closes_to_timeframe(closes, timestamps, timeframe_minutes)

    # Need enough aggregated candles for RSI calculation
    min_candles = rsi_period + 5  # RSI needs period+1, plus buffer
    if len(agg_closes) < min_candles:
        return {
            **no_signal_base,
            'reasoning': f"Insufficient aggregated data: {len(agg_closes)} candles at {timeframe_minutes}min (need {min_candles}+)",
            'metrics': {}
        }

    # Calculate RSI on aggregated closes
    rsi_value = calculate_rsi(agg_closes, period=rsi_period)

    if rsi_value is None:
        return {
            **no_signal_base,
            'reasoning': "Failed to calculate RSI",
            'metrics': {}
        }

    # Use current_price if provided, otherwise use latest close
    entry_price = current_price if current_price else agg_closes[-1]

    # Check entry condition: RSI <= rsi_entry
    if rsi_value > rsi_entry:
        return {
            **no_signal_base,
            'reasoning': f"RSI({rsi_period}) = {rsi_value:.1f} > entry threshold {rsi_entry} (waiting for oversold)",
            'metrics': {
                'rsi': rsi_value,
                'rsi_entry_threshold': rsi_entry,
                'timeframe_minutes': timeframe_minutes,
                'aggregated_candles': len(agg_closes),
                'current_price': entry_price
            }
        }

    # ENTRY SIGNAL CONFIRMED
    stop_loss = entry_price * (1 - disaster_stop_pct / 100)
    # Estimate profit target based on historical RSI recovery
    # Conservative: expect ~2-3% move from oversold RSI recovery
    estimated_target_pct = 2.0
    profit_target = entry_price * (1 + estimated_target_pct / 100)

    # Confidence based on RSI depth
    if rsi_value <= 15:
        confidence = 'high'
    elif rsi_value <= rsi_entry:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'signal': 'buy',
        'strategy': 'rsi_mean_reversion',
        'confidence': confidence,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'reasoning': (
            f"RSI MEAN REVERSION (LONG): RSI({rsi_period}) = {rsi_value:.1f} <= {rsi_entry} (oversold). "
            f"Entry: ${entry_price:.4f}, Disaster stop: ${stop_loss:.4f} (-{disaster_stop_pct}%), "
            f"RSI exit target: {rsi_exit}, Partial exit: {rsi_partial_exit}. "
            f"Timeframe: {timeframe_minutes}min, Max hold: {max_hold_bars} bars."
        ),
        'metrics': {
            'rsi': rsi_value,
            'rsi_entry_threshold': rsi_entry,
            'rsi_exit_threshold': rsi_exit,
            'rsi_partial_exit_threshold': rsi_partial_exit,
            'disaster_stop_pct': disaster_stop_pct,
            'max_hold_bars': max_hold_bars,
            'trailing_activate_pct': trailing_activate_pct,
            'trailing_stop_pct': trailing_stop_pct,
            'timeframe_minutes': timeframe_minutes,
            'aggregated_candles': len(agg_closes),
            'current_price': entry_price
        }
    }


def check_rsi_exit_signal(
    symbol: str,
    timeframe_minutes: int,
    config_params: Dict,
    entry_price: float,
    current_price: float,
    data_directory: str = 'coinbase-data',
    max_age_hours: int = 5040,
    min_profit_usd: float = 0.0,
    net_profit_usd: float = 0.0
) -> Dict:
    """
    Check if an open RSI mean reversion position should be exited based on RSI recovery.

    This handles exit conditions 3 and 4 from the strategy:
      3. RSI full recovery (RSI >= rsi_exit AND net_profit_usd >= min_profit_usd)
      4. RSI partial recovery + profitable (RSI >= rsi_partial AND net_profit_usd >= min_profit_usd)

    Conditions 1 (disaster stop), 2 (trailing stop), and 5 (max hold) are
    handled by existing index.py logic.

    Args:
        symbol: Product ID
        timeframe_minutes: Timeframe in minutes
        config_params: Symbol-specific config with rsi_exit, rsi_partial_exit
        entry_price: Buy price
        current_price: Current market price
        data_directory: Directory containing price data
        max_age_hours: Maximum data age
        min_profit_usd: Minimum net profit in USD required for RSI exits
        net_profit_usd: Current net profit in USD (after fees/taxes)

    Returns:
        Dict with 'should_exit', 'reason', 'rsi_value'
    """
    rsi_period = config_params.get('rsi_period', 14)
    rsi_exit = config_params.get('rsi_exit', 48)
    rsi_partial_exit = config_params.get('rsi_partial_exit', 35)
    meets_min_profit = net_profit_usd >= min_profit_usd

    closes = get_property_values_from_crypto_file(
        data_directory, symbol, 'price', max_age_hours=max_age_hours
    )
    timestamps = get_property_values_from_crypto_file(
        data_directory, symbol, 'timestamp', max_age_hours=max_age_hours
    )

    if not closes or len(closes) < 50:
        return {'should_exit': False, 'reason': 'Insufficient data for RSI', 'rsi_value': None}

    agg_closes = aggregate_closes_to_timeframe(closes, timestamps, timeframe_minutes)

    if len(agg_closes) < rsi_period + 1:
        return {'should_exit': False, 'reason': 'Insufficient aggregated data', 'rsi_value': None}

    rsi_value = calculate_rsi(agg_closes, period=rsi_period)

    if rsi_value is None:
        return {'should_exit': False, 'reason': 'RSI calculation failed', 'rsi_value': None}

    # Calculate profit metrics
    profit_pct = ((current_price - entry_price) / entry_price) * 100

    # Check RSI full recovery - requires min_profit_usd threshold
    if rsi_value >= rsi_exit:
        if meets_min_profit:
            return {
                'should_exit': True,
                'reason': f'RSI full recovery: {rsi_value:.1f} >= {rsi_exit} (exit threshold) AND profit ${net_profit_usd:.2f} >= ${min_profit_usd:.2f}',
                'rsi_value': rsi_value
            }
        else:
            return {
                'should_exit': False,
                'reason': f'RSI full recovery: {rsi_value:.1f} >= {rsi_exit} BUT profit ${net_profit_usd:.2f} < ${min_profit_usd:.2f} min',
                'rsi_value': rsi_value
            }

    # Check RSI partial recovery - also requires min_profit_usd threshold
    if rsi_value >= rsi_partial_exit:
        if meets_min_profit:
            return {
                'should_exit': True,
                'reason': f'RSI partial recovery: {rsi_value:.1f} >= {rsi_partial_exit} AND profit ${net_profit_usd:.2f} >= ${min_profit_usd:.2f}',
                'rsi_value': rsi_value
            }

    return {
        'should_exit': False,
        'reason': f'RSI = {rsi_value:.1f} (exit: {rsi_exit}, partial: {rsi_partial_exit}, profit: ${net_profit_usd:.2f}/${min_profit_usd:.2f})',
        'rsi_value': rsi_value
    }
