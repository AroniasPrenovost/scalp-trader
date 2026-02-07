"""
Close-Only Reversion Strategy (co_revert)

Walk-forward validated RSI + Bollinger Band mean reversion.
Uses fixed profit target/stop loss instead of trailing stops.

Entry: RSI deeply oversold + price below BB lower band
Exit priority:
  1. Stop loss (fixed price-based, handled by index.py)
  2. Profit target (fixed price-based, handled by index.py)
  3. RSI recovery + min profit (RSI >= rsi_exit AND profit >= min_profit_for_rsi_exit)
  4. Max hold time (handled by index.py)

Validated symbols:
  - NEAR-USD @ 30min: Val +6.23% (adv2), +7.37% (adv3), 67-83% WR
  - CRV-USD @ 30min: Val +6.01% (adv2), +7.15% (adv3), 67% WR
"""

from typing import Dict
from utils.file_helpers import get_property_values_from_crypto_file
from utils.price_helpers import calculate_rsi
from utils.technical_indicators import calculate_bollinger_bands
from utils.rsi_mean_reversion_strategy import aggregate_closes_to_timeframe


def check_co_revert_signal(
    symbol: str,
    timeframe_minutes: int,
    config_params: Dict,
    data_directory: str = 'coinbase-data',
    current_price: float = None,
    max_age_hours: int = 5040
) -> Dict:
    """
    Check for co_revert entry signal.

    Entry requires:
    1. RSI <= rsi_entry (deeply oversold)
    2. Price <= BB lower band (if use_bb_filter is True)

    Args:
        symbol: Product ID (e.g., 'CRV-USD')
        timeframe_minutes: Timeframe in minutes
        config_params: Symbol-specific config dict
        data_directory: Directory containing price data files
        current_price: Current market price (optional)
        max_age_hours: Maximum data age in hours

    Returns:
        Signal dictionary with signal, strategy, confidence, etc.
    """
    rsi_period = config_params.get('rsi_period', 14)
    rsi_entry = config_params.get('rsi_entry', 25)
    rsi_exit = config_params.get('rsi_exit', 45)
    use_bb_filter = config_params.get('use_bb_filter', True)
    bb_period = config_params.get('bb_period', 20)
    bb_std = config_params.get('bb_std', 2.0)
    profit_target_pct = config_params.get('profit_target_pct', 1.0)
    stop_loss_pct = config_params.get('stop_loss_pct', 1.0)
    max_hold_bars = config_params.get('max_hold_bars', 24)
    min_profit_for_rsi_exit = config_params.get('min_profit_for_rsi_exit', 0.3)

    no_signal_base = {
        'signal': 'no_signal',
        'strategy': 'co_revert',
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

    min_candles = max(rsi_period + 5, bb_period + 5)
    if len(agg_closes) < min_candles:
        return {
            **no_signal_base,
            'reasoning': f"Insufficient aggregated data: {len(agg_closes)} candles at {timeframe_minutes}min (need {min_candles}+)",
            'metrics': {}
        }

    # Calculate RSI
    rsi_value = calculate_rsi(agg_closes, period=rsi_period)
    if rsi_value is None:
        return {
            **no_signal_base,
            'reasoning': "Failed to calculate RSI",
            'metrics': {}
        }

    entry_price = current_price if current_price else agg_closes[-1]

    # Check RSI entry condition
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

    # Check BB filter
    if use_bb_filter:
        bb = calculate_bollinger_bands(agg_closes, period=bb_period, std_dev=bb_std)
        if bb is None:
            return {
                **no_signal_base,
                'reasoning': "RSI oversold but insufficient data for BB calculation",
                'metrics': {'rsi': rsi_value, 'current_price': entry_price}
            }

        bb_lower = bb['lower']
        if agg_closes[-1] > bb_lower:
            return {
                **no_signal_base,
                'reasoning': (
                    f"RSI({rsi_period}) = {rsi_value:.1f} <= {rsi_entry} BUT price ${agg_closes[-1]:.4f} "
                    f"above BB lower ${bb_lower:.4f} (need below for entry)"
                ),
                'metrics': {
                    'rsi': rsi_value,
                    'bb_lower': bb_lower,
                    'price': agg_closes[-1],
                    'timeframe_minutes': timeframe_minutes,
                    'current_price': entry_price
                }
            }

    # ENTRY SIGNAL CONFIRMED (RSI oversold + below BB lower)
    stop_loss = entry_price * (1 - stop_loss_pct / 100)
    profit_target = entry_price * (1 + profit_target_pct / 100)

    if rsi_value <= 15:
        confidence = 'high'
    elif rsi_value <= 20:
        confidence = 'high'
    elif rsi_value <= rsi_entry:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'signal': 'buy',
        'strategy': 'co_revert',
        'confidence': confidence,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'reasoning': (
            f"CO_REVERT (LONG): RSI({rsi_period}) = {rsi_value:.1f} <= {rsi_entry} + below BB lower. "
            f"Entry: ${entry_price:.4f}, Stop: ${stop_loss:.4f} (-{stop_loss_pct}%), "
            f"Target: ${profit_target:.4f} (+{profit_target_pct}%). "
            f"RSI exit: {rsi_exit} (if profit >= {min_profit_for_rsi_exit}%). "
            f"Max hold: {max_hold_bars} bars."
        ),
        'metrics': {
            'rsi': rsi_value,
            'rsi_entry_threshold': rsi_entry,
            'rsi_exit_threshold': rsi_exit,
            'profit_target_pct': profit_target_pct,
            'stop_loss_pct': stop_loss_pct,
            'max_hold_bars': max_hold_bars,
            'min_profit_for_rsi_exit': min_profit_for_rsi_exit,
            'timeframe_minutes': timeframe_minutes,
            'aggregated_candles': len(agg_closes),
            'current_price': entry_price
        }
    }


def check_co_revert_exit_signal(
    symbol: str,
    timeframe_minutes: int,
    config_params: Dict,
    entry_price: float,
    current_price: float,
    data_directory: str = 'coinbase-data',
    max_age_hours: int = 5040
) -> Dict:
    """
    Check if co_revert position should exit based on RSI recovery.

    Stop loss and profit target are handled by index.py's built-in logic.
    This checks the RSI recovery exit: RSI >= rsi_exit AND profit >= min_profit_for_rsi_exit.

    Args:
        symbol: Product ID
        timeframe_minutes: Timeframe in minutes
        config_params: Symbol-specific config
        entry_price: Buy price
        current_price: Current market price
        data_directory: Directory containing price data
        max_age_hours: Maximum data age

    Returns:
        Dict with 'should_exit', 'reason', 'rsi_value'
    """
    rsi_period = config_params.get('rsi_period', 14)
    rsi_exit = config_params.get('rsi_exit', 45)
    min_profit_for_rsi_exit = config_params.get('min_profit_for_rsi_exit', 0.3)

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

    # RSI recovery exit: RSI normalized AND trade is profitable enough
    gross_pnl_pct = ((current_price - entry_price) / entry_price) * 100
    if rsi_value >= rsi_exit and gross_pnl_pct >= min_profit_for_rsi_exit:
        return {
            'should_exit': True,
            'reason': f'RSI recovery: {rsi_value:.1f} >= {rsi_exit} AND profit +{gross_pnl_pct:.2f}% >= {min_profit_for_rsi_exit}%',
            'rsi_value': rsi_value
        }

    return {
        'should_exit': False,
        'reason': f'RSI = {rsi_value:.1f} (exit: {rsi_exit}, profit: {gross_pnl_pct:.2f}%, min: {min_profit_for_rsi_exit}%)',
        'rsi_value': rsi_value
    }
