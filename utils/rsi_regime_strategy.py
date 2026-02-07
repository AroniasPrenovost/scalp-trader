"""
RSI Regime Strategy

Walk-forward validated RSI mean reversion with EMA regime filter.
Prevents entering during market freefall by requiring:
1. Price within X% of EMA (not too far below = not in crash)
2. EMA not declining steeply (no sustained downtrend)

Otherwise identical to rsi_mean_reversion for exits.

Validated symbols:
  - ICP-USD @ 5min:  Val +6.03% (adv2), +7.63% (adv3), 67-81% WR
  - ATOM-USD @ 30min: Val +3.31% (adv2), +4.53% (adv3), 69% WR
  - NEAR-USD @ 15min: Val +2.15% (adv2), +7.91% (adv3), 64-86% WR
"""

from typing import Dict
from utils.file_helpers import get_property_values_from_crypto_file
from utils.price_helpers import calculate_rsi
from utils.technical_indicators import calculate_ema
from utils.rsi_mean_reversion_strategy import aggregate_closes_to_timeframe


def check_rsi_regime_signal(
    symbol: str,
    timeframe_minutes: int,
    config_params: Dict,
    data_directory: str = 'coinbase-data',
    current_price: float = None,
    max_age_hours: int = 5040
) -> Dict:
    """
    Check for RSI regime strategy entry signal.

    Same as rsi_mean_reversion but adds EMA regime filters:
    - Price must be within max_below_ema_pct of EMA (not in freefall)
    - EMA must not be declining more than max_ema_decline_pct over slope_bars

    Args:
        symbol: Product ID (e.g., 'ICP-USD')
        timeframe_minutes: Timeframe in minutes
        config_params: Symbol-specific config dict
        data_directory: Directory containing price data files
        current_price: Current market price (optional)
        max_age_hours: Maximum data age in hours

    Returns:
        Signal dictionary with signal, strategy, confidence, etc.
    """
    rsi_period = config_params.get('rsi_period', 14)
    rsi_entry = config_params.get('rsi_entry', 20)
    rsi_exit = config_params.get('rsi_exit', 50)
    rsi_partial_exit = config_params.get('rsi_partial_exit', 35)
    disaster_stop_pct = config_params.get('disaster_stop_pct', 5.0)
    max_hold_bars = config_params.get('max_hold_bars', 24)
    trailing_activate_pct = config_params.get('trailing_activate_pct', 0.3)
    trailing_stop_pct = config_params.get('trailing_stop_pct', 0.2)

    # Regime filter params
    regime_ema_period = config_params.get('regime_ema', 50)
    max_below_ema_pct = config_params.get('max_below_ema_pct', 5.0)
    ema_slope_bars = config_params.get('ema_slope_bars', 10)
    max_ema_decline_pct = config_params.get('max_ema_decline_pct', 3.0)

    no_signal_base = {
        'signal': 'no_signal',
        'strategy': 'rsi_regime',
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

    # Need enough candles for RSI + EMA calculation
    min_candles = max(rsi_period + 5, regime_ema_period + ema_slope_bars + 5)
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

    # REGIME FILTER: Check EMA proximity and slope
    current_ema = calculate_ema(agg_closes, period=regime_ema_period)

    if current_ema is not None:
        # Check price proximity to EMA
        distance_below = (current_ema - agg_closes[-1]) / current_ema
        if distance_below > max_below_ema_pct / 100.0:
            return {
                **no_signal_base,
                'reasoning': (
                    f"RSI({rsi_period}) = {rsi_value:.1f} <= {rsi_entry} BUT regime BLOCKED: "
                    f"price {distance_below*100:.1f}% below EMA({regime_ema_period}) (max: {max_below_ema_pct}%)"
                ),
                'metrics': {
                    'rsi': rsi_value,
                    'rsi_entry_threshold': rsi_entry,
                    'regime_ema': current_ema,
                    'distance_below_ema_pct': distance_below * 100,
                    'max_below_ema_pct': max_below_ema_pct,
                    'timeframe_minutes': timeframe_minutes,
                    'current_price': entry_price
                }
            }

        # Check EMA slope (is the trend declining too fast?)
        if len(agg_closes) > ema_slope_bars:
            earlier_ema = calculate_ema(agg_closes[:-ema_slope_bars], period=regime_ema_period)
            if earlier_ema is not None:
                ema_change = (current_ema - earlier_ema) / earlier_ema
                if ema_change < -(max_ema_decline_pct / 100.0):
                    return {
                        **no_signal_base,
                        'reasoning': (
                            f"RSI({rsi_period}) = {rsi_value:.1f} <= {rsi_entry} BUT regime BLOCKED: "
                            f"EMA({regime_ema_period}) declining {ema_change*100:.2f}% over {ema_slope_bars} bars (max: -{max_ema_decline_pct}%)"
                        ),
                        'metrics': {
                            'rsi': rsi_value,
                            'regime_ema': current_ema,
                            'ema_decline_pct': ema_change * 100,
                            'max_ema_decline_pct': max_ema_decline_pct,
                            'timeframe_minutes': timeframe_minutes,
                            'current_price': entry_price
                        }
                    }

    # ENTRY SIGNAL CONFIRMED (RSI oversold + regime filter passed)
    stop_loss = entry_price * (1 - disaster_stop_pct / 100)
    estimated_target_pct = 2.0
    profit_target = entry_price * (1 + estimated_target_pct / 100)

    if rsi_value <= 15:
        confidence = 'high'
    elif rsi_value <= rsi_entry:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'signal': 'buy',
        'strategy': 'rsi_regime',
        'confidence': confidence,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'reasoning': (
            f"RSI REGIME (LONG): RSI({rsi_period}) = {rsi_value:.1f} <= {rsi_entry} (oversold) + regime OK. "
            f"Entry: ${entry_price:.4f}, Disaster stop: ${stop_loss:.4f} (-{disaster_stop_pct}%), "
            f"RSI exit: {rsi_exit}, Partial: {rsi_partial_exit}. "
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
            'regime_ema_period': regime_ema_period,
            'timeframe_minutes': timeframe_minutes,
            'aggregated_candles': len(agg_closes),
            'current_price': entry_price
        }
    }
