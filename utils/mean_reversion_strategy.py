#!/usr/bin/env python3
"""
Mean Reversion Strategy - Buy Oversold, Sell at Mean

Strategy Logic:
1. ENTRY: Buy when price is oversold (RSI < 30) with volume confirmation
2. EXIT: Sell when price reverts to mean (RSI > 50 or hit target)
3. STOP: 0.5% to prevent catching falling knives
4. TARGET: 0.7% NET profit

This should achieve 60-70% win rate by exploiting crypto's oscillating nature.
"""

from typing import Dict, List, Optional
from utils.profit_calculator import calculate_required_price_for_target_profit
from utils.price_helpers import calculate_rsi


def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[Dict]:
    """
    Calculate Bollinger Bands.

    Price below lower band = Oversold
    Price above upper band = Overbought
    Price at middle band = Mean

    Args:
        prices: Historical prices
        period: MA period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        {'upper': float, 'middle': float, 'lower': float, 'position': str}
    """
    if len(prices) < period:
        return None

    recent = prices[-period:]
    middle = sum(recent) / period

    # Calculate standard deviation
    variance = sum((p - middle) ** 2 for p in recent) / period
    std = variance ** 0.5

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    current_price = prices[-1]

    # Determine position
    if current_price < lower:
        position = 'below'  # Oversold
    elif current_price > upper:
        position = 'above'  # Overbought
    else:
        position = 'inside'  # Normal

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'current': current_price,
        'position': position,
        'distance_from_middle_pct': ((current_price - middle) / middle) * 100
    }


def calculate_moving_average(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate simple moving average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_volume_ratio(volumes: List[float], current_volume: float, lookback: int = 24) -> float:
    """Calculate current volume / average volume ratio."""
    if len(volumes) < lookback:
        return 1.0

    avg_volume = sum(volumes[-lookback:]) / lookback
    if avg_volume == 0:
        return 1.0

    return current_volume / avg_volume


def check_mean_reversion_entry(
    prices: List[float],
    volumes: List[float],
    current_price: float,
    current_volume: float,
    target_net_pct: float = 0.7,
    stop_gross_pct: float = 0.5,
    entry_fee_pct: float = 0.6,
    exit_fee_pct: float = 0.6,
    tax_rate_pct: float = 37.0,
    rsi_oversold: int = 30,
    min_volume_ratio: float = 1.0
) -> Optional[Dict]:
    """
    Check for mean reversion entry signal.

    ENTRY RULES:
    1. RSI < 30 (oversold)
    2. Price below lower Bollinger Band OR significantly below MA
    3. Volume >= 1.0x average (confirmation)
    4. Not in strong downtrend (24h momentum > -3%)

    Args:
        prices: Historical prices (hourly)
        volumes: Historical 24h volumes
        current_price: Current market price
        current_volume: Current 24h volume
        target_net_pct: Target NET profit %
        stop_gross_pct: Stop loss GROSS %
        rsi_oversold: RSI threshold for oversold (default 30)
        min_volume_ratio: Minimum volume ratio (default 1.0x)

    Returns:
        Signal dictionary or None
    """

    if len(prices) < 24:
        return {
            'signal': 'no_signal',
            'reason': 'Insufficient data (need 24+ hours)'
        }

    # Calculate indicators
    rsi = calculate_rsi(prices, period=14)
    bb = calculate_bollinger_bands(prices, period=20)
    ma20 = calculate_moving_average(prices, period=20)
    volume_ratio = calculate_volume_ratio(volumes, current_volume, lookback=24)

    if rsi is None or bb is None or ma20 is None:
        return {
            'signal': 'no_signal',
            'reason': 'Cannot calculate indicators'
        }

    # Calculate 24h momentum (prevent catching falling knife)
    momentum_24h = ((prices[-1] - prices[-24]) / prices[-24]) * 100

    # FILTER 1: Check if oversold (RSI < 30)
    if rsi >= rsi_oversold:
        return {
            'signal': 'no_signal',
            'reason': f'Not oversold: RSI {rsi:.1f} (need < {rsi_oversold})',
            'metrics': {'rsi': rsi}
        }

    # FILTER 2: Check if price is below mean (BB lower band OR 2%+ below MA)
    distance_from_ma = ((current_price - ma20) / ma20) * 100

    if bb['position'] != 'below' and distance_from_ma > -2.0:
        return {
            'signal': 'no_signal',
            'reason': f'Not below mean: {distance_from_ma:.2f}% from MA20',
            'metrics': {
                'rsi': rsi,
                'bb_position': bb['position'],
                'distance_from_ma': distance_from_ma
            }
        }

    # FILTER 3: Volume confirmation
    if volume_ratio < min_volume_ratio:
        return {
            'signal': 'no_signal',
            'reason': f'Volume too low: {volume_ratio:.2f}x average (need {min_volume_ratio}x+)',
            'metrics': {'rsi': rsi, 'volume_ratio': volume_ratio}
        }

    # FILTER 4: Not in free fall (prevent catching falling knife)
    if momentum_24h < -3.0:
        return {
            'signal': 'no_signal',
            'reason': f'Strong downtrend: {momentum_24h:.2f}% (24h) - possible falling knife',
            'metrics': {'rsi': rsi, 'momentum_24h': momentum_24h}
        }

    # ENTRY APPROVED!
    entry_price = current_price
    stop_loss = entry_price * (1 - stop_gross_pct / 100)

    # Calculate profit target
    target_calc = calculate_required_price_for_target_profit(
        entry_price=entry_price,
        target_net_profit_pct=target_net_pct,
        entry_fee_pct=entry_fee_pct,
        exit_fee_pct=exit_fee_pct,
        tax_rate_pct=tax_rate_pct
    )
    profit_target = target_calc['required_exit_price']

    # Calculate mean reversion target (RSI 50 level)
    # This is where we expect price to revert to
    mean_reversion_target = ma20  # Simple: revert to MA20

    return {
        'signal': 'buy',
        'strategy': 'mean_reversion',
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'mean_reversion_target': mean_reversion_target,
        'confidence': 'high' if rsi < 25 and bb['position'] == 'below' else 'medium',
        'reasoning': (
            f'MEAN REVERSION: RSI oversold at {rsi:.1f} (< {rsi_oversold}). '
            f'Price {distance_from_ma:.2f}% below MA20. '
            f'BB position: {bb["position"]}. '
            f'Volume {volume_ratio:.1f}x average. '
            f'24h momentum: {momentum_24h:.2f}% (not falling knife).'
        ),
        'metrics': {
            'rsi': rsi,
            'bb_upper': bb['upper'],
            'bb_middle': bb['middle'],
            'bb_lower': bb['lower'],
            'bb_position': bb['position'],
            'ma20': ma20,
            'distance_from_ma_pct': distance_from_ma,
            'volume_ratio': volume_ratio,
            'momentum_24h': momentum_24h
        }
    }


def check_mean_reversion_exit(
    prices: List[float],
    entry_price: float,
    current_price: float,
    hours_held: int,
    target_price: float,
    stop_price: float,
    max_hold_hours: int = 24,
    rsi_mean: int = 50
) -> Optional[Dict]:
    """
    Check if we should exit a mean reversion trade.

    EXIT RULES:
    1. RSI > 50 (reverted to mean)
    2. Price crossed above MA20 (reverted to mean)
    3. Hit profit target (0.7% NET)
    4. Hit stop loss (0.5% GROSS)
    5. Max hold time (24h - don't become bagholder)

    Args:
        prices: Historical prices
        entry_price: Entry price
        current_price: Current price
        hours_held: Hours since entry
        target_price: Profit target price
        stop_price: Stop loss price
        max_hold_hours: Maximum hold time
        rsi_mean: RSI level for "mean" (default 50)

    Returns:
        Exit signal or None
    """

    # Check stop loss first
    if current_price <= stop_price:
        return {
            'exit': True,
            'exit_price': stop_price,
            'exit_reason': 'stop_loss',
            'reasoning': f'Hit stop loss at {stop_price:.4f}'
        }

    # Check profit target
    if current_price >= target_price:
        return {
            'exit': True,
            'exit_price': target_price,
            'exit_reason': 'target',
            'reasoning': f'Hit profit target at {target_price:.4f}'
        }

    # Check max hold time
    if hours_held >= max_hold_hours:
        return {
            'exit': True,
            'exit_price': current_price,
            'exit_reason': 'max_hold',
            'reasoning': f'Max hold time reached ({hours_held}h)'
        }

    # Check mean reversion indicators
    rsi = calculate_rsi(prices, period=14)
    ma20 = calculate_moving_average(prices, period=20)

    if rsi and rsi >= rsi_mean:
        # RSI reverted to mean - take profit
        return {
            'exit': True,
            'exit_price': current_price,
            'exit_reason': 'rsi_reversion',
            'reasoning': f'RSI reverted to mean: {rsi:.1f} (>= {rsi_mean})'
        }

    if ma20 and current_price >= ma20:
        # Price crossed above MA - take profit
        return {
            'exit': True,
            'exit_price': current_price,
            'exit_reason': 'ma_crossover',
            'reasoning': f'Price crossed above MA20: {current_price:.4f} >= {ma20:.4f}'
        }

    # No exit signal
    return None


def get_strategy_info() -> Dict:
    """Get strategy information."""
    return {
        'name': 'Mean Reversion Strategy',
        'version': '1.0',
        'approach': 'Buy oversold, sell when price reverts to mean',
        'target_holding_time': '4-24 hours',
        'target_profit': '0.7% NET',
        'max_stop_loss': '0.5% GROSS',
        'expected_win_rate': '60-70% (typical for mean reversion)',
        'entry_criteria': [
            'RSI < 30 (oversold)',
            'Price below lower Bollinger Band OR 2%+ below MA20',
            'Volume >= 1.0x average (confirmation)',
            'Not in strong downtrend (24h momentum > -3%)'
        ],
        'exit_criteria': [
            'RSI >= 50 (reverted to mean)',
            'Price crossed above MA20',
            'Hit 0.7% NET profit target',
            'Hit 0.5% stop loss',
            'Max hold 24 hours'
        ]
    }
