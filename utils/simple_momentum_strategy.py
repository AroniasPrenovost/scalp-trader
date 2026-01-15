#!/usr/bin/env python3
"""
Simple Momentum Strategy - Acceleration + Volume

Strategy:
- ENTRY: Price accelerating (5-min momentum > 15-min momentum) + volume confirmation
- EXIT: Hit target, stop, or momentum reverses
- TARGET: 0.7% NET
- STOP: 0.5% GROSS
- HOLD: 1-3 hours max

This strategy requires 1-minute price data.
"""

from typing import Dict, List, Optional
from utils.profit_calculator import calculate_required_price_for_target_profit


def calculate_momentum(prices: List[float], periods: int) -> Optional[float]:
    """
    Calculate momentum (% price change) over N periods.

    Args:
        prices: List of prices (1-minute intervals)
        periods: Number of periods to look back

    Returns:
        Momentum as percentage, or None if insufficient data
    """
    if len(prices) < periods + 1:
        return None

    old_price = prices[-periods - 1]
    current_price = prices[-1]

    if old_price == 0:
        return None

    momentum = ((current_price - old_price) / old_price) * 100
    return momentum


def calculate_volume_ratio(volumes: List[float], lookback: int = 60) -> Optional[float]:
    """
    Calculate current volume vs average volume.

    Args:
        volumes: List of 24h volumes
        lookback: Number of periods for average (default 60 = 1 hour)

    Returns:
        Volume ratio (current / average)
    """
    if len(volumes) < lookback + 1:
        return None

    current_volume = volumes[-1]
    avg_volume = sum(volumes[-lookback-1:-1]) / lookback

    if avg_volume == 0:
        return None

    return current_volume / avg_volume


def check_momentum_entry(
    prices: List[float],
    volumes: List[float],
    current_price: float,
    current_volume: float,
    target_net_pct: float = 0.7,
    stop_gross_pct: float = 0.5,
    entry_fee_pct: float = 0.25,
    exit_fee_pct: float = 0.25,
    tax_rate_pct: float = 24.0
) -> Optional[Dict]:
    """
    Check for simple momentum entry signal.

    ENTRY RULES:
    1. 5-min momentum > 15-min momentum (acceleration)
    2. 5-min momentum > 0.3% (meaningful move)
    3. Volume > 1.5x average (confirmation)
    4. Recent momentum positive (last 3 min)

    Args:
        prices: Historical prices at 1-minute intervals
        volumes: Historical 24h volumes at 1-minute intervals
        current_price: Current market price
        current_volume: Current 24h volume
        target_net_pct: Target NET profit %
        stop_gross_pct: Stop loss GROSS %

    Returns:
        Signal dictionary or None
    """

    # Need at least 20 minutes of data (for 15-min momentum + buffer)
    if len(prices) < 20:
        return {
            'signal': 'no_signal',
            'reason': 'Insufficient data (need 20+ minutes)'
        }

    # FILTER 1: Calculate momentum at different timeframes
    momentum_5min = calculate_momentum(prices, 5)
    momentum_15min = calculate_momentum(prices, 15)
    momentum_3min = calculate_momentum(prices, 3)

    if momentum_5min is None or momentum_15min is None or momentum_3min is None:
        return {
            'signal': 'no_signal',
            'reason': 'Cannot calculate momentum'
        }

    # FILTER 2: Check for acceleration (5-min > 15-min)
    if momentum_5min <= momentum_15min:
        return {
            'signal': 'no_signal',
            'reason': f'No acceleration: 5-min ({momentum_5min:.3f}%) <= 15-min ({momentum_15min:.3f}%)',
            'metrics': {
                'momentum_5min': momentum_5min,
                'momentum_15min': momentum_15min,
                'momentum_3min': momentum_3min
            }
        }

    # FILTER 3: Check 5-min momentum is meaningful (> 0.3%)
    if momentum_5min < 0.3:
        return {
            'signal': 'no_signal',
            'reason': f'5-min momentum too weak: {momentum_5min:.3f}% (need > 0.3%)',
            'metrics': {
                'momentum_5min': momentum_5min,
                'momentum_15min': momentum_15min,
                'momentum_3min': momentum_3min
            }
        }

    # FILTER 4: Recent momentum must be positive (confirming direction)
    if momentum_3min < 0:
        return {
            'signal': 'no_signal',
            'reason': f'Recent momentum negative: {momentum_3min:.3f}% (losing steam)',
            'metrics': {
                'momentum_5min': momentum_5min,
                'momentum_15min': momentum_15min,
                'momentum_3min': momentum_3min
            }
        }

    # FILTER 5: Volume confirmation
    volume_ratio = calculate_volume_ratio(volumes, lookback=60)

    if volume_ratio is None:
        return {
            'signal': 'no_signal',
            'reason': 'Cannot calculate volume ratio'
        }

    if volume_ratio < 1.5:
        return {
            'signal': 'no_signal',
            'reason': f'Volume too low: {volume_ratio:.2f}x average (need 1.5x+)',
            'metrics': {
                'momentum_5min': momentum_5min,
                'momentum_15min': momentum_15min,
                'momentum_3min': momentum_3min,
                'volume_ratio': volume_ratio
            }
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

    # Calculate acceleration magnitude
    acceleration = momentum_5min - momentum_15min

    # Determine confidence
    if acceleration > 0.5 and volume_ratio > 2.0:
        confidence = 'high'
    elif acceleration > 0.3 and volume_ratio > 1.8:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'signal': 'buy',
        'strategy': 'simple_momentum',
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'confidence': confidence,
        'reasoning': (
            f"MOMENTUM ACCELERATION: 5-min momentum ({momentum_5min:.3f}%) > "
            f"15-min momentum ({momentum_15min:.3f}%), acceleration = +{acceleration:.3f}%. "
            f"Recent 3-min: {momentum_3min:.3f}% (confirming). "
            f"Volume {volume_ratio:.1f}x average."
        ),
        'metrics': {
            'momentum_5min': momentum_5min,
            'momentum_15min': momentum_15min,
            'momentum_3min': momentum_3min,
            'acceleration': acceleration,
            'volume_ratio': volume_ratio
        }
    }


def check_momentum_exit(
    prices: List[float],
    entry_price: float,
    current_price: float,
    minutes_held: int,
    target_price: float,
    stop_price: float,
    max_hold_minutes: int = 180  # 3 hours
) -> Optional[Dict]:
    """
    Check if we should exit a momentum trade.

    EXIT RULES:
    1. Hit 0.7% NET profit target
    2. Hit 0.5% GROSS stop loss
    3. 5-min momentum turns negative (reversal)
    4. Max hold 3 hours

    Args:
        prices: Historical prices at 1-minute intervals
        entry_price: Entry price
        current_price: Current price
        minutes_held: Minutes since entry
        target_price: Profit target price
        stop_price: Stop loss price
        max_hold_minutes: Max hold time (default 180 min = 3 hours)

    Returns:
        Exit signal or None
    """

    # Check stop loss first
    if current_price <= stop_price:
        return {
            'exit': True,
            'exit_price': stop_price,
            'exit_reason': 'stop_loss',
            'reasoning': f'Hit stop loss at ${stop_price:.2f}'
        }

    # Check profit target
    if current_price >= target_price:
        return {
            'exit': True,
            'exit_price': target_price,
            'exit_reason': 'target',
            'reasoning': f'Hit profit target at ${target_price:.2f}'
        }

    # Check max hold time
    if minutes_held >= max_hold_minutes:
        return {
            'exit': True,
            'exit_price': current_price,
            'exit_reason': 'max_hold',
            'reasoning': f'Max hold time reached ({minutes_held} minutes)'
        }

    # Check momentum reversal (5-min momentum turns negative)
    if len(prices) >= 6:
        momentum_5min = calculate_momentum(prices, 5)

        if momentum_5min is not None and momentum_5min < -0.2:
            # Momentum reversed (now falling)
            return {
                'exit': True,
                'exit_price': current_price,
                'exit_reason': 'momentum_reversal',
                'reasoning': f'Momentum reversed: 5-min = {momentum_5min:.3f}% (falling)'
            }

    # No exit signal
    return None


def get_strategy_info() -> Dict:
    """Get strategy information."""
    return {
        'name': 'Simple Momentum Strategy',
        'version': '1.0',
        'approach': 'Buy when price accelerates with volume confirmation',
        'data_frequency': '1-minute intervals',
        'holding_time': '1-3 hours (60-180 minutes)',
        'target_profit': '0.7% NET',
        'stop_loss': '0.5% GROSS',
        'expected_win_rate': '50-60% (typical for momentum)',
        'entry_criteria': [
            '5-min momentum > 15-min momentum (acceleration)',
            '5-min momentum > 0.3% (meaningful move)',
            'Recent 3-min momentum > 0% (confirming direction)',
            'Volume >= 1.5x average (confirmation)'
        ],
        'exit_criteria': [
            'Hit 0.7% NET profit target',
            'Hit 0.5% GROSS stop loss',
            '5-min momentum turns negative (reversal)',
            'Max hold 3 hours'
        ]
    }
