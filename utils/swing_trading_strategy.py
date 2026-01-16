#!/usr/bin/env python3
"""
Swing Trading Strategy - Hold 2-7 Days for 3-5% Moves

Strategy:
- IDENTIFY: Strong uptrend (7-day MA > 30-day MA)
- ENTER: Pullback to 7-day MA (buy the dip in uptrend)
- TARGET: 3% NET profit (~4.5% gross move)
- STOP: 1.5% GROSS (give room for volatility)
- HOLD: 2-10 days max
- POSITIONS: Max 2 concurrent

Expected win rate: 55-65% (typical for swing trading)
"""

from typing import Dict, List, Optional
from utils.profit_calculator import calculate_required_price_for_target_profit
from utils.price_helpers import calculate_rsi


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return None

    # Start with SMA
    sma = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)

    ema = sma
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def identify_trend(prices: List[float], volumes: List[float] = None) -> Dict:
    """
    Identify if we're in a strong uptrend suitable for swing trading.

    UPTREND CRITERIA:
    - 7-day SMA > 30-day SMA (short-term above long-term)
    - Price > 7-day SMA (price above short-term MA)
    - 7-day momentum > 3% (strong move)
    - Rising trend (7-day SMA today > 7-day SMA 3 days ago)

    Args:
        prices: Historical prices (hourly)
        volumes: Historical volumes (optional)

    Returns:
        {
            'trend': 'strong_up' | 'weak_up' | 'neutral' | 'down',
            'sma7': float,
            'sma30': float,
            'momentum_7d': float (% change over 7 days),
            'distance_from_sma7': float (% above/below SMA7),
            'is_rising': bool (SMA7 sloping upward),
            'strength_score': float (0-100)
        }
    """
    if len(prices) < 30 * 24:  # Need 30 days of hourly data
        return {'trend': 'unknown', 'reason': 'Insufficient data'}

    # Convert hourly to daily (take every 24th price for simplicity)
    daily_prices = [prices[i] for i in range(0, len(prices), 24)]

    if len(daily_prices) < 30:
        return {'trend': 'unknown', 'reason': 'Insufficient daily data'}

    current_price = prices[-1]

    # Calculate SMAs
    sma7 = calculate_sma(daily_prices, 7)
    sma30 = calculate_sma(daily_prices, 30)

    if not sma7 or not sma30:
        return {'trend': 'unknown', 'reason': 'Cannot calculate SMAs'}

    # Calculate 7-day momentum
    if len(daily_prices) >= 7:
        price_7d_ago = daily_prices[-7]
        momentum_7d = ((daily_prices[-1] - price_7d_ago) / price_7d_ago) * 100
    else:
        momentum_7d = 0

    # Check if SMA7 is rising
    if len(daily_prices) >= 10:
        sma7_3d_ago = calculate_sma(daily_prices[:-3], 7)
        is_rising = sma7 > sma7_3d_ago if sma7_3d_ago else False
    else:
        is_rising = False

    # Distance from SMA7
    distance_from_sma7 = ((current_price - sma7) / sma7) * 100

    # Calculate strength score (0-100)
    score = 0

    # 1. SMA alignment (30 points)
    if sma7 > sma30:
        sma_diff_pct = ((sma7 - sma30) / sma30) * 100
        score += min(sma_diff_pct * 10, 30)  # Max 30 points

    # 2. Price above SMA7 (20 points)
    if current_price > sma7:
        score += 20

    # 3. Momentum (30 points)
    if momentum_7d > 0:
        score += min(momentum_7d * 6, 30)  # Max 30 points

    # 4. Rising SMA7 (20 points)
    if is_rising:
        score += 20

    # Determine trend
    if sma7 > sma30 and current_price > sma7 and momentum_7d > 3 and is_rising:
        trend = 'strong_up'
    elif sma7 > sma30 and momentum_7d > 0:
        trend = 'weak_up'
    elif sma7 < sma30:
        trend = 'down'
    else:
        trend = 'neutral'

    return {
        'trend': trend,
        'sma7': sma7,
        'sma30': sma30,
        'momentum_7d': momentum_7d,
        'distance_from_sma7': distance_from_sma7,
        'is_rising': is_rising,
        'strength_score': min(score, 100)
    }


def check_bounce_confirmation(prices: List[float], sma7: float) -> Dict:
    """
    Check if price is bouncing off the 7-day SMA support.

    BOUNCE CRITERIA:
    1. Price touched/went below SMA7 in last 6 hours (pullback happened)
    2. Price is now above SMA7 (bounce started)
    3. Last 3 hours showing higher lows (momentum turning up)

    Args:
        prices: Recent hourly prices (need at least 12 hours)
        sma7: Current 7-day SMA value

    Returns:
        {'bouncing': bool, 'reason': str, 'bounce_strength': float}
    """
    if len(prices) < 12:
        return {'bouncing': False, 'reason': 'Insufficient data', 'bounce_strength': 0}

    current_price = prices[-1]

    # Check 1: Did price touch/go below SMA7 in last 6 hours?
    recent_prices = prices[-6:]
    touched_support = any(p <= sma7 * 1.01 for p in recent_prices[:-1])  # Allow 1% above

    if not touched_support:
        return {
            'bouncing': False,
            'reason': 'Price did not touch SMA7 support',
            'bounce_strength': 0
        }

    # Check 2: Is price now above SMA7? (bounce started)
    if current_price < sma7:
        return {
            'bouncing': False,
            'reason': f'Price still below SMA7: ${current_price:.2f} < ${sma7:.2f}',
            'bounce_strength': 0
        }

    # Check 3: Are we seeing higher lows? (momentum turning up)
    last_3_hours = prices[-3:]

    # Check for higher lows pattern
    if len(last_3_hours) >= 3:
        # Find local low in last 3 hours
        low_idx = last_3_hours.index(min(last_3_hours))

        # If low was at the start and we're higher now = bouncing
        if low_idx == 0 and current_price > last_3_hours[0]:
            bounce_strength = ((current_price - last_3_hours[0]) / last_3_hours[0]) * 100

            return {
                'bouncing': True,
                'reason': f'Bounce confirmed: touched SMA7, now {bounce_strength:.2f}% higher',
                'bounce_strength': bounce_strength
            }

    # Check for simple momentum: last hour > hour before
    if len(prices) >= 2 and prices[-1] > prices[-2]:
        momentum_1h = ((prices[-1] - prices[-2]) / prices[-2]) * 100

        return {
            'bouncing': True,
            'reason': f'Bounce started: +{momentum_1h:.2f}% last hour after touching SMA7',
            'bounce_strength': momentum_1h
        }

    return {
        'bouncing': False,
        'reason': 'Price at SMA7 but no bounce momentum yet',
        'bounce_strength': 0
    }


def check_swing_entry(
    prices: List[float],
    volumes: List[float],
    current_price: float,
    current_volume: float,
    target_net_pct: float = 3.0,
    stop_gross_pct: float = 2.0,
    entry_fee_pct: float = 0.6,
    exit_fee_pct: float = 0.6,
    tax_rate_pct: float = 37.0
) -> Optional[Dict]:
    """
    Check for swing trading entry signal WITH BOUNCE CONFIRMATION.

    NEW ENTRY LOGIC (Bounce Confirmation):
    1. Strong uptrend (trend score > 60)
    2. Price TOUCHED 7-day SMA recently (pullback happened)
    3. Price BOUNCING off SMA (reversal confirmed)
    4. RSI 35-65 (healthy, recovering)
    5. Volume >= 0.5x average (some activity)

    Args:
        prices: Historical hourly prices
        volumes: Historical hourly volumes
        current_price: Current price
        current_volume: Current 24h volume
        target_net_pct: Target NET profit (default 3%)
        stop_gross_pct: Stop GROSS loss (default 2.0% - wider for volatility)

    Returns:
        Signal dictionary or None
    """

    if len(prices) < 30 * 24:  # 30 days
        return {
            'signal': 'no_signal',
            'reason': 'Insufficient data (need 30+ days)'
        }

    # FILTER 1: Identify trend
    trend_info = identify_trend(prices, volumes)

    if trend_info['trend'] not in ['strong_up', 'weak_up']:
        return {
            'signal': 'no_signal',
            'reason': f"Not in uptrend: {trend_info['trend']}",
            'metrics': trend_info
        }

    if trend_info['strength_score'] < 60:
        return {
            'signal': 'no_signal',
            'reason': f"Trend too weak: score {trend_info['strength_score']:.0f}/100 (need 60+)",
            'metrics': trend_info
        }

    # FILTER 2: BOUNCE CONFIRMATION (NEW!)
    # Must see price touch SMA7 and start bouncing
    bounce_check = check_bounce_confirmation(prices, trend_info['sma7'])

    if not bounce_check['bouncing']:
        return {
            'signal': 'no_signal',
            'reason': f"No bounce confirmation: {bounce_check['reason']}",
            'metrics': {**trend_info, 'bounce': bounce_check}
        }

    # FILTER 3: Price position (not too extended after bounce)
    distance = trend_info['distance_from_sma7']

    if distance > 5:  # More than 5% above SMA7 = bounce already happened
        return {
            'signal': 'no_signal',
            'reason': f"Bounce already ran: {distance:.1f}% above SMA7 (need < 5%)",
            'metrics': trend_info
        }

    # FILTER 4: RSI check (want recovering, not extreme)
    rsi = calculate_rsi(prices, period=14)

    if not rsi:
        return {'signal': 'no_signal', 'reason': 'Cannot calculate RSI'}

    if rsi < 30:  # Too oversold = might keep falling
        return {
            'signal': 'no_signal',
            'reason': f"RSI too low: {rsi:.0f} (need 30+, avoid falling knives)",
            'metrics': {'rsi': rsi, **trend_info}
        }

    if rsi > 65:  # Too high = pullback not deep enough
        return {
            'signal': 'no_signal',
            'reason': f"RSI too high: {rsi:.0f} (need < 65, want pullback)",
            'metrics': {'rsi': rsi, **trend_info}
        }

    # FILTER 5: Volume check (want some activity)
    if len(volumes) >= 24:
        avg_volume = sum(volumes[-24:]) / 24
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    else:
        volume_ratio = 1.0

    if volume_ratio < 0.5:  # Dead volume = low conviction
        return {
            'signal': 'no_signal',
            'reason': f"Volume too low: {volume_ratio:.2f}x average (need 0.5x+)",
            'metrics': {'volume_ratio': volume_ratio, 'rsi': rsi, **trend_info}
        }

    # ENTRY APPROVED! (Bounce confirmed)
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

    # Determine confidence based on bounce strength and trend
    if trend_info['strength_score'] >= 80 and bounce_check['bounce_strength'] > 0.5:
        confidence = 'high'
    elif trend_info['strength_score'] >= 70 and bounce_check['bounce_strength'] > 0.3:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'signal': 'buy',
        'strategy': 'swing_trade_bounce',
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'confidence': confidence,
        'trend_score': trend_info['strength_score'],
        'reasoning': (
            f"SWING TRADE (BOUNCE CONFIRMED): Strong uptrend (score {trend_info['strength_score']:.0f}/100). "
            f"Price bounced off 7-day SMA (${trend_info['sma7']:.2f}), now {distance:.1f}% above. "
            f"{bounce_check['reason']}. "
            f"RSI: {rsi:.0f} (recovering). "
            f"Volume {volume_ratio:.1f}x average."
        ),
        'metrics': {
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'bounce_strength': bounce_check['bounce_strength'],
            **trend_info
        }
    }


def check_swing_exit(
    prices: List[float],
    entry_price: float,
    current_price: float,
    hours_held: int,
    target_price: float,
    stop_price: float,
    entry_sma7: float,
    max_hold_hours: int = 240  # 10 days
) -> Optional[Dict]:
    """
    Check if we should exit a swing trade.

    EXIT RULES:
    1. Hit 3% NET profit target
    2. Hit 1.5% GROSS stop loss
    3. Trend reversal (price breaks 2% below 7-day SMA)
    4. Max hold 10 days (don't become bagholder)

    Args:
        prices: Historical prices
        entry_price: Entry price
        current_price: Current price
        hours_held: Hours since entry
        target_price: Profit target
        stop_price: Stop loss
        entry_sma7: 7-day SMA at entry (for trailing)
        max_hold_hours: Max hold (default 240h = 10 days)

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
    if hours_held >= max_hold_hours:
        return {
            'exit': True,
            'exit_price': current_price,
            'exit_reason': 'max_hold',
            'reasoning': f'Max hold time reached ({hours_held/24:.1f} days)'
        }

    # Check trend reversal (price breaks below SMA7)
    if len(prices) >= 7 * 24:
        daily_prices = [prices[i] for i in range(0, len(prices), 24)]
        current_sma7 = calculate_sma(daily_prices, 7)

        if current_sma7:
            distance_from_sma = ((current_price - current_sma7) / current_sma7) * 100

            # Exit if price drops 2%+ below SMA7 (trend broken)
            if distance_from_sma < -2:
                return {
                    'exit': True,
                    'exit_price': current_price,
                    'exit_reason': 'trend_break',
                    'reasoning': f'Trend broken: price {distance_from_sma:.1f}% below SMA7'
                }

    # No exit signal
    return None


def get_strategy_info() -> Dict:
    """Get strategy information."""
    return {
        'name': 'Swing Trading Strategy',
        'version': '1.0',
        'approach': 'Buy pullbacks in strong uptrends, hold 2-10 days',
        'holding_time': '2-10 days (48-240 hours)',
        'target_profit': '3% NET (~4.5% gross)',
        'stop_loss': '1.5% GROSS',
        'expected_win_rate': '55-65% (typical for swing trading)',
        'max_concurrent_positions': 2,
        'position_size': '$2,250 each (split $4,500 capital)',
        'entry_criteria': [
            'Strong uptrend (7-day SMA > 30-day SMA, score > 60)',
            'Price 0-8% above 7-day SMA (pullback to support)',
            'RSI 35-70 (healthy, not extreme)',
            'Volume >= 0.5x average',
            '7-day momentum > 3%'
        ],
        'exit_criteria': [
            'Hit 3% NET profit target',
            'Hit 1.5% GROSS stop loss',
            'Price breaks 2% below 7-day SMA (trend reversal)',
            'Max hold 10 days'
        ]
    }
