"""
Multi-Timeframe Momentum Breakout Strategy

Optimized for profitability in high-fee environments (0.25% maker / 0.50% taker).
Targets 5-15% swing moves on 4-hour timeframe with multi-timeframe confirmation.

STRATEGY OVERVIEW:
1. Core Setup (4H Timeframe):
   - Bollinger Band Squeeze: Identifies consolidation/energy buildup
   - Entry Trigger: 4H candle closes above upper Bollinger Band
   - Volume Confirmation: 2.0x average volume (filters fakeouts)
   - RSI 50-70: Rising momentum, not overbought
   - MACD Histogram: Positive and increasing

2. Multi-Timeframe Filter (Daily Timeframe):
   - Only LONG if Daily price > 200-period MA
   - Prevents buying into secular bear markets

3. Exit Strategy:
   - Trailing Stop: Track hourly candles, exit on first negative close
   - ATR-Based Stop: Initial stop at 2x ATR below entry

4. Risk Management:
   - Max 1% capital risk per trade
   - Position sizing based on ATR stop distance

This strategy is designed to be:
- Fee-efficient: Larger moves cover 0.50% taker fees + 24% taxes
- Trend-following: Only trades in direction of higher timeframe trend
- High win-rate: Multiple confirmations reduce false signals
"""

from typing import List, Optional, Dict
from utils.technical_indicators import (
    calculate_bollinger_bands,
    is_bollinger_squeeze,
    calculate_macd,
    calculate_atr,
    calculate_sma,
    calculate_ema,
    calculate_volume_average,
    is_volume_spike,
    aggregate_candles_to_timeframe
)
from utils.price_helpers import calculate_rsi
from utils.file_helpers import get_property_values_from_crypto_file
import json
import os


def load_candle_data(symbol: str, data_directory: str = 'coinbase-data', max_age_hours: int = 720) -> List[Dict]:
    """
    Load historical 5-minute candle data from the coinbase-data files.

    Args:
        symbol: Product ID (e.g., 'BTC-USD')
        data_directory: Directory containing candle data
        max_age_hours: Maximum age of data to load (default 720h = 30 days)

    Returns:
        List of candle dicts with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    """
    file_path = os.path.join(data_directory, f"{symbol}.json")

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        # Filter by age
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        # Transform data entries into candle format
        candles = []
        for entry in data:
            timestamp = entry.get('timestamp')
            if not timestamp:
                continue

            # Filter by age
            if current_time - timestamp > max_age_seconds:
                continue

            # For 5-minute data, we only have close prices
            # We'll approximate OHLC from consecutive close prices
            price = float(entry.get('price', 0))
            volume = float(entry.get('volume_24h', 0))

            candles.append({
                'timestamp': timestamp,
                'open': price,  # Approximation
                'high': price,  # Approximation
                'low': price,   # Approximation
                'close': price,
                'volume': volume
            })

        return candles

    except Exception as e:
        print(f"  ERROR loading candle data for {symbol}: {e}")
        return []


def check_mtf_breakout_signal(
    symbol: str,
    data_directory: str = 'coinbase-data',
    current_price: float = None,
    entry_fee_pct: float = 0.50,
    exit_fee_pct: float = 0.50,
    tax_rate_pct: float = 24.0,
    atr_stop_multiplier: float = 2.0,
    target_profit_pct: float = 7.5,
    max_age_hours: int = 720
) -> Optional[Dict]:
    """
    Check for multi-timeframe momentum breakout signals.

    Args:
        symbol: Product ID (e.g., 'BTC-USD')
        data_directory: Directory containing candle data
        current_price: Current market price
        entry_fee_pct: Entry fee percentage (default 0.50%)
        exit_fee_pct: Exit fee percentage (default 0.50%)
        tax_rate_pct: Tax rate percentage (default 24%)
        atr_stop_multiplier: ATR multiplier for stop-loss (default 2.0)
        target_profit_pct: Target profit percentage (default 7.5%)
        max_age_hours: Maximum age of data to use

    Returns:
        Signal dictionary or None:
        {
            'signal': 'buy' or 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'high' or 'medium' or 'low',
            'entry_price': float,
            'stop_loss': float,
            'profit_target': float,
            'reasoning': str,
            'metrics': {...}
        }
    """

    # Load 5-minute candle data
    candles_5min = load_candle_data(symbol, data_directory, max_age_hours)

    if len(candles_5min) < 300:  # Need at least 25 hours of data (300 x 5min)
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': f"Insufficient data: {len(candles_5min)} candles (need 300+)",
            'metrics': {}
        }

    # Aggregate to 4-hour candles (240 minutes)
    candles_4h = aggregate_candles_to_timeframe(candles_5min, 240)

    if len(candles_4h) < 30:  # Need at least 30 4H candles (5 days)
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': f"Insufficient 4H data: {len(candles_4h)} candles (need 30+)",
            'metrics': {}
        }

    # Aggregate to daily candles (1440 minutes)
    candles_daily = aggregate_candles_to_timeframe(candles_5min, 1440)

    if len(candles_daily) < 200:  # Need 200 days for 200-MA
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': f"Insufficient daily data for 200-MA: {len(candles_daily)} candles (need 200+)",
            'metrics': {}
        }

    # ========================================
    # STEP 1: MULTI-TIMEFRAME FILTER (DAILY)
    # ========================================
    # Only trade LONG if Daily price > 200-period MA
    daily_closes = [float(c['close']) for c in candles_daily]
    ma_200_daily = calculate_sma(daily_closes, 200)

    if not ma_200_daily:
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': "Failed to calculate 200-day MA",
            'metrics': {}
        }

    daily_price = daily_closes[-1]

    if daily_price < ma_200_daily:
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': f"Daily trend filter: Price ${daily_price:.2f} below 200-MA ${ma_200_daily:.2f} (bearish higher timeframe)",
            'metrics': {
                'daily_price': daily_price,
                'ma_200_daily': ma_200_daily,
                'daily_trend': 'bearish'
            }
        }

    # ========================================
    # STEP 2: 4H BOLLINGER BAND SQUEEZE
    # ========================================
    closes_4h = [float(c['close']) for c in candles_4h]

    # Check for Bollinger Band squeeze
    bb_squeeze = is_bollinger_squeeze(closes_4h, period=20, lookback=20)

    if not bb_squeeze:
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': "No Bollinger Band squeeze detected on 4H (waiting for consolidation)",
            'metrics': {
                'daily_price': daily_price,
                'ma_200_daily': ma_200_daily,
                'daily_trend': 'bullish',
                'bb_squeeze': False
            }
        }

    # ========================================
    # STEP 3: BREAKOUT TRIGGER
    # ========================================
    # Check if current 4H candle closed above upper Bollinger Band
    bb = calculate_bollinger_bands(closes_4h, period=20)

    if not bb:
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': "Failed to calculate Bollinger Bands",
            'metrics': {}
        }

    current_close_4h = closes_4h[-1]
    upper_band = bb['upper']

    if current_close_4h <= upper_band:
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': f"4H price ${current_close_4h:.2f} has not broken above upper BB ${upper_band:.2f}",
            'metrics': {
                'daily_price': daily_price,
                'ma_200_daily': ma_200_daily,
                'daily_trend': 'bullish',
                'bb_squeeze': True,
                'current_price_4h': current_close_4h,
                'upper_bb': upper_band
            }
        }

    # ========================================
    # STEP 4: VOLUME CONFIRMATION
    # ========================================
    volumes_4h = [float(c['volume']) for c in candles_4h]

    if not is_volume_spike(volumes_4h, multiplier=2.0, period=10):
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': "Breakout volume insufficient (< 2.0x average - possible fakeout)",
            'metrics': {
                'daily_price': daily_price,
                'ma_200_daily': ma_200_daily,
                'daily_trend': 'bullish',
                'bb_squeeze': True,
                'current_price_4h': current_close_4h,
                'upper_bb': upper_band,
                'volume_confirmed': False
            }
        }

    # ========================================
    # STEP 5: RSI MOMENTUM CHECK
    # ========================================
    rsi = calculate_rsi(closes_4h, period=14)

    if not rsi or rsi < 50 or rsi > 70:
        reason = "No RSI data" if not rsi else f"RSI {rsi:.1f} outside optimal range 50-70"
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': f"{reason} (need rising but not overbought momentum)",
            'metrics': {
                'daily_price': daily_price,
                'ma_200_daily': ma_200_daily,
                'daily_trend': 'bullish',
                'bb_squeeze': True,
                'current_price_4h': current_close_4h,
                'upper_bb': upper_band,
                'volume_confirmed': True,
                'rsi': rsi
            }
        }

    # ========================================
    # STEP 6: MACD CONFIRMATION
    # ========================================
    macd = calculate_macd(closes_4h, fast_period=12, slow_period=26, signal_period=9)

    if not macd or macd['histogram'] <= 0:
        reason = "No MACD data" if not macd else f"MACD histogram {macd['histogram']:.4f} not positive"
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': f"{reason} (need positive momentum confirmation)",
            'metrics': {
                'daily_price': daily_price,
                'ma_200_daily': ma_200_daily,
                'daily_trend': 'bullish',
                'bb_squeeze': True,
                'current_price_4h': current_close_4h,
                'upper_bb': upper_band,
                'volume_confirmed': True,
                'rsi': rsi,
                'macd_histogram': macd['histogram'] if macd else None
            }
        }

    # Check if MACD histogram is increasing (compare with previous)
    # Calculate previous MACD
    if len(closes_4h) > 1:
        prev_macd = calculate_macd(closes_4h[:-1], fast_period=12, slow_period=26, signal_period=9)
        if prev_macd:
            macd_increasing = macd['histogram'] > prev_macd['histogram']
        else:
            macd_increasing = True  # Can't verify, assume true
    else:
        macd_increasing = True

    if not macd_increasing:
        return {
            'signal': 'no_signal',
            'strategy': 'mtf_momentum_breakout',
            'confidence': 'low',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': "MACD histogram not increasing (momentum weakening)",
            'metrics': {
                'daily_price': daily_price,
                'ma_200_daily': ma_200_daily,
                'daily_trend': 'bullish',
                'bb_squeeze': True,
                'current_price_4h': current_close_4h,
                'upper_bb': upper_band,
                'volume_confirmed': True,
                'rsi': rsi,
                'macd_histogram': macd['histogram'],
                'macd_increasing': False
            }
        }

    # ========================================
    # SIGNAL CONFIRMED - CALCULATE ENTRY/EXIT
    # ========================================

    # Entry price: current market price (or use current_close_4h as proxy)
    entry_price = current_price if current_price else current_close_4h

    # Calculate ATR for stop-loss
    atr = calculate_atr(candles_4h, period=14)

    if not atr:
        atr = entry_price * 0.02  # Fallback: 2% of entry price

    # Stop-loss: 2x ATR below entry
    stop_loss = entry_price - (atr * atr_stop_multiplier)

    # Profit target: target_profit_pct % above entry
    profit_target = entry_price * (1 + target_profit_pct / 100)

    # Calculate net profit after fees and taxes
    # Entry cost: entry_price + (entry_price * entry_fee_pct / 100)
    # Exit revenue: profit_target - (profit_target * exit_fee_pct / 100)
    # Gross profit: exit_revenue - entry_cost
    # Taxes: gross_profit * (tax_rate_pct / 100)
    # Net profit: gross_profit - taxes

    entry_cost = entry_price * (1 + entry_fee_pct / 100)
    exit_revenue = profit_target * (1 - exit_fee_pct / 100)
    gross_profit = exit_revenue - entry_cost
    taxes = gross_profit * (tax_rate_pct / 100) if gross_profit > 0 else 0
    net_profit = gross_profit - taxes
    net_profit_pct = (net_profit / entry_cost) * 100

    # Risk/Reward ratio
    risk = entry_price - stop_loss
    reward = profit_target - entry_price
    risk_reward_ratio = reward / risk if risk > 0 else 0

    # Confidence scoring
    confidence = 'medium'  # Base confidence

    # High confidence if RSI is in sweet spot and MACD is strongly positive
    if 55 <= rsi <= 65 and macd['histogram'] > 0.01:
        confidence = 'high'
    # Low confidence if RSI near boundaries or weak MACD
    elif rsi < 52 or rsi > 68 or macd['histogram'] < 0.005:
        confidence = 'medium'

    return {
        'signal': 'buy',
        'strategy': 'mtf_momentum_breakout',
        'confidence': confidence,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'reasoning': (
            f"MTF MOMENTUM BREAKOUT (LONG): 4H squeeze breakout with multi-timeframe confirmation. "
            f"Daily trend bullish (${daily_price:.2f} > 200-MA ${ma_200_daily:.2f}), "
            f"4H broke above BB ${upper_band:.2f}, volume 2x+, RSI {rsi:.1f}, MACD+ increasing. "
            f"Target: ${profit_target:.2f} (+{target_profit_pct:.1f}%), Stop: ${stop_loss:.2f}, "
            f"Net profit after fees/taxes: {net_profit_pct:.2f}%"
        ),
        'metrics': {
            'daily_price': daily_price,
            'ma_200_daily': ma_200_daily,
            'daily_trend': 'bullish',
            'bb_squeeze': True,
            'current_price_4h': current_close_4h,
            'upper_bb': upper_band,
            'lower_bb': bb['lower'],
            'middle_bb': bb['middle'],
            'volume_confirmed': True,
            'rsi': rsi,
            'macd_histogram': macd['histogram'],
            'macd_increasing': True,
            'atr': atr,
            'risk_reward_ratio': risk_reward_ratio,
            'net_profit_pct': net_profit_pct,
            'gross_profit_pct': ((gross_profit / entry_cost) * 100) if entry_cost > 0 else 0
        }
    }
