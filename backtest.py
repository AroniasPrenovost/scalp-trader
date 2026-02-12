#!/usr/bin/env python3
"""
High-Performance Backtesting Framework for Crypto Scalping Strategies

Vectorized indicator pre-computation for fast iteration.
Designed for Coinbase Advanced 2 fee tier (0.125% maker per side).

Usage:
    python backtest.py                          # Run default strategy on all symbols
    python backtest.py --strategy rsi_mr        # Specific strategy
    python backtest.py --optimize               # Grid search optimization
    python backtest.py --walk-forward           # Walk-forward validation
    python backtest.py --symbol BTC-USD         # Single symbol
    python backtest.py --show-trades            # Show individual trades

Author: Trading Research Engine
"""

import json
import os
import sys
import math
import statistics
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# CONSTANTS
# ==============================================================================

FEE_TIERS = {
    'intro2': {'name': 'Intro 2', 'maker': 0.004, 'taker': 0.008},
    'adv1': {'name': 'Advanced 1', 'maker': 0.0025, 'taker': 0.005},
    'adv2': {'name': 'Advanced 2', 'maker': 0.00125, 'taker': 0.0025},
    'adv3': {'name': 'Advanced 3', 'maker': 0.00075, 'taker': 0.0015},
}

TAX_RATE = 0.24  # 24% short-term capital gains

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coinbase-data')

SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD', 'ADA-USD', 'LTC-USD', 'ATOM-USD']


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Trade:
    symbol: str
    entry_time: float
    entry_price: float
    exit_time: float
    exit_price: float
    direction: str  # 'long' or 'short'
    exit_reason: str
    gross_pnl_pct: float  # percentage move
    fee_pct: float  # total round-trip fee percentage
    net_pnl_pct: float  # after fees
    net_pnl_usd: float  # dollar P&L on position
    hold_bars: int  # number of 5-min bars held


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict
    symbol: str
    period: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    gross_profit_pct: float = 0.0
    total_fees_pct: float = 0.0
    net_profit_pct: float = 0.0
    net_profit_after_tax_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    trades_per_day: float = 0.0
    avg_hold_bars: float = 0.0
    avg_hold_hours: float = 0.0
    net_profit_usd: float = 0.0
    net_profit_after_tax_usd: float = 0.0
    trades: List[Trade] = field(default_factory=list)


# ==============================================================================
# VECTORIZED INDICATOR COMPUTATION
# ==============================================================================

def compute_rsi_array(closes: list, period: int = 14) -> list:
    """Compute RSI for every bar. Returns list same length as closes (None for early bars)."""
    n = len(closes)
    rsi = [None] * n
    if n < period + 1:
        return rsi

    # Compute all price changes
    changes = [0.0] * n
    for i in range(1, n):
        changes[i] = closes[i] - closes[i - 1]

    # First RSI using simple average
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        if changes[i] > 0:
            avg_gain += changes[i]
        else:
            avg_loss += abs(changes[i])
    avg_gain /= period
    avg_loss /= period

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's smoothed RSI for remaining bars
    for i in range(period + 1, n):
        change = changes[i]
        gain = change if change > 0 else 0.0
        loss = abs(change) if change < 0 else 0.0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_ema_array(closes: list, period: int) -> list:
    """Compute EMA for every bar."""
    n = len(closes)
    ema = [None] * n
    if n < period:
        return ema

    # Seed with SMA
    sma = sum(closes[:period]) / period
    ema[period - 1] = sma

    multiplier = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = closes[i] * multiplier + ema[i - 1] * (1 - multiplier)

    return ema


def compute_sma_array(closes: list, period: int) -> list:
    """Compute SMA for every bar using rolling sum."""
    n = len(closes)
    sma = [None] * n
    if n < period:
        return sma

    rolling_sum = sum(closes[:period])
    sma[period - 1] = rolling_sum / period

    for i in range(period, n):
        rolling_sum += closes[i] - closes[i - period]
        sma[i] = rolling_sum / period

    return sma


def compute_bb_array(closes: list, period: int = 20, num_std: float = 2.0) -> Tuple[list, list, list, list]:
    """Compute Bollinger Bands arrays. Returns (upper, middle, lower, bandwidth)."""
    n = len(closes)
    upper = [None] * n
    middle = [None] * n
    lower = [None] * n
    bandwidth = [None] * n

    if n < period:
        return upper, middle, lower, bandwidth

    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        mid = sum(window) / period
        variance = sum((x - mid) ** 2 for x in window) / period
        std = math.sqrt(variance)

        middle[i] = mid
        upper[i] = mid + num_std * std
        lower[i] = mid - num_std * std
        bandwidth[i] = ((upper[i] - lower[i]) / mid * 100) if mid > 0 else 0

    return upper, middle, lower, bandwidth


def compute_atr_array(highs: list, lows: list, closes: list, period: int = 14) -> list:
    """Compute ATR for every bar."""
    n = len(closes)
    atr = [None] * n
    if n < period + 1:
        return atr

    # Compute true ranges
    tr = [0.0] * n
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    # First ATR = simple average
    atr_val = sum(tr[1:period + 1]) / period
    atr[period] = atr_val

    # Wilder's smoothing
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        atr[i] = atr_val

    return atr


def compute_stoch_rsi_array(rsi_arr: list, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[list, list]:
    """Compute Stochastic RSI (%K, %D) arrays."""
    n = len(rsi_arr)
    stoch_k_raw = [None] * n
    stoch_k = [None] * n
    stoch_d = [None] * n

    for i in range(period - 1, n):
        window = []
        for j in range(i - period + 1, i + 1):
            if rsi_arr[j] is not None:
                window.append(rsi_arr[j])
        if len(window) < period:
            continue
        rsi_min = min(window)
        rsi_max = max(window)
        if rsi_max - rsi_min == 0:
            stoch_k_raw[i] = 50.0
        else:
            stoch_k_raw[i] = ((rsi_arr[i] - rsi_min) / (rsi_max - rsi_min)) * 100.0

    # Smooth %K
    for i in range(n):
        if stoch_k_raw[i] is None:
            continue
        window = []
        for j in range(max(0, i - smooth_k + 1), i + 1):
            if stoch_k_raw[j] is not None:
                window.append(stoch_k_raw[j])
        if len(window) >= smooth_k:
            stoch_k[i] = sum(window) / len(window)

    # %D = SMA of %K
    for i in range(n):
        if stoch_k[i] is None:
            continue
        window = []
        for j in range(max(0, i - smooth_d + 1), i + 1):
            if stoch_k[j] is not None:
                window.append(stoch_k[j])
        if len(window) >= smooth_d:
            stoch_d[i] = sum(window) / len(window)

    return stoch_k, stoch_d


def compute_volume_sma_array(volumes: list, period: int = 20) -> list:
    """Compute volume SMA array."""
    return compute_sma_array(volumes, period)


def precompute_indicators(closes: list, highs: list, lows: list, volumes: list, params: dict) -> dict:
    """Pre-compute all indicators needed for strategy evaluation."""
    indicators = {}

    # RSI
    rsi_period = params.get('rsi_period', 14)
    indicators['rsi'] = compute_rsi_array(closes, rsi_period)

    # Bollinger Bands
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'], indicators['bb_bandwidth'] = \
        compute_bb_array(closes, bb_period, bb_std)

    # EMAs
    for ema_period in params.get('ema_periods', [9, 21, 50]):
        indicators[f'ema_{ema_period}'] = compute_ema_array(closes, ema_period)

    # SMAs
    for sma_period in params.get('sma_periods', [50, 200]):
        indicators[f'sma_{sma_period}'] = compute_sma_array(closes, sma_period)

    # ATR
    atr_period = params.get('atr_period', 14)
    indicators['atr'] = compute_atr_array(highs, lows, closes, atr_period)

    # ATR as percentage of price
    indicators['atr_pct'] = [None] * len(closes)
    for i in range(len(closes)):
        if indicators['atr'][i] is not None and closes[i] > 0:
            indicators['atr_pct'][i] = (indicators['atr'][i] / closes[i]) * 100.0

    # Volume SMA
    vol_period = params.get('volume_sma_period', 20)
    indicators['vol_sma'] = compute_volume_sma_array(volumes, vol_period)

    # Volume ratio (current / average)
    indicators['vol_ratio'] = [None] * len(volumes)
    for i in range(len(volumes)):
        if indicators['vol_sma'][i] is not None and indicators['vol_sma'][i] > 0:
            indicators['vol_ratio'][i] = volumes[i] / indicators['vol_sma'][i]

    # Stochastic RSI
    indicators['stoch_k'], indicators['stoch_d'] = compute_stoch_rsi_array(indicators['rsi'], rsi_period)

    # Price rate of change (momentum)
    roc_period = params.get('roc_period', 12)
    indicators['roc'] = [None] * len(closes)
    for i in range(roc_period, len(closes)):
        if closes[i - roc_period] > 0:
            indicators['roc'][i] = ((closes[i] - closes[i - roc_period]) / closes[i - roc_period]) * 100.0

    return indicators


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_symbol_data(symbol: str) -> Tuple[list, list, list, list, list]:
    """Load and return (timestamps, opens, highs, lows, closes, volumes) as float lists."""
    filepath = os.path.join(DATA_DIR, f'{symbol}.json')
    if not os.path.exists(filepath):
        return [], [], [], [], [], []

    with open(filepath, 'r') as f:
        raw = json.load(f)

    raw = sorted(raw, key=lambda x: x.get('timestamp', 0))

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    for i, entry in enumerate(raw):
        price = float(entry.get('price', 0))
        vol = float(entry.get('volume_24h', 0))
        ts = float(entry.get('timestamp', 0))
        if price <= 0 or ts <= 0:
            continue

        prev_price = float(raw[i - 1].get('price', price)) if i > 0 else price
        next_price = float(raw[i + 1].get('price', price)) if i < len(raw) - 1 else price

        timestamps.append(ts)
        opens.append(prev_price)
        highs.append(max(prev_price, price, next_price))
        lows.append(min(prev_price, price, next_price))
        closes.append(price)
        volumes.append(vol)

    return timestamps, opens, highs, lows, closes, volumes


def aggregate_to_timeframe(timestamps, opens, highs, lows, closes, volumes, target_minutes: int):
    """Aggregate 5-min data to higher timeframe using clock-aligned boundaries.

    Candles are aligned to UTC clock boundaries (e.g., 00:00, 02:00, 04:00 for 2H).
    This ensures stable candle boundaries regardless of data start/end shifts.
    """
    target_seconds = target_minutes * 60
    n = len(closes)
    new_ts, new_o, new_h, new_l, new_c, new_v = [], [], [], [], [], []

    if n == 0:
        return new_ts, new_o, new_h, new_l, new_c, new_v

    # Group bars by clock-aligned candle boundaries
    current_candle_start = (timestamps[0] // target_seconds) * target_seconds
    candle_indices = []  # indices of bars in current candle

    for i in range(n):
        candle_boundary = (timestamps[i] // target_seconds) * target_seconds
        if candle_boundary != current_candle_start and candle_indices:
            # Emit completed candle
            s, e = candle_indices[0], candle_indices[-1]
            new_ts.append(current_candle_start)
            new_o.append(opens[s])
            new_h.append(max(highs[s:e + 1]))
            new_l.append(min(lows[s:e + 1]))
            new_c.append(closes[e])
            new_v.append(sum(volumes[s:e + 1]))
            candle_indices = []
            current_candle_start = candle_boundary
        candle_indices.append(i)

    # Emit last candle (may be partial)
    if candle_indices:
        s, e = candle_indices[0], candle_indices[-1]
        new_ts.append(current_candle_start)
        new_o.append(opens[s])
        new_h.append(max(highs[s:e + 1]))
        new_l.append(min(lows[s:e + 1]))
        new_c.append(closes[e])
        new_v.append(sum(volumes[s:e + 1]))

    return new_ts, new_o, new_h, new_l, new_c, new_v


# ==============================================================================
# STRATEGY IMPLEMENTATIONS
# ==============================================================================

class VectorizedStrategy:
    """Base class for vectorized strategies."""
    name = "base"

    def __init__(self, params: dict):
        self.params = params

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators) -> List[dict]:
        """Generate all trade signals. Returns list of signal dicts."""
        raise NotImplementedError


class RSIMeanReversionStrategy(VectorizedStrategy):
    """
    RSI Mean Reversion Scalper.

    Entry: RSI drops below oversold threshold + price near/below BB lower band.
    Exit: Profit target, stop loss, RSI normalization, or max hold time.

    Optimized for 0.25% round-trip fees.
    """
    name = "rsi_mr"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_exit': 50,  # Exit when RSI recovers to this level
        'bb_period': 20,
        'bb_std': 2.0,
        'profit_target_pct': 0.8,
        'stop_loss_pct': 0.5,
        'max_hold_bars': 36,  # 3 hours at 5-min
        'min_atr_pct': 0.15,  # minimum volatility
        'require_bb_touch': False,
        'trend_filter': False,  # require price above SMA
        'trend_sma_period': 50,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 2,  # minimum bars between trades
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = max(self.params.get('bb_period', 20), self.params.get('rsi_period', 14), 50) + 10

        rsi = indicators['rsi']
        bb_lower = indicators['bb_lower']
        bb_upper = indicators['bb_upper']
        bb_middle = indicators['bb_middle']
        atr_pct = indicators['atr_pct']

        trend_sma_key = f"sma_{self.params.get('trend_sma_period', 50)}"
        trend_sma = indicators.get(trend_sma_key, [None] * n)

        rsi_oversold = self.params.get('rsi_oversold', 30)
        rsi_exit = self.params.get('rsi_exit', 50)
        profit_target = self.params.get('profit_target_pct', 0.8) / 100.0
        stop_loss = self.params.get('stop_loss_pct', 0.5) / 100.0
        max_hold = self.params.get('max_hold_bars', 36)
        min_atr = self.params.get('min_atr_pct', 0.15)
        require_bb = self.params.get('require_bb_touch', False)
        use_trend = self.params.get('trend_filter', False)
        cooldown = self.params.get('min_cooldown_bars', 2)

        i = warmup
        last_exit_bar = 0

        while i < n:
            # Skip if indicators not ready
            if rsi[i] is None or bb_lower[i] is None or atr_pct[i] is None:
                i += 1
                continue

            # Cooldown
            if i - last_exit_bar < cooldown:
                i += 1
                continue

            # Volatility filter
            if atr_pct[i] < min_atr:
                i += 1
                continue

            # Trend filter
            if use_trend and trend_sma[i] is not None:
                if closes[i] < trend_sma[i] * 0.98:
                    i += 1
                    continue

            # Entry: RSI oversold
            if rsi[i] <= rsi_oversold:
                # Optional BB touch requirement
                if require_bb and bb_lower[i] is not None and closes[i] > bb_lower[i] * 1.005:
                    i += 1
                    continue

                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                target_price = entry_price * (1.0 + profit_target)
                stop_price = entry_price * (1.0 - stop_loss)

                # Simulate holding
                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    # Check stop loss first (use low of bar)
                    if lows[j] <= stop_price:
                        exit_bar = j
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break

                    # Check profit target (use high of bar)
                    if highs[j] >= target_price:
                        exit_bar = j
                        exit_price = target_price
                        exit_reason = 'profit_target'
                        break

                    # RSI exit (mean reverted)
                    if rsi[j] is not None and rsi[j] >= rsi_exit:
                        current_pnl = (closes[j] - entry_price) / entry_price
                        if current_pnl > 0:  # Only exit if profitable
                            exit_bar = j
                            exit_price = closes[j]
                            exit_reason = 'rsi_exit'
                            break

                # Max hold time exit
                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='',
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=timestamps[exit_bar],
                    exit_price=exit_price,
                    direction='long',
                    exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct,
                    fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct,
                    net_pnl_usd=0,  # calculated later
                    hold_bars=exit_bar - entry_bar,
                ))

                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class BBBounceStrategy(VectorizedStrategy):
    """
    Bollinger Band Bounce Strategy.

    Entry: Price touches or breaks below lower BB with RSI confirmation.
    Exit: Price returns toward middle BB, profit target, or stop loss.

    More selective than pure RSI - requires both BB and RSI alignment.
    """
    name = "bb_bounce"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_oversold': 35,
        'bb_period': 20,
        'bb_std': 2.0,
        'profit_target_pct': 0.7,
        'stop_loss_pct': 0.4,
        'max_hold_bars': 24,  # 2 hours at 5-min
        'min_atr_pct': 0.1,
        'exit_at_middle_bb': True,  # Exit when price returns to middle BB
        'bb_penetration_pct': 0.0,  # How far below BB to enter (0 = at BB)
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 2,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = max(self.params.get('bb_period', 20) + 5, 50)

        rsi = indicators['rsi']
        bb_lower = indicators['bb_lower']
        bb_upper = indicators['bb_upper']
        bb_middle = indicators['bb_middle']
        atr_pct = indicators['atr_pct']

        rsi_oversold = self.params.get('rsi_oversold', 35)
        profit_target = self.params.get('profit_target_pct', 0.7) / 100.0
        stop_loss = self.params.get('stop_loss_pct', 0.4) / 100.0
        max_hold = self.params.get('max_hold_bars', 24)
        min_atr = self.params.get('min_atr_pct', 0.1)
        exit_at_mid = self.params.get('exit_at_middle_bb', True)
        bb_pen = self.params.get('bb_penetration_pct', 0.0) / 100.0
        cooldown = self.params.get('min_cooldown_bars', 2)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or bb_lower[i] is None or atr_pct[i] is None or bb_middle[i] is None:
                i += 1
                continue

            if i - last_exit_bar < cooldown:
                i += 1
                continue

            if atr_pct[i] < min_atr:
                i += 1
                continue

            # Entry: Price at or below lower BB + RSI oversold
            entry_threshold = bb_lower[i] * (1.0 + bb_pen)
            if closes[i] <= entry_threshold and rsi[i] <= rsi_oversold:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                target_price = entry_price * (1.0 + profit_target)
                stop_price = entry_price * (1.0 - stop_loss)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    if lows[j] <= stop_price:
                        exit_bar = j
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break

                    if highs[j] >= target_price:
                        exit_bar = j
                        exit_price = target_price
                        exit_reason = 'profit_target'
                        break

                    # Exit at middle BB
                    if exit_at_mid and bb_middle[j] is not None:
                        if highs[j] >= bb_middle[j]:
                            pnl = (bb_middle[j] - entry_price) / entry_price
                            if pnl > 0:
                                exit_bar = j
                                exit_price = bb_middle[j]
                                exit_reason = 'bb_middle'
                                break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))

                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class EMAMomentumStrategy(VectorizedStrategy):
    """
    EMA Crossover Momentum Strategy.

    Entry: Fast EMA crosses above slow EMA with RSI confirmation (not overbought).
    Exit: Fast EMA crosses below slow EMA, profit target, or stop loss.

    Trades in the direction of short-term momentum.
    """
    name = "ema_momentum"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'fast_ema': 9,
        'slow_ema': 21,
        'rsi_max_entry': 65,  # Don't enter if already overbought
        'rsi_min_entry': 40,  # Need some recovery momentum
        'bb_period': 20,
        'bb_std': 2.0,
        'profit_target_pct': 1.0,
        'stop_loss_pct': 0.5,
        'max_hold_bars': 48,
        'min_atr_pct': 0.15,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 3,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)

        fast_key = f"ema_{self.params.get('fast_ema', 9)}"
        slow_key = f"ema_{self.params.get('slow_ema', 21)}"
        fast_ema = indicators.get(fast_key, [None] * n)
        slow_ema = indicators.get(slow_key, [None] * n)
        rsi = indicators['rsi']
        atr_pct = indicators['atr_pct']

        rsi_max = self.params.get('rsi_max_entry', 65)
        rsi_min = self.params.get('rsi_min_entry', 40)
        profit_target = self.params.get('profit_target_pct', 1.0) / 100.0
        stop_loss = self.params.get('stop_loss_pct', 0.5) / 100.0
        max_hold = self.params.get('max_hold_bars', 48)
        min_atr = self.params.get('min_atr_pct', 0.15)
        cooldown = self.params.get('min_cooldown_bars', 3)

        warmup = max(self.params.get('slow_ema', 21), 50) + 5
        i = warmup
        last_exit_bar = 0

        while i < n:
            if (fast_ema[i] is None or slow_ema[i] is None or
                fast_ema[i-1] is None or slow_ema[i-1] is None or
                rsi[i] is None or atr_pct[i] is None):
                i += 1
                continue

            if i - last_exit_bar < cooldown:
                i += 1
                continue

            if atr_pct[i] < min_atr:
                i += 1
                continue

            # Bullish crossover: fast crosses above slow
            if fast_ema[i-1] <= slow_ema[i-1] and fast_ema[i] > slow_ema[i]:
                if rsi_min <= rsi[i] <= rsi_max:
                    entry_price = closes[i]
                    entry_bar = i
                    entry_time = timestamps[i]
                    target_price = entry_price * (1.0 + profit_target)
                    stop_price = entry_price * (1.0 - stop_loss)

                    exit_bar = None
                    exit_price = None
                    exit_reason = None

                    for j in range(i + 1, min(i + max_hold + 1, n)):
                        if lows[j] <= stop_price:
                            exit_bar = j
                            exit_price = stop_price
                            exit_reason = 'stop_loss'
                            break

                        if highs[j] >= target_price:
                            exit_bar = j
                            exit_price = target_price
                            exit_reason = 'profit_target'
                            break

                        # Bearish crossover exit
                        if (fast_ema[j] is not None and slow_ema[j] is not None and
                            fast_ema[j] < slow_ema[j]):
                            exit_bar = j
                            exit_price = closes[j]
                            exit_reason = 'ema_cross_exit'
                            break

                    if exit_bar is None:
                        exit_bar = min(i + max_hold, n - 1)
                        exit_price = closes[exit_bar]
                        exit_reason = 'max_hold'

                    gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                    net_pnl_pct = gross_pnl_pct - fee_pct

                    trades.append(Trade(
                        symbol='', entry_time=entry_time, entry_price=entry_price,
                        exit_time=timestamps[exit_bar], exit_price=exit_price,
                        direction='long', exit_reason=exit_reason,
                        gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                        net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                        hold_bars=exit_bar - entry_bar,
                    ))

                    last_exit_bar = exit_bar
                    i = exit_bar + 1
                    continue

            i += 1

        return trades


class AdaptiveRSIStrategy(VectorizedStrategy):
    """
    Adaptive RSI Strategy with Dynamic Exits.

    Key insight: Use RSI for entries but adapt exit strategy based on volatility.
    In high vol: use wider targets. In low vol: use tighter targets.

    Entry: RSI oversold + StochRSI crossover confirmation.
    Exit: Dynamic target based on ATR, or RSI normalization, or stop loss.
    """
    name = "adaptive_rsi"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'stoch_rsi_oversold': 20,  # StochRSI %K threshold
        'use_stoch_cross': True,  # Require stochRSI %K cross above %D
        'bb_period': 20,
        'bb_std': 2.0,
        'base_profit_target_pct': 0.6,  # Base target (modified by ATR)
        'atr_target_multiplier': 1.5,  # Target = ATR * this
        'stop_loss_atr_mult': 1.0,  # Stop = ATR * this
        'max_hold_bars': 36,
        'min_atr_pct': 0.1,
        'rsi_exit_threshold': 55,  # exit when RSI recovers
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 2,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']
        atr = indicators['atr']
        atr_pct = indicators['atr_pct']
        stoch_k = indicators['stoch_k']
        stoch_d = indicators['stoch_d']

        rsi_oversold = self.params.get('rsi_oversold', 30)
        stoch_oversold = self.params.get('stoch_rsi_oversold', 20)
        use_stoch = self.params.get('use_stoch_cross', True)
        base_target = self.params.get('base_profit_target_pct', 0.6) / 100.0
        atr_target_mult = self.params.get('atr_target_multiplier', 1.5)
        stop_atr_mult = self.params.get('stop_loss_atr_mult', 1.0)
        max_hold = self.params.get('max_hold_bars', 36)
        min_atr = self.params.get('min_atr_pct', 0.1)
        rsi_exit = self.params.get('rsi_exit_threshold', 55)
        cooldown = self.params.get('min_cooldown_bars', 2)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if (rsi[i] is None or atr[i] is None or atr_pct[i] is None):
                i += 1
                continue

            if i - last_exit_bar < cooldown:
                i += 1
                continue

            if atr_pct[i] < min_atr:
                i += 1
                continue

            # Entry conditions
            rsi_ok = rsi[i] <= rsi_oversold

            stoch_ok = True
            if use_stoch:
                if stoch_k[i] is None or stoch_d[i] is None:
                    stoch_ok = False
                elif stoch_k[i] > stoch_oversold:
                    stoch_ok = False
                elif (stoch_k[i-1] is not None and stoch_d[i-1] is not None):
                    # Require bullish cross: %K crosses above %D
                    if not (stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i]):
                        # Or just require both below threshold
                        if stoch_k[i] > stoch_oversold:
                            stoch_ok = False

            if rsi_ok and stoch_ok:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]

                # Dynamic target based on ATR
                atr_based_target = (atr[i] * atr_target_mult) / entry_price
                target_pct = max(base_target, atr_based_target)
                # Cap target to avoid unrealistic expectations
                target_pct = min(target_pct, 0.03)  # Max 3%
                target_price = entry_price * (1.0 + target_pct)

                # Dynamic stop based on ATR
                stop_dist = (atr[i] * stop_atr_mult) / entry_price
                stop_dist = max(stop_dist, 0.003)  # min 0.3% stop
                stop_dist = min(stop_dist, 0.02)  # max 2% stop
                stop_price = entry_price * (1.0 - stop_dist)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    if lows[j] <= stop_price:
                        exit_bar = j
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break

                    if highs[j] >= target_price:
                        exit_bar = j
                        exit_price = target_price
                        exit_reason = 'profit_target'
                        break

                    # RSI normalization exit
                    if rsi[j] is not None and rsi[j] >= rsi_exit:
                        pnl = (closes[j] - entry_price) / entry_price
                        if pnl > 0.002:  # At least 0.2% profit
                            exit_bar = j
                            exit_price = closes[j]
                            exit_reason = 'rsi_exit'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))

                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class MTFMeanReversionStrategy(VectorizedStrategy):
    """
    Multi-Timeframe Mean Reversion Strategy.

    Key insight: Use higher timeframe (1H) for trend direction,
    lower timeframe (5min) for entry timing.

    Only take mean reversion entries that align with higher TF trend.

    Entry: 5min RSI oversold + 1H trend is bullish (price above 1H EMA).
    Exit: Profit target, stop loss, RSI normalization.
    """
    name = "mtf_mr"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_oversold': 28,
        'htf_minutes': 60,  # Higher timeframe in minutes
        'htf_ema_period': 21,  # EMA period on higher TF
        'bb_period': 20,
        'bb_std': 2.0,
        'profit_target_pct': 0.8,
        'stop_loss_pct': 0.5,
        'max_hold_bars': 36,
        'min_atr_pct': 0.12,
        'rsi_exit_threshold': 50,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 2,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        """Need to also receive higher TF data."""
        # This strategy requires custom handling - HTF indicators passed in params
        htf_trend = self.params.get('_htf_trend_array', None)

        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']
        bb_lower = indicators['bb_lower']
        atr_pct = indicators['atr_pct']

        rsi_oversold = self.params.get('rsi_oversold', 28)
        profit_target = self.params.get('profit_target_pct', 0.8) / 100.0
        stop_loss = self.params.get('stop_loss_pct', 0.5) / 100.0
        max_hold = self.params.get('max_hold_bars', 36)
        min_atr = self.params.get('min_atr_pct', 0.12)
        rsi_exit = self.params.get('rsi_exit_threshold', 50)
        cooldown = self.params.get('min_cooldown_bars', 2)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or atr_pct[i] is None:
                i += 1
                continue

            if i - last_exit_bar < cooldown:
                i += 1
                continue

            if atr_pct[i] < min_atr:
                i += 1
                continue

            # HTF trend filter
            if htf_trend is not None:
                htf_idx = i  # Map 5min bar to HTF bar
                if htf_idx < len(htf_trend) and htf_trend[htf_idx] is not None:
                    if not htf_trend[htf_idx]:  # Not bullish on HTF
                        i += 1
                        continue

            # Entry: RSI oversold
            if rsi[i] <= rsi_oversold:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                target_price = entry_price * (1.0 + profit_target)
                stop_price = entry_price * (1.0 - stop_loss)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    if lows[j] <= stop_price:
                        exit_bar = j
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break

                    if highs[j] >= target_price:
                        exit_bar = j
                        exit_price = target_price
                        exit_reason = 'profit_target'
                        break

                    if rsi[j] is not None and rsi[j] >= rsi_exit:
                        pnl = (closes[j] - entry_price) / entry_price
                        if pnl > 0:
                            exit_bar = j
                            exit_price = closes[j]
                            exit_reason = 'rsi_exit'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))

                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class DualBBStrategy(VectorizedStrategy):
    """
    Dual Bollinger Band Strategy.

    Uses two sets of BBs: tight (1.5 std) and wide (2.5 std).

    Entry: Price breaks below wide BB lower + RSI < 35.
    Exit: Price returns to tight BB middle or profit target.

    The dual BB structure provides better entry and exit zones.
    """
    name = "dual_bb"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_oversold': 35,
        'bb_period': 20,
        'bb_std': 2.0,  # Wide BB for entries
        'bb_tight_std': 1.5,  # Tight BB for exits
        'profit_target_pct': 0.8,
        'stop_loss_pct': 0.5,
        'max_hold_bars': 30,
        'min_atr_pct': 0.1,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 2,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']
        bb_lower = indicators['bb_lower']  # wide BB
        bb_middle = indicators['bb_middle']
        atr_pct = indicators['atr_pct']

        # Compute tight BB
        tight_std = self.params.get('bb_tight_std', 1.5)
        bb_period = self.params.get('bb_period', 20)
        _, tight_middle, _, _ = compute_bb_array(closes, bb_period, tight_std)

        rsi_oversold = self.params.get('rsi_oversold', 35)
        profit_target = self.params.get('profit_target_pct', 0.8) / 100.0
        stop_loss = self.params.get('stop_loss_pct', 0.5) / 100.0
        max_hold = self.params.get('max_hold_bars', 30)
        min_atr = self.params.get('min_atr_pct', 0.1)
        cooldown = self.params.get('min_cooldown_bars', 2)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or bb_lower[i] is None or atr_pct[i] is None:
                i += 1
                continue

            if i - last_exit_bar < cooldown:
                i += 1
                continue

            if atr_pct[i] < min_atr:
                i += 1
                continue

            # Entry: price below wide BB lower + RSI oversold
            if closes[i] <= bb_lower[i] and rsi[i] <= rsi_oversold:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                target_price = entry_price * (1.0 + profit_target)
                stop_price = entry_price * (1.0 - stop_loss)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    if lows[j] <= stop_price:
                        exit_bar = j
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break

                    if highs[j] >= target_price:
                        exit_bar = j
                        exit_price = target_price
                        exit_reason = 'profit_target'
                        break

                    # Exit when price returns to BB middle
                    if bb_middle[j] is not None and highs[j] >= bb_middle[j]:
                        pnl = (bb_middle[j] - entry_price) / entry_price
                        if pnl > 0:
                            exit_bar = j
                            exit_price = bb_middle[j]
                            exit_reason = 'bb_middle'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))

                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


# ==============================================================================
# CLOSE-PRICE-ONLY STRATEGIES (no synthetic OHLC lookahead bias)
#
# These strategies ONLY use close prices for entry/exit decisions.
# The raw data only has close prices (highs/lows are synthesized with
# lookahead bias from adjacent candle closes), so close-only strategies
# are the most realistic for backtesting this dataset.
# ==============================================================================

class CloseOnlyReversionStrategy(VectorizedStrategy):
    """
    Close-price-only RSI + BB mean reversion scalper.

    Entry: RSI deeply oversold + optionally price below BB lower band.
    Exit: All exits based on CLOSE prices only (no synthetic high/low).

    Key design: wider stop than target to maximize win rate at extreme
    oversold entries where bounce probability is highest.
    """
    name = "co_revert"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_entry': 25,            # Deep oversold for high-probability bounce
        'rsi_exit': 50,             # Exit when RSI normalizes (if profitable)
        'use_bb_filter': True,      # Also require price < BB lower
        'bb_period': 20,
        'bb_std': 2.0,
        'profit_target_pct': 1.0,   # Gross target
        'stop_loss_pct': 0.7,       # Gross stop
        'max_hold_bars': 24,        # Max bars to hold
        'min_profit_for_rsi_exit': 0.3,  # Min gross profit for RSI-based exit
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 3,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']
        bb_lower = indicators['bb_lower']

        rsi_entry = self.params.get('rsi_entry', 25)
        rsi_exit_thresh = self.params.get('rsi_exit', 50)
        use_bb = self.params.get('use_bb_filter', True)
        profit_target = self.params.get('profit_target_pct', 1.0) / 100.0
        stop_loss = self.params.get('stop_loss_pct', 0.7) / 100.0
        max_hold = self.params.get('max_hold_bars', 24)
        min_profit_rsi = self.params.get('min_profit_for_rsi_exit', 0.3) / 100.0
        cooldown = self.params.get('min_cooldown_bars', 3)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None:
                i += 1
                continue
            if i - last_exit_bar < cooldown:
                i += 1
                continue

            # Entry: deep RSI oversold + optional BB filter
            rsi_ok = rsi[i] <= rsi_entry
            bb_ok = True
            if use_bb and bb_lower[i] is not None:
                bb_ok = closes[i] <= bb_lower[i]
            elif use_bb:
                bb_ok = False

            if rsi_ok and bb_ok:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                target_price = entry_price * (1.0 + profit_target)
                stop_price = entry_price * (1.0 - stop_loss)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    # CLOSE-ONLY stop loss
                    if closes[j] <= stop_price:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'stop_loss'
                        break
                    # CLOSE-ONLY profit target
                    if closes[j] >= target_price:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'profit_target'
                        break
                    # RSI recovery exit (only if profitable)
                    if rsi[j] is not None and rsi[j] >= rsi_exit_thresh:
                        gross_pnl = (closes[j] - entry_price) / entry_price
                        if gross_pnl >= min_profit_rsi:
                            exit_bar = j
                            exit_price = closes[j]
                            exit_reason = 'rsi_exit'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))
                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class CloseOnlyTrendDipStrategy(VectorizedStrategy):
    """
    Close-price-only Trend Dip Buyer.

    Buys dips in uptrends: higher timeframe trend (EMA direction) determines
    direction, lower timeframe RSI oversold for entry timing.

    Only uses close prices for all decisions.
    """
    name = "co_trend_dip"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_entry': 35,            # Mild oversold is enough when trend is with us
        'rsi_exit': 55,             # RSI recovery
        'trend_ema': 50,            # Price must be above this EMA (uptrend)
        'pullback_ema': 9,          # Price pulled back below this fast EMA
        'bb_period': 20,
        'bb_std': 2.0,
        'profit_target_pct': 1.0,
        'stop_loss_pct': 0.6,
        'max_hold_bars': 24,
        'min_profit_for_rsi_exit': 0.3,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 3,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']
        trend_key = f"ema_{self.params.get('trend_ema', 50)}"
        pullback_key = f"ema_{self.params.get('pullback_ema', 9)}"
        trend_ema = indicators.get(trend_key, [None] * n)
        pullback_ema = indicators.get(pullback_key, [None] * n)

        rsi_entry = self.params.get('rsi_entry', 35)
        rsi_exit_thresh = self.params.get('rsi_exit', 55)
        profit_target = self.params.get('profit_target_pct', 1.0) / 100.0
        stop_loss = self.params.get('stop_loss_pct', 0.6) / 100.0
        max_hold = self.params.get('max_hold_bars', 24)
        min_profit_rsi = self.params.get('min_profit_for_rsi_exit', 0.3) / 100.0
        cooldown = self.params.get('min_cooldown_bars', 3)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or trend_ema[i] is None or pullback_ema[i] is None:
                i += 1
                continue
            if i - last_exit_bar < cooldown:
                i += 1
                continue

            # Entry: uptrend + RSI dip + pulled back below fast EMA
            in_uptrend = closes[i] > trend_ema[i]
            is_dip = rsi[i] <= rsi_entry
            pulled_back = closes[i] < pullback_ema[i]

            if in_uptrend and is_dip and pulled_back:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                target_price = entry_price * (1.0 + profit_target)
                stop_price = entry_price * (1.0 - stop_loss)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    # CLOSE-ONLY exits
                    if closes[j] <= stop_price:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'stop_loss'
                        break
                    # Trend break stop: close below trend EMA
                    if trend_ema[j] is not None and closes[j] < trend_ema[j] * 0.995:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'trend_break'
                        break
                    if closes[j] >= target_price:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'profit_target'
                        break
                    # RSI recovery exit
                    if rsi[j] is not None and rsi[j] >= rsi_exit_thresh:
                        gross_pnl = (closes[j] - entry_price) / entry_price
                        if gross_pnl >= min_profit_rsi:
                            exit_bar = j
                            exit_price = closes[j]
                            exit_reason = 'rsi_exit'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))
                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class CloseOnlyBBReversionStrategy(VectorizedStrategy):
    """
    Close-price-only Bollinger Band mean reversion.

    Enter when price closes below lower BB with RSI confirmation.
    Exit when price returns to BB middle (natural mean reversion target).

    All entries/exits on close prices only.
    """
    name = "co_bb"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_entry': 35,            # RSI confirmation
        'bb_period': 20,
        'bb_std': 2.0,
        'profit_target_pct': 1.0,
        'stop_loss_pct': 0.6,
        'max_hold_bars': 24,
        'exit_at_bb_mid': True,     # Exit when close reaches BB middle
        'min_profit_for_bb_exit': 0.2,  # Min profit for BB mid exit
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 2,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']
        bb_lower = indicators['bb_lower']
        bb_middle = indicators['bb_middle']

        rsi_entry = self.params.get('rsi_entry', 35)
        profit_target = self.params.get('profit_target_pct', 1.0) / 100.0
        stop_loss = self.params.get('stop_loss_pct', 0.6) / 100.0
        max_hold = self.params.get('max_hold_bars', 24)
        exit_at_mid = self.params.get('exit_at_bb_mid', True)
        min_profit_bb = self.params.get('min_profit_for_bb_exit', 0.2) / 100.0
        cooldown = self.params.get('min_cooldown_bars', 2)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or bb_lower[i] is None or bb_middle[i] is None:
                i += 1
                continue
            if i - last_exit_bar < cooldown:
                i += 1
                continue

            # Entry: close below BB lower + RSI oversold
            if closes[i] <= bb_lower[i] and rsi[i] <= rsi_entry:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                target_price = entry_price * (1.0 + profit_target)
                stop_price = entry_price * (1.0 - stop_loss)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    # CLOSE-ONLY stop
                    if closes[j] <= stop_price:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'stop_loss'
                        break
                    # CLOSE-ONLY profit target
                    if closes[j] >= target_price:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'profit_target'
                        break
                    # BB middle reversion exit
                    if exit_at_mid and bb_middle[j] is not None:
                        if closes[j] >= bb_middle[j]:
                            gross_pnl = (closes[j] - entry_price) / entry_price
                            if gross_pnl >= min_profit_bb:
                                exit_bar = j
                                exit_price = closes[j]
                                exit_reason = 'bb_middle'
                                break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))
                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class RSIOnlyStrategy(VectorizedStrategy):
    """
    Pure RSI mean reversion with trailing stop and disaster stop.

    Key insight: at extreme oversold (RSI < 20-25), mean reversion is
    highly probable if you don't get stopped out prematurely. Fixed stops
    on close-price-only data gap through and destroy profitability.

    Instead of a tight fixed stop, uses:
    1. RSI normalization exit (primary exit — take the mean reversion)
    2. Trailing stop (lock in profits when trade goes well)
    3. Disaster stop (moderate close-based stop to prevent catastrophic losses)
    4. Time-decay exit: if RSI partially recovers, accept smaller gains/losses

    Entry: RSI crosses below oversold threshold.
    """
    name = "rsi_pure"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_entry': 25,
        'rsi_exit': 50,             # RSI normalization target
        'rsi_partial_exit': 40,     # Accept exit at partial recovery if in profit
        'disaster_stop_pct': 2.5,   # Close-based disaster stop (wide!)
        'max_hold_bars': 48,
        'trailing_activate_pct': 0.5,
        'trailing_stop_pct': 0.3,
        'min_net_profit_pct': 0.3,  # Min net profit % for RSI exits
        'bb_period': 20,
        'bb_std': 2.0,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 4,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']

        rsi_entry = self.params.get('rsi_entry', 25)
        rsi_exit = self.params.get('rsi_exit', 50)
        rsi_partial = self.params.get('rsi_partial_exit', 40)
        disaster_stop = self.params.get('disaster_stop_pct', 2.5) / 100.0
        max_hold = self.params.get('max_hold_bars', 48)
        trailing_activate = self.params.get('trailing_activate_pct', 0.5) / 100.0
        trailing_pct = self.params.get('trailing_stop_pct', 0.3) / 100.0
        cooldown = self.params.get('min_cooldown_bars', 4)
        min_net_profit = self.params.get('min_net_profit_pct', 0.3) / 100.0
        fee_pct = self.params.get('round_trip_fee_pct', 0.25) / 100.0

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or i - last_exit_bar < cooldown:
                i += 1
                continue

            if rsi[i] <= rsi_entry:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                peak_price = entry_price
                trailing_active = False
                disaster_price = entry_price * (1.0 - disaster_stop)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    current_price = closes[j]
                    current_pnl = (current_price - entry_price) / entry_price

                    # 1. Disaster stop (close-based, wide)
                    if current_price <= disaster_price:
                        exit_bar = j
                        exit_price = current_price
                        exit_reason = 'disaster_stop'
                        break

                    # Track peak for trailing
                    if current_price > peak_price:
                        peak_price = current_price

                    # 2. Trailing stop activation and check
                    gain_from_entry = (peak_price - entry_price) / entry_price
                    if gain_from_entry >= trailing_activate:
                        trailing_active = True
                    if trailing_active:
                        trail_price = peak_price * (1.0 - trailing_pct)
                        if current_price <= trail_price:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'trailing_stop'
                            break

                    # 3. RSI full recovery exit - requires min net profit
                    if rsi[j] is not None and rsi[j] >= rsi_exit:
                        net_pnl = current_pnl - fee_pct
                        if net_pnl >= min_net_profit:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'rsi_exit'
                            break

                    # 4. RSI partial recovery - also requires min net profit
                    if rsi[j] is not None and rsi[j] >= rsi_partial:
                        net_pnl = current_pnl - fee_pct
                        if net_pnl >= min_net_profit:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'rsi_partial'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct_final = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct_final

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct_final,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))
                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class StochRSIDivergenceStrategy(VectorizedStrategy):
    """
    Stochastic RSI crossover strategy.

    Entry: StochRSI %K crosses above %D from oversold zone.
    Exit: StochRSI %K crosses below %D from overbought zone, or max hold.
    No hard stop — momentum-based entries/exits.
    """
    name = "stoch_cross"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'stoch_oversold': 20,
        'stoch_overbought': 80,
        'max_hold_bars': 36,
        'bb_period': 20,
        'bb_std': 2.0,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 3,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        stoch_k = indicators['stoch_k']
        stoch_d = indicators['stoch_d']

        stoch_oversold = self.params.get('stoch_oversold', 20)
        stoch_overbought = self.params.get('stoch_overbought', 80)
        max_hold = self.params.get('max_hold_bars', 36)
        cooldown = self.params.get('min_cooldown_bars', 3)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if (stoch_k[i] is None or stoch_d[i] is None or
                stoch_k[i-1] is None or stoch_d[i-1] is None or
                i - last_exit_bar < cooldown):
                i += 1
                continue

            # Entry: bullish crossover from oversold zone
            bullish_cross = stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i]
            in_oversold = stoch_k[i] < stoch_oversold or stoch_k[i-1] < stoch_oversold

            if bullish_cross and in_oversold:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    if stoch_k[j] is None or stoch_d[j] is None:
                        continue

                    # Exit: bearish crossover from overbought
                    if (stoch_k[j-1] is not None and stoch_d[j-1] is not None):
                        bearish_cross = stoch_k[j-1] >= stoch_d[j-1] and stoch_k[j] < stoch_d[j]
                        if bearish_cross and stoch_k[j] > stoch_overbought * 0.7:
                            exit_bar = j
                            exit_price = closes[j]
                            exit_reason = 'stoch_exit'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))
                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class EMARSIComboStrategy(VectorizedStrategy):
    """
    EMA trend + RSI dip with relaxed conditions.

    Entry: Price above slow EMA (uptrend) AND RSI dips below threshold.
    No requirement to be below fast EMA (which was too restrictive).
    Exit: RSI recovery or max hold. No hard stop.
    """
    name = "ema_rsi"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_entry': 35,
        'rsi_exit': 55,
        'trend_ema': 50,
        'max_hold_bars': 36,
        'bb_period': 20,
        'bb_std': 2.0,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 4,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']
        trend_key = f"ema_{self.params.get('trend_ema', 50)}"
        trend_ema = indicators.get(trend_key, [None] * n)

        rsi_entry = self.params.get('rsi_entry', 35)
        rsi_exit = self.params.get('rsi_exit', 55)
        max_hold = self.params.get('max_hold_bars', 36)
        cooldown = self.params.get('min_cooldown_bars', 4)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or trend_ema[i] is None or i - last_exit_bar < cooldown:
                i += 1
                continue

            # Entry: uptrend + RSI dip
            if closes[i] > trend_ema[i] and rsi[i] <= rsi_entry:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    # RSI recovery exit
                    if rsi[j] is not None and rsi[j] >= rsi_exit:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'rsi_exit'
                        break
                    # Trend break emergency exit
                    if trend_ema[j] is not None and closes[j] < trend_ema[j] * 0.99:
                        exit_bar = j
                        exit_price = closes[j]
                        exit_reason = 'trend_break'
                        break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))
                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class RSIBidirectionalStrategy(VectorizedStrategy):
    """
    Bidirectional RSI mean reversion: LONG when oversold, SHORT when overbought.

    This addresses the key weakness of long-only strategies in bear markets.
    When BTC/ETH/SOL drop 18%+, overbought RSI entries (short) can profit
    from the mean reversion back down.

    Long entry: RSI <= rsi_entry_long (deep oversold)
    Short entry: RSI >= rsi_entry_short (deep overbought)

    Same exit logic as RSIOnlyStrategy but mirrored for shorts.
    Close-price-only for all decisions.
    """
    name = "rsi_bidir"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        # Long side
        'rsi_entry_long': 25,
        'rsi_exit_long': 50,
        'rsi_partial_exit_long': 40,
        # Short side
        'rsi_entry_short': 75,
        'rsi_exit_short': 50,
        'rsi_partial_exit_short': 60,
        # Shared
        'disaster_stop_pct': 5.0,
        'max_hold_bars': 24,
        'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2,
        'bb_period': 20,
        'bb_std': 2.0,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 3,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']

        # Long params
        rsi_entry_long = self.params.get('rsi_entry_long', 25)
        rsi_exit_long = self.params.get('rsi_exit_long', 50)
        rsi_partial_long = self.params.get('rsi_partial_exit_long', 40)

        # Short params
        rsi_entry_short = self.params.get('rsi_entry_short', 75)
        rsi_exit_short = self.params.get('rsi_exit_short', 50)
        rsi_partial_short = self.params.get('rsi_partial_exit_short', 60)

        # Shared
        disaster_stop = self.params.get('disaster_stop_pct', 5.0) / 100.0
        max_hold = self.params.get('max_hold_bars', 24)
        trailing_activate = self.params.get('trailing_activate_pct', 0.3) / 100.0
        trailing_pct = self.params.get('trailing_stop_pct', 0.2) / 100.0
        cooldown = self.params.get('min_cooldown_bars', 3)

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or i - last_exit_bar < cooldown:
                i += 1
                continue

            direction = None
            if rsi[i] <= rsi_entry_long:
                direction = 'long'
            elif rsi[i] >= rsi_entry_short:
                direction = 'short'

            if direction is not None:
                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                peak_price = entry_price  # Best price in our favor
                trailing_active = False

                if direction == 'long':
                    disaster_price = entry_price * (1.0 - disaster_stop)
                else:
                    disaster_price = entry_price * (1.0 + disaster_stop)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    current_price = closes[j]

                    if direction == 'long':
                        current_pnl = (current_price - entry_price) / entry_price

                        # 1. Disaster stop
                        if current_price <= disaster_price:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'disaster_stop'
                            break

                        # Track peak
                        if current_price > peak_price:
                            peak_price = current_price

                        # 2. Trailing stop
                        gain_from_entry = (peak_price - entry_price) / entry_price
                        if gain_from_entry >= trailing_activate:
                            trailing_active = True
                        if trailing_active:
                            trail_price = peak_price * (1.0 - trailing_pct)
                            if current_price <= trail_price:
                                exit_bar = j
                                exit_price = current_price
                                exit_reason = 'trailing_stop'
                                break

                        # 3. RSI full recovery
                        if rsi[j] is not None and rsi[j] >= rsi_exit_long:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'rsi_exit'
                            break

                        # 4. RSI partial + profitable
                        if rsi[j] is not None and rsi[j] >= rsi_partial_long and current_pnl > 0:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'rsi_partial'
                            break

                    else:  # short
                        current_pnl = (entry_price - current_price) / entry_price

                        # 1. Disaster stop (price goes up too much)
                        if current_price >= disaster_price:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'disaster_stop'
                            break

                        # Track trough (best price for short = lowest)
                        if current_price < peak_price:
                            peak_price = current_price

                        # 2. Trailing stop (price bounces up from trough)
                        gain_from_entry = (entry_price - peak_price) / entry_price
                        if gain_from_entry >= trailing_activate:
                            trailing_active = True
                        if trailing_active:
                            trail_price = peak_price * (1.0 + trailing_pct)
                            if current_price >= trail_price:
                                exit_bar = j
                                exit_price = current_price
                                exit_reason = 'trailing_stop'
                                break

                        # 3. RSI full recovery (drops back to neutral)
                        if rsi[j] is not None and rsi[j] <= rsi_exit_short:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'rsi_exit'
                            break

                        # 4. RSI partial + profitable
                        if rsi[j] is not None and rsi[j] <= rsi_partial_short and current_pnl > 0:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'rsi_partial'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                if direction == 'long':
                    gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                else:
                    gross_pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0

                fee_pct = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction=direction, exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))
                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


class RSIRegimeStrategy(VectorizedStrategy):
    """
    RSI mean reversion with regime filter to avoid entering during freefall.

    Same core as RSIOnlyStrategy, but adds:
    1. EMA proximity filter: only enter long when price is within X% of EMA
       (prevents buying the dip in a crashing market)
    2. Slope filter: EMA must not be declining steeply

    This should help LTC, LINK, ADA, and possibly others pass walk-forward.
    """
    name = "rsi_regime"

    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_entry': 20,
        'rsi_exit': 48,
        'rsi_partial_exit': 35,
        'disaster_stop_pct': 5.0,
        'max_hold_bars': 24,
        'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2,
        'min_net_profit_pct': 0.3,   # Min net profit % for RSI exits (matches ~$1.50 on $500 position after fees)
        # Regime filter params
        'regime_ema': 50,            # EMA to use for regime detection
        'max_below_ema_pct': 5.0,    # Max % price can be below EMA to still enter
        'ema_slope_bars': 10,        # Number of bars to measure EMA slope
        'max_ema_decline_pct': 3.0,  # Max % EMA can decline over slope_bars
        'bb_period': 20,
        'bb_std': 2.0,
        'ema_periods': [9, 21, 50],
        'sma_periods': [50, 200],
        'atr_period': 14,
        'volume_sma_period': 20,
        'roc_period': 12,
        'min_cooldown_bars': 3,
    }

    def generate_signals(self, timestamps, opens, highs, lows, closes, volumes, indicators):
        trades = []
        n = len(closes)
        warmup = 60

        rsi = indicators['rsi']
        regime_ema_period = self.params.get('regime_ema', 50)
        regime_ema_key = f"ema_{regime_ema_period}"
        regime_ema = indicators.get(regime_ema_key, [None] * n)

        rsi_entry = self.params.get('rsi_entry', 20)
        rsi_exit = self.params.get('rsi_exit', 48)
        rsi_partial = self.params.get('rsi_partial_exit', 35)
        disaster_stop = self.params.get('disaster_stop_pct', 5.0) / 100.0
        max_hold = self.params.get('max_hold_bars', 24)
        trailing_activate = self.params.get('trailing_activate_pct', 0.3) / 100.0
        trailing_pct = self.params.get('trailing_stop_pct', 0.2) / 100.0
        cooldown = self.params.get('min_cooldown_bars', 3)
        min_net_profit = self.params.get('min_net_profit_pct', 0.3) / 100.0
        fee_pct = self.params.get('round_trip_fee_pct', 0.25) / 100.0
        max_below_ema = self.params.get('max_below_ema_pct', 5.0) / 100.0
        slope_bars = self.params.get('ema_slope_bars', 10)
        max_decline = self.params.get('max_ema_decline_pct', 3.0) / 100.0

        i = warmup
        last_exit_bar = 0

        while i < n:
            if rsi[i] is None or i - last_exit_bar < cooldown:
                i += 1
                continue

            if rsi[i] <= rsi_entry:
                # Regime filter: check if we're not in freefall
                regime_ok = True

                if regime_ema[i] is not None:
                    # Price proximity: don't buy if price is way below EMA
                    distance_below = (regime_ema[i] - closes[i]) / regime_ema[i]
                    if distance_below > max_below_ema:
                        regime_ok = False

                    # EMA slope: don't buy if EMA is declining steeply
                    if regime_ok and i >= slope_bars and regime_ema[i - slope_bars] is not None:
                        ema_change = (regime_ema[i] - regime_ema[i - slope_bars]) / regime_ema[i - slope_bars]
                        if ema_change < -max_decline:
                            regime_ok = False

                if not regime_ok:
                    i += 1
                    continue

                entry_price = closes[i]
                entry_bar = i
                entry_time = timestamps[i]
                peak_price = entry_price
                trailing_active = False
                disaster_price = entry_price * (1.0 - disaster_stop)

                exit_bar = None
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + max_hold + 1, n)):
                    current_price = closes[j]
                    current_pnl = (current_price - entry_price) / entry_price

                    # 1. Disaster stop
                    if current_price <= disaster_price:
                        exit_bar = j
                        exit_price = current_price
                        exit_reason = 'disaster_stop'
                        break

                    # Track peak
                    if current_price > peak_price:
                        peak_price = current_price

                    # 2. Trailing stop
                    gain_from_entry = (peak_price - entry_price) / entry_price
                    if gain_from_entry >= trailing_activate:
                        trailing_active = True
                    if trailing_active:
                        trail_price = peak_price * (1.0 - trailing_pct)
                        if current_price <= trail_price:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'trailing_stop'
                            break

                    # 3. RSI full recovery - requires min net profit
                    if rsi[j] is not None and rsi[j] >= rsi_exit:
                        net_pnl = current_pnl - fee_pct
                        if net_pnl >= min_net_profit:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'rsi_exit'
                            break

                    # 4. RSI partial - also requires min net profit
                    if rsi[j] is not None and rsi[j] >= rsi_partial:
                        net_pnl = current_pnl - fee_pct
                        if net_pnl >= min_net_profit:
                            exit_bar = j
                            exit_price = current_price
                            exit_reason = 'rsi_partial'
                            break

                if exit_bar is None:
                    exit_bar = min(i + max_hold, n - 1)
                    exit_price = closes[exit_bar]
                    exit_reason = 'max_hold'

                gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                fee_pct_final = self.params.get('round_trip_fee_pct', 0.25)
                net_pnl_pct = gross_pnl_pct - fee_pct_final

                trades.append(Trade(
                    symbol='', entry_time=entry_time, entry_price=entry_price,
                    exit_time=timestamps[exit_bar], exit_price=exit_price,
                    direction='long', exit_reason=exit_reason,
                    gross_pnl_pct=gross_pnl_pct, fee_pct=fee_pct_final,
                    net_pnl_pct=net_pnl_pct, net_pnl_usd=0,
                    hold_bars=exit_bar - entry_bar,
                ))
                last_exit_bar = exit_bar
                i = exit_bar + 1
            else:
                i += 1

        return trades


# ==============================================================================
# BACKTESTING ENGINE
# ==============================================================================

STRATEGY_MAP = {
    'rsi_mr': RSIMeanReversionStrategy,
    'bb_bounce': BBBounceStrategy,
    'ema_momentum': EMAMomentumStrategy,
    'adaptive_rsi': AdaptiveRSIStrategy,
    'mtf_mr': MTFMeanReversionStrategy,
    'dual_bb': DualBBStrategy,
    'co_revert': CloseOnlyReversionStrategy,
    'co_trend_dip': CloseOnlyTrendDipStrategy,
    'co_bb': CloseOnlyBBReversionStrategy,
    'rsi_pure': RSIOnlyStrategy,
    'stoch_cross': StochRSIDivergenceStrategy,
    'ema_rsi': EMARSIComboStrategy,
    'rsi_bidir': RSIBidirectionalStrategy,
    'rsi_regime': RSIRegimeStrategy,
}


def run_backtest(symbol: str, strategy_cls, params: dict,
                 fee_tier: str = 'adv2', capital: float = 4500.0,
                 start_ts: float = None, end_ts: float = None,
                 timeframe_minutes: int = 5) -> BacktestResult:
    """Run a single backtest. Returns BacktestResult."""

    # Load data
    timestamps, opens, highs, lows, closes, volumes = load_symbol_data(symbol)
    if not closes:
        return BacktestResult(strategy_name=strategy_cls.name if hasattr(strategy_cls, 'name') else 'unknown',
                              params=params, symbol=symbol, period='no data')

    # Aggregate if needed
    if timeframe_minutes > 5:
        timestamps, opens, highs, lows, closes, volumes = aggregate_to_timeframe(
            timestamps, opens, highs, lows, closes, volumes, timeframe_minutes)

    # Filter date range
    if start_ts:
        mask = [i for i in range(len(timestamps)) if timestamps[i] >= start_ts]
        if mask:
            s = mask[0]
            timestamps, opens, highs, lows, closes, volumes = \
                timestamps[s:], opens[s:], highs[s:], lows[s:], closes[s:], volumes[s:]

    if end_ts:
        mask = [i for i in range(len(timestamps)) if timestamps[i] <= end_ts]
        if mask:
            e = mask[-1] + 1
            timestamps, opens, highs, lows, closes, volumes = \
                timestamps[:e], opens[:e], highs[:e], lows[:e], closes[:e], volumes[:e]

    if len(closes) < 100:
        return BacktestResult(strategy_name='', params=params, symbol=symbol, period='insufficient data')

    # Fee setup
    tier = FEE_TIERS.get(fee_tier, FEE_TIERS['adv2'])
    round_trip_fee_pct = (tier['maker'] + tier['maker']) * 100.0  # Both sides maker
    params_with_fees = dict(params)
    params_with_fees['round_trip_fee_pct'] = round_trip_fee_pct

    # Pre-compute indicators
    indicators = precompute_indicators(closes, highs, lows, volumes, params_with_fees)

    # For MTF strategy, compute HTF trend
    if strategy_cls == MTFMeanReversionStrategy or (hasattr(strategy_cls, 'name') and strategy_cls.name == 'mtf_mr'):
        htf_minutes = params.get('htf_minutes', 60)
        if timeframe_minutes < htf_minutes:
            htf_ts, htf_o, htf_h, htf_l, htf_c, htf_v = load_symbol_data(symbol)
            htf_ts, htf_o, htf_h, htf_l, htf_c, htf_v = aggregate_to_timeframe(
                htf_ts, htf_o, htf_h, htf_l, htf_c, htf_v, htf_minutes)
            htf_ema_period = params.get('htf_ema_period', 21)
            htf_ema = compute_ema_array(htf_c, htf_ema_period)

            # Create trend array mapped to LTF bars
            bars_per_htf = htf_minutes // timeframe_minutes
            htf_trend = [None] * len(closes)
            for hi in range(len(htf_c)):
                if htf_ema[hi] is None:
                    continue
                bullish = htf_c[hi] > htf_ema[hi]
                ltf_start = hi * bars_per_htf
                ltf_end = min(ltf_start + bars_per_htf, len(closes))
                for li in range(ltf_start, ltf_end):
                    if li < len(htf_trend):
                        htf_trend[li] = bullish

            params_with_fees['_htf_trend_array'] = htf_trend

    # Instantiate and run strategy
    strategy = strategy_cls(params_with_fees)
    trades = strategy.generate_signals(timestamps, opens, highs, lows, closes, volumes, indicators)

    # Set symbol on trades and compute USD P&L
    for t in trades:
        t.symbol = symbol
        t.net_pnl_usd = (t.net_pnl_pct / 100.0) * capital

    # Compute result metrics
    result = compute_metrics(trades, strategy_cls.name if hasattr(strategy_cls, 'name') else strategy_cls.__name__,
                             params, symbol, timestamps, capital, timeframe_minutes, fee_tier)
    return result


def compute_metrics(trades: List[Trade], strategy_name: str, params: dict,
                    symbol: str, timestamps: list, capital: float,
                    timeframe_minutes: int, fee_tier: str) -> BacktestResult:
    """Compute comprehensive metrics from trade list."""

    if not timestamps:
        return BacktestResult(strategy_name=strategy_name, params=params, symbol=symbol, period='')

    first_dt = datetime.fromtimestamp(timestamps[0], tz=timezone.utc)
    last_dt = datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)
    period_str = f"{first_dt.strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d')}"
    total_days = (timestamps[-1] - timestamps[0]) / 86400.0

    result = BacktestResult(
        strategy_name=strategy_name,
        params=params,
        symbol=symbol,
        period=period_str,
        trades=trades,
    )

    if not trades:
        return result

    result.total_trades = len(trades)

    wins = [t for t in trades if t.net_pnl_pct > 0]
    losses = [t for t in trades if t.net_pnl_pct <= 0]

    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = len(wins) / len(trades) * 100.0

    result.gross_profit_pct = sum(t.gross_pnl_pct for t in trades)
    result.total_fees_pct = sum(t.fee_pct for t in trades)
    result.net_profit_pct = sum(t.net_pnl_pct for t in trades)

    # Tax: on NET profit (gains and losses offset within tax year)
    result.net_profit_usd = sum(t.net_pnl_usd for t in trades)
    net_taxable = max(0, result.net_profit_pct)
    tax_owed = net_taxable * TAX_RATE
    result.net_profit_after_tax_pct = result.net_profit_pct - tax_owed

    net_taxable_usd = max(0, result.net_profit_usd)
    result.net_profit_after_tax_usd = result.net_profit_usd - (net_taxable_usd * TAX_RATE)

    result.avg_win_pct = sum(t.net_pnl_pct for t in wins) / len(wins) if wins else 0
    result.avg_loss_pct = sum(t.net_pnl_pct for t in losses) / len(losses) if losses else 0

    total_win_pct = sum(t.net_pnl_pct for t in wins) if wins else 0
    total_loss_pct = abs(sum(t.net_pnl_pct for t in losses)) if losses else 0
    result.profit_factor = total_win_pct / total_loss_pct if total_loss_pct > 0 else float('inf')

    result.avg_hold_bars = sum(t.hold_bars for t in trades) / len(trades)
    result.avg_hold_hours = result.avg_hold_bars * timeframe_minutes / 60.0

    if total_days > 0:
        result.trades_per_day = len(trades) / total_days

    # Max drawdown (on equity curve)
    equity = capital
    peak = capital
    max_dd = 0
    for t in trades:
        equity += t.net_pnl_usd
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
    result.max_drawdown_pct = max_dd

    # Sharpe ratio (annualized from per-trade returns)
    if len(trades) > 1:
        returns = [t.net_pnl_pct for t in trades]
        avg_ret = statistics.mean(returns)
        std_ret = statistics.stdev(returns)
        if std_ret > 0:
            # Annualize: multiply by sqrt(trades_per_year)
            trades_per_year = result.trades_per_day * 365.0 if result.trades_per_day > 0 else 1
            result.sharpe_ratio = (avg_ret / std_ret) * math.sqrt(trades_per_year)
        else:
            result.sharpe_ratio = 0

    return result


# ==============================================================================
# WALK-FORWARD VALIDATION
# ==============================================================================

def walk_forward_test(symbol: str, strategy_cls, params: dict,
                      fee_tier: str = 'adv2', train_pct: float = 0.7,
                      timeframe_minutes: int = 5) -> Tuple[BacktestResult, BacktestResult]:
    """Split data into training and validation sets."""
    timestamps, _, _, _, _, _ = load_symbol_data(symbol)
    if not timestamps:
        empty = BacktestResult(strategy_name='', params=params, symbol=symbol, period='')
        return empty, empty

    split_idx = int(len(timestamps) * train_pct)
    split_ts = timestamps[split_idx]

    train_result = run_backtest(symbol, strategy_cls, params, fee_tier,
                                end_ts=split_ts, timeframe_minutes=timeframe_minutes)
    val_result = run_backtest(symbol, strategy_cls, params, fee_tier,
                              start_ts=split_ts, timeframe_minutes=timeframe_minutes)

    return train_result, val_result


# ==============================================================================
# GRID SEARCH OPTIMIZATION
# ==============================================================================

def grid_search(symbol: str, strategy_cls, param_grid: dict,
                fee_tier: str = 'adv2', timeframe_minutes: int = 5,
                min_trades: int = 20) -> List[Tuple[dict, BacktestResult]]:
    """Run grid search over parameter combinations. Returns sorted results."""

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"  Grid search: {len(combinations)} combinations for {symbol}...")

    results = []
    for combo in combinations:
        params = dict(zip(param_names, combo))
        # Merge with defaults
        defaults = strategy_cls.DEFAULT_PARAMS.copy() if hasattr(strategy_cls, 'DEFAULT_PARAMS') else {}
        merged = {**defaults, **params}

        result = run_backtest(symbol, strategy_cls, merged, fee_tier,
                              timeframe_minutes=timeframe_minutes)

        if result.total_trades >= min_trades:
            results.append((merged, result))

    # Sort by net profit after tax
    results.sort(key=lambda x: x[1].net_profit_after_tax_pct, reverse=True)
    return results


# ==============================================================================
# DISPLAY
# ==============================================================================

def print_result(result: BacktestResult, show_trades: bool = False, indent: str = ''):
    """Print formatted backtest result."""
    p = indent
    print(f"{p}{'='*75}")
    print(f"{p}Strategy: {result.strategy_name} | Symbol: {result.symbol}")
    print(f"{p}Period: {result.period}")
    print(f"{p}{'-'*75}")
    print(f"{p}Trades: {result.total_trades} | Wins: {result.winning_trades} | Losses: {result.losing_trades}")
    print(f"{p}Win Rate: {result.win_rate:.1f}% | Profit Factor: {result.profit_factor:.2f}")
    print(f"{p}Trades/Day: {result.trades_per_day:.2f} | Avg Hold: {result.avg_hold_hours:.1f}h ({result.avg_hold_bars:.0f} bars)")
    print(f"{p}{'-'*75}")
    print(f"{p}Gross Profit: {result.gross_profit_pct:+.2f}%")
    print(f"{p}Total Fees:   {result.total_fees_pct:.2f}%")
    print(f"{p}Net Profit:   {result.net_profit_pct:+.2f}% (${result.net_profit_usd:+.2f})")
    print(f"{p}After Tax:    {result.net_profit_after_tax_pct:+.2f}% (${result.net_profit_after_tax_usd:+.2f})")
    print(f"{p}{'-'*75}")
    print(f"{p}Avg Win: {result.avg_win_pct:+.3f}% | Avg Loss: {result.avg_loss_pct:+.3f}%")
    print(f"{p}Max Drawdown: {result.max_drawdown_pct:.2f}% | Sharpe: {result.sharpe_ratio:.2f}")
    print(f"{p}{'='*75}")

    if show_trades and result.trades:
        print(f"\n{p}Last 30 Trades:")
        print(f"{p}{'Date':<18} {'Entry':>10} {'Exit':>10} {'Gross%':>8} {'Net%':>8} {'$P&L':>9} {'Bars':>5} {'Reason':<15}")
        print(f"{p}{'-'*90}")
        for t in result.trades[-30:]:
            dt = datetime.fromtimestamp(t.entry_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            print(f"{p}{dt:<18} {t.entry_price:>10.2f} {t.exit_price:>10.2f} {t.gross_pnl_pct:>+7.3f}% {t.net_pnl_pct:>+7.3f}% ${t.net_pnl_usd:>+8.2f} {t.hold_bars:>5} {t.exit_reason:<15}")

    # Exit reason breakdown
    if result.trades:
        reasons = {}
        for t in result.trades:
            r = t.exit_reason
            if r not in reasons:
                reasons[r] = {'count': 0, 'net_pct': 0.0}
            reasons[r]['count'] += 1
            reasons[r]['net_pct'] += t.net_pnl_pct

        print(f"\n{p}Exit Reason Breakdown:")
        print(f"{p}{'Reason':<20} {'Count':>6} {'Avg Net%':>10} {'Total Net%':>12}")
        print(f"{p}{'-'*55}")
        for reason, data in sorted(reasons.items(), key=lambda x: -x[1]['count']):
            avg = data['net_pct'] / data['count'] if data['count'] > 0 else 0
            print(f"{p}{reason:<20} {data['count']:>6} {avg:>+9.3f}% {data['net_pct']:>+11.3f}%")


def print_fee_tier_projection(result: BacktestResult, indent: str = ''):
    """Show how strategy would perform at different fee tiers."""
    p = indent
    if result.total_trades == 0:
        return
    print(f"\n{p}Fee Tier Projections:")
    print(f"{p}{'Tier':<15} {'RT Fee%':>8} {'Net%':>10} {'After Tax%':>12} {'Net USD':>12}")
    print(f"{p}{'-'*60}")
    for tier_id, tier in FEE_TIERS.items():
        rt_fee = (tier['maker'] + tier['maker']) * 100.0  # both maker
        total_fees = rt_fee * result.total_trades
        net_pct = result.gross_profit_pct - total_fees
        tax = sum(1 for t in result.trades if t.gross_pnl_pct - rt_fee > 0) * \
              (sum(t.gross_pnl_pct - rt_fee for t in result.trades if t.gross_pnl_pct - rt_fee > 0) /
               max(1, sum(1 for t in result.trades if t.gross_pnl_pct - rt_fee > 0))) * TAX_RATE if result.trades else 0
        # Simpler: approximate tax
        taxable = max(0, sum(max(0, t.gross_pnl_pct - rt_fee) for t in result.trades))
        after_tax = net_pct - taxable * TAX_RATE
        net_usd = (net_pct / 100.0) * 4500.0
        print(f"{p}{tier_id + ' (' + tier['name'] + ')':<15} {rt_fee:>7.3f}% {net_pct:>+9.2f}% {after_tax:>+11.2f}% ${net_usd:>+11.2f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='High-Performance Crypto Backtester')
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--strategy', type=str, default='all',
                        choices=list(STRATEGY_MAP.keys()) + ['all'])
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--walk-forward', action='store_true')
    parser.add_argument('--fee-tier', type=str, default='adv2', choices=['intro2', 'adv1', 'adv2', 'adv3'])
    parser.add_argument('--timeframe', type=int, default=5, help='Timeframe in minutes')
    parser.add_argument('--show-trades', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    tier = FEE_TIERS[args.fee_tier]
    rt_fee = (tier['maker'] + tier['maker']) * 100.0
    print("=" * 75)
    print("HIGH-PERFORMANCE CRYPTO SCALPING BACKTESTER")
    print("=" * 75)
    print(f"Fee Tier: {args.fee_tier} ({tier['name']})")
    print(f"Maker Fee: {tier['maker']*100:.3f}% | Round-trip (maker/maker): {rt_fee:.3f}%")
    print(f"Tax Rate: {TAX_RATE*100:.0f}%")
    print(f"Timeframe: {args.timeframe}min")
    print(f"Symbols: {', '.join(symbols)}")

    if args.optimize:
        run_optimization(symbols, args)
    elif args.walk_forward:
        run_walk_forward(symbols, args)
    else:
        run_standard_backtest(symbols, args)


def run_standard_backtest(symbols, args):
    """Run standard backtest across symbols and strategies."""
    strategies = list(STRATEGY_MAP.keys()) if args.strategy == 'all' else [args.strategy]

    summary = []

    for strat_name in strategies:
        strategy_cls = STRATEGY_MAP[strat_name]
        defaults = strategy_cls.DEFAULT_PARAMS.copy() if hasattr(strategy_cls, 'DEFAULT_PARAMS') else {}

        for symbol in symbols:
            result = run_backtest(symbol, strategy_cls, defaults, args.fee_tier,
                                  timeframe_minutes=args.timeframe)
            print_result(result, show_trades=args.show_trades)
            print_fee_tier_projection(result)
            summary.append(result)

    # Summary table
    if len(summary) > 1:
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"{'Strategy':<16} {'Symbol':<10} {'Trades':>7} {'Win%':>7} {'Net%':>9} {'AfterTax%':>10} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'T/Day':>6}")
        print("-" * 100)
        for r in sorted(summary, key=lambda x: x.net_profit_after_tax_pct, reverse=True):
            print(f"{r.strategy_name:<16} {r.symbol:<10} {r.total_trades:>7} {r.win_rate:>6.1f}% {r.net_profit_pct:>+8.2f}% {r.net_profit_after_tax_pct:>+9.2f}% {r.profit_factor:>5.2f} {r.sharpe_ratio:>6.2f} {r.max_drawdown_pct:>6.2f}% {r.trades_per_day:>5.2f}")

        # Aggregate by strategy
        print("\n" + "=" * 100)
        print("AGGREGATE BY STRATEGY (across all symbols)")
        print("=" * 100)
        strat_agg = {}
        for r in summary:
            if r.strategy_name not in strat_agg:
                strat_agg[r.strategy_name] = {
                    'results': [], 'total_trades': 0, 'net_pct': 0, 'after_tax_pct': 0,
                    'profitable_symbols': 0, 'total_symbols': 0
                }
            strat_agg[r.strategy_name]['results'].append(r)
            strat_agg[r.strategy_name]['total_trades'] += r.total_trades
            strat_agg[r.strategy_name]['net_pct'] += r.net_profit_pct
            strat_agg[r.strategy_name]['after_tax_pct'] += r.net_profit_after_tax_pct
            strat_agg[r.strategy_name]['total_symbols'] += 1
            if r.net_profit_after_tax_pct > 0:
                strat_agg[r.strategy_name]['profitable_symbols'] += 1

        print(f"{'Strategy':<16} {'Symbols':>10} {'Profitable':>12} {'Total Trades':>13} {'Total Net%':>11} {'AfterTax%':>11}")
        print("-" * 80)
        for name, data in sorted(strat_agg.items(), key=lambda x: x[1]['after_tax_pct'], reverse=True):
            print(f"{name:<16} {data['total_symbols']:>10} {data['profitable_symbols']:>12} {data['total_trades']:>13} {data['net_pct']:>+10.2f}% {data['after_tax_pct']:>+10.2f}%")


def run_optimization(symbols, args):
    """Run grid search optimization."""
    strategies_to_optimize = list(STRATEGY_MAP.keys()) if args.strategy == 'all' else [args.strategy]

    # Define grids for each strategy
    grids = {
        'rsi_mr': {
            'rsi_period': [7, 14],
            'rsi_oversold': [25, 28, 30, 33, 35],
            'rsi_exit': [45, 50, 55],
            'profit_target_pct': [0.5, 0.7, 0.9, 1.2],
            'stop_loss_pct': [0.3, 0.5, 0.7],
            'max_hold_bars': [24, 36, 48],
            'min_atr_pct': [0.1, 0.15, 0.2],
        },
        'bb_bounce': {
            'rsi_oversold': [30, 35, 40],
            'bb_std': [1.8, 2.0, 2.2],
            'profit_target_pct': [0.5, 0.7, 0.9],
            'stop_loss_pct': [0.3, 0.5, 0.7],
            'max_hold_bars': [18, 24, 36],
            'exit_at_middle_bb': [True, False],
        },
        'ema_momentum': {
            'fast_ema': [5, 9],
            'slow_ema': [15, 21],
            'rsi_min_entry': [35, 40, 45],
            'rsi_max_entry': [60, 65, 70],
            'profit_target_pct': [0.7, 1.0, 1.5],
            'stop_loss_pct': [0.4, 0.5, 0.7],
        },
        'adaptive_rsi': {
            'rsi_oversold': [25, 28, 30, 33],
            'base_profit_target_pct': [0.5, 0.6, 0.8],
            'atr_target_multiplier': [1.0, 1.5, 2.0],
            'stop_loss_atr_mult': [0.8, 1.0, 1.5],
            'rsi_exit_threshold': [45, 50, 55],
            'use_stoch_cross': [True, False],
        },
        'mtf_mr': {
            'rsi_oversold': [25, 28, 30, 33],
            'htf_minutes': [30, 60],
            'htf_ema_period': [14, 21],
            'profit_target_pct': [0.6, 0.8, 1.0],
            'stop_loss_pct': [0.4, 0.5, 0.7],
            'max_hold_bars': [24, 36],
        },
        'dual_bb': {
            'rsi_oversold': [30, 35, 40],
            'bb_std': [1.8, 2.0, 2.5],
            'bb_tight_std': [1.0, 1.5],
            'profit_target_pct': [0.5, 0.7, 1.0],
            'stop_loss_pct': [0.3, 0.5],
            'max_hold_bars': [24, 36],
        },
        'co_revert': {
            'rsi_entry': [20, 23, 25, 28, 30],
            'rsi_exit': [45, 50, 55],
            'use_bb_filter': [True, False],
            'profit_target_pct': [0.6, 0.8, 1.0, 1.3],
            'stop_loss_pct': [0.5, 0.7, 1.0],
            'max_hold_bars': [16, 24, 36],
        },
        'co_trend_dip': {
            'rsi_entry': [30, 35, 38, 40],
            'rsi_exit': [50, 55, 60],
            'trend_ema': [21, 50],
            'pullback_ema': [9, 21],
            'profit_target_pct': [0.6, 0.8, 1.0, 1.3],
            'stop_loss_pct': [0.4, 0.6, 0.8],
            'max_hold_bars': [16, 24, 36],
        },
        'co_bb': {
            'rsi_entry': [30, 35, 40],
            'bb_std': [1.8, 2.0, 2.2, 2.5],
            'profit_target_pct': [0.6, 0.8, 1.0, 1.3],
            'stop_loss_pct': [0.4, 0.6, 0.8],
            'max_hold_bars': [16, 24, 36],
            'exit_at_bb_mid': [True, False],
        },
        'rsi_pure': {
            'rsi_entry': [20, 23, 25, 28, 30],
            'rsi_exit': [45, 50, 55],
            'rsi_partial_exit': [35, 40, 45],
            'disaster_stop_pct': [1.5, 2.0, 2.5, 3.5],
            'max_hold_bars': [24, 36, 48],
            'trailing_activate_pct': [0.3, 0.5, 0.8],
            'trailing_stop_pct': [0.2, 0.3, 0.5],
        },
        'stoch_cross': {
            'stoch_oversold': [15, 20, 25],
            'stoch_overbought': [70, 80],
            'max_hold_bars': [24, 36, 48],
            'min_cooldown_bars': [2, 4],
        },
        'ema_rsi': {
            'rsi_entry': [30, 33, 35, 38, 40],
            'rsi_exit': [48, 50, 55, 60],
            'trend_ema': [21, 50],
            'max_hold_bars': [24, 36, 48],
            'min_cooldown_bars': [3, 6],
        },
        'rsi_bidir': {
            'rsi_entry_long': [20, 23, 25],
            'rsi_entry_short': [75, 77, 80],
            'rsi_exit_long': [45, 48, 50],
            'rsi_exit_short': [50, 52, 55],
            'rsi_partial_exit_long': [35, 40],
            'rsi_partial_exit_short': [60, 65],
            'disaster_stop_pct': [3.0, 5.0],
            'max_hold_bars': [24, 36],
        },
        'rsi_regime': {
            'rsi_entry': [18, 20, 23, 25],
            'rsi_exit': [45, 48, 50],
            'rsi_partial_exit': [33, 35, 40],
            'disaster_stop_pct': [3.0, 5.0, 7.0],
            'max_hold_bars': [18, 24, 36],
            'max_below_ema_pct': [3.0, 5.0, 8.0],
            'max_ema_decline_pct': [1.5, 3.0, 5.0],
        },
    }

    all_best = []

    for strat_name in strategies_to_optimize:
        if strat_name not in grids:
            continue
        strategy_cls = STRATEGY_MAP[strat_name]
        grid = grids[strat_name]
        total_combos = 1
        for v in grid.values():
            total_combos *= len(v)

        print(f"\n{'='*75}")
        print(f"OPTIMIZING: {strat_name} ({total_combos} combinations)")
        print(f"{'='*75}")

        # Test on each symbol
        for symbol in symbols:
            results = grid_search(symbol, strategy_cls, grid, args.fee_tier,
                                  args.timeframe, min_trades=10)

            profitable = [(p, r) for p, r in results if r.net_profit_after_tax_pct > 0]

            print(f"\n  {symbol}: {len(profitable)}/{len(results)} profitable combos")

            if profitable:
                best_params, best_result = profitable[0]
                print(f"  Best: Net after tax: {best_result.net_profit_after_tax_pct:+.2f}% | "
                      f"Win: {best_result.win_rate:.1f}% | PF: {best_result.profit_factor:.2f} | "
                      f"Trades: {best_result.total_trades} | Sharpe: {best_result.sharpe_ratio:.2f}")
                # Print key params
                key_params = {k: v for k, v in best_params.items()
                              if k in grid and not k.startswith('_')}
                print(f"  Params: {key_params}")
                all_best.append((strat_name, symbol, best_params, best_result))
            elif results:
                p, r = results[0]
                print(f"  Best (unprofitable): Net: {r.net_profit_after_tax_pct:+.2f}% | Trades: {r.total_trades}")

    # Summary of best results
    if all_best:
        print(f"\n{'='*100}")
        print("OPTIMIZATION SUMMARY - BEST RESULTS")
        print(f"{'='*100}")
        print(f"{'Strategy':<16} {'Symbol':<10} {'Trades':>7} {'Win%':>7} {'Net%':>9} {'AfterTax%':>10} {'PF':>6} {'Sharpe':>7}")
        print("-" * 80)
        for strat_name, symbol, params, result in sorted(all_best, key=lambda x: x[3].net_profit_after_tax_pct, reverse=True):
            print(f"{strat_name:<16} {symbol:<10} {result.total_trades:>7} {result.win_rate:>6.1f}% {result.net_profit_pct:>+8.2f}% {result.net_profit_after_tax_pct:>+9.2f}% {result.profit_factor:>5.2f} {result.sharpe_ratio:>6.2f}")


def run_walk_forward(symbols, args):
    """Run walk-forward validation."""
    strategies = list(STRATEGY_MAP.keys()) if args.strategy == 'all' else [args.strategy]

    for strat_name in strategies:
        strategy_cls = STRATEGY_MAP[strat_name]
        defaults = strategy_cls.DEFAULT_PARAMS.copy() if hasattr(strategy_cls, 'DEFAULT_PARAMS') else {}

        print(f"\n{'='*75}")
        print(f"WALK-FORWARD VALIDATION: {strat_name}")
        print(f"{'='*75}")

        for symbol in symbols:
            train_result, val_result = walk_forward_test(
                symbol, strategy_cls, defaults, args.fee_tier,
                timeframe_minutes=args.timeframe)

            print(f"\n  {symbol}:")
            print(f"  Training  : Trades={train_result.total_trades:>4} Win={train_result.win_rate:>5.1f}% Net={train_result.net_profit_pct:>+7.2f}% AfterTax={train_result.net_profit_after_tax_pct:>+7.2f}%")
            print(f"  Validation: Trades={val_result.total_trades:>4} Win={val_result.win_rate:>5.1f}% Net={val_result.net_profit_pct:>+7.2f}% AfterTax={val_result.net_profit_after_tax_pct:>+7.2f}%")

            if train_result.net_profit_pct > 0 and val_result.net_profit_pct <= 0:
                print(f"  ** WARNING: Possible overfitting - profitable in-sample but not out-of-sample")
            elif val_result.net_profit_after_tax_pct > 0:
                print(f"  ** PASSED: Strategy profitable in both training and validation")


if __name__ == '__main__':
    main()
