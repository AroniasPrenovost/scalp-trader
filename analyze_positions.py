#!/usr/bin/env python3
"""
Position Count Analysis: 2 vs 3 Concurrent Positions

Analyzes signal overlap frequency and expected profit difference
between running 2 vs 3 concurrent positions.

Usage:
    python analyze_positions.py
"""

import json
import os
import sys
import math
import statistics
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# CONSTANTS & CONFIG
# ==============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coinbase-data')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# Fee tier (Adv 1)
MAKER_FEE = 0.0025
TAKER_FEE = 0.005
ROUND_TRIP_FEE = (MAKER_FEE + TAKER_FEE) * 100  # 0.75%

# Walk-forward validated returns (from config comments)
VALIDATED_RETURNS = {
    'TAO-USD': {'return_pct': 8.60, 'win_rate': 63},
    'ICP-USD': {'return_pct': 6.03, 'win_rate': 67},
    'CRV-USD': {'return_pct': 6.01, 'win_rate': 67},
    'NEAR-USD': {'return_pct': 3.57, 'win_rate': 60},
    'ATOM-USD': {'return_pct': 3.31, 'win_rate': 69},
    'ZEC-USD': {'return_pct': 1.47, 'win_rate': 67},
}

# Symbol -> Strategy mapping
SYMBOL_STRATEGY = {
    'TAO-USD': 'rsi_mean_reversion',
    'NEAR-USD': 'rsi_mean_reversion',
    'ZEC-USD': 'rsi_mean_reversion',
    'ICP-USD': 'rsi_regime',
    'ATOM-USD': 'rsi_regime',
    'CRV-USD': 'co_revert',
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_symbol_data(symbol: str) -> Tuple[list, list, list, list, list, list]:
    """Load raw 5-min data."""
    filepath = os.path.join(DATA_DIR, f'{symbol}.json')
    if not os.path.exists(filepath):
        return [], [], [], [], [], []

    with open(filepath, 'r') as f:
        raw = json.load(f)

    raw = sorted(raw, key=lambda x: x.get('timestamp', 0))

    timestamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []

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
    """Aggregate 5-min data to larger timeframe using clock-aligned candles."""
    if not timestamps or target_minutes <= 5:
        return timestamps, opens, highs, lows, closes, volumes

    target_seconds = target_minutes * 60
    agg_ts, agg_o, agg_h, agg_l, agg_c, agg_v = [], [], [], [], [], []

    i = 0
    n = len(timestamps)

    while i < n:
        # Find candle start aligned to clock
        candle_start = (int(timestamps[i]) // target_seconds) * target_seconds
        candle_end = candle_start + target_seconds

        # Collect all bars in this candle
        candle_opens = []
        candle_highs = []
        candle_lows = []
        candle_closes = []
        candle_volumes = []
        candle_timestamps = []

        while i < n and timestamps[i] < candle_end:
            candle_timestamps.append(timestamps[i])
            candle_opens.append(opens[i])
            candle_highs.append(highs[i])
            candle_lows.append(lows[i])
            candle_closes.append(closes[i])
            candle_volumes.append(volumes[i])
            i += 1

        if candle_closes:
            agg_ts.append(candle_start)
            agg_o.append(candle_opens[0])
            agg_h.append(max(candle_highs))
            agg_l.append(min(candle_lows))
            agg_c.append(candle_closes[-1])
            agg_v.append(sum(candle_volumes))

    return agg_ts, agg_o, agg_h, agg_l, agg_c, agg_v


# ==============================================================================
# INDICATOR COMPUTATION
# ==============================================================================

def compute_rsi(closes: list, period: int = 14) -> list:
    """Compute RSI for every bar."""
    n = len(closes)
    rsi = [None] * n
    if n < period + 1:
        return rsi

    changes = [0.0] * n
    for i in range(1, n):
        changes[i] = closes[i] - closes[i - 1]

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


def compute_ema(closes: list, period: int) -> list:
    """Compute EMA for every bar."""
    n = len(closes)
    ema = [None] * n
    if n < period:
        return ema

    sma = sum(closes[:period]) / period
    ema[period - 1] = sma

    multiplier = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = closes[i] * multiplier + ema[i - 1] * (1 - multiplier)

    return ema


def compute_bb(closes: list, period: int = 20, num_std: float = 2.0) -> Tuple[list, list, list]:
    """Compute Bollinger Bands. Returns (upper, middle, lower)."""
    n = len(closes)
    upper = [None] * n
    middle = [None] * n
    lower = [None] * n

    if n < period:
        return upper, middle, lower

    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        mid = sum(window) / period
        variance = sum((x - mid) ** 2 for x in window) / period
        std = math.sqrt(variance)

        middle[i] = mid
        upper[i] = mid + num_std * std
        lower[i] = mid - num_std * std

    return upper, middle, lower


# ==============================================================================
# SIGNAL DETECTION (matches index.py logic)
# ==============================================================================

@dataclass
class Signal:
    symbol: str
    strategy: str
    timestamp: float
    bar_index: int
    rsi: float
    price: float


def load_config() -> dict:
    """Load config.json."""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def detect_signals_rsi_mean_reversion(symbol: str, config: dict, timestamps, closes, rsi) -> List[Signal]:
    """Detect entry signals for rsi_mean_reversion strategy."""
    signals = []
    symbol_config = config['rsi_mean_reversion']['symbols'].get(symbol, {})

    rsi_entry = symbol_config.get('rsi_entry', 25)
    cooldown_bars = symbol_config.get('min_cooldown_bars', 4)

    last_signal_bar = -cooldown_bars

    for i in range(60, len(closes)):
        if rsi[i] is None:
            continue

        if i - last_signal_bar < cooldown_bars:
            continue

        if rsi[i] <= rsi_entry:
            signals.append(Signal(
                symbol=symbol,
                strategy='rsi_mean_reversion',
                timestamp=timestamps[i],
                bar_index=i,
                rsi=rsi[i],
                price=closes[i]
            ))
            last_signal_bar = i

    return signals


def detect_signals_rsi_regime(symbol: str, config: dict, timestamps, closes, rsi, ema50) -> List[Signal]:
    """Detect entry signals for rsi_regime strategy."""
    signals = []
    symbol_config = config['rsi_regime']['symbols'].get(symbol, {})

    rsi_entry = symbol_config.get('rsi_entry', 20)
    cooldown_bars = symbol_config.get('min_cooldown_bars', 3)
    max_below_ema_pct = symbol_config.get('max_below_ema_pct', 5.0) / 100.0
    ema_slope_bars = symbol_config.get('ema_slope_bars', 10)
    max_ema_decline_pct = symbol_config.get('max_ema_decline_pct', 3.0) / 100.0

    last_signal_bar = -cooldown_bars

    for i in range(60, len(closes)):
        if rsi[i] is None or ema50[i] is None:
            continue

        if i - last_signal_bar < cooldown_bars:
            continue

        if rsi[i] <= rsi_entry:
            # Regime filter
            regime_ok = True

            # Price proximity check
            distance_below = (ema50[i] - closes[i]) / ema50[i]
            if distance_below > max_below_ema_pct:
                regime_ok = False

            # EMA slope check
            if regime_ok and i >= ema_slope_bars and ema50[i - ema_slope_bars] is not None:
                ema_change = (ema50[i] - ema50[i - ema_slope_bars]) / ema50[i - ema_slope_bars]
                if ema_change < -max_ema_decline_pct:
                    regime_ok = False

            if regime_ok:
                signals.append(Signal(
                    symbol=symbol,
                    strategy='rsi_regime',
                    timestamp=timestamps[i],
                    bar_index=i,
                    rsi=rsi[i],
                    price=closes[i]
                ))
                last_signal_bar = i

    return signals


def detect_signals_co_revert(symbol: str, config: dict, timestamps, closes, rsi, bb_lower) -> List[Signal]:
    """Detect entry signals for co_revert strategy."""
    signals = []
    symbol_config = config['co_revert']['symbols'].get(symbol, {})

    rsi_entry = symbol_config.get('rsi_entry', 25)
    use_bb = symbol_config.get('use_bb_filter', True)
    cooldown_bars = symbol_config.get('min_cooldown_bars', 3)

    last_signal_bar = -cooldown_bars

    for i in range(60, len(closes)):
        if rsi[i] is None:
            continue

        if i - last_signal_bar < cooldown_bars:
            continue

        rsi_ok = rsi[i] <= rsi_entry
        bb_ok = True
        if use_bb:
            if bb_lower[i] is not None:
                bb_ok = closes[i] <= bb_lower[i]
            else:
                bb_ok = False

        if rsi_ok and bb_ok:
            signals.append(Signal(
                symbol=symbol,
                strategy='co_revert',
                timestamp=timestamps[i],
                bar_index=i,
                rsi=rsi[i],
                price=closes[i]
            ))
            last_signal_bar = i

    return signals


# ==============================================================================
# ANALYSIS
# ==============================================================================

def get_estimated_hold_hours(strategy: str) -> float:
    """Estimated average hold time based on strategy parameters."""
    hold_times = {
        'rsi_mean_reversion': 6.0,   # 30min bars * ~12 bars avg
        'rsi_regime': 2.5,           # Mixed 5min and 30min
        'co_revert': 4.0,            # 30min bars with 1.0% target
    }
    return hold_times.get(strategy, 4.0)


def analyze_signal_overlap(config: dict):
    """Main analysis: detect all signals and analyze overlap patterns."""

    print("=" * 80)
    print("POSITION COUNT ANALYSIS: 2 vs 3 CONCURRENT POSITIONS")
    print("=" * 80)
    print()

    # Load data and detect signals for each symbol
    all_signals = []
    symbol_signals = {}

    for symbol, strategy in SYMBOL_STRATEGY.items():
        symbol_config = None
        timeframe = 30  # default

        if strategy == 'rsi_mean_reversion':
            symbol_config = config['rsi_mean_reversion']['symbols'].get(symbol, {})
            timeframe = symbol_config.get('timeframe_minutes', 30)
        elif strategy == 'rsi_regime':
            symbol_config = config['rsi_regime']['symbols'].get(symbol, {})
            timeframe = symbol_config.get('timeframe_minutes', 30)
        elif strategy == 'co_revert':
            symbol_config = config['co_revert']['symbols'].get(symbol, {})
            timeframe = symbol_config.get('timeframe_minutes', 30)

        # Load and aggregate data
        ts, opens, highs, lows, closes, volumes = load_symbol_data(symbol)
        if not closes:
            print(f"  [!] No data for {symbol}")
            continue

        if timeframe > 5:
            ts, opens, highs, lows, closes, volumes = aggregate_to_timeframe(
                ts, opens, highs, lows, closes, volumes, timeframe)

        # Compute indicators
        rsi = compute_rsi(closes, 14)
        ema50 = compute_ema(closes, 50)
        _, _, bb_lower = compute_bb(closes, 20, 2.0)

        # Detect signals based on strategy
        if strategy == 'rsi_mean_reversion':
            signals = detect_signals_rsi_mean_reversion(symbol, config, ts, closes, rsi)
        elif strategy == 'rsi_regime':
            signals = detect_signals_rsi_regime(symbol, config, ts, closes, rsi, ema50)
        elif strategy == 'co_revert':
            signals = detect_signals_co_revert(symbol, config, ts, closes, rsi, bb_lower)
        else:
            signals = []

        symbol_signals[symbol] = signals
        all_signals.extend(signals)

        print(f"  {symbol} ({strategy}@{timeframe}min): {len(signals)} signals detected")

    print()

    if not all_signals:
        print("No signals detected!")
        return

    # Sort all signals by timestamp
    all_signals.sort(key=lambda s: s.timestamp)

    # Get data date range
    first_ts = all_signals[0].timestamp
    last_ts = all_signals[-1].timestamp
    total_days = (last_ts - first_ts) / 86400.0

    print(f"Data range: {datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime('%Y-%m-%d')} to "
          f"{datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime('%Y-%m-%d')} ({total_days:.0f} days)")
    print(f"Total signals: {len(all_signals)}")
    print(f"Signals per day: {len(all_signals) / total_days:.2f}")
    print()

    # ==============================================================================
    # OVERLAP ANALYSIS
    # ==============================================================================

    print("=" * 80)
    print("SIGNAL OVERLAP ANALYSIS")
    print("=" * 80)
    print()

    # Estimate position duration windows (in seconds)
    avg_hold_hours = 4.0  # Conservative average across all strategies
    position_window = avg_hold_hours * 3600  # seconds

    # Count overlapping signals using sliding window
    overlap_counts = defaultdict(int)  # {num_concurrent: count}

    for i, signal in enumerate(all_signals):
        # Count how many signals are "active" at this signal's time
        # A position is active if it started within position_window before this signal
        concurrent = 0
        for j in range(max(0, i - 50), i + 1):  # Look back at recent signals
            other = all_signals[j]
            if other.timestamp >= signal.timestamp - position_window:
                concurrent += 1
        overlap_counts[concurrent] += 1

    # More precise overlap: look at 1-hour windows
    print("Hourly window overlap analysis (how many signals fire within 1-hour windows):")
    hourly_counts = defaultdict(int)
    window_size = 3600  # 1 hour

    current_window_start = first_ts
    while current_window_start < last_ts:
        window_end = current_window_start + window_size
        signals_in_window = [s for s in all_signals if current_window_start <= s.timestamp < window_end]
        if signals_in_window:
            hourly_counts[len(signals_in_window)] += 1
        current_window_start = window_end

    print(f"  1 signal in window:  {hourly_counts.get(1, 0)} windows")
    print(f"  2 signals in window: {hourly_counts.get(2, 0)} windows")
    print(f"  3 signals in window: {hourly_counts.get(3, 0)} windows")
    print(f"  4+ signals in window: {sum(hourly_counts.get(k, 0) for k in range(4, 10))} windows")
    print()

    # 4-hour windows (more realistic for position overlap)
    print("4-hour window overlap analysis (typical position duration):")
    four_hour_counts = defaultdict(int)
    window_size = 4 * 3600  # 4 hours

    current_window_start = first_ts
    while current_window_start < last_ts:
        window_end = current_window_start + window_size
        signals_in_window = [s for s in all_signals if current_window_start <= s.timestamp < window_end]
        if signals_in_window:
            four_hour_counts[len(signals_in_window)] += 1
        current_window_start = window_size  # Non-overlapping windows
        current_window_start = current_window_start + window_size - window_size + window_end

    # Recalculate with overlapping windows
    four_hour_counts = defaultdict(int)
    window_size = 4 * 3600
    step = 3600  # Slide by 1 hour

    current_window_start = first_ts
    total_windows = 0
    while current_window_start < last_ts:
        window_end = current_window_start + window_size
        signals_in_window = [s for s in all_signals if current_window_start <= s.timestamp < window_end]
        four_hour_counts[len(signals_in_window)] += 1
        total_windows += 1
        current_window_start += step

    windows_with_signals = total_windows - four_hour_counts.get(0, 0)
    print(f"  Windows with 0 signals: {four_hour_counts.get(0, 0)} ({100*four_hour_counts.get(0, 0)/total_windows:.1f}%)")
    print(f"  Windows with 1 signal:  {four_hour_counts.get(1, 0)} ({100*four_hour_counts.get(1, 0)/total_windows:.1f}%)")
    print(f"  Windows with 2 signals: {four_hour_counts.get(2, 0)} ({100*four_hour_counts.get(2, 0)/total_windows:.1f}%)")
    print(f"  Windows with 3 signals: {four_hour_counts.get(3, 0)} ({100*four_hour_counts.get(3, 0)/total_windows:.1f}%)")
    print(f"  Windows with 4+ signals: {sum(four_hour_counts.get(k, 0) for k in range(4, 15))} ({100*sum(four_hour_counts.get(k, 0) for k in range(4, 15))/total_windows:.1f}%)")
    print()

    # ==============================================================================
    # SIGNAL QUALITY RANKING
    # ==============================================================================

    print("=" * 80)
    print("SIGNAL PRIORITIZATION (by expected return)")
    print("=" * 80)
    print()

    # Rank symbols by expected return
    ranked_symbols = sorted(VALIDATED_RETURNS.items(), key=lambda x: -x[1]['return_pct'])

    print("Priority order for signal selection:")
    for rank, (symbol, data) in enumerate(ranked_symbols, 1):
        strategy = SYMBOL_STRATEGY[symbol]
        print(f"  {rank}. {symbol}: +{data['return_pct']:.2f}% return, {data['win_rate']}% WR ({strategy})")
    print()

    # ==============================================================================
    # PROFIT CALCULATION: 2 vs 3 POSITIONS
    # ==============================================================================

    print("=" * 80)
    print("EXPECTED PROFIT ANALYSIS: 2 vs 3 POSITIONS")
    print("=" * 80)
    print()

    TOTAL_CAPITAL = 6010
    CAPITAL_2POS = TOTAL_CAPITAL / 2  # 3005 per position
    CAPITAL_3POS = TOTAL_CAPITAL / 3  # 2003.33 per position

    print(f"Total trading capital: ${TOTAL_CAPITAL:,.0f}")
    print(f"Capital per position (2 positions): ${CAPITAL_2POS:,.2f}")
    print(f"Capital per position (3 positions): ${CAPITAL_3POS:,.2f}")
    print()

    # Simulation: process signals in priority order
    # Track how many signals would be missed with each setup

    print("-" * 60)
    print("SIMULATION: Signal execution with position limits")
    print("-" * 60)
    print()

    # Simplified simulation using estimated hold times
    HOLD_HOURS = {
        'TAO-USD': 6.0,
        'ICP-USD': 1.5,  # 5min timeframe, faster
        'CRV-USD': 4.0,
        'NEAR-USD': 6.0,
        'ATOM-USD': 5.0,
        'ZEC-USD': 5.0,
    }

    def simulate_trading(max_positions: int, signals: List[Signal], capital_per_pos: float) -> Dict:
        """Simulate trading with position limits."""

        # Track active positions: list of (symbol, exit_time)
        active_positions = []
        executed_signals = []
        missed_signals = []
        missed_by_symbol = defaultdict(int)

        for signal in signals:
            # Clean up expired positions
            active_positions = [(sym, exit_t) for sym, exit_t in active_positions
                              if exit_t > signal.timestamp]

            # Check if symbol already has active position
            active_symbols = [sym for sym, _ in active_positions]

            if signal.symbol in active_symbols:
                # Already have position in this symbol
                missed_signals.append(signal)
                missed_by_symbol[signal.symbol] += 1
                continue

            if len(active_positions) >= max_positions:
                # Position limit reached
                missed_signals.append(signal)
                missed_by_symbol[signal.symbol] += 1
                continue

            # Execute signal
            hold_hours = HOLD_HOURS.get(signal.symbol, 4.0)
            exit_time = signal.timestamp + hold_hours * 3600
            active_positions.append((signal.symbol, exit_time))
            executed_signals.append(signal)

        # Calculate expected profit
        total_profit = 0.0
        for signal in executed_signals:
            symbol_return = VALIDATED_RETURNS[signal.symbol]['return_pct'] / 100.0
            # Approximate per-trade return based on total return / signal count
            num_symbol_signals = len(symbol_signals.get(signal.symbol, []))
            if num_symbol_signals > 0:
                per_trade_return = symbol_return / num_symbol_signals * len(symbol_signals[signal.symbol])
                # Simplified: use average expected return per trade
                avg_per_trade = symbol_return * capital_per_pos / 30  # ~30 trades over validation period
                total_profit += avg_per_trade

        # Better calculation using validated returns
        executed_by_symbol = defaultdict(int)
        for signal in executed_signals:
            executed_by_symbol[signal.symbol] += 1

        expected_profit = 0.0
        for symbol, count in executed_by_symbol.items():
            total_signals = len(symbol_signals.get(symbol, [1]))
            if total_signals > 0:
                execution_rate = count / total_signals
                symbol_profit = VALIDATED_RETURNS[symbol]['return_pct'] / 100.0 * capital_per_pos * execution_rate * total_days / 30
                expected_profit += symbol_profit

        return {
            'executed': len(executed_signals),
            'missed': len(missed_signals),
            'missed_by_symbol': dict(missed_by_symbol),
            'executed_by_symbol': dict(executed_by_symbol),
            'expected_profit': expected_profit,
            'execution_rate': len(executed_signals) / len(signals) if signals else 0,
        }

    # Run simulations
    results_2pos = simulate_trading(2, all_signals, CAPITAL_2POS)
    results_3pos = simulate_trading(3, all_signals, CAPITAL_3POS)

    print("With 2 CONCURRENT POSITIONS:")
    print(f"  Signals executed: {results_2pos['executed']}")
    print(f"  Signals missed:   {results_2pos['missed']}")
    print(f"  Execution rate:   {results_2pos['execution_rate']*100:.1f}%")
    print(f"  Missed by symbol: {results_2pos['missed_by_symbol']}")
    print()

    print("With 3 CONCURRENT POSITIONS:")
    print(f"  Signals executed: {results_3pos['executed']}")
    print(f"  Signals missed:   {results_3pos['missed']}")
    print(f"  Execution rate:   {results_3pos['execution_rate']*100:.1f}%")
    print(f"  Missed by symbol: {results_3pos['missed_by_symbol']}")
    print()

    # ==============================================================================
    # CAPITAL EFFICIENCY ANALYSIS
    # ==============================================================================

    print("=" * 80)
    print("CAPITAL EFFICIENCY ANALYSIS")
    print("=" * 80)
    print()

    # Calculate expected profit using walk-forward validated returns

    # Method 1: Simple pro-rata based on execution rate
    def calc_expected_monthly_profit(max_pos: int, capital_per_pos: float, execution_rate: float) -> float:
        """Calculate expected monthly profit."""
        # Weight by validated return (prioritize high-return assets)
        weighted_monthly_return = 0.0
        for symbol, data in VALIDATED_RETURNS.items():
            # Validation period was ~30 days, so this is roughly monthly
            weighted_monthly_return += data['return_pct'] / 6  # 6 symbols

        # Per-position expected monthly return
        per_pos_monthly = weighted_monthly_return / 100 * capital_per_pos * execution_rate

        return per_pos_monthly * max_pos

    # Method 2: Use actual execution rates from simulation
    monthly_2pos = calc_expected_monthly_profit(2, CAPITAL_2POS, results_2pos['execution_rate'])
    monthly_3pos = calc_expected_monthly_profit(3, CAPITAL_3POS, results_3pos['execution_rate'])

    print("Expected monthly profit (based on walk-forward validation):")
    print()
    print("  2 POSITIONS:")
    print(f"    Capital per position: ${CAPITAL_2POS:,.2f}")
    print(f"    Execution rate: {results_2pos['execution_rate']*100:.1f}%")
    print(f"    Expected monthly profit: ${monthly_2pos:,.2f}")
    print()
    print("  3 POSITIONS:")
    print(f"    Capital per position: ${CAPITAL_3POS:,.2f}")
    print(f"    Execution rate: {results_3pos['execution_rate']*100:.1f}%")
    print(f"    Expected monthly profit: ${monthly_3pos:,.2f}")
    print()

    # ==============================================================================
    # RISK ANALYSIS
    # ==============================================================================

    print("=" * 80)
    print("RISK ANALYSIS")
    print("=" * 80)
    print()

    # Single position max loss
    avg_disaster_stop = 4.0  # Average across strategies

    max_loss_2pos = CAPITAL_2POS * (avg_disaster_stop / 100) * 2  # Both positions stop out
    max_loss_3pos = CAPITAL_3POS * (avg_disaster_stop / 100) * 3  # All positions stop out

    print("Worst-case scenario (all positions hit disaster stop):")
    print(f"  2 positions max loss: ${max_loss_2pos:,.2f} ({100*max_loss_2pos/TOTAL_CAPITAL:.1f}% of capital)")
    print(f"  3 positions max loss: ${max_loss_3pos:,.2f} ({100*max_loss_3pos/TOTAL_CAPITAL:.1f}% of capital)")
    print()

    print("Concentration risk:")
    print(f"  2 positions: Each position = {100*CAPITAL_2POS/TOTAL_CAPITAL:.1f}% of capital")
    print(f"  3 positions: Each position = {100*CAPITAL_3POS/TOTAL_CAPITAL:.1f}% of capital")
    print()

    # ==============================================================================
    # FINAL RECOMMENDATION
    # ==============================================================================

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    # Calculate advantage
    profit_advantage = monthly_3pos - monthly_2pos
    execution_advantage = results_3pos['execution_rate'] - results_2pos['execution_rate']

    print("Key findings:")
    print()
    print(f"1. Signal overlap: In 4-hour windows, {four_hour_counts.get(3, 0)} windows had 3+ signals")
    print(f"   ({100*four_hour_counts.get(3, 0)/total_windows:.1f}% of time)")
    print()
    print(f"2. Execution rate improvement with 3 positions: +{execution_advantage*100:.1f}%")
    print()
    print(f"3. Expected monthly profit difference: ${profit_advantage:,.2f}")
    print()

    if profit_advantage > 0:
        print("RECOMMENDATION: 3 CONCURRENT POSITIONS")
        print()
        print("Reasoning:")
        print("  - Higher execution rate captures more opportunities")
        print("  - Signal overlap is frequent enough to benefit from 3rd slot")
        print("  - Diversification across 3 positions reduces single-trade risk")
        print("  - Fee tier progression is faster with more trades")
    else:
        print("RECOMMENDATION: 2 CONCURRENT POSITIONS")
        print()
        print("Reasoning:")
        print("  - Larger position sizes can compound faster")
        print("  - Signal overlap is not frequent enough to justify 3rd slot")
        print("  - Simpler to manage with fewer positions")

    print()
    print("=" * 80)

    # Return key metrics
    return {
        '2_positions': {
            'capital_per_pos': CAPITAL_2POS,
            'execution_rate': results_2pos['execution_rate'],
            'expected_monthly_profit': monthly_2pos,
            'executed_signals': results_2pos['executed'],
            'missed_signals': results_2pos['missed'],
        },
        '3_positions': {
            'capital_per_pos': CAPITAL_3POS,
            'execution_rate': results_3pos['execution_rate'],
            'expected_monthly_profit': monthly_3pos,
            'executed_signals': results_3pos['executed'],
            'missed_signals': results_3pos['missed'],
        },
        'recommendation': '3_positions' if profit_advantage > 0 else '2_positions',
        'profit_difference': profit_advantage,
    }


if __name__ == '__main__':
    config = load_config()
    results = analyze_signal_overlap(config)
