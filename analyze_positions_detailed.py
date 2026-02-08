#!/usr/bin/env python3
"""
Detailed Position Count Analysis with Capital Utilization Tracking

This provides a deeper dive into:
1. Time-weighted capital utilization
2. Trade-by-trade profit simulation
3. Opportunity cost of idle capital
"""

import json
import os
import sys
import math
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coinbase-data')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

MAKER_FEE = 0.0025
TAKER_FEE = 0.005
ROUND_TRIP_FEE_PCT = (MAKER_FEE + TAKER_FEE) * 100  # 0.75%

VALIDATED_RETURNS = {
    'TAO-USD': {'return_pct': 8.60, 'win_rate': 63, 'avg_trade_pct': 0.82},
    'ICP-USD': {'return_pct': 6.03, 'win_rate': 67, 'avg_trade_pct': 0.68},
    'CRV-USD': {'return_pct': 6.01, 'win_rate': 67, 'avg_trade_pct': 0.72},
    'NEAR-USD': {'return_pct': 3.57, 'win_rate': 60, 'avg_trade_pct': 0.45},
    'ATOM-USD': {'return_pct': 3.31, 'win_rate': 69, 'avg_trade_pct': 0.55},
    'ZEC-USD': {'return_pct': 1.47, 'win_rate': 67, 'avg_trade_pct': 0.25},
}

SYMBOL_STRATEGY = {
    'TAO-USD': 'rsi_mean_reversion',
    'NEAR-USD': 'rsi_mean_reversion',
    'ZEC-USD': 'rsi_mean_reversion',
    'ICP-USD': 'rsi_regime',
    'ATOM-USD': 'rsi_regime',
    'CRV-USD': 'co_revert',
}

# Estimated average hold time in hours per symbol (from backtest data)
HOLD_HOURS = {
    'TAO-USD': 5.5,
    'ICP-USD': 1.5,
    'CRV-USD': 3.5,
    'NEAR-USD': 6.5,
    'ATOM-USD': 5.0,
    'ZEC-USD': 4.0,
}


def load_symbol_data(symbol: str):
    filepath = os.path.join(DATA_DIR, f'{symbol}.json')
    if not os.path.exists(filepath):
        return [], [], [], [], [], []

    with open(filepath, 'r') as f:
        raw = json.load(f)

    raw = sorted(raw, key=lambda x: x.get('timestamp', 0))

    timestamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []

    for i, entry in enumerate(raw):
        price = float(entry.get('price', 0))
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

    return timestamps, opens, highs, lows, closes, []


def aggregate_to_timeframe(timestamps, opens, highs, lows, closes, volumes, target_minutes):
    if not timestamps or target_minutes <= 5:
        return timestamps, opens, highs, lows, closes, volumes

    target_seconds = target_minutes * 60
    agg_ts, agg_o, agg_h, agg_l, agg_c, agg_v = [], [], [], [], [], []

    i = 0
    n = len(timestamps)

    while i < n:
        candle_start = (int(timestamps[i]) // target_seconds) * target_seconds
        candle_end = candle_start + target_seconds

        candle_opens, candle_highs, candle_lows, candle_closes = [], [], [], []

        while i < n and timestamps[i] < candle_end:
            candle_opens.append(opens[i])
            candle_highs.append(highs[i])
            candle_lows.append(lows[i])
            candle_closes.append(closes[i])
            i += 1

        if candle_closes:
            agg_ts.append(candle_start)
            agg_o.append(candle_opens[0])
            agg_h.append(max(candle_highs))
            agg_l.append(min(candle_lows))
            agg_c.append(candle_closes[-1])
            agg_v.append(0)

    return agg_ts, agg_o, agg_h, agg_l, agg_c, agg_v


def compute_rsi(closes, period=14):
    n = len(closes)
    rsi = [None] * n
    if n < period + 1:
        return rsi

    changes = [0.0] * n
    for i in range(1, n):
        changes[i] = closes[i] - closes[i - 1]

    avg_gain = avg_loss = 0.0
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
        rsi[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    for i in range(period + 1, n):
        change = changes[i]
        gain = change if change > 0 else 0.0
        loss = abs(change) if change < 0 else 0.0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    return rsi


def compute_ema(closes, period):
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


def compute_bb(closes, period=20, num_std=2.0):
    n = len(closes)
    lower = [None] * n

    if n < period:
        return lower

    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        mid = sum(window) / period
        variance = sum((x - mid) ** 2 for x in window) / period
        std = math.sqrt(variance)
        lower[i] = mid - num_std * std

    return lower


@dataclass
class Signal:
    symbol: str
    strategy: str
    timestamp: float
    rsi: float
    price: float
    expected_return: float


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def detect_all_signals(config):
    """Detect signals for all symbols and return sorted by timestamp."""
    all_signals = []
    signal_counts = {}

    for symbol, strategy in SYMBOL_STRATEGY.items():
        if strategy == 'rsi_mean_reversion':
            symbol_config = config['rsi_mean_reversion']['symbols'].get(symbol, {})
        elif strategy == 'rsi_regime':
            symbol_config = config['rsi_regime']['symbols'].get(symbol, {})
        elif strategy == 'co_revert':
            symbol_config = config['co_revert']['symbols'].get(symbol, {})
        else:
            continue

        timeframe = symbol_config.get('timeframe_minutes', 30)

        ts, opens, highs, lows, closes, _ = load_symbol_data(symbol)
        if not closes:
            continue

        if timeframe > 5:
            ts, opens, highs, lows, closes, _ = aggregate_to_timeframe(
                ts, opens, highs, lows, closes, [], timeframe)

        rsi = compute_rsi(closes, 14)
        ema50 = compute_ema(closes, 50)
        bb_lower = compute_bb(closes, 20, 2.0)

        rsi_entry = symbol_config.get('rsi_entry', 25)
        cooldown = symbol_config.get('min_cooldown_bars', 3)
        max_below_ema = symbol_config.get('max_below_ema_pct', 5.0) / 100.0 if strategy == 'rsi_regime' else None
        use_bb = symbol_config.get('use_bb_filter', True) if strategy == 'co_revert' else False

        last_signal_bar = -cooldown
        symbol_signals = []

        for i in range(60, len(closes)):
            if rsi[i] is None:
                continue
            if i - last_signal_bar < cooldown:
                continue

            should_signal = False

            if strategy == 'rsi_mean_reversion':
                should_signal = rsi[i] <= rsi_entry

            elif strategy == 'rsi_regime':
                if rsi[i] <= rsi_entry and ema50[i] is not None:
                    distance_below = (ema50[i] - closes[i]) / ema50[i]
                    if distance_below <= max_below_ema:
                        should_signal = True

            elif strategy == 'co_revert':
                rsi_ok = rsi[i] <= rsi_entry
                bb_ok = not use_bb or (bb_lower[i] is not None and closes[i] <= bb_lower[i])
                should_signal = rsi_ok and bb_ok

            if should_signal:
                expected_return = VALIDATED_RETURNS[symbol]['avg_trade_pct']
                symbol_signals.append(Signal(
                    symbol=symbol,
                    strategy=strategy,
                    timestamp=ts[i],
                    rsi=rsi[i],
                    price=closes[i],
                    expected_return=expected_return
                ))
                last_signal_bar = i

        all_signals.extend(symbol_signals)
        signal_counts[symbol] = len(symbol_signals)

    all_signals.sort(key=lambda s: s.timestamp)
    return all_signals, signal_counts


def simulate_detailed(max_positions: int, capital_per_pos: float, signals: List[Signal]):
    """
    Detailed simulation tracking:
    - Active positions over time
    - Capital utilization
    - Trade-by-trade P&L
    """

    @dataclass
    class Position:
        symbol: str
        entry_time: float
        exit_time: float
        entry_price: float
        expected_return: float

    active_positions: List[Position] = []
    completed_trades = []
    missed_signals = []
    capital_utilization_samples = []  # (timestamp, deployed_capital, idle_capital)

    total_capital = capital_per_pos * max_positions

    for signal in signals:
        # Clean up expired positions
        expired = [p for p in active_positions if p.exit_time <= signal.timestamp]
        for p in expired:
            # Record trade result
            trade_pnl = capital_per_pos * (p.expected_return / 100.0) - (capital_per_pos * ROUND_TRIP_FEE_PCT / 100.0)
            completed_trades.append({
                'symbol': p.symbol,
                'entry_time': p.entry_time,
                'exit_time': p.exit_time,
                'pnl': trade_pnl,
                'duration_hours': (p.exit_time - p.entry_time) / 3600,
            })

        active_positions = [p for p in active_positions if p.exit_time > signal.timestamp]

        # Track capital utilization
        deployed = len(active_positions) * capital_per_pos
        idle = total_capital - deployed
        capital_utilization_samples.append((signal.timestamp, deployed, idle))

        # Check if symbol already has position
        if signal.symbol in [p.symbol for p in active_positions]:
            missed_signals.append(signal)
            continue

        # Check position limit
        if len(active_positions) >= max_positions:
            missed_signals.append(signal)
            continue

        # Open new position
        hold_hours = HOLD_HOURS.get(signal.symbol, 4.0)
        exit_time = signal.timestamp + hold_hours * 3600

        active_positions.append(Position(
            symbol=signal.symbol,
            entry_time=signal.timestamp,
            exit_time=exit_time,
            entry_price=signal.price,
            expected_return=signal.expected_return
        ))

    # Close remaining positions
    for p in active_positions:
        trade_pnl = capital_per_pos * (p.expected_return / 100.0) - (capital_per_pos * ROUND_TRIP_FEE_PCT / 100.0)
        completed_trades.append({
            'symbol': p.symbol,
            'entry_time': p.entry_time,
            'exit_time': p.exit_time,
            'pnl': trade_pnl,
            'duration_hours': (p.exit_time - p.entry_time) / 3600,
        })

    return {
        'completed_trades': completed_trades,
        'missed_signals': missed_signals,
        'capital_utilization_samples': capital_utilization_samples,
    }


def main():
    print("=" * 80)
    print("DETAILED POSITION COUNT ANALYSIS")
    print("=" * 80)
    print()

    config = load_config()
    all_signals, signal_counts = detect_all_signals(config)

    if not all_signals:
        print("No signals detected!")
        return

    total_days = (all_signals[-1].timestamp - all_signals[0].timestamp) / 86400.0

    print(f"Analysis period: {total_days:.0f} days")
    print(f"Total signals detected: {len(all_signals)}")
    print()
    print("Signals per symbol:")
    for symbol, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
        ret = VALIDATED_RETURNS[symbol]['return_pct']
        print(f"  {symbol}: {count} signals ({count/total_days:.2f}/day) - WF return: +{ret:.2f}%")
    print()

    TOTAL_CAPITAL = 6010

    # 2 positions
    capital_2 = TOTAL_CAPITAL / 2
    results_2 = simulate_detailed(2, capital_2, all_signals)

    # 3 positions
    capital_3 = TOTAL_CAPITAL / 3
    results_3 = simulate_detailed(3, capital_3, all_signals)

    print("=" * 80)
    print("TRADE-BY-TRADE ANALYSIS")
    print("=" * 80)
    print()

    # 2 Position Results
    trades_2 = results_2['completed_trades']
    total_pnl_2 = sum(t['pnl'] for t in trades_2)
    avg_trade_2 = total_pnl_2 / len(trades_2) if trades_2 else 0

    print("2 CONCURRENT POSITIONS:")
    print(f"  Trades executed: {len(trades_2)}")
    print(f"  Signals missed: {len(results_2['missed_signals'])}")
    print(f"  Total P&L: ${total_pnl_2:.2f}")
    print(f"  Avg P&L per trade: ${avg_trade_2:.2f}")
    print(f"  Monthly rate: ${total_pnl_2 / (total_days/30):.2f}/month")
    print()

    # P&L by symbol
    pnl_by_symbol_2 = defaultdict(float)
    trades_by_symbol_2 = defaultdict(int)
    for t in trades_2:
        pnl_by_symbol_2[t['symbol']] += t['pnl']
        trades_by_symbol_2[t['symbol']] += 1

    print("  P&L breakdown by symbol:")
    for symbol in sorted(pnl_by_symbol_2.keys(), key=lambda s: -pnl_by_symbol_2[s]):
        print(f"    {symbol}: ${pnl_by_symbol_2[symbol]:.2f} ({trades_by_symbol_2[symbol]} trades)")
    print()

    # 3 Position Results
    trades_3 = results_3['completed_trades']
    total_pnl_3 = sum(t['pnl'] for t in trades_3)
    avg_trade_3 = total_pnl_3 / len(trades_3) if trades_3 else 0

    print("3 CONCURRENT POSITIONS:")
    print(f"  Trades executed: {len(trades_3)}")
    print(f"  Signals missed: {len(results_3['missed_signals'])}")
    print(f"  Total P&L: ${total_pnl_3:.2f}")
    print(f"  Avg P&L per trade: ${avg_trade_3:.2f}")
    print(f"  Monthly rate: ${total_pnl_3 / (total_days/30):.2f}/month")
    print()

    # P&L by symbol
    pnl_by_symbol_3 = defaultdict(float)
    trades_by_symbol_3 = defaultdict(int)
    for t in trades_3:
        pnl_by_symbol_3[t['symbol']] += t['pnl']
        trades_by_symbol_3[t['symbol']] += 1

    print("  P&L breakdown by symbol:")
    for symbol in sorted(pnl_by_symbol_3.keys(), key=lambda s: -pnl_by_symbol_3[s]):
        print(f"    {symbol}: ${pnl_by_symbol_3[symbol]:.2f} ({trades_by_symbol_3[symbol]} trades)")
    print()

    # ==============================================================================
    # CAPITAL UTILIZATION
    # ==============================================================================

    print("=" * 80)
    print("CAPITAL UTILIZATION ANALYSIS")
    print("=" * 80)
    print()

    # Calculate average capital utilization
    util_2 = results_2['capital_utilization_samples']
    util_3 = results_3['capital_utilization_samples']

    if util_2:
        avg_deployed_2 = sum(d for _, d, _ in util_2) / len(util_2)
        avg_idle_2 = sum(i for _, _, i in util_2) / len(util_2)
        util_pct_2 = avg_deployed_2 / TOTAL_CAPITAL * 100
    else:
        avg_deployed_2 = avg_idle_2 = util_pct_2 = 0

    if util_3:
        avg_deployed_3 = sum(d for _, d, _ in util_3) / len(util_3)
        avg_idle_3 = sum(i for _, _, i in util_3) / len(util_3)
        util_pct_3 = avg_deployed_3 / TOTAL_CAPITAL * 100
    else:
        avg_deployed_3 = avg_idle_3 = util_pct_3 = 0

    print("2 POSITIONS:")
    print(f"  Average deployed capital: ${avg_deployed_2:,.2f}")
    print(f"  Average idle capital: ${avg_idle_2:,.2f}")
    print(f"  Capital utilization: {util_pct_2:.1f}%")
    print()

    print("3 POSITIONS:")
    print(f"  Average deployed capital: ${avg_deployed_3:,.2f}")
    print(f"  Average idle capital: ${avg_idle_3:,.2f}")
    print(f"  Capital utilization: {util_pct_3:.1f}%")
    print()

    # ==============================================================================
    # OPPORTUNITY COST ANALYSIS
    # ==============================================================================

    print("=" * 80)
    print("OPPORTUNITY COST: MISSED SIGNALS")
    print("=" * 80)
    print()

    # Estimate value of missed signals
    missed_value_2 = sum(
        capital_2 * (s.expected_return / 100.0) - (capital_2 * ROUND_TRIP_FEE_PCT / 100.0)
        for s in results_2['missed_signals']
    )
    missed_value_3 = sum(
        capital_3 * (s.expected_return / 100.0) - (capital_3 * ROUND_TRIP_FEE_PCT / 100.0)
        for s in results_3['missed_signals']
    )

    missed_by_symbol_2 = defaultdict(int)
    for s in results_2['missed_signals']:
        missed_by_symbol_2[s.symbol] += 1

    missed_by_symbol_3 = defaultdict(int)
    for s in results_3['missed_signals']:
        missed_by_symbol_3[s.symbol] += 1

    print("2 POSITIONS - Missed signals value:")
    print(f"  Total missed: {len(results_2['missed_signals'])} signals")
    print(f"  Estimated missed profit: ${missed_value_2:.2f}")
    print(f"  By symbol: {dict(missed_by_symbol_2)}")
    print()

    print("3 POSITIONS - Missed signals value:")
    print(f"  Total missed: {len(results_3['missed_signals'])} signals")
    print(f"  Estimated missed profit: ${missed_value_3:.2f}")
    print(f"  By symbol: {dict(missed_by_symbol_3)}")
    print()

    # ==============================================================================
    # FINAL COMPARISON
    # ==============================================================================

    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print()

    monthly_pnl_2 = total_pnl_2 / (total_days / 30)
    monthly_pnl_3 = total_pnl_3 / (total_days / 30)

    print(f"{'Metric':<35} {'2 Positions':>15} {'3 Positions':>15} {'Difference':>12}")
    print("-" * 80)
    print(f"{'Capital per position':<35} ${capital_2:>13,.2f} ${capital_3:>13,.2f} ${capital_2 - capital_3:>10,.2f}")
    print(f"{'Trades executed':<35} {len(trades_2):>15} {len(trades_3):>15} {len(trades_3) - len(trades_2):>+12}")
    print(f"{'Signals missed':<35} {len(results_2['missed_signals']):>15} {len(results_3['missed_signals']):>15} {len(results_3['missed_signals']) - len(results_2['missed_signals']):>+12}")
    print(f"{'Total P&L':<35} ${total_pnl_2:>13,.2f} ${total_pnl_3:>13,.2f} ${total_pnl_3 - total_pnl_2:>+10,.2f}")
    print(f"{'Monthly P&L':<35} ${monthly_pnl_2:>13,.2f} ${monthly_pnl_3:>13,.2f} ${monthly_pnl_3 - monthly_pnl_2:>+10,.2f}")
    print(f"{'Avg P&L per trade':<35} ${avg_trade_2:>13,.2f} ${avg_trade_3:>13,.2f} ${avg_trade_3 - avg_trade_2:>+10,.2f}")
    print(f"{'Capital utilization':<35} {util_pct_2:>14.1f}% {util_pct_3:>14.1f}% {util_pct_3 - util_pct_2:>+10.1f}%")
    print(f"{'Missed opportunity cost':<35} ${missed_value_2:>13,.2f} ${missed_value_3:>13,.2f} ${missed_value_2 - missed_value_3:>+10,.2f}")
    print()

    # ==============================================================================
    # RECOMMENDATION
    # ==============================================================================

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    pnl_diff = total_pnl_3 - total_pnl_2
    trade_diff = len(trades_3) - len(trades_2)
    missed_reduction = len(results_2['missed_signals']) - len(results_3['missed_signals'])

    if pnl_diff > 0:
        print("RECOMMENDED: 3 CONCURRENT POSITIONS")
        print()
        print(f"Advantages of 3 positions:")
        print(f"  - +${pnl_diff:.2f} total profit over {total_days:.0f} days")
        print(f"  - +{trade_diff} additional trades executed")
        print(f"  - {missed_reduction} fewer missed signals")
        print(f"  - Better diversification (33% per position vs 50%)")
        print(f"  - Faster fee tier progression due to higher volume")
        print()
        print("Trade-offs:")
        print(f"  - ${avg_trade_2 - avg_trade_3:.2f} less profit per trade (smaller position size)")
        print(f"  - More positions to monitor")
    else:
        print("RECOMMENDED: 2 CONCURRENT POSITIONS")
        print()
        print(f"Advantages of 2 positions:")
        print(f"  - ${abs(pnl_diff):.2f} higher total profit")
        print(f"  - Larger position sizes = bigger individual wins")
        print(f"  - Simpler position management")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
