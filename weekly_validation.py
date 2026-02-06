#!/usr/bin/env python3
"""
Weekly Strategy Validation Script

Runs walk-forward validation on active trading symbols to confirm the
RSI mean reversion strategy is still profitable. Also scans inactive
symbols for potential new opportunities.

Run via cron:
  0 0 * * 0 cd /path/to/scalp-scripts && python3 weekly_validation.py >> logs/weekly_validation.log 2>&1

Uses backtest.py functions: run_backtest(), RSIOnlyStrategy, load_symbol_data(), aggregate_to_timeframe()
"""

import sys
import os
import json
import time
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (
    run_backtest,
    RSIOnlyStrategy,
    load_symbol_data,
    aggregate_to_timeframe,
    BacktestResult,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Active trading symbols with their validated parameters
ACTIVE_SYMBOLS = {
    'ATOM-USD': {
        'timeframe_minutes': 15,
        'params': {
            'rsi_period': 14,
            'rsi_entry': 20,
            'rsi_exit': 48,
            'rsi_partial_exit': 35,
            'disaster_stop_pct': 5.0,
            'max_hold_bars': 24,
            'trailing_activate_pct': 0.3,
            'trailing_stop_pct': 0.2,
            'min_cooldown_bars': 3,
        }
    },
    'LINK-USD': {
        'timeframe_minutes': 120,
        'params': {
            'rsi_period': 14,
            'rsi_entry': 20,
            'rsi_exit': 45,
            'rsi_partial_exit': 33,
            'disaster_stop_pct': 5.0,
            'max_hold_bars': 12,
            'trailing_activate_pct': 0.3,
            'trailing_stop_pct': 0.2,
            'min_cooldown_bars': 2,
        }
    },
    'HBAR-USD': {
        'timeframe_minutes': 120,
        'params': {
            'rsi_period': 14,
            'rsi_entry': 25,
            'rsi_exit': 50,
            'rsi_partial_exit': 38,
            'disaster_stop_pct': 5.0,
            'max_hold_bars': 12,
            'trailing_activate_pct': 0.3,
            'trailing_stop_pct': 0.2,
            'min_cooldown_bars': 2,
        }
    },
}

# Inactive symbols to scan for new opportunities
SCAN_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'ADA-USD', 'LTC-USD',
                'ZEC-USD', 'SUI-USD', 'TAO-USD', 'BCH-USD', 'ICP-USD', 'CRV-USD']

# Walk-forward window settings
# 3 rolling test windows, each ~30 days, stepping by ~15 days
WALK_FORWARD_WINDOWS = 3
WINDOW_SIZE_DAYS = 30
WINDOW_STEP_DAYS = 15

FEE_TIER = 'adv2'
CAPITAL = 6000.0
TAX_RATE = 0.24


def get_walk_forward_windows(symbol: str):
    """
    Generate rolling walk-forward test windows from available data.

    Returns list of (start_ts, end_ts, label) tuples for the most recent
    WALK_FORWARD_WINDOWS windows.
    """
    timestamps, _, _, _, closes, _ = load_symbol_data(symbol)
    if not timestamps or len(timestamps) < 100:
        return []

    data_end = timestamps[-1]
    data_start = timestamps[0]
    total_days = (data_end - data_start) / 86400.0

    if total_days < WINDOW_SIZE_DAYS * 2:
        return []

    windows = []
    # Work backwards from most recent data
    for i in range(WALK_FORWARD_WINDOWS):
        end_ts = data_end - (i * WINDOW_STEP_DAYS * 86400)
        start_ts = end_ts - (WINDOW_SIZE_DAYS * 86400)

        if start_ts < data_start:
            break

        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        label = f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d')}"

        windows.append((start_ts, end_ts, label))

    windows.reverse()  # Chronological order
    return windows


def validate_symbol(symbol: str, timeframe_minutes: int, params: dict):
    """
    Run full backtest and walk-forward validation on a symbol.

    Returns:
        dict with 'full_result', 'window_results', 'all_windows_pass', 'status'
    """
    # Full period backtest
    full_result = run_backtest(
        symbol=symbol,
        strategy_cls=RSIOnlyStrategy,
        params=params,
        fee_tier=FEE_TIER,
        capital=CAPITAL,
        timeframe_minutes=timeframe_minutes
    )

    # Walk-forward windows
    windows = get_walk_forward_windows(symbol)
    window_results = []
    all_pass = True

    for start_ts, end_ts, label in windows:
        wf_result = run_backtest(
            symbol=symbol,
            strategy_cls=RSIOnlyStrategy,
            params=params,
            fee_tier=FEE_TIER,
            capital=CAPITAL,
            start_ts=start_ts,
            end_ts=end_ts,
            timeframe_minutes=timeframe_minutes
        )

        # Pass if profitable OR no trades in window (can't fail without trading)
        window_pass = wf_result.total_trades == 0 or wf_result.net_profit_after_tax_pct >= 0
        if not window_pass:
            all_pass = False

        window_results.append({
            'label': label,
            'result': wf_result,
            'passed': window_pass
        })

    status = 'VALIDATED' if all_pass and full_result.total_trades > 0 else 'FAILED'

    return {
        'full_result': full_result,
        'window_results': window_results,
        'all_windows_pass': all_pass,
        'status': status
    }


def format_result_line(result: BacktestResult) -> str:
    """Format a BacktestResult into a single summary line."""
    if result.total_trades == 0:
        return "0 trades"

    return (
        f"{result.total_trades}t "
        f"WR={result.win_rate:.0f}% "
        f"AT={result.net_profit_after_tax_pct:+.2f}% "
        f"PF={result.profit_factor:.2f} "
        f"DD={result.max_drawdown_pct:.2f}%"
    )


def print_validation_report(active_results: dict, scan_results: dict):
    """Print the full validation report."""
    now = datetime.now(timezone.utc)
    print(f"\n{'=' * 70}")
    print(f"  WEEKLY STRATEGY VALIDATION ({now.strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"{'=' * 70}")
    print(f"  Strategy: RSI Pure Mean Reversion (rsi_pure)")
    print(f"  Fee Tier: {FEE_TIER} | Capital: ${CAPITAL:,.0f} | Tax: {TAX_RATE*100:.0f}%")
    print(f"{'=' * 70}\n")

    # Active symbols
    any_failure = False
    for symbol, validation in active_results.items():
        config = ACTIVE_SYMBOLS[symbol]
        tf = config['timeframe_minutes']
        tf_label = f"{tf}min" if tf < 60 else f"{tf // 60}H"

        full = validation['full_result']
        status = validation['status']
        status_mark = 'PASS' if status == 'VALIDATED' else '** FAIL **'

        print(f"  {symbol} @ {tf_label}: [ACTIVE]")
        print(f"    Full: {format_result_line(full)}")

        if full.total_trades > 0 and full.sharpe_ratio:
            print(f"    Sharpe: {full.sharpe_ratio:.2f} | "
                  f"Avg Hold: {full.avg_hold_hours:.1f}h | "
                  f"Trades/Day: {full.trades_per_day:.2f}")

        for wf in validation['window_results']:
            wf_mark = 'PASS' if wf['passed'] else 'FAIL'
            wf_result = wf['result']
            trades_str = f"{wf_result.total_trades}t" if wf_result.total_trades > 0 else "no trades"
            print(f"    Window ({wf['label']}): "
                  f"{trades_str} AT={wf_result.net_profit_after_tax_pct:+.2f}% [{wf_mark}]")

        print(f"    -> STATUS: {status_mark}")

        if status != 'VALIDATED':
            any_failure = True

        print()

    # Warnings
    if any_failure:
        print(f"  {'!' * 50}")
        print(f"  WARNING: One or more active symbols FAILED validation!")
        print(f"  Review performance and consider pausing trading.")
        print(f"  {'!' * 50}\n")

    # Scan inactive symbols
    if scan_results:
        print(f"  --- Scanning inactive symbols for new opportunities ---\n")

        for symbol, validation in scan_results.items():
            full = validation['full_result']
            status = validation['status']
            tf = 15  # Default scan timeframe

            if full.total_trades == 0:
                print(f"  {symbol} @ {tf}min: No trades")
            else:
                at_pct = full.net_profit_after_tax_pct
                status_str = 'PASS' if at_pct > 0 else 'FAIL'
                print(f"  {symbol} @ {tf}min: AT={at_pct:+.2f}% [{status_str}]")

                if status == 'VALIDATED':
                    print(f"    -> POTENTIAL NEW OPPORTUNITY (passes all windows)")
                    print(f"    {format_result_line(full)}")

        print()

    print(f"{'=' * 70}")
    print(f"  Validation complete.")
    print(f"{'=' * 70}\n")

    return not any_failure


def main():
    print(f"\nStarting weekly validation at {datetime.now(timezone.utc).isoformat()}")

    # Validate active symbols
    active_results = {}
    for symbol, config in ACTIVE_SYMBOLS.items():
        print(f"  Validating {symbol}...", end='', flush=True)
        try:
            result = validate_symbol(
                symbol=symbol,
                timeframe_minutes=config['timeframe_minutes'],
                params=config['params']
            )
            active_results[symbol] = result
            print(f" {result['status']}")
        except Exception as e:
            print(f" ERROR: {e}")
            active_results[symbol] = {
                'full_result': BacktestResult(strategy_name='rsi_pure', params={}, symbol=symbol, period='error'),
                'window_results': [],
                'all_windows_pass': False,
                'status': 'ERROR'
            }

    # Scan inactive symbols (use default RSI params at 15min)
    default_scan_params = {
        'rsi_period': 14,
        'rsi_entry': 22,
        'rsi_exit': 48,
        'rsi_partial_exit': 35,
        'disaster_stop_pct': 5.0,
        'max_hold_bars': 24,
        'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2,
        'min_cooldown_bars': 3,
    }

    scan_results = {}
    for symbol in SCAN_SYMBOLS:
        print(f"  Scanning {symbol}...", end='', flush=True)
        try:
            result = validate_symbol(
                symbol=symbol,
                timeframe_minutes=15,
                params=default_scan_params
            )
            scan_results[symbol] = result
            status = result['full_result'].net_profit_after_tax_pct
            print(f" AT={status:+.2f}%")
        except Exception as e:
            print(f" ERROR: {e}")

    # Print report
    all_valid = print_validation_report(active_results, scan_results)

    # Exit code: 0 = all pass, 1 = failure detected
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
