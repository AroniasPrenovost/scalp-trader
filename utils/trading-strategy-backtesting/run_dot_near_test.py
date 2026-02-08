#!/usr/bin/env python3
"""
Test DOT and NEAR with clock-aligned aggregation at 30min.

Run from project root: python3 utils/trading-strategy-backtesting/run_dot_near_test.py
"""

import sys
import os

# Add project root to path for imports (two levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from backtest import run_backtest, walk_forward_test, print_result, STRATEGY_MAP

strategy_cls = STRATEGY_MAP['rsi_pure']

# Param grid — key variables that differ between symbols
PARAM_GRID = [
    {'rsi_entry': 20, 'rsi_exit': 48, 'rsi_partial_exit': 35},
    {'rsi_entry': 20, 'rsi_exit': 45, 'rsi_partial_exit': 33},
    {'rsi_entry': 25, 'rsi_exit': 50, 'rsi_partial_exit': 38},
    {'rsi_entry': 25, 'rsi_exit': 48, 'rsi_partial_exit': 35},
    {'rsi_entry': 22, 'rsi_exit': 48, 'rsi_partial_exit': 35},
    {'rsi_entry': 20, 'rsi_exit': 50, 'rsi_partial_exit': 38},
]

# Fixed params that work across symbols
FIXED_PARAMS = {
    'rsi_period': 14,
    'disaster_stop_pct': 5.0,
    'trailing_activate_pct': 0.3,
    'trailing_stop_pct': 0.2,
    'min_cooldown_bars': 2,
}

SYMBOLS = ['DOT-USD', 'NEAR-USD']
TIMEFRAME = 30

# Also test max_hold variants
MAX_HOLD_OPTIONS = [12, 24, 36]

print("=" * 130)
print("DOT & NEAR PARAM SWEEP @ 30min (clock-aligned)")
print("=" * 130)

best_results = {}  # symbol -> (params, result)

for symbol in SYMBOLS:
    print(f"\n{'='*130}")
    print(f"  {symbol} @ {TIMEFRAME}min")
    print(f"{'='*130}")
    print(f"{'RSI_entry':>10} {'RSI_exit':>9} {'Partial':>8} {'MaxHold':>8} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Gross%':>9} {'Net%':>9} {'AT%':>9} {'Sharpe':>8} {'MaxDD%':>8}")
    print("-" * 130)

    for grid in PARAM_GRID:
        for max_hold in MAX_HOLD_OPTIONS:
            params = strategy_cls.DEFAULT_PARAMS.copy()
            params.update(FIXED_PARAMS)
            params.update(grid)
            params['max_hold_bars'] = max_hold

            r = run_backtest(symbol, strategy_cls, params, 'adv2', timeframe_minutes=TIMEFRAME)

            marker = ""
            if r.total_trades >= 8 and r.net_profit_after_tax_pct > 0 and r.profit_factor > 1.0:
                marker = " <-- CANDIDATE"
                key = f"{symbol}"
                if key not in best_results or r.net_profit_after_tax_pct > best_results[key][1].net_profit_after_tax_pct:
                    best_results[key] = ({**grid, 'max_hold_bars': max_hold}, r)

            print(f"{grid['rsi_entry']:>10} {grid['rsi_exit']:>9} {grid['rsi_partial_exit']:>8} {max_hold:>8} "
                  f"{r.total_trades:>7} {r.win_rate:>6.1f}% {r.profit_factor:>6.2f} "
                  f"{r.gross_profit_pct:>+8.2f}% {r.net_profit_pct:>+8.2f}% "
                  f"{r.net_profit_after_tax_pct:>+8.2f}% {r.sharpe_ratio:>7.2f} {r.max_drawdown_pct:>7.2f}%{marker}")

# Walk-forward validate best candidates
print("\n\n" + "=" * 130)
print("WALK-FORWARD VALIDATION OF BEST CANDIDATES")
print("=" * 130)

for symbol, (best_grid, best_r) in best_results.items():
    params = strategy_cls.DEFAULT_PARAMS.copy()
    params.update(FIXED_PARAMS)
    params.update(best_grid)

    rsi_e = best_grid['rsi_entry']
    rsi_x = best_grid['rsi_exit']
    rsi_p = best_grid['rsi_partial_exit']
    mh = best_grid['max_hold_bars']

    print(f"\n  {symbol} @ {TIMEFRAME}min — rsi_entry={rsi_e}, rsi_exit={rsi_x}, partial={rsi_p}, max_hold={mh}")
    print(f"  Full period: {best_r.total_trades}t, {best_r.win_rate:.1f}% WR, PF {best_r.profit_factor:.2f}, AT {best_r.net_profit_after_tax_pct:+.2f}%")

    train_r, val_r = walk_forward_test(symbol, strategy_cls, params, 'adv2', timeframe_minutes=TIMEFRAME)
    print(f"  TRAIN : {train_r.total_trades:>3}t | WR {train_r.win_rate:>5.1f}% | PF {train_r.profit_factor:>5.2f} | Net {train_r.net_profit_pct:>+7.2f}% | AT {train_r.net_profit_after_tax_pct:>+7.2f}% | Sharpe {train_r.sharpe_ratio:>5.2f} | MaxDD {train_r.max_drawdown_pct:>5.2f}%")
    print(f"  VALID : {val_r.total_trades:>3}t | WR {val_r.win_rate:>5.1f}% | PF {val_r.profit_factor:>5.2f} | Net {val_r.net_profit_pct:>+7.2f}% | AT {val_r.net_profit_after_tax_pct:>+7.2f}% | Sharpe {val_r.sharpe_ratio:>5.2f} | MaxDD {val_r.max_drawdown_pct:>5.2f}%")

    if val_r.net_profit_after_tax_pct > 0:
        print(f"  ** PASSED walk-forward")
    else:
        print(f"  ** FAILED walk-forward")

    # Show trades for context
    print_result(best_r, show_trades=True, indent='    ')

print()
