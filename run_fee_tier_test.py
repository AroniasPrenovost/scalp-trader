#!/usr/bin/env python3
"""Backtest + walk-forward for active RSI symbols with correct per-symbol params."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (
    run_backtest, walk_forward_test, print_result, FEE_TIERS, STRATEGY_MAP
)

# Per-symbol optimized params from config.json
SYMBOL_CONFIGS = {
    'ATOM-USD': {
        'timeframe_minutes': 15,
        'rsi_period': 14,
        'rsi_entry': 20,
        'rsi_exit': 48,
        'rsi_partial_exit': 35,
        'disaster_stop_pct': 5.0,
        'max_hold_bars': 24,
        'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2,
        'min_cooldown_bars': 3,
    },
    'LINK-USD': {
        'timeframe_minutes': 120,
        'rsi_period': 14,
        'rsi_entry': 20,
        'rsi_exit': 45,
        'rsi_partial_exit': 33,
        'disaster_stop_pct': 5.0,
        'max_hold_bars': 12,
        'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2,
        'min_cooldown_bars': 2,
    },
    'HBAR-USD': {
        'timeframe_minutes': 120,
        'rsi_period': 14,
        'rsi_entry': 25,
        'rsi_exit': 50,
        'rsi_partial_exit': 38,
        'disaster_stop_pct': 5.0,
        'max_hold_bars': 12,
        'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2,
        'min_cooldown_bars': 2,
    },
}

strategy_cls = STRATEGY_MAP['rsi_pure']

print("=" * 90)
print("FULL-PERIOD BACKTEST + WALK-FORWARD: Per-Symbol Optimized Params @ adv2")
print("=" * 90)

for symbol, sym_config in SYMBOL_CONFIGS.items():
    tf = sym_config['timeframe_minutes']
    params = strategy_cls.DEFAULT_PARAMS.copy()
    params.update(sym_config)

    print(f"\n{'='*90}")
    print(f"  {symbol} @ {tf}min  (rsi_entry={sym_config['rsi_entry']}, rsi_exit={sym_config['rsi_exit']}, "
          f"rsi_partial={sym_config['rsi_partial_exit']}, disaster={sym_config['disaster_stop_pct']}%)")
    print(f"{'='*90}")

    # Full-period backtest
    print(f"\n  --- FULL PERIOD ---")
    full_result = run_backtest(symbol, strategy_cls, params, 'adv2', timeframe_minutes=tf)
    print_result(full_result, show_trades=True)

    # Walk-forward 70/30
    print(f"\n  --- WALK-FORWARD (70/30 split) ---")
    train_result, val_result = walk_forward_test(symbol, strategy_cls, params, 'adv2', timeframe_minutes=tf)
    print(f"  TRAIN : {train_result.total_trades:>3}t | WR {train_result.win_rate:>5.1f}% | PF {train_result.profit_factor:>5.2f} | Net {train_result.net_profit_pct:>+7.2f}% | AT {train_result.net_profit_after_tax_pct:>+7.2f}% | Sharpe {train_result.sharpe_ratio:>5.2f} | MaxDD {train_result.max_drawdown_pct:>5.2f}%")
    print(f"  VALID : {val_result.total_trades:>3}t | WR {val_result.win_rate:>5.1f}% | PF {val_result.profit_factor:>5.2f} | Net {val_result.net_profit_pct:>+7.2f}% | AT {val_result.net_profit_after_tax_pct:>+7.2f}% | Sharpe {val_result.sharpe_ratio:>5.2f} | MaxDD {val_result.max_drawdown_pct:>5.2f}%")
    if val_result.net_profit_after_tax_pct > 0:
        print(f"  ** PASSED")
    else:
        print(f"  ** FAILED validation")

print()
