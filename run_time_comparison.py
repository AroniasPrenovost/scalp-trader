#!/usr/bin/env python3
"""Compare backtest results at different data cutoffs to find when strategy broke."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timezone
from backtest import run_backtest, STRATEGY_MAP

strategy_cls = STRATEGY_MAP['rsi_pure']

SYMBOL_CONFIGS = {
    'ATOM-USD': {
        'timeframe_minutes': 15,
        'rsi_period': 14, 'rsi_entry': 20, 'rsi_exit': 48,
        'rsi_partial_exit': 35, 'disaster_stop_pct': 5.0,
        'max_hold_bars': 24, 'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2, 'min_cooldown_bars': 3,
    },
    'LINK-USD': {
        'timeframe_minutes': 120,
        'rsi_period': 14, 'rsi_entry': 20, 'rsi_exit': 45,
        'rsi_partial_exit': 33, 'disaster_stop_pct': 5.0,
        'max_hold_bars': 12, 'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2,
    },
    'HBAR-USD': {
        'timeframe_minutes': 120,
        'rsi_period': 14, 'rsi_entry': 25, 'rsi_exit': 50,
        'rsi_partial_exit': 38, 'disaster_stop_pct': 5.0,
        'max_hold_bars': 12, 'trailing_activate_pct': 0.3,
        'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2,
    },
}

# Test at different data cutoff dates
CUTOFF_DATES = [
    ('2025-11-01', 'Nov 1 (early)'),
    ('2025-12-01', 'Dec 1'),
    ('2025-12-15', 'Dec 15'),
    ('2026-01-01', 'Jan 1'),
    ('2026-01-15', 'Jan 15'),
    ('2026-01-25', 'Jan 25'),
    ('2026-02-06', 'Feb 6 (now)'),
]

print("=" * 130)
print("BACKTEST RESULTS BY DATA CUTOFF DATE")
print("=" * 130)

for symbol, sym_config in SYMBOL_CONFIGS.items():
    tf = sym_config['timeframe_minutes']
    params = strategy_cls.DEFAULT_PARAMS.copy()
    params.update(sym_config)

    print(f"\n{'='*130}")
    print(f"  {symbol} @ {tf}min")
    print(f"{'='*130}")
    print(f"{'Cutoff':<18} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Gross%':>9} {'Fees%':>8} {'Net%':>9} {'AT%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'AvgWin':>8} {'AvgLoss':>9}")
    print("-" * 130)

    for date_str, label in CUTOFF_DATES:
        end_ts = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()
        result = run_backtest(symbol, strategy_cls, params, 'adv2',
                              end_ts=end_ts, timeframe_minutes=tf)
        print(f"{label:<18} {result.total_trades:>7} {result.win_rate:>6.1f}% {result.profit_factor:>6.2f} "
              f"{result.gross_profit_pct:>+8.2f}% {result.total_fees_pct:>7.2f}% {result.net_profit_pct:>+8.2f}% "
              f"{result.net_profit_after_tax_pct:>+8.2f}% {result.sharpe_ratio:>7.2f} {result.max_drawdown_pct:>7.2f}% "
              f"{result.avg_win_pct:>+7.3f}% {result.avg_loss_pct:>+8.3f}%")

print()
