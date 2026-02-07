#!/usr/bin/env python3
"""
Pre-launch walk-forward validation of all 6 active trading configs.
Uses exact parameters from config.json. Tests at both adv2 and adv3 fee tiers.
"""
import sys
import json
from datetime import datetime, timezone

from backtest import (
    RSIOnlyStrategy, RSIRegimeStrategy, CloseOnlyReversionStrategy,
    walk_forward_test, run_backtest, print_result
)

# Load config.json to get exact live parameters
with open('config.json') as f:
    config = json.load(f)

# Build test configs from config.json
CONFIGS = []

# rsi_mean_reversion symbols -> rsi_pure strategy
rsi_mr = config.get('rsi_mean_reversion', {})
if rsi_mr.get('enabled'):
    for symbol, params in rsi_mr.get('symbols', {}).items():
        merged = dict(RSIOnlyStrategy.DEFAULT_PARAMS)
        merged.update(params)
        merged['rsi_period'] = rsi_mr.get('rsi_period', 14)
        CONFIGS.append({
            'symbol': symbol,
            'strategy_name': 'rsi_pure',
            'strategy_cls': RSIOnlyStrategy,
            'timeframe': params['timeframe_minutes'],
            'params': merged,
        })

# rsi_regime symbols
rsi_reg = config.get('rsi_regime', {})
if rsi_reg.get('enabled'):
    for symbol, params in rsi_reg.get('symbols', {}).items():
        merged = dict(RSIRegimeStrategy.DEFAULT_PARAMS)
        merged.update(params)
        merged['rsi_period'] = rsi_reg.get('rsi_period', 14)
        CONFIGS.append({
            'symbol': symbol,
            'strategy_name': 'rsi_regime',
            'strategy_cls': RSIRegimeStrategy,
            'timeframe': params['timeframe_minutes'],
            'params': merged,
        })

# co_revert symbols
co_rev = config.get('co_revert', {})
if co_rev.get('enabled'):
    for symbol, params in co_rev.get('symbols', {}).items():
        merged = dict(CloseOnlyReversionStrategy.DEFAULT_PARAMS)
        merged.update(params)
        merged['rsi_period'] = co_rev.get('rsi_period', 14)
        CONFIGS.append({
            'symbol': symbol,
            'strategy_name': 'co_revert',
            'strategy_cls': CloseOnlyReversionStrategy,
            'timeframe': params['timeframe_minutes'],
            'params': merged,
        })

FEE_TIERS = ['adv2', 'adv3']

print("=" * 100)
print("PRE-LAUNCH WALK-FORWARD VALIDATION")
print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"Configs: {len(CONFIGS)} | Fee tiers: {FEE_TIERS}")
print("=" * 100)

all_pass = True
results_summary = []

for i, cfg in enumerate(CONFIGS, 1):
    symbol = cfg['symbol']
    strat_name = cfg['strategy_name']
    strat_cls = cfg['strategy_cls']
    tf = cfg['timeframe']
    params = cfg['params']

    print(f"\n{'='*100}")
    print(f"[{i}/{len(CONFIGS)}] {symbol} | {strat_name} @ {tf}min")
    print(f"{'='*100}")

    # Key params summary
    key_params = ['rsi_entry', 'rsi_exit', 'disaster_stop_pct', 'trailing_stop_pct']
    if strat_name == 'rsi_regime':
        key_params += ['max_below_ema_pct', 'max_ema_decline_pct']
    elif strat_name == 'co_revert':
        key_params = ['rsi_entry', 'rsi_exit', 'profit_target_pct', 'stop_loss_pct']
    print(f"  Params: {', '.join(f'{k}={params.get(k)}' for k in key_params if k in params)}")

    for fee_tier in FEE_TIERS:
        print(f"\n  --- {fee_tier.upper()} ---")

        # Full period backtest
        full = run_backtest(symbol, strat_cls, params, fee_tier, timeframe_minutes=tf)
        print(f"  Full:  {full.total_trades}t, {full.win_rate:.0f}%WR, PF{full.profit_factor:.2f}, "
              f"AT{full.net_profit_after_tax_pct:+.2f}%, Sharpe{full.sharpe_ratio:.2f}, DD{full.max_drawdown_pct:.2f}%")

        # Walk-forward 70/30
        train, val = walk_forward_test(symbol, strat_cls, params, fee_tier, timeframe_minutes=tf)

        print(f"  Train: {train.total_trades}t, {train.win_rate:.0f}%WR, PF{train.profit_factor:.2f}, "
              f"AT{train.net_profit_after_tax_pct:+.2f}%")
        print(f"  Valid: {val.total_trades}t, {val.win_rate:.0f}%WR, PF{val.profit_factor:.2f}, "
              f"AT{val.net_profit_after_tax_pct:+.2f}%, Sharpe{val.sharpe_ratio:.2f}, DD{val.max_drawdown_pct:.2f}%")

        # PASS/FAIL criteria
        passed = (
            val.net_profit_after_tax_pct > 0
            and val.total_trades >= 5
            and val.win_rate >= 50
        )

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  >> {status} {'<<' if not passed else ''}")

        results_summary.append({
            'symbol': symbol,
            'strategy': strat_name,
            'timeframe': tf,
            'fee_tier': fee_tier,
            'full_trades': full.total_trades,
            'full_at': full.net_profit_after_tax_pct,
            'val_trades': val.total_trades,
            'val_wr': val.win_rate,
            'val_pf': val.profit_factor,
            'val_at': val.net_profit_after_tax_pct,
            'val_sharpe': val.sharpe_ratio,
            'val_dd': val.max_drawdown_pct,
            'passed': passed,
        })

# Summary table
print(f"\n\n{'='*120}")
print("SUMMARY")
print(f"{'='*120}")
print(f"{'Symbol':<12} {'Strategy':<16} {'TF':<6} {'Tier':<6} {'Full AT%':<10} {'Val Trades':<11} {'Val WR%':<9} {'Val PF':<8} {'Val AT%':<10} {'Sharpe':<8} {'MaxDD%':<8} {'Result':<8}")
print("-" * 120)

for r in results_summary:
    status_str = "PASS" if r['passed'] else "** FAIL **"
    print(f"{r['symbol']:<12} {r['strategy']:<16} {r['timeframe']:<6} {r['fee_tier']:<6} "
          f"{r['full_at']:>+8.2f}%  {r['val_trades']:<11} {r['val_wr']:<9.0f} {r['val_pf']:<8.2f} "
          f"{r['val_at']:>+8.2f}%  {r['val_sharpe']:<8.2f} {r['val_dd']:<8.2f} {status_str}")

print(f"\n{'='*120}")
if all_pass:
    print("ALL CONFIGS PASS - Ready to trade!")
else:
    failing = [r for r in results_summary if not r['passed']]
    print(f"WARNING: {len(failing)} config(s) FAILED walk-forward validation:")
    for r in failing:
        print(f"  - {r['symbol']} {r['strategy']} @ {r['timeframe']}min ({r['fee_tier']}): Val AT {r['val_at']:+.2f}%, {r['val_trades']}t, {r['val_wr']:.0f}%WR")
print(f"{'='*120}")
