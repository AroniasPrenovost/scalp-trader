#!/usr/bin/env python3
"""
Targeted optimization on the top candidates from Phase 1.
Smaller grids focused on the winning symbol/timeframe/strategy combos.
"""

import sys
import os
import json
import time
from datetime import datetime, timezone
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (
    run_backtest, walk_forward_test, grid_search,
    RSIOnlyStrategy, RSIMeanReversionStrategy, CloseOnlyReversionStrategy,
    RSIRegimeStrategy, EMARSIComboStrategy,
    FEE_TIERS, STRATEGY_MAP
)

# Top candidates from Phase 1 (adv2 profitable, sorted by AT%)
TOP_CONFIGS = [
    # (strategy_name, strategy_cls, symbol, timeframe)
    ('rsi_pure', RSIOnlyStrategy, 'ZEC-USD', 15),
    ('co_revert', CloseOnlyReversionStrategy, 'ZEC-USD', 15),
    ('rsi_pure', RSIOnlyStrategy, 'NEAR-USD', 30),
    ('co_revert', CloseOnlyReversionStrategy, 'NEAR-USD', 30),
    ('rsi_pure', RSIOnlyStrategy, 'TAO-USD', 15),
    ('rsi_pure', RSIOnlyStrategy, 'TAO-USD', 30),
    ('rsi_regime', RSIRegimeStrategy, 'ICP-USD', 5),
    ('rsi_pure', RSIOnlyStrategy, 'XLM-USD', 60),
    ('rsi_pure', RSIOnlyStrategy, 'RENDER-USD', 120),
    ('co_revert', CloseOnlyReversionStrategy, 'CRV-USD', 30),
    ('co_revert', CloseOnlyReversionStrategy, 'HBAR-USD', 5),
    ('rsi_regime', RSIRegimeStrategy, 'NEAR-USD', 15),
    ('rsi_pure', RSIOnlyStrategy, 'LTC-USD', 60),
    ('rsi_regime', RSIRegimeStrategy, 'ATOM-USD', 30),
    ('rsi_pure', RSIOnlyStrategy, 'DOT-USD', 30),
    ('co_revert', CloseOnlyReversionStrategy, 'AAVE-USD', 30),
    ('rsi_pure', RSIOnlyStrategy, 'ZEC-USD', 120),
    ('co_revert', CloseOnlyReversionStrategy, 'ICP-USD', 30),
]

# Focused grids (smaller than comprehensive, ~2000-5000 combos each)
GRIDS = {
    'rsi_pure': {
        'rsi_entry': [20, 25, 28],
        'rsi_exit': [48, 50],
        'rsi_partial_exit': [38, 40],
        'disaster_stop_pct': [2.5, 3.5, 5.0],
        'max_hold_bars': [36, 48],
        'trailing_activate_pct': [0.3, 0.5],
        'trailing_stop_pct': [0.2, 0.3],
    },  # 288 combos
    'co_revert': {
        'rsi_entry': [23, 25, 28],
        'rsi_exit': [45, 50],
        'use_bb_filter': [True, False],
        'profit_target_pct': [0.8, 1.0, 1.5],
        'stop_loss_pct': [0.8, 1.0, 1.5],
        'max_hold_bars': [24, 36, 48],
    },  # 324 combos
    'rsi_regime': {
        'rsi_entry': [20, 23, 25],
        'rsi_exit': [45, 50],
        'rsi_partial_exit': [35, 40],
        'disaster_stop_pct': [3.0, 5.0],
        'max_hold_bars': [24, 36],
        'max_below_ema_pct': [5.0, 8.0],
        'max_ema_decline_pct': [3.0, 5.0],
    },  # 192 combos
    'ema_rsi': {
        'rsi_entry': [30, 33, 35],
        'rsi_exit': [50, 55],
        'trend_ema': [21, 50],
        'max_hold_bars': [24, 36],
        'min_cooldown_bars': [3, 5],
    },  # 48 combos
}

FEE_TIER_LIST = ['adv2', 'adv3']


def optimize_and_validate():
    start = time.time()

    print("=" * 120)
    print("TARGETED OPTIMIZATION + WALK-FORWARD VALIDATION")
    print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Top configs: {len(TOP_CONFIGS)} | Fee Tiers: {FEE_TIER_LIST}")
    print("=" * 120)

    all_results = []

    for idx, (strat_name, strat_cls, symbol, tf) in enumerate(TOP_CONFIGS):
        grid = GRIDS.get(strat_name)
        if not grid:
            continue

        total_combos = 1
        for v in grid.values():
            total_combos *= len(v)

        print(f"\n{'='*100}")
        print(f"[{idx+1}/{len(TOP_CONFIGS)}] {strat_name} | {symbol} @ {tf}min ({total_combos} combos)")
        print(f"{'='*100}")

        for fee_tier in FEE_TIER_LIST:
            tier_info = FEE_TIERS[fee_tier]
            rt_fee = (tier_info['maker'] * 2) * 100

            # Grid search
            results = grid_search(symbol, strat_cls, grid, fee_tier,
                                  timeframe_minutes=tf, min_trades=5)

            profitable = [(p, r) for p, r in results if r.net_profit_after_tax_pct > 0]

            if not profitable:
                print(f"  {fee_tier} (RT={rt_fee:.3f}%): No profitable configs")
                continue

            # Show top 5 for parameter stability
            print(f"\n  {fee_tier} (RT={rt_fee:.3f}%): {len(profitable)}/{len(results)} profitable")
            print(f"  {'Rank':<5} {'AT%':>9} {'Net%':>9} {'WR':>7} {'Trades':>7} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7} | Key Params")
            print(f"  {'-'*90}")

            for rank, (params, result) in enumerate(profitable[:5]):
                key_p = {k: v for k, v in params.items() if k in grid}
                print(f"  {rank+1:<5} {result.net_profit_after_tax_pct:>+8.2f}% {result.net_profit_pct:>+8.2f}% "
                      f"{result.win_rate:>6.1f}% {result.total_trades:>7} {result.profit_factor:>5.2f} "
                      f"{result.sharpe_ratio:>6.2f} {result.max_drawdown_pct:>6.2f}% | {key_p}")

            # Walk-forward validate top 3
            print(f"\n  Walk-Forward Validation (70/30 split):")
            for rank, (params, result) in enumerate(profitable[:3]):
                train_r, val_r = walk_forward_test(symbol, strat_cls, params, fee_tier,
                                                    timeframe_minutes=tf)

                status = "FAIL"
                if train_r.net_profit_pct > 0 and val_r.net_profit_pct > 0:
                    status = "PASS"
                elif train_r.net_profit_pct > 0 and val_r.net_profit_pct > -1:
                    status = "MARGINAL"

                key_p = {k: v for k, v in params.items() if k in grid}
                print(f"  #{rank+1} [{status}] Train={train_r.net_profit_after_tax_pct:>+6.2f}% ({train_r.total_trades}t, WR={train_r.win_rate:.0f}%) "
                      f"Val={val_r.net_profit_after_tax_pct:>+6.2f}% ({val_r.total_trades}t, WR={val_r.win_rate:.0f}%) "
                      f"PF(V)={val_r.profit_factor:.2f} Sharpe(V)={val_r.sharpe_ratio:.2f}")

                all_results.append({
                    'strategy': strat_name,
                    'symbol': symbol,
                    'timeframe': tf,
                    'fee_tier': fee_tier,
                    'opt_rank': rank + 1,
                    'full_at_pct': result.net_profit_after_tax_pct,
                    'full_net_pct': result.net_profit_pct,
                    'full_trades': result.total_trades,
                    'full_wr': result.win_rate,
                    'full_pf': result.profit_factor,
                    'full_sharpe': result.sharpe_ratio,
                    'full_max_dd': result.max_drawdown_pct,
                    'full_net_usd_6k': (result.net_profit_pct / 100) * 6000,
                    'full_at_usd_6k': (result.net_profit_after_tax_pct / 100) * 6000,
                    'train_at_pct': train_r.net_profit_after_tax_pct,
                    'train_trades': train_r.total_trades,
                    'train_wr': train_r.win_rate,
                    'train_pf': train_r.profit_factor,
                    'val_at_pct': val_r.net_profit_after_tax_pct,
                    'val_trades': val_r.total_trades,
                    'val_wr': val_r.win_rate,
                    'val_pf': val_r.profit_factor,
                    'val_sharpe': val_r.sharpe_ratio,
                    'val_max_dd': val_r.max_drawdown_pct,
                    'val_net_usd_6k': (val_r.net_profit_pct / 100) * 6000,
                    'status': status,
                    'key_params': key_p,
                    'full_params': {k: v for k, v in params.items() if not k.startswith('_')},
                })

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    passed = [r for r in all_results if r['status'] == 'PASS']
    marginal = [r for r in all_results if r['status'] == 'MARGINAL']

    passed.sort(key=lambda x: x['val_at_pct'], reverse=True)
    marginal.sort(key=lambda x: x['val_at_pct'], reverse=True)

    print(f"\n{'='*140}")
    print(f"FINAL RESULTS: {len(passed)} PASSED | {len(marginal)} MARGINAL | {len(all_results) - len(passed) - len(marginal)} FAILED")
    print(f"{'='*140}")

    if passed:
        print(f"\n*** WALK-FORWARD VALIDATED CONFIGS (sorted by validation after-tax %) ***")
        print(f"{'#':<3} {'Strategy':<12} {'Symbol':<12} {'TF':>5} {'Fee':>5} | {'Full AT%':>9} {'T':>4} {'WR':>6} | {'Train AT%':>10} {'Val AT%':>9} {'Val T':>6} {'Val WR':>7} {'Val PF':>7} | {'$6K AT':>8}")
        print("-" * 130)
        for i, r in enumerate(passed[:25]):
            print(f"{i+1:<3} {r['strategy']:<12} {r['symbol']:<12} {r['timeframe']:>4}m {r['fee_tier']:>5} | "
                  f"{r['full_at_pct']:>+8.2f}% {r['full_trades']:>4} {r['full_wr']:>5.1f}% | "
                  f"{r['train_at_pct']:>+9.2f}% {r['val_at_pct']:>+8.2f}% {r['val_trades']:>5} {r['val_wr']:>6.1f}% {r['val_pf']:>6.2f} | "
                  f"${r['full_at_usd_6k']:>+7.0f}")

    if marginal:
        print(f"\nMARGINAL CONFIGS (watch-list):")
        for i, r in enumerate(marginal[:10]):
            print(f"  {r['strategy']} | {r['symbol']} @ {r['timeframe']}min | {r['fee_tier']} | "
                  f"Full AT={r['full_at_pct']:+.2f}% | Train={r['train_at_pct']:+.2f}% Val={r['val_at_pct']:+.2f}%")

    # =========================================================================
    # CAPITAL ALLOCATION COMPARISON
    # =========================================================================
    print(f"\n{'='*100}")
    print("CAPITAL ALLOCATION ANALYSIS")
    print(f"{'='*100}")

    for tier in FEE_TIER_LIST:
        tier_passed = [p for p in passed if p['fee_tier'] == tier]
        if not tier_passed:
            continue

        tier_info = FEE_TIERS[tier]
        print(f"\n  --- {tier.upper()} ({tier_info['name']}) ---")

        # Option A: Single $6000 best performer
        best = tier_passed[0]
        single_usd = best['full_at_usd_6k']
        print(f"\n  OPTION A: Single $6,000 position")
        print(f"    {best['strategy']} | {best['symbol']} @ {best['timeframe']}min")
        print(f"    Full period AT: {best['full_at_pct']:+.2f}% = ${single_usd:+.0f}")
        print(f"    Validation AT: {best['val_at_pct']:+.2f}% | WR: {best['val_wr']:.1f}% | PF: {best['val_pf']:.2f}")
        print(f"    Params: {best['key_params']}")

        # Option B: Best dual $3000×2 (different symbols)
        best_dual = None
        best_dual_usd = 0
        for i, p1 in enumerate(tier_passed):
            for p2 in tier_passed[i+1:]:
                if p1['symbol'] != p2['symbol']:
                    dual_usd = (p1['full_at_pct'] + p2['full_at_pct']) / 100 * 3000
                    if dual_usd > best_dual_usd:
                        best_dual = (p1, p2)
                        best_dual_usd = dual_usd

        if best_dual:
            p1, p2 = best_dual
            print(f"\n  OPTION B: Dual $3,000 × 2 positions")
            print(f"    Slot 1: {p1['strategy']} | {p1['symbol']} @ {p1['timeframe']}min")
            print(f"      Val AT: {p1['val_at_pct']:+.2f}% | WR: {p1['val_wr']:.1f}%")
            print(f"      Params: {p1['key_params']}")
            print(f"    Slot 2: {p2['strategy']} | {p2['symbol']} @ {p2['timeframe']}min")
            print(f"      Val AT: {p2['val_at_pct']:+.2f}% | WR: {p2['val_wr']:.1f}%")
            print(f"      Params: {p2['key_params']}")
            print(f"    Combined AT: ${best_dual_usd:+.0f}")

            if best_dual_usd > single_usd:
                print(f"\n    >>> DUAL is better by ${best_dual_usd - single_usd:.0f} with diversification")
            else:
                print(f"\n    >>> SINGLE is better by ${single_usd - best_dual_usd:.0f} (concentrated)")

        # Option C: Top 3 with $2000 each
        top3 = []
        syms_used = set()
        for p in tier_passed:
            if p['symbol'] not in syms_used:
                top3.append(p)
                syms_used.add(p['symbol'])
            if len(top3) >= 3:
                break

        if len(top3) >= 3:
            triple_usd = sum(p['full_at_pct'] for p in top3) / 100 * 2000
            print(f"\n  OPTION C: Triple $2,000 × 3 positions")
            for j, p in enumerate(top3):
                print(f"    Slot {j+1}: {p['strategy']} | {p['symbol']} @ {p['timeframe']}min | Val AT: {p['val_at_pct']:+.2f}%")
            print(f"    Combined AT: ${triple_usd:+.0f}")

    # =========================================================================
    # ADV3 UPGRADE IMPACT
    # =========================================================================
    print(f"\n{'='*100}")
    print("ADV3 UPGRADE IMPACT (fee savings projection)")
    print(f"{'='*100}")

    # Find matching configs at both tiers
    adv2_map = {}
    adv3_map = {}
    for r in all_results:
        key = (r['strategy'], r['symbol'], r['timeframe'], r['opt_rank'])
        if r['fee_tier'] == 'adv2':
            adv2_map[key] = r
        else:
            adv3_map[key] = r

    for key in sorted(adv2_map.keys()):
        if key in adv3_map:
            r2 = adv2_map[key]
            r3 = adv3_map[key]
            improvement = r3['full_at_pct'] - r2['full_at_pct']
            if r2['status'] == 'PASS' or r3['status'] == 'PASS':
                usd_gain = improvement / 100 * 6000
                print(f"  {r2['strategy']:<12} {r2['symbol']:<12} {r2['timeframe']:>3}m #{r2['opt_rank']} | "
                      f"adv2={r2['full_at_pct']:>+7.2f}% adv3={r3['full_at_pct']:>+7.2f}% | "
                      f"Gain: {improvement:>+5.2f}% (${usd_gain:>+.0f}/period) | "
                      f"adv2={r2['status']:<8} adv3={r3['status']}")

    # Save all results
    with open('optimization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - start
    print(f"\n{'='*100}")
    print(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*100}")


if __name__ == '__main__':
    optimize_and_validate()
