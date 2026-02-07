#!/usr/bin/env python3
"""
Comprehensive Strategy Optimization - Find the absolute best trading configurations.

Tests ALL 22 symbols × multiple timeframes × both fee tiers × key strategies.
Then optimizes parameters for top performers and walk-forward validates.

Phase 1: Broad sweep with default params
Phase 2: Grid search optimization on top performers
Phase 3: Walk-forward validation on optimized configs
"""

import sys
import os
import json
import time
from datetime import datetime, timezone
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (
    run_backtest, walk_forward_test, grid_search, print_result,
    RSIOnlyStrategy, RSIMeanReversionStrategy, CloseOnlyReversionStrategy,
    RSIRegimeStrategy, EMARSIComboStrategy, StochRSIDivergenceStrategy,
    AdaptiveRSIStrategy, CloseOnlyTrendDipStrategy, CloseOnlyBBReversionStrategy,
    FEE_TIERS, STRATEGY_MAP
)

# All 22 symbols with data
ALL_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD', 'ADA-USD',
    'LTC-USD', 'ATOM-USD', 'DOT-USD', 'NEAR-USD', 'HBAR-USD', 'UNI-USD',
    'AAVE-USD', 'BCH-USD', 'CRV-USD', 'ICP-USD', 'ONDO-USD', 'RENDER-USD',
    'SUI-USD', 'TAO-USD', 'XLM-USD', 'ZEC-USD'
]

# Timeframes to test (in minutes)
TIMEFRAMES = [5, 15, 30, 60, 120]

# Fee tiers to compare
FEE_TIER_LIST = ['adv2', 'adv3']

# Top strategies to test (focused on close-only data proven winners)
STRATEGIES_TO_TEST = {
    'rsi_pure': RSIOnlyStrategy,
    'rsi_regime': RSIRegimeStrategy,
    'co_revert': CloseOnlyReversionStrategy,
    'ema_rsi': EMARSIComboStrategy,
}


def phase1_broad_sweep():
    """Phase 1: Run all symbols × timeframes × strategies with default params."""
    print("=" * 100)
    print("PHASE 1: BROAD SWEEP - All Symbols × Timeframes × Strategies")
    print("=" * 100)

    results = []
    total_combos = len(ALL_SYMBOLS) * len(TIMEFRAMES) * len(STRATEGIES_TO_TEST) * len(FEE_TIER_LIST)
    count = 0
    start_time = time.time()

    for fee_tier in FEE_TIER_LIST:
        tier_info = FEE_TIERS[fee_tier]
        rt_fee = (tier_info['maker'] * 2) * 100
        print(f"\n{'='*80}")
        print(f"Fee Tier: {fee_tier} ({tier_info['name']}) - RT Fee: {rt_fee:.3f}%")
        print(f"{'='*80}")

        for strat_name, strat_cls in STRATEGIES_TO_TEST.items():
            defaults = strat_cls.DEFAULT_PARAMS.copy() if hasattr(strat_cls, 'DEFAULT_PARAMS') else {}

            for tf in TIMEFRAMES:
                for symbol in ALL_SYMBOLS:
                    count += 1
                    result = run_backtest(symbol, strat_cls, defaults, fee_tier,
                                         capital=6000.0, timeframe_minutes=tf)

                    if result.total_trades >= 5:
                        results.append({
                            'symbol': symbol,
                            'strategy': strat_name,
                            'timeframe': tf,
                            'fee_tier': fee_tier,
                            'trades': result.total_trades,
                            'win_rate': result.win_rate,
                            'net_pct': result.net_profit_pct,
                            'after_tax_pct': result.net_profit_after_tax_pct,
                            'profit_factor': result.profit_factor,
                            'sharpe': result.sharpe_ratio,
                            'max_dd': result.max_drawdown_pct,
                            'trades_per_day': result.trades_per_day,
                            'avg_hold_hours': result.avg_hold_hours,
                            'net_usd': result.net_profit_usd,
                            'after_tax_usd': result.net_profit_after_tax_usd,
                        })

                    if count % 50 == 0:
                        elapsed = time.time() - start_time
                        pct = count / total_combos * 100
                        print(f"  Progress: {count}/{total_combos} ({pct:.0f}%) - {elapsed:.0f}s elapsed")

    # Sort by after-tax profit
    results.sort(key=lambda x: x['after_tax_pct'], reverse=True)

    print(f"\n{'='*120}")
    print(f"PHASE 1 RESULTS: {len(results)} profitable+ configs found (sorted by after-tax %)")
    print(f"{'='*120}")
    print(f"{'Rank':<5} {'Strategy':<12} {'Symbol':<12} {'TF':>5} {'Fee':>5} {'Trades':>7} {'Win%':>7} {'Net%':>9} {'AT%':>9} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'$/Day':>8}")
    print("-" * 120)

    for i, r in enumerate(results[:80]):
        net_per_day = r['net_usd'] / max(1, r['trades'] / max(0.001, r['trades_per_day'])) * r['trades_per_day'] if r['trades_per_day'] > 0 else 0
        # Simpler: total net / total days
        total_days = r['trades'] / r['trades_per_day'] if r['trades_per_day'] > 0 else 1
        daily_usd = r['net_usd'] / total_days if total_days > 0 else 0
        print(f"{i+1:<5} {r['strategy']:<12} {r['symbol']:<12} {r['timeframe']:>4}m {r['fee_tier']:>5} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['net_pct']:>+8.2f}% {r['after_tax_pct']:>+8.2f}% {r['profit_factor']:>5.2f} {r['sharpe']:>6.2f} {r['max_dd']:>6.2f}% ${daily_usd:>+7.2f}")

    # Save results to JSON for phase 2
    with open('phase1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Filter to unique symbol/tf/strategy combos that are profitable at adv2
    # (if profitable at adv2, they'll be even better at adv3)
    top_candidates = []
    seen = set()
    for r in results:
        key = (r['strategy'], r['symbol'], r['timeframe'])
        if key not in seen and r['fee_tier'] == 'adv2' and r['after_tax_pct'] > 0:
            seen.add(key)
            top_candidates.append(r)

    # Also include ones that are profitable at adv3 but not adv2
    # (they become viable when you hit adv3)
    for r in results:
        key = (r['strategy'], r['symbol'], r['timeframe'])
        if key not in seen and r['fee_tier'] == 'adv3' and r['after_tax_pct'] > 0:
            seen.add(key)
            top_candidates.append(r)

    top_candidates.sort(key=lambda x: x['after_tax_pct'], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP CANDIDATES FOR OPTIMIZATION: {len(top_candidates)} (profitable after tax)")
    print(f"{'='*100}")
    for i, r in enumerate(top_candidates[:30]):
        print(f"  {i+1}. {r['strategy']} | {r['symbol']} @ {r['timeframe']}min | {r['fee_tier']} | "
              f"AT={r['after_tax_pct']:+.2f}% | WR={r['win_rate']:.1f}% | T={r['trades']} | PF={r['profit_factor']:.2f}")

    return top_candidates


def phase2_optimize(candidates, max_candidates=15):
    """Phase 2: Grid search optimize top candidates."""
    print(f"\n{'='*100}")
    print("PHASE 2: GRID SEARCH OPTIMIZATION ON TOP CANDIDATES")
    print(f"{'='*100}")

    # Extended grid for rsi_pure (the proven winner)
    rsi_pure_grid = {
        'rsi_entry': [18, 20, 22, 25, 28, 30],
        'rsi_exit': [42, 45, 48, 50, 55],
        'rsi_partial_exit': [33, 35, 38, 40, 43],
        'disaster_stop_pct': [1.5, 2.0, 2.5, 3.0, 3.5, 5.0],
        'max_hold_bars': [18, 24, 36, 48, 72],
        'trailing_activate_pct': [0.3, 0.5, 0.7, 1.0],
        'trailing_stop_pct': [0.15, 0.2, 0.3, 0.4],
    }

    rsi_regime_grid = {
        'rsi_entry': [18, 20, 23, 25, 28],
        'rsi_exit': [42, 45, 48, 50],
        'rsi_partial_exit': [33, 35, 38, 40],
        'disaster_stop_pct': [2.0, 3.0, 5.0, 7.0],
        'max_hold_bars': [18, 24, 36, 48],
        'max_below_ema_pct': [3.0, 5.0, 8.0],
        'max_ema_decline_pct': [1.5, 3.0, 5.0],
    }

    co_revert_grid = {
        'rsi_entry': [18, 20, 23, 25, 28, 30],
        'rsi_exit': [42, 45, 48, 50, 55],
        'use_bb_filter': [True, False],
        'profit_target_pct': [0.5, 0.7, 0.9, 1.2, 1.5],
        'stop_loss_pct': [0.5, 0.7, 1.0, 1.5],
        'max_hold_bars': [16, 24, 36, 48],
    }

    ema_rsi_grid = {
        'rsi_entry': [25, 28, 30, 33, 35, 38],
        'rsi_exit': [45, 48, 50, 55, 60],
        'trend_ema': [21, 50],
        'max_hold_bars': [18, 24, 36, 48],
        'min_cooldown_bars': [2, 4, 6],
    }

    grids = {
        'rsi_pure': rsi_pure_grid,
        'rsi_regime': rsi_regime_grid,
        'co_revert': co_revert_grid,
        'ema_rsi': ema_rsi_grid,
    }

    optimized = []

    # Deduplicate: use only the best fee tier per symbol/strategy/tf combo
    seen = set()
    unique_candidates = []
    for c in candidates[:max_candidates]:
        key = (c['strategy'], c['symbol'], c['timeframe'])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for i, candidate in enumerate(unique_candidates[:max_candidates]):
        strat_name = candidate['strategy']
        symbol = candidate['symbol']
        tf = candidate['timeframe']

        if strat_name not in grids:
            continue

        strategy_cls = STRATEGIES_TO_TEST[strat_name]
        grid = grids[strat_name]

        total_combos = 1
        for v in grid.values():
            total_combos *= len(v)

        print(f"\n  [{i+1}/{min(len(unique_candidates), max_candidates)}] Optimizing {strat_name} | {symbol} @ {tf}min ({total_combos} combos)")

        # Test at both fee tiers
        for fee_tier in FEE_TIER_LIST:
            results = grid_search(symbol, strategy_cls, grid, fee_tier,
                                  timeframe_minutes=tf, min_trades=5)

            profitable = [(p, r) for p, r in results if r.net_profit_after_tax_pct > 0]

            if profitable:
                best_params, best_result = profitable[0]
                key_params = {k: v for k, v in best_params.items() if k in grid}

                print(f"    {fee_tier}: AT={best_result.net_profit_after_tax_pct:+.2f}% | "
                      f"WR={best_result.win_rate:.1f}% | PF={best_result.profit_factor:.2f} | "
                      f"T={best_result.total_trades} | Sharpe={best_result.sharpe_ratio:.2f} | "
                      f"MaxDD={best_result.max_drawdown_pct:.2f}%")
                print(f"    Params: {key_params}")

                optimized.append({
                    'strategy': strat_name,
                    'symbol': symbol,
                    'timeframe': tf,
                    'fee_tier': fee_tier,
                    'params': best_params,
                    'key_params': key_params,
                    'trades': best_result.total_trades,
                    'win_rate': best_result.win_rate,
                    'net_pct': best_result.net_profit_pct,
                    'after_tax_pct': best_result.net_profit_after_tax_pct,
                    'profit_factor': best_result.profit_factor,
                    'sharpe': best_result.sharpe_ratio,
                    'max_dd': best_result.max_drawdown_pct,
                    'trades_per_day': best_result.trades_per_day,
                    'net_usd': best_result.net_profit_usd,
                    'after_tax_usd': best_result.net_profit_after_tax_usd,
                })

                # Also show top 3 for parameter stability
                if len(profitable) >= 3:
                    print(f"    Top 3 stability check:")
                    for rank, (p, r) in enumerate(profitable[:3]):
                        kp = {k: v for k, v in p.items() if k in grid}
                        print(f"      #{rank+1}: AT={r.net_profit_after_tax_pct:+.2f}% WR={r.win_rate:.1f}% T={r.total_trades} Params={kp}")
            else:
                print(f"    {fee_tier}: No profitable configurations found")

    # Sort and summarize
    optimized.sort(key=lambda x: x['after_tax_pct'], reverse=True)

    print(f"\n{'='*120}")
    print(f"PHASE 2 OPTIMIZATION RESULTS: {len(optimized)} optimized configs")
    print(f"{'='*120}")
    print(f"{'Rank':<5} {'Strategy':<12} {'Symbol':<12} {'TF':>5} {'Fee':>5} {'Trades':>7} {'Win%':>7} {'Net%':>9} {'AT%':>9} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7}")
    print("-" * 110)

    for i, r in enumerate(optimized[:30]):
        print(f"{i+1:<5} {r['strategy']:<12} {r['symbol']:<12} {r['timeframe']:>4}m {r['fee_tier']:>5} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['net_pct']:>+8.2f}% {r['after_tax_pct']:>+8.2f}% {r['profit_factor']:>5.2f} {r['sharpe']:>6.2f} {r['max_dd']:>6.2f}%")

    # Save for phase 3
    # Convert params to serializable format
    serializable = []
    for o in optimized:
        s = dict(o)
        s['params'] = {k: v for k, v in o['params'].items() if not k.startswith('_')}
        serializable.append(s)

    with open('phase2_results.json', 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    return optimized


def phase3_walk_forward(optimized, max_configs=20):
    """Phase 3: Walk-forward validate the optimized configs."""
    print(f"\n{'='*100}")
    print("PHASE 3: WALK-FORWARD VALIDATION (70/30 Train/Test Split)")
    print(f"{'='*100}")

    validated = []

    # Deduplicate by symbol/strategy/tf (prefer adv2 first for conservative testing)
    seen = set()
    unique = []
    for o in optimized:
        key = (o['strategy'], o['symbol'], o['timeframe'])
        if key not in seen:
            seen.add(key)
            unique.append(o)

    for i, config in enumerate(unique[:max_configs]):
        strat_name = config['strategy']
        symbol = config['symbol']
        tf = config['timeframe']
        params = config['params']
        strategy_cls = STRATEGIES_TO_TEST[strat_name]

        print(f"\n  [{i+1}/{min(len(unique), max_configs)}] Validating {strat_name} | {symbol} @ {tf}min")

        for fee_tier in FEE_TIER_LIST:
            train_result, val_result = walk_forward_test(
                symbol, strategy_cls, params, fee_tier,
                timeframe_minutes=tf)

            train_at = train_result.net_profit_after_tax_pct
            val_at = val_result.net_profit_after_tax_pct

            status = "FAIL"
            if train_result.net_profit_pct > 0 and val_result.net_profit_pct > 0:
                status = "PASS"
            elif train_result.net_profit_pct > 0 and val_result.net_profit_pct > -2:
                status = "MARGINAL"

            print(f"    {fee_tier}: Train={train_at:+.2f}% ({train_result.total_trades}t, WR={train_result.win_rate:.1f}%) | "
                  f"Val={val_at:+.2f}% ({val_result.total_trades}t, WR={val_result.win_rate:.1f}%) | "
                  f"[{status}]")

            if val_result.total_trades >= 3:
                validated.append({
                    'strategy': strat_name,
                    'symbol': symbol,
                    'timeframe': tf,
                    'fee_tier': fee_tier,
                    'params': {k: v for k, v in params.items() if not k.startswith('_')},
                    'key_params': config.get('key_params', {}),
                    'train_trades': train_result.total_trades,
                    'train_win_rate': train_result.win_rate,
                    'train_net_pct': train_result.net_profit_pct,
                    'train_after_tax': train_at,
                    'train_pf': train_result.profit_factor,
                    'train_sharpe': train_result.sharpe_ratio,
                    'val_trades': val_result.total_trades,
                    'val_win_rate': val_result.win_rate,
                    'val_net_pct': val_result.net_profit_pct,
                    'val_after_tax': val_at,
                    'val_pf': val_result.profit_factor,
                    'val_sharpe': val_result.sharpe_ratio,
                    'val_max_dd': val_result.max_drawdown_pct,
                    'status': status,
                    'combined_at': train_at + val_at,
                    'full_period_at': config.get('after_tax_pct', 0),
                })

    # Summary
    passed = [v for v in validated if v['status'] == 'PASS']
    marginal = [v for v in validated if v['status'] == 'MARGINAL']

    passed.sort(key=lambda x: x['val_after_tax'], reverse=True)
    marginal.sort(key=lambda x: x['val_after_tax'], reverse=True)

    print(f"\n{'='*140}")
    print(f"WALK-FORWARD RESULTS: {len(passed)} PASSED | {len(marginal)} MARGINAL | {len(validated) - len(passed) - len(marginal)} FAILED")
    print(f"{'='*140}")

    if passed:
        print(f"\nPASSED CONFIGURATIONS:")
        print(f"{'Rank':<5} {'Strategy':<12} {'Symbol':<12} {'TF':>5} {'Fee':>5} | {'Train':>13} {'Val':>13} | {'Train':>8} {'Val':>8} | {'PF(V)':>6} {'Sharpe(V)':>9} {'MaxDD(V)':>8}")
        print(f"{'':>5} {'':>12} {'':>12} {'':>5} {'':>5} | {'AT%':>13} {'AT%':>13} | {'WR':>8} {'WR':>8} |")
        print("-" * 130)
        for i, v in enumerate(passed):
            print(f"{i+1:<5} {v['strategy']:<12} {v['symbol']:<12} {v['timeframe']:>4}m {v['fee_tier']:>5} | "
                  f"{v['train_after_tax']:>+8.2f}% ({v['train_trades']:>3}t) {v['val_after_tax']:>+8.2f}% ({v['val_trades']:>3}t) | "
                  f"{v['train_win_rate']:>6.1f}% {v['val_win_rate']:>6.1f}% | "
                  f"{v['val_pf']:>5.2f} {v['val_sharpe']:>8.2f} {v['val_max_dd']:>7.2f}%")

    if marginal:
        print(f"\nMARGINAL CONFIGURATIONS (watch-list):")
        for i, v in enumerate(marginal):
            print(f"  {v['strategy']} | {v['symbol']} @ {v['timeframe']}min | {v['fee_tier']} | "
                  f"Train={v['train_after_tax']:+.2f}% Val={v['val_after_tax']:+.2f}%")

    # Save validated results
    with open('phase3_validated.json', 'w') as f:
        json.dump(validated, f, indent=2, default=str)

    return passed, marginal


def phase4_capital_allocation(passed):
    """Phase 4: Compare $6000 single position vs $3000×2 concurrent positions."""
    print(f"\n{'='*100}")
    print("PHASE 4: CAPITAL ALLOCATION COMPARISON")
    print(f"{'='*100}")

    if not passed:
        print("  No passed configurations to analyze.")
        return

    # Get unique symbol/strategy combos for adv2 (current tier)
    adv2_passed = [p for p in passed if p['fee_tier'] == 'adv2']
    adv3_passed = [p for p in passed if p['fee_tier'] == 'adv3']

    for tier_label, tier_passed in [('ADV2 (Current)', adv2_passed), ('ADV3 (Future)', adv3_passed)]:
        if not tier_passed:
            continue

        print(f"\n  --- {tier_label} ---")

        # Single position: best single symbol
        best_single = tier_passed[0] if tier_passed else None

        # Dual position: best 2 non-overlapping symbols
        # (Different symbols or at least different timeframes to avoid conflicts)
        best_dual = None
        for i, p1 in enumerate(tier_passed):
            for p2 in tier_passed[i+1:]:
                if p1['symbol'] != p2['symbol']:
                    combined_daily = 0
                    # Approximate: each gets $3000 so half the return
                    p1_at_3k = p1['val_after_tax'] / 2  # rough: half capital = half USD return
                    p2_at_3k = p2['val_after_tax'] / 2
                    combined = p1_at_3k + p2_at_3k
                    if best_dual is None or combined > best_dual[2]:
                        best_dual = (p1, p2, combined)

        if best_single:
            print(f"\n  OPTION A: Single $6000 Position")
            print(f"    Best: {best_single['strategy']} | {best_single['symbol']} @ {best_single['timeframe']}min")
            print(f"    Val After-Tax: {best_single['val_after_tax']:+.2f}% ({best_single['val_trades']} trades)")
            print(f"    Val Win Rate: {best_single['val_win_rate']:.1f}% | PF: {best_single['val_pf']:.2f}")
            print(f"    Full-period AT: {best_single['full_period_at']:+.2f}%")
            single_usd = best_single['full_period_at'] / 100 * 6000
            print(f"    Est. USD return (full period, $6000): ${single_usd:+.2f}")

        if best_dual:
            p1, p2, _ = best_dual
            print(f"\n  OPTION B: Dual $3000×2 Positions")
            print(f"    Slot 1: {p1['strategy']} | {p1['symbol']} @ {p1['timeframe']}min")
            print(f"      Val AT: {p1['val_after_tax']:+.2f}% ({p1['val_trades']} trades, WR={p1['val_win_rate']:.1f}%)")
            print(f"    Slot 2: {p2['strategy']} | {p2['symbol']} @ {p2['timeframe']}min")
            print(f"      Val AT: {p2['val_after_tax']:+.2f}% ({p2['val_trades']} trades, WR={p2['val_win_rate']:.1f}%)")
            dual_usd = (p1['full_period_at'] + p2['full_period_at']) / 100 * 3000
            print(f"    Est. combined USD return ($3000 each): ${dual_usd:+.2f}")

            # Diversification benefit
            print(f"\n  COMPARISON:")
            print(f"    Single: ${single_usd:+.2f} | Risk: concentrated in one asset")
            print(f"    Dual:   ${dual_usd:+.2f} | Risk: diversified across 2 assets")

            if dual_usd > single_usd:
                print(f"    >>> DUAL is better by ${dual_usd - single_usd:.2f} with diversification benefit")
            else:
                print(f"    >>> SINGLE is better by ${single_usd - dual_usd:.2f} (concentrated but higher return)")

    # Also show: top 5 symbols that could be traded in the future
    print(f"\n  ALL PASSED CONFIGS (sorted by validation after-tax %):")
    for v in passed[:10]:
        print(f"    {v['strategy']} | {v['symbol']} @ {v['timeframe']}min | {v['fee_tier']} | "
              f"Val AT={v['val_after_tax']:+.2f}% | Full AT={v['full_period_at']:+.2f}% | "
              f"WR={v['val_win_rate']:.1f}% | PF={v['val_pf']:.2f}")


def main():
    start = time.time()

    print("=" * 100)
    print("COMPREHENSIVE STRATEGY OPTIMIZATION")
    print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Symbols: {len(ALL_SYMBOLS)} | Timeframes: {TIMEFRAMES} | Strategies: {list(STRATEGIES_TO_TEST.keys())}")
    print(f"Fee Tiers: {FEE_TIER_LIST}")
    print("=" * 100)

    # Phase 1: Broad sweep
    top_candidates = phase1_broad_sweep()
    p1_time = time.time() - start
    print(f"\n  Phase 1 completed in {p1_time:.0f}s")

    # Phase 2: Optimize top performers
    optimized = phase2_optimize(top_candidates, max_candidates=15)
    p2_time = time.time() - start - p1_time
    print(f"\n  Phase 2 completed in {p2_time:.0f}s")

    # Phase 3: Walk-forward validate
    passed, marginal = phase3_walk_forward(optimized, max_configs=20)
    p3_time = time.time() - start - p1_time - p2_time
    print(f"\n  Phase 3 completed in {p3_time:.0f}s")

    # Phase 4: Capital allocation
    phase4_capital_allocation(passed)

    total_time = time.time() - start
    print(f"\n{'='*100}")
    print(f"TOTAL RUNTIME: {total_time:.0f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
