#!/usr/bin/env python3
"""
Scan new symbols for RSI mean reversion viability.

Backfills 210 days of 5-minute candles from Coinbase, then runs
walk-forward validation at multiple timeframes.

Usage:
    python3 scan_new_symbols.py
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from coinbase.rest import RESTClient

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (
    run_backtest,
    RSIOnlyStrategy,
    load_symbol_data,
    BacktestResult,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Symbols to scan (top Coinbase products by volume, excluding already tested/meme/stablecoin)
SCAN_SYMBOLS = [
    'ZEC-USD',      # #7  - Zcash, privacy coin, $84M vol
    'SUI-USD',      # #8  - SUI, newer L1, $83M vol
    'HBAR-USD',     # #11 - Hedera, enterprise, $54M vol
    'XLM-USD',      # #13 - Stellar, payments, $24M vol
    'TAO-USD',      # #15 - Bittensor, AI, $22M vol
    'AAVE-USD',     # #16 - AAVE, DeFi lending, $22M vol
    'BCH-USD',      # #17 - Bitcoin Cash, $21M vol
    'ONDO-USD',     # #18 - Ondo, RWA, $21M vol
    'UNI-USD',      # #30 - Uniswap, DeFi, $10M vol
    'DOT-USD',      # #31 - Polkadot, interop, $10M vol
    'NEAR-USD',     # #33 - NEAR Protocol, L1, $9M vol
    'ICP-USD',      # #35 - Internet Computer, $9M vol
    'CRV-USD',      # #36 - Curve, DeFi, $8M vol
    'RENDER-USD',   # #37 - Render, AI/computing, $7M vol
]

# Already tested (skip these)
ALREADY_TESTED = {'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'ADA-USD', 'LTC-USD', 'ATOM-USD', 'LINK-USD'}

DAYS_BACK = 210
FEE_TIER = 'adv2'
CAPITAL = 6000.0

# Timeframes to test (minutes)
TIMEFRAMES = [15, 30, 60, 120]

# RSI parameter sets to test at each timeframe
PARAM_SETS = {
    15: [
        {'rsi_period': 14, 'rsi_entry': 20, 'rsi_exit': 48, 'rsi_partial_exit': 35,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 24, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 3},
        {'rsi_period': 14, 'rsi_entry': 22, 'rsi_exit': 48, 'rsi_partial_exit': 35,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 24, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 3},
        {'rsi_period': 14, 'rsi_entry': 25, 'rsi_exit': 50, 'rsi_partial_exit': 38,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 24, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 3},
    ],
    30: [
        {'rsi_period': 14, 'rsi_entry': 20, 'rsi_exit': 48, 'rsi_partial_exit': 35,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 18, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 3},
        {'rsi_period': 14, 'rsi_entry': 22, 'rsi_exit': 45, 'rsi_partial_exit': 33,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 18, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2},
        {'rsi_period': 14, 'rsi_entry': 25, 'rsi_exit': 50, 'rsi_partial_exit': 38,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 18, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2},
    ],
    60: [
        {'rsi_period': 14, 'rsi_entry': 20, 'rsi_exit': 48, 'rsi_partial_exit': 35,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 15, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2},
        {'rsi_period': 14, 'rsi_entry': 22, 'rsi_exit': 45, 'rsi_partial_exit': 33,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 15, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2},
        {'rsi_period': 14, 'rsi_entry': 25, 'rsi_exit': 50, 'rsi_partial_exit': 38,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 15, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2},
    ],
    120: [
        {'rsi_period': 14, 'rsi_entry': 20, 'rsi_exit': 45, 'rsi_partial_exit': 33,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 12, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2},
        {'rsi_period': 14, 'rsi_entry': 22, 'rsi_exit': 48, 'rsi_partial_exit': 35,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 12, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2},
        {'rsi_period': 14, 'rsi_entry': 25, 'rsi_exit': 50, 'rsi_partial_exit': 38,
         'disaster_stop_pct': 5.0, 'max_hold_bars': 12, 'trailing_activate_pct': 0.3,
         'trailing_stop_pct': 0.2, 'min_cooldown_bars': 2},
    ],
}

# Walk-forward settings (same as weekly_validation.py)
WF_WINDOWS = 3
WF_WINDOW_DAYS = 30
WF_STEP_DAYS = 15


# ==============================================================================
# BACKFILL
# ==============================================================================

def fetch_candles(client, product_id, start_time, end_time):
    """Fetch 5-min candles from Coinbase API."""
    try:
        response = client.get_candles(
            product_id=product_id,
            start=int(start_time),
            end=int(end_time),
            granularity="FIVE_MINUTE"
        )
        if hasattr(response, 'to_dict'):
            data = response.to_dict()
        elif hasattr(response, '__dict__'):
            data = response.__dict__
        else:
            data = response
        return data.get('candles', [])
    except Exception as e:
        print(f"    ERROR fetching {product_id}: {e}")
        return []


def backfill_symbol(client, symbol, days_back=210):
    """Backfill 5-min candles for a symbol. Returns number of candles saved."""
    data_file = f"coinbase-data/{symbol}.json"

    # Load existing data
    existing = []
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    if len(existing) > 50000:
        print(f"  {symbol}: Already have {len(existing)} candles, skipping backfill")
        return len(existing)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    all_data = []
    current_start = start_time
    chunk = 0

    while current_start < end_time:
        chunk += 1
        current_end = min(current_start + timedelta(minutes=1500), end_time)

        candles = fetch_candles(client, symbol,
                                current_start.timestamp(), current_end.timestamp())

        for c in candles:
            if isinstance(c, dict):
                ts = c.get('start')
                close = c.get('close')
                vol = c.get('volume')
            else:
                ts = getattr(c, 'start', None)
                close = getattr(c, 'close', None)
                vol = getattr(c, 'volume', None)

            if ts and close:
                all_data.append({
                    'timestamp': float(ts),
                    'product_id': symbol,
                    'price': str(close),
                    'volume_24h': str(vol) if vol else "0"
                })

        current_start = current_end
        if current_start < end_time:
            time.sleep(0.3)

        # Print progress every 20 chunks
        if chunk % 20 == 0:
            print(f"    {symbol}: chunk {chunk}, {len(all_data)} candles so far...")

    # Merge with existing
    data_dict = {}
    for p in existing:
        ts = p.get('timestamp')
        if ts:
            data_dict[ts] = p
    for p in all_data:
        ts = p.get('timestamp')
        if ts:
            data_dict[ts] = p

    merged = sorted(data_dict.values(), key=lambda x: x.get('timestamp', 0))

    os.makedirs('coinbase-data', exist_ok=True)
    with open(data_file, 'w') as f:
        json.dump(merged, f, indent=4)

    new_count = len(merged) - len(existing)
    print(f"  {symbol}: {len(merged)} total candles ({new_count} new)")
    return len(merged)


# ==============================================================================
# WALK-FORWARD VALIDATION
# ==============================================================================

def get_wf_windows(symbol):
    """Generate rolling walk-forward test windows."""
    timestamps, _, _, _, closes, _ = load_symbol_data(symbol)
    if not timestamps or len(timestamps) < 100:
        return []

    data_end = timestamps[-1]
    data_start = timestamps[0]
    total_days = (data_end - data_start) / 86400.0

    if total_days < WF_WINDOW_DAYS * 2:
        return []

    windows = []
    for i in range(WF_WINDOWS):
        end_ts = data_end - (i * WF_STEP_DAYS * 86400)
        start_ts = end_ts - (WF_WINDOW_DAYS * 86400)
        if start_ts < data_start:
            break
        windows.append((start_ts, end_ts))

    windows.reverse()
    return windows


def validate_symbol(symbol, timeframe, params):
    """
    Run full backtest + walk-forward validation.
    Returns dict with results or None if insufficient data.
    """
    try:
        full = run_backtest(
            symbol=symbol,
            strategy_cls=RSIOnlyStrategy,
            params=params,
            fee_tier=FEE_TIER,
            capital=CAPITAL,
            timeframe_minutes=timeframe
        )
    except Exception as e:
        return None

    if full.total_trades == 0:
        return None

    # Walk-forward windows
    windows = get_wf_windows(symbol)
    if not windows:
        return None

    all_pass = True
    window_results = []
    for start_ts, end_ts in windows:
        try:
            wf = run_backtest(
                symbol=symbol,
                strategy_cls=RSIOnlyStrategy,
                params=params,
                fee_tier=FEE_TIER,
                capital=CAPITAL,
                start_ts=start_ts,
                end_ts=end_ts,
                timeframe_minutes=timeframe
            )
        except:
            all_pass = False
            window_results.append(None)
            continue

        passed = wf.total_trades == 0 or wf.net_profit_after_tax_pct >= 0
        if not passed:
            all_pass = False
        window_results.append(wf)

    return {
        'full': full,
        'windows': window_results,
        'all_pass': all_pass,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 75)
    print("  RSI MEAN REVERSION - NEW SYMBOL SCANNER")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 75)

    # Filter out already tested symbols
    symbols = [s for s in SCAN_SYMBOLS if s not in ALREADY_TESTED]
    print(f"\nScanning {len(symbols)} new symbols:")
    for s in symbols:
        print(f"  - {s}")

    # Init Coinbase client
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    if not api_key or not api_secret:
        print("\nERROR: Missing COINBASE_API_KEY/COINBASE_API_SECRET in .env")
        return

    client = RESTClient(api_key=api_key, api_secret=api_secret)

    # Phase 1: Backfill
    print(f"\n{'=' * 75}")
    print("  PHASE 1: BACKFILLING {0} DAYS OF 5-MIN CANDLES".format(DAYS_BACK))
    print(f"{'=' * 75}")

    for symbol in symbols:
        try:
            count = backfill_symbol(client, symbol, DAYS_BACK)
            if count < 1000:
                print(f"  WARNING: Only {count} candles for {symbol}, may be insufficient")
        except Exception as e:
            print(f"  ERROR backfilling {symbol}: {e}")

    # Phase 2: Walk-forward validation
    print(f"\n{'=' * 75}")
    print("  PHASE 2: WALK-FORWARD VALIDATION")
    print(f"{'=' * 75}")

    results = []  # (symbol, timeframe, params, validation_result)

    for symbol in symbols:
        print(f"\n  --- {symbol} ---")
        for tf in TIMEFRAMES:
            tf_label = f"{tf}min" if tf < 60 else f"{tf // 60}H"
            param_list = PARAM_SETS.get(tf, [])

            for pi, params in enumerate(param_list):
                v = validate_symbol(symbol, tf, params)
                if v is None:
                    continue

                full = v['full']
                status = "PASS" if v['all_pass'] else "FAIL"

                # Only print interesting results (some trades + decent stats)
                if full.total_trades >= 3:
                    win_detail = ""
                    for wi, w in enumerate(v['windows']):
                        if w:
                            wp = "+" if (w.total_trades == 0 or w.net_profit_after_tax_pct >= 0) else "-"
                            win_detail += wp
                        else:
                            win_detail += "?"

                    print(f"  {tf_label} p{pi}: {full.total_trades}t "
                          f"WR={full.win_rate:.0f}% "
                          f"AT={full.net_profit_after_tax_pct:+.2f}% "
                          f"PF={full.profit_factor:.2f} "
                          f"Sharpe={full.sharpe_ratio:.2f} "
                          f"DD={full.max_drawdown_pct:.2f}% "
                          f"WF=[{win_detail}] [{status}]")

                results.append((symbol, tf, params, v))

    # Phase 3: Summary of passing symbols
    print(f"\n{'=' * 75}")
    print("  RESULTS: SYMBOLS PASSING ALL WALK-FORWARD WINDOWS")
    print(f"{'=' * 75}")

    passing = [(s, tf, p, v) for s, tf, p, v in results
                if v['all_pass'] and v['full'].total_trades >= 5]

    # Sort by after-tax profit
    passing.sort(key=lambda x: x[3]['full'].net_profit_after_tax_pct, reverse=True)

    if not passing:
        print("\n  No new symbols passed walk-forward validation with >= 5 trades.")
        print("  This is consistent with RSI mean reversion being selective.")
    else:
        print(f"\n  {'Symbol':<12} {'TF':<6} {'Trades':>7} {'WR%':>6} {'AT%':>9} {'PF':>7} {'Sharpe':>7} {'DD%':>7} {'AvgHold':>8}")
        print(f"  {'-' * 75}")
        for s, tf, p, v in passing:
            f = v['full']
            tf_label = f"{tf}min" if tf < 60 else f"{tf // 60}H"
            print(f"  {s:<12} {tf_label:<6} {f.total_trades:>7} {f.win_rate:>5.0f}% "
                  f"{f.net_profit_after_tax_pct:>+8.2f}% {f.profit_factor:>6.2f} "
                  f"{f.sharpe_ratio:>6.2f} {f.max_drawdown_pct:>6.2f}% "
                  f"{f.avg_hold_hours:>7.1f}h")
            print(f"    params: entry={p['rsi_entry']} exit={p['rsi_exit']} "
                  f"partial={p['rsi_partial_exit']} hold={p['max_hold_bars']}bars")

    # Also show "almost passing" (2 of 3 windows)
    print(f"\n  --- Near-misses (2 of 3 windows pass, >= 5 trades) ---")
    near_miss = []
    for s, tf, p, v in results:
        if v['all_pass'] or v['full'].total_trades < 5:
            continue
        passed_count = sum(1 for w in v['windows']
                          if w and (w.total_trades == 0 or w.net_profit_after_tax_pct >= 0))
        if passed_count >= 2:
            near_miss.append((s, tf, p, v, passed_count))

    near_miss.sort(key=lambda x: x[3]['full'].net_profit_after_tax_pct, reverse=True)

    if near_miss:
        for s, tf, p, v, pc in near_miss[:10]:
            f = v['full']
            tf_label = f"{tf}min" if tf < 60 else f"{tf // 60}H"
            print(f"  {s:<12} {tf_label:<6} {f.total_trades:>3}t "
                  f"WR={f.win_rate:.0f}% AT={f.net_profit_after_tax_pct:+.2f}% "
                  f"PF={f.profit_factor:.2f} ({pc}/3 windows)")
    else:
        print("  None")

    print(f"\n{'=' * 75}")
    print("  Scan complete.")
    print(f"{'=' * 75}")


if __name__ == '__main__':
    main()
