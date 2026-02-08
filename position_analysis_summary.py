#!/usr/bin/env python3
"""
Position Count Analysis Summary

Uses walk-forward validated returns directly to compare 2 vs 3 positions.
The WF validation returns are: TAO +8.60%, ICP +6.03%, CRV +6.01%,
NEAR +3.57%, ATOM +3.31%, ZEC +1.47% over ~30 day validation period.
"""

import json
import os
from collections import defaultdict

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# Walk-forward validated returns (30-day validation period, from Feb 7 2026)
# These represent NET returns after fees on ~$4125 capital per symbol
VALIDATED_DATA = {
    'TAO-USD': {
        'return_pct': 8.60,
        'win_rate': 63,
        'profit_factor': 1.57,
        'strategy': 'rsi_mean_reversion',
        'timeframe': '30min',
        'signals_per_day': 0.58,  # From our detection
    },
    'ICP-USD': {
        'return_pct': 6.03,
        'win_rate': 67,
        'sharpe': 7.11,
        'strategy': 'rsi_regime',
        'timeframe': '5min',
        'signals_per_day': 0.46,
    },
    'CRV-USD': {
        'return_pct': 6.01,
        'win_rate': 67,
        'profit_factor': 1.72,
        'strategy': 'co_revert',
        'timeframe': '30min',
        'signals_per_day': 0.25,
    },
    'NEAR-USD': {
        'return_pct': 3.57,
        'win_rate': 60,
        'profit_factor': 1.41,
        'strategy': 'rsi_mean_reversion',
        'timeframe': '30min',
        'signals_per_day': 0.41,
    },
    'ATOM-USD': {
        'return_pct': 3.31,
        'win_rate': 69,
        'profit_factor': 2.40,
        'strategy': 'rsi_regime',
        'timeframe': '30min',
        'signals_per_day': 0.38,
    },
    'ZEC-USD': {
        'return_pct': 1.47,
        'win_rate': 67,
        'profit_factor': 1.12,
        'strategy': 'rsi_mean_reversion',
        'timeframe': '15min',
        'signals_per_day': 0.43,
    },
}


def main():
    print("=" * 80)
    print("2 vs 3 CONCURRENT POSITIONS: FINAL ANALYSIS")
    print("=" * 80)
    print()

    TOTAL_CAPITAL = 6010
    CAPITAL_2POS = TOTAL_CAPITAL / 2  # $3,005
    CAPITAL_3POS = TOTAL_CAPITAL / 3  # $2,003.33

    # Sum of validated returns
    total_wf_return = sum(d['return_pct'] for d in VALIDATED_DATA.values())
    avg_wf_return = total_wf_return / len(VALIDATED_DATA)

    print("WALK-FORWARD VALIDATION DATA (30-day period):")
    print("-" * 60)
    print(f"{'Symbol':<12} {'Return':>10} {'WR':>8} {'PF':>8} {'Timeframe':>10}")
    print("-" * 60)

    for symbol in sorted(VALIDATED_DATA.keys(), key=lambda s: -VALIDATED_DATA[s]['return_pct']):
        data = VALIDATED_DATA[symbol]
        pf = data.get('profit_factor', data.get('sharpe', '-'))
        if isinstance(pf, float):
            pf_str = f"{pf:.2f}"
        else:
            pf_str = str(pf)
        print(f"{symbol:<12} {data['return_pct']:>9.2f}% {data['win_rate']:>7}% {pf_str:>8} {data['timeframe']:>10}")

    print("-" * 60)
    print(f"{'Combined':<12} {total_wf_return:>9.2f}%")
    print(f"{'Average':<12} {avg_wf_return:>9.2f}%")
    print()

    # ==============================================================================
    # KEY INSIGHT: Signal overlap analysis results
    # ==============================================================================

    print("=" * 80)
    print("SIGNAL OVERLAP FINDINGS (from 202-day analysis)")
    print("=" * 80)
    print()

    print("Total signals detected: 505 over 202 days (2.5 signals/day)")
    print()
    print("4-hour window overlap distribution:")
    print("  - 0 signals: 82.2% of windows (market quiet)")
    print("  - 1 signal:   8.9% of windows")
    print("  - 2 signals:  3.9% of windows")
    print("  - 3+ signals: 5.0% of windows")
    print()
    print("KEY FINDING: When signals DO fire, there's often overlap.")
    print("  - 44% of windows with signals have 2+ concurrent opportunities")
    print("  - This is because RSI oversold conditions often correlate across assets")
    print()

    # ==============================================================================
    # EXECUTION RATE ANALYSIS
    # ==============================================================================

    print("=" * 80)
    print("EXECUTION RATE COMPARISON")
    print("=" * 80)
    print()

    # From simulation results
    EXEC_2POS = 0.491  # 49.1%
    EXEC_3POS = 0.578  # 57.8%

    print("Simulation results (505 signals over 202 days):")
    print()
    print("  2 POSITIONS:")
    print(f"    - Signals executed: 248 (49.1%)")
    print(f"    - Signals missed:   257 (50.9%)")
    print()
    print("  3 POSITIONS:")
    print(f"    - Signals executed: 292 (57.8%)")
    print(f"    - Signals missed:   213 (42.2%)")
    print()
    print(f"  Execution improvement: +{(EXEC_3POS - EXEC_2POS)*100:.1f}% with 3rd position")
    print()

    # ==============================================================================
    # PROFIT CALCULATION
    # ==============================================================================

    print("=" * 80)
    print("EXPECTED PROFIT CALCULATION")
    print("=" * 80)
    print()

    # The WF validation was done on ~$4125 capital per symbol
    # Our capital per position differs, so we scale

    WF_CAPITAL_BASE = 4125  # Capital used in WF validation

    # Method: Scale WF returns to our position sizes, then adjust by execution rate

    print("SCENARIO A: 2 CONCURRENT POSITIONS")
    print("-" * 60)
    print(f"  Capital per position: ${CAPITAL_2POS:,.2f}")
    print(f"  Position size vs WF baseline: {CAPITAL_2POS/WF_CAPITAL_BASE:.2%}")
    print(f"  Execution rate: {EXEC_2POS:.1%}")
    print()

    # Calculate scaled monthly return for 2 positions
    # WF returns were for 30 days, so monthly = WF return
    # But with only 2 positions, we can only capture 2 of 6 opportunities at once
    # Average return per symbol per month: avg_wf_return (scaled to position size)

    # The key insight: with 2 positions, we execute 49.1% of signals
    # With the signals spread across 6 symbols, and returns proportional to signals

    # Weight returns by signal frequency (more signals = more opportunity)
    total_signals_per_day = sum(d['signals_per_day'] for d in VALIDATED_DATA.values())

    weighted_return_2pos = 0
    for symbol, data in VALIDATED_DATA.items():
        signal_weight = data['signals_per_day'] / total_signals_per_day
        symbol_contribution = data['return_pct'] * signal_weight * EXEC_2POS * (CAPITAL_2POS / WF_CAPITAL_BASE)
        weighted_return_2pos += symbol_contribution

    # Since we have 2 positions potentially earning simultaneously
    monthly_profit_2pos = CAPITAL_2POS * (weighted_return_2pos / 100) * 2

    print(f"  Weighted monthly return (per position): {weighted_return_2pos:.2f}%")
    print(f"  Expected monthly profit (2 slots): ${monthly_profit_2pos:.2f}")
    print()

    print("SCENARIO B: 3 CONCURRENT POSITIONS")
    print("-" * 60)
    print(f"  Capital per position: ${CAPITAL_3POS:,.2f}")
    print(f"  Position size vs WF baseline: {CAPITAL_3POS/WF_CAPITAL_BASE:.2%}")
    print(f"  Execution rate: {EXEC_3POS:.1%}")
    print()

    weighted_return_3pos = 0
    for symbol, data in VALIDATED_DATA.items():
        signal_weight = data['signals_per_day'] / total_signals_per_day
        symbol_contribution = data['return_pct'] * signal_weight * EXEC_3POS * (CAPITAL_3POS / WF_CAPITAL_BASE)
        weighted_return_3pos += symbol_contribution

    # Since we have 3 positions potentially earning simultaneously
    monthly_profit_3pos = CAPITAL_3POS * (weighted_return_3pos / 100) * 3

    print(f"  Weighted monthly return (per position): {weighted_return_3pos:.2f}%")
    print(f"  Expected monthly profit (3 slots): ${monthly_profit_3pos:.2f}")
    print()

    # ==============================================================================
    # ALTERNATIVE CALCULATION: Direct scaling from WF data
    # ==============================================================================

    print("=" * 80)
    print("ALTERNATIVE CALCULATION: Portfolio-Level Scaling")
    print("=" * 80)
    print()

    # The WF validation showed combined ~29% return over 30 days on 6 symbols
    # This was with individual symbol allocation (each symbol got its own capital)
    #
    # With concurrent positions, we're rotating capital across symbols based on signals
    # The question is: how much of this 29% can we capture?

    # If we could always have optimal positions (6 concurrent), we'd get full return
    # With 2 positions, we capture roughly 2/6 * execution_rate_adjustment
    # With 3 positions, we capture roughly 3/6 * execution_rate_adjustment

    # But it's not linear because of signal overlap correlation

    print("Combined monthly return from all 6 symbols: +29.00%")
    print("(On $24,750 total capital if each symbol had $4,125)")
    print()

    # With position limits, our capture is:
    # - 2 positions: Can hold max 2 of 6 symbols at once
    # - 3 positions: Can hold max 3 of 6 symbols at once

    # Expected capture rate (accounting for overlap):
    capture_2pos = 0.33  # ~33% (2/6 base, plus some benefit from overlap)
    capture_3pos = 0.50  # ~50% (3/6 base, plus some benefit from overlap)

    combined_monthly_return = total_wf_return  # 29%
    combined_monthly_usd = TOTAL_CAPITAL * (combined_monthly_return / 100)

    expected_2pos = combined_monthly_usd * capture_2pos
    expected_3pos = combined_monthly_usd * capture_3pos

    print("Expected capture with position limits:")
    print(f"  2 positions: {capture_2pos:.0%} capture = ${expected_2pos:.2f}/month")
    print(f"  3 positions: {capture_3pos:.0%} capture = ${expected_3pos:.2f}/month")
    print()

    # ==============================================================================
    # RISK COMPARISON
    # ==============================================================================

    print("=" * 80)
    print("RISK COMPARISON")
    print("=" * 80)
    print()

    avg_disaster_stop = 4.0  # Average across strategies

    print("Single position worst-case (disaster stop):")
    print(f"  2 positions: ${CAPITAL_2POS * avg_disaster_stop / 100:.2f} per position")
    print(f"  3 positions: ${CAPITAL_3POS * avg_disaster_stop / 100:.2f} per position")
    print()

    print("All positions stop out simultaneously:")
    print(f"  2 positions: ${CAPITAL_2POS * avg_disaster_stop / 100 * 2:.2f} ({2 * avg_disaster_stop:.1f}% of capital)")
    print(f"  3 positions: ${CAPITAL_3POS * avg_disaster_stop / 100 * 3:.2f} ({3 * avg_disaster_stop:.1f}% of capital)")
    print()

    print("Diversification benefit:")
    print(f"  2 positions: 50% concentration per position")
    print(f"  3 positions: 33% concentration per position")
    print()

    # ==============================================================================
    # FEE TIER PROGRESSION
    # ==============================================================================

    print("=" * 80)
    print("FEE TIER PROGRESSION IMPACT")
    print("=" * 80)
    print()

    # More trades = faster volume accumulation
    trades_per_month_2pos = 248 / 202 * 30  # ~37 trades/month
    trades_per_month_3pos = 292 / 202 * 30  # ~43 trades/month

    volume_per_trade_2pos = CAPITAL_2POS
    volume_per_trade_3pos = CAPITAL_3POS

    monthly_volume_2pos = trades_per_month_2pos * volume_per_trade_2pos * 2  # buy + sell
    monthly_volume_3pos = trades_per_month_3pos * volume_per_trade_3pos * 2

    print(f"Estimated monthly trading volume:")
    print(f"  2 positions: ${monthly_volume_2pos:,.0f}")
    print(f"  3 positions: ${monthly_volume_3pos:,.0f}")
    print()

    # Fee tier thresholds
    print("Fee tier thresholds:")
    print("  Adv 2 (0.125%/0.25%): $75,000 volume")
    print("  Adv 3 (0.075%/0.15%): $250,000 volume")
    print()

    months_to_adv2_2pos = 75000 / monthly_volume_2pos if monthly_volume_2pos > 0 else float('inf')
    months_to_adv2_3pos = 75000 / monthly_volume_3pos if monthly_volume_3pos > 0 else float('inf')

    print(f"Months to reach Adv 2:")
    print(f"  2 positions: {months_to_adv2_2pos:.1f} months")
    print(f"  3 positions: {months_to_adv2_3pos:.1f} months")
    print()

    # ==============================================================================
    # FINAL RECOMMENDATION
    # ==============================================================================

    print("=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print()

    profit_diff = expected_3pos - expected_2pos

    print(f"Monthly profit advantage for 3 positions: ${profit_diff:.2f}")
    print()

    print("RECOMMENDED: 3 CONCURRENT POSITIONS")
    print()
    print("Supporting factors:")
    print("  1. Higher execution rate (+8.7%): Captures more signals")
    print("  2. Signal overlap is frequent: 44% of active windows have 2+ signals")
    print("  3. Diversification: 33% per position vs 50% reduces single-trade risk")
    print("  4. Faster fee tier progression: More trades = faster volume")
    print(f"  5. Expected monthly profit improvement: ~${profit_diff:.0f}")
    print()
    print("Trade-offs (acceptable):")
    print("  1. Smaller position sizes ($2,003 vs $3,005)")
    print("  2. More positions to manage")
    print("  3. Slightly higher worst-case risk (12% vs 8% if all stop out)")
    print()

    print("=" * 80)
    print("CONFIG RECOMMENDATION")
    print("=" * 80)
    print()
    print("Update config.json market_rotation settings:")
    print()
    print('  "market_rotation": {')
    print('    "total_trading_capital_usd": 6010,')
    print('    "capital_per_position": 2003,        // 6010 / 3')
    print('    "max_concurrent_orders": 3,          // Changed from 2')
    print('    ...')
    print('  }')
    print()


if __name__ == '__main__':
    main()
