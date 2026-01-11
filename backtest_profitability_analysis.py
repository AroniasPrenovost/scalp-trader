#!/usr/bin/env python3
"""
Comprehensive Backtest for 0.6% Profitability Threshold Analysis
Tests current AMR strategy against required minimum gain after fees
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# TRADING FEES (Coinbase Advanced Trade)
TAKER_FEE = 0.006  # 0.6% per trade
MAKER_FEE = 0.004  # 0.4% per trade
TOTAL_ROUND_TRIP_FEE = (TAKER_FEE + MAKER_FEE)  # 1.0% total for buy+sell
MIN_PROFIT_AFTER_FEES = 0.006  # 0.6% minimum to be profitable


def load_price_data(symbol: str) -> Optional[List[float]]:
    """Load historical price data for a symbol."""
    filepath = f"coinbase-data/{symbol}.json"
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract prices in chronological order
    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    prices = [float(entry['price']) for entry in data_sorted]
    timestamps = [entry['timestamp'] for entry in data_sorted]

    return prices, timestamps


def calculate_fees(entry_price: float, exit_price: float, position_size_usd: float) -> Dict:
    """Calculate trading fees for a round trip."""
    # Buy fees
    buy_fee = position_size_usd * TAKER_FEE
    coins_bought = (position_size_usd - buy_fee) / entry_price

    # Sell fees
    sell_value = coins_bought * exit_price
    sell_fee = sell_value * MAKER_FEE
    final_value = sell_value - sell_fee

    # Net P/L
    gross_pnl = sell_value - position_size_usd
    net_pnl = final_value - position_size_usd
    total_fees = buy_fee + sell_fee

    gross_pnl_pct = (gross_pnl / position_size_usd) * 100
    net_pnl_pct = (net_pnl / position_size_usd) * 100
    fee_pct = (total_fees / position_size_usd) * 100

    return {
        'gross_pnl_usd': gross_pnl,
        'net_pnl_usd': net_pnl,
        'total_fees_usd': total_fees,
        'gross_pnl_pct': gross_pnl_pct,
        'net_pnl_pct': net_pnl_pct,
        'fee_pct': fee_pct,
        'profitable_after_fees': net_pnl_pct >= MIN_PROFIT_AFTER_FEES * 100
    }


def detect_trend(prices: List[float], lookback: int = 168) -> str:
    """Detect market trend (uptrend/downtrend/sideways)."""
    if len(prices) < lookback:
        return 'sideways'

    recent = prices[-lookback:]
    start = recent[0]
    end = recent[-1]
    price_change = (end - start) / start

    # Calculate volatility
    price_range = (max(recent) - min(recent)) / min(recent)

    # Trend detection
    if price_change > 0.05 and price_range > 0.10:
        return 'uptrend'
    elif price_change < -0.05 and price_range > 0.10:
        return 'downtrend'
    else:
        return 'sideways'


def check_buy_signal(prices: List[float], current_price: float,
                     min_dip: float = 0.02, max_dip: float = 0.03,
                     ma_period: int = 24) -> Optional[Dict]:
    """Check for AMR buy signal based on config.json settings."""
    if len(prices) < max(168, ma_period):
        return None

    # Detect trend
    trend = detect_trend(prices, lookback=168)

    # Skip downtrends
    if trend == 'downtrend':
        return None

    # Calculate MA
    ma = sum(prices[-ma_period:]) / ma_period

    # Calculate deviation
    deviation_from_ma = (current_price - ma) / ma

    # BUY SIGNAL: min_dip to max_dip below MA
    if -max_dip <= deviation_from_ma <= -min_dip:
        return {
            'entry': current_price,
            'ma': ma,
            'trend': trend,
            'deviation_pct': deviation_from_ma * 100
        }

    return None


def backtest_symbol(symbol: str, position_size_usd: float = 4609,
                   profit_target_pct: float = 1.7,
                   stop_loss_pct: float = 1.0,
                   max_hold_hours: int = 72,
                   max_hours: int = 4320) -> Optional[Dict]:
    """Backtest a single symbol with fee analysis."""

    data = load_price_data(symbol)
    if not data:
        return None

    prices, timestamps = data

    if len(prices) < 200:
        return None

    # Use only last max_hours
    if len(prices) > max_hours:
        prices = prices[-max_hours:]
        timestamps = timestamps[-max_hours:]

    trades = []
    position = None

    # Start from hour 168 (need data for trend detection)
    for i in range(168, len(prices)):
        current_price = prices[i]
        historical = prices[:i]

        # ENTRY
        if position is None:
            signal = check_buy_signal(historical, current_price)
            if signal:
                target_price = signal['entry'] * (1 + profit_target_pct / 100)
                stop_price = signal['entry'] * (1 - stop_loss_pct / 100)

                position = {
                    'entry_idx': i,
                    'entry_price': signal['entry'],
                    'entry_timestamp': timestamps[i],
                    'target': target_price,
                    'stop': stop_price,
                    'trend': signal['trend'],
                    'ma': signal['ma'],
                    'deviation_pct': signal['deviation_pct']
                }

        # EXIT
        elif position:
            hours_held = i - position['entry_idx']

            # Max hold time
            if hours_held >= max_hold_hours:
                fees = calculate_fees(position['entry_price'], current_price, position_size_usd)
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': timestamps[i],
                    'gross_pnl_pct': fees['gross_pnl_pct'],
                    'net_pnl_pct': fees['net_pnl_pct'],
                    'fee_pct': fees['fee_pct'],
                    'net_pnl_usd': fees['net_pnl_usd'],
                    'total_fees_usd': fees['total_fees_usd'],
                    'outcome': 'win' if fees['net_pnl_pct'] > 0 else 'loss',
                    'profitable_after_fees': fees['profitable_after_fees'],
                    'exit_reason': 'max_hold',
                    'trend': position['trend'],
                    'hours_held': hours_held,
                    'deviation_at_entry': position['deviation_pct']
                })
                position = None

            # Stop loss
            elif current_price <= position['stop']:
                fees = calculate_fees(position['entry_price'], current_price, position_size_usd)
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': timestamps[i],
                    'gross_pnl_pct': fees['gross_pnl_pct'],
                    'net_pnl_pct': fees['net_pnl_pct'],
                    'fee_pct': fees['fee_pct'],
                    'net_pnl_usd': fees['net_pnl_usd'],
                    'total_fees_usd': fees['total_fees_usd'],
                    'outcome': 'loss',
                    'profitable_after_fees': fees['profitable_after_fees'],
                    'exit_reason': 'stop_loss',
                    'trend': position['trend'],
                    'hours_held': hours_held,
                    'deviation_at_entry': position['deviation_pct']
                })
                position = None

            # Profit target
            elif current_price >= position['target']:
                fees = calculate_fees(position['entry_price'], current_price, position_size_usd)
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': timestamps[i],
                    'gross_pnl_pct': fees['gross_pnl_pct'],
                    'net_pnl_pct': fees['net_pnl_pct'],
                    'fee_pct': fees['fee_pct'],
                    'net_pnl_usd': fees['net_pnl_usd'],
                    'total_fees_usd': fees['total_fees_usd'],
                    'outcome': 'win',
                    'profitable_after_fees': fees['profitable_after_fees'],
                    'exit_reason': 'target',
                    'trend': position['trend'],
                    'hours_held': hours_held,
                    'deviation_at_entry': position['deviation_pct']
                })
                position = None

    # Close remaining position
    if position:
        final_price = prices[-1]
        hours_held = len(prices) - 1 - position['entry_idx']
        fees = calculate_fees(position['entry_price'], final_price, position_size_usd)

        trades.append({
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': timestamps[-1],
            'gross_pnl_pct': fees['gross_pnl_pct'],
            'net_pnl_pct': fees['net_pnl_pct'],
            'fee_pct': fees['fee_pct'],
            'net_pnl_usd': fees['net_pnl_usd'],
            'total_fees_usd': fees['total_fees_usd'],
            'outcome': 'win' if fees['net_pnl_pct'] > 0 else 'loss',
            'profitable_after_fees': fees['profitable_after_fees'],
            'exit_reason': 'eod',
            'trend': position['trend'],
            'hours_held': hours_held,
            'deviation_at_entry': position['deviation_pct']
        })

    if not trades:
        return None

    # Calculate comprehensive stats
    total_trades = len(trades)
    profitable_trades = [t for t in trades if t['profitable_after_fees']]
    unprofitable_trades = [t for t in trades if not t['profitable_after_fees']]

    wins = [t for t in trades if t['outcome'] == 'win']
    losses = [t for t in trades if t['outcome'] == 'loss']

    target_exits = [t for t in trades if t['exit_reason'] == 'target']
    stop_exits = [t for t in trades if t['exit_reason'] == 'stop_loss']

    # Net P/L calculations
    total_net_pnl_usd = sum(t['net_pnl_usd'] for t in trades)
    total_fees_usd = sum(t['total_fees_usd'] for t in trades)
    avg_net_pnl_pct = sum(t['net_pnl_pct'] for t in trades) / total_trades

    # Profitable vs unprofitable
    profitability_rate = (len(profitable_trades) / total_trades) * 100
    avg_profitable_pnl = sum(t['net_pnl_usd'] for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
    avg_unprofitable_pnl = sum(t['net_pnl_usd'] for t in unprofitable_trades) / len(unprofitable_trades) if unprofitable_trades else 0

    # Win/loss stats
    win_rate = (len(wins) / total_trades) * 100
    avg_win_pct = sum(t['net_pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss_pct = sum(t['net_pnl_pct'] for t in losses) / len(losses) if losses else 0

    # By trend
    uptrend = [t for t in trades if t['trend'] == 'uptrend']
    sideways = [t for t in trades if t['trend'] == 'sideways']

    uptrend_profitable = [t for t in uptrend if t['profitable_after_fees']]
    sideways_profitable = [t for t in sideways if t['profitable_after_fees']]

    uptrend_pr = (len(uptrend_profitable) / len(uptrend)) * 100 if uptrend else 0
    sideways_pr = (len(sideways_profitable) / len(sideways)) * 100 if sideways else 0

    avg_hold_hours = sum(t['hours_held'] for t in trades) / total_trades

    return {
        'symbol': symbol,
        'total_trades': total_trades,
        'position_size_usd': position_size_usd,

        # Fee analysis
        'total_fees_usd': total_fees_usd,
        'avg_fee_per_trade_usd': total_fees_usd / total_trades,
        'fee_pct_per_trade': TOTAL_ROUND_TRIP_FEE * 100,

        # Profitability (>= 0.6% net gain)
        'profitable_trades': len(profitable_trades),
        'unprofitable_trades': len(unprofitable_trades),
        'profitability_rate': profitability_rate,
        'avg_profitable_pnl_usd': avg_profitable_pnl,
        'avg_unprofitable_pnl_usd': avg_unprofitable_pnl,

        # Win/loss
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,

        # Exit reasons
        'target_exits': len(target_exits),
        'stop_exits': len(stop_exits),
        'target_exit_rate': (len(target_exits) / total_trades) * 100,

        # Net performance
        'total_net_pnl_usd': total_net_pnl_usd,
        'avg_net_pnl_pct': avg_net_pnl_pct,
        'expected_value_per_trade_usd': total_net_pnl_usd / total_trades,

        # By trend
        'uptrend_trades': len(uptrend),
        'uptrend_profitable': len(uptrend_profitable),
        'uptrend_profitability_rate': uptrend_pr,
        'sideways_trades': len(sideways),
        'sideways_profitable': len(sideways_profitable),
        'sideways_profitability_rate': sideways_pr,

        'avg_hold_hours': avg_hold_hours,
        'trades': trades
    }


def print_profitability_analysis(results: List[Dict]):
    """Print detailed profitability analysis."""
    print("\n" + "="*140)
    print("PROFITABILITY ANALYSIS - 0.6% NET GAIN THRESHOLD (AFTER FEES)")
    print("="*140)
    print(f"Fee Structure: {TAKER_FEE*100:.1f}% taker + {MAKER_FEE*100:.1f}% maker = {TOTAL_ROUND_TRIP_FEE*100:.1f}% round trip")
    print(f"Minimum Required Net Gain: {MIN_PROFIT_AFTER_FEES*100:.1f}% (to make $2+ per trade)")
    print("="*140)

    # Header
    print(f"\n{'Symbol':<10} {'Trades':<8} {'Prof Rate':<12} {'Win Rate':<10} "
          f"{'Net P/L $':<12} {'Avg EV $':<12} {'Total Fees $':<14} {'Target %':<10}")
    print("-"*140)

    # Sort by profitability rate
    results.sort(key=lambda x: x['profitability_rate'], reverse=True)

    for r in results:
        profitable_emoji = "✅" if r['profitability_rate'] >= 50 else "⚠️" if r['profitability_rate'] >= 40 else "❌"

        print(f"{profitable_emoji} {r['symbol']:<8} {r['total_trades']:<8} "
              f"{r['profitability_rate']:>6.1f}%     "
              f"{r['win_rate']:>6.1f}%    "
              f"${r['total_net_pnl_usd']:>+9.2f}  "
              f"${r['expected_value_per_trade_usd']:>+8.2f}   "
              f"${r['total_fees_usd']:>10.2f}    "
              f"{r['target_exit_rate']:>6.1f}%")

    # Overall stats
    total_trades = sum(r['total_trades'] for r in results)
    total_profitable = sum(r['profitable_trades'] for r in results)
    overall_prof_rate = (total_profitable / total_trades) * 100 if total_trades else 0
    total_net_pnl = sum(r['total_net_pnl_usd'] for r in results)
    total_fees = sum(r['total_fees_usd'] for r in results)
    overall_ev = total_net_pnl / total_trades if total_trades else 0

    print("-"*140)
    print(f"OVERALL: {total_trades} trades | {overall_prof_rate:.1f}% profitable | "
          f"${total_net_pnl:+.2f} net P/L | ${overall_ev:+.2f} avg EV | "
          f"${total_fees:.2f} total fees")
    print("="*140)


def print_detailed_breakdown(results: List[Dict]):
    """Print detailed per-symbol breakdown."""
    print("\n" + "="*140)
    print("DETAILED BREAKDOWN BY SYMBOL")
    print("="*140)

    for r in results:
        print(f"\n📊 {r['symbol']}")
        print(f"  Total Trades: {r['total_trades']}")
        print(f"  Profitability Rate: {r['profitability_rate']:.1f}% ({r['profitable_trades']} profitable, {r['unprofitable_trades']} unprofitable)")
        print(f"  Win Rate: {r['win_rate']:.1f}% ({r['wins']} wins, {r['losses']} losses)")
        print(f"  Exit Breakdown: {r['target_exits']} targets ({r['target_exit_rate']:.1f}%), {r['stop_exits']} stops")
        print(f"\n  Net Performance:")
        print(f"    Total Net P/L: ${r['total_net_pnl_usd']:+.2f}")
        print(f"    Avg Net P/L: {r['avg_net_pnl_pct']:+.2f}%")
        print(f"    Expected Value per Trade: ${r['expected_value_per_trade_usd']:+.2f}")
        print(f"\n  Fee Impact:")
        print(f"    Total Fees Paid: ${r['total_fees_usd']:.2f}")
        print(f"    Avg Fee per Trade: ${r['avg_fee_per_trade_usd']:.2f} ({r['fee_pct_per_trade']:.1f}%)")
        print(f"\n  Profitable Trades:")
        print(f"    Avg Profitable Trade: ${r['avg_profitable_pnl_usd']:+.2f}")
        print(f"    Avg Win: {r['avg_win_pct']:+.2f}%")
        print(f"\n  Unprofitable Trades:")
        print(f"    Avg Unprofitable Trade: ${r['avg_unprofitable_pnl_usd']:+.2f}")
        print(f"    Avg Loss: {r['avg_loss_pct']:+.2f}%")
        print(f"\n  By Market Condition:")
        print(f"    Uptrend: {r['uptrend_trades']} trades, {r['uptrend_profitability_rate']:.1f}% profitable")
        print(f"    Sideways: {r['sideways_trades']} trades, {r['sideways_profitability_rate']:.1f}% profitable")
        print(f"\n  Avg Hold Time: {r['avg_hold_hours']:.1f} hours")
        print("-"*140)


def main():
    """Run comprehensive profitability backtest."""

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    amr_config = config.get('adaptive_mean_reversion', {})
    market_rotation = config.get('market_rotation', {})

    profit_target = amr_config.get('profit_target_percentage', 1.7)
    stop_loss = amr_config.get('stop_loss_percentage', 1.0)
    max_hold = amr_config.get('max_hold_hours', 72)
    position_size = market_rotation.get('total_trading_capital_usd', 4609)

    print(f"\nBacktest Configuration:")
    print(f"  Position Size: ${position_size}")
    print(f"  Profit Target: {profit_target}%")
    print(f"  Stop Loss: {stop_loss}%")
    print(f"  Max Hold: {max_hold} hours")
    print(f"  Period: 180 days (4320 hours)")

    # Get enabled symbols
    symbols = []
    for wallet in config['wallets']:
        if wallet.get('enabled', False):
            symbols.append(wallet['symbol'])

    print(f"\nBacktesting {len(symbols)} symbols: {', '.join(symbols)}")
    print("\nRunning backtest...")

    results = []

    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        result = backtest_symbol(
            symbol,
            position_size_usd=position_size,
            profit_target_pct=profit_target,
            stop_loss_pct=stop_loss,
            max_hold_hours=max_hold,
            max_hours=4320
        )

        if result:
            results.append(result)
            emoji = "✅" if result['profitability_rate'] >= 50 else "⚠️" if result['profitability_rate'] >= 40 else "❌"
            print(f"{emoji} {result['total_trades']} trades, {result['profitability_rate']:.1f}% profitable, "
                  f"${result['total_net_pnl_usd']:+.2f} net P/L")
        else:
            print("⚠️  No data or trades")

    if not results:
        print("\n⚠️  No backtest results available")
        return

    # Print analyses
    print_profitability_analysis(results)
    print_detailed_breakdown(results)

    # Key insights
    print("\n" + "="*140)
    print("KEY INSIGHTS")
    print("="*140)

    viable_symbols = [r for r in results if r['profitability_rate'] >= 50]
    marginal_symbols = [r for r in results if 40 <= r['profitability_rate'] < 50]
    unviable_symbols = [r for r in results if r['profitability_rate'] < 40]

    print(f"\n✅ VIABLE ({len(viable_symbols)}): ≥50% profitable trades (meeting 0.6% threshold)")
    for r in viable_symbols:
        print(f"   {r['symbol']}: {r['profitability_rate']:.1f}% profitable, "
              f"${r['expected_value_per_trade_usd']:+.2f} EV per trade")

    print(f"\n⚠️  MARGINAL ({len(marginal_symbols)}): 40-50% profitable trades")
    for r in marginal_symbols:
        print(f"   {r['symbol']}: {r['profitability_rate']:.1f}% profitable, "
              f"${r['expected_value_per_trade_usd']:+.2f} EV per trade")

    print(f"\n❌ NOT VIABLE ({len(unviable_symbols)}): <40% profitable trades")
    for r in unviable_symbols:
        print(f"   {r['symbol']}: {r['profitability_rate']:.1f}% profitable, "
              f"${r['expected_value_per_trade_usd']:+.2f} EV per trade")

    # Overall verdict
    total_trades = sum(r['total_trades'] for r in results)
    total_profitable = sum(r['profitable_trades'] for r in results)
    overall_prof_rate = (total_profitable / total_trades) * 100
    total_net_pnl = sum(r['total_net_pnl_usd'] for r in results)
    overall_ev = total_net_pnl / total_trades

    print(f"\n{'='*140}")
    print(f"OVERALL VERDICT")
    print(f"{'='*140}")
    print(f"Profitability Rate: {overall_prof_rate:.1f}% ({total_profitable}/{total_trades} trades)")
    print(f"Total Net P/L: ${total_net_pnl:+.2f}")
    print(f"Expected Value per Trade: ${overall_ev:+.2f}")

    if overall_prof_rate >= 55:
        print(f"\n✅ STRATEGY IS PROFITABLE: {overall_prof_rate:.1f}% of trades meet 0.6% threshold")
        print(f"   With proper execution, expect ${overall_ev:+.2f} per trade on average")
    elif overall_prof_rate >= 45:
        print(f"\n⚠️  STRATEGY IS MARGINAL: {overall_prof_rate:.1f}% profitability rate")
        print(f"   Consider focusing only on high-performing symbols")
    else:
        print(f"\n❌ STRATEGY NEEDS IMPROVEMENT: Only {overall_prof_rate:.1f}% profitable")
        print(f"   Current settings do not reliably achieve 0.6% net gains")

    print(f"{'='*140}\n")

    # Save results
    output = {
        'backtest_date': datetime.now().isoformat(),
        'period_days': 180,
        'position_size_usd': position_size,
        'profit_target_pct': profit_target,
        'stop_loss_pct': stop_loss,
        'fee_structure': {
            'taker_fee_pct': TAKER_FEE * 100,
            'maker_fee_pct': MAKER_FEE * 100,
            'round_trip_fee_pct': TOTAL_ROUND_TRIP_FEE * 100
        },
        'min_profit_threshold_pct': MIN_PROFIT_AFTER_FEES * 100,
        'results': results,
        'overall': {
            'total_trades': total_trades,
            'profitable_trades': total_profitable,
            'profitability_rate': overall_prof_rate,
            'total_net_pnl_usd': total_net_pnl,
            'expected_value_per_trade_usd': overall_ev
        }
    }

    with open('profitability_backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"📊 Detailed results saved to profitability_backtest_results.json\n")


if __name__ == '__main__':
    main()
