#!/usr/bin/env python3
"""
Backtest with ACTUAL fee structure from live trading
- Coinbase Taker Fee: 0.25%
- Federal Tax Rate: 24%
- Minimum profitability: 0.40% gain for $2+ profit
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# ACTUAL TRADING COSTS
TAKER_FEE = 0.0025  # 0.25% Coinbase taker fee
TAX_RATE = 0.24     # 24% federal tax rate
POSITION_SIZE_USD = 4609

# Calculate minimum gain needed for $2 profit
# Formula derived from index.py profit calculation
# net_profit = gain_pct * cost_basis - (taker_fee * current_value) - (tax_rate * gain_pct * cost_basis)
# For $2 profit: gain_pct = (2/cost_basis + taker_fee) / (1 - tax_rate - taker_fee)
MIN_NET_PROFIT_USD = 2.00
net_pct = MIN_NET_PROFIT_USD / POSITION_SIZE_USD
MIN_GAIN_PCT = (net_pct + TAKER_FEE) / (1 - TAX_RATE - TAKER_FEE)


def load_price_data(symbol: str) -> Optional[tuple]:
    """Load historical price data."""
    filepath = f"coinbase-data/{symbol}.json"
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    prices = [float(entry['price']) for entry in data_sorted]
    timestamps = [entry['timestamp'] for entry in data_sorted]

    return prices, timestamps


def calculate_net_profit(entry_price: float, exit_price: float, position_size: float) -> Dict:
    """
    Calculate net profit using EXACT logic from index.py lines 1620-1659
    """
    # STEP 1: Current market value (if we sold now)
    # Note: In reality, entry includes entry fee, but we simplify here
    # by assuming position_size already accounts for entry costs
    cost_basis = position_size
    shares = position_size / entry_price
    current_value = exit_price * shares

    # STEP 3: Gross profit before exit costs
    gross_profit = current_value - cost_basis

    # STEP 4: Exit fee (taker fee on current value)
    exit_fee = TAKER_FEE * current_value

    # STEP 5: Unrealized capital gain (for tax calculation)
    capital_gain = current_value - cost_basis

    # STEP 6: Capital gains tax
    tax = TAX_RATE * capital_gain

    # STEP 7: Net profit (index.py:1652)
    net_profit = current_value - cost_basis - exit_fee - tax

    # STEP 8: Percentage return
    net_pct = (net_profit / cost_basis) * 100
    gross_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'gross_pnl_pct': gross_pct,
        'net_pnl_pct': net_pct,
        'net_pnl_usd': net_profit,
        'exit_fee_usd': exit_fee,
        'tax_usd': tax,
        'profitable': net_profit >= MIN_NET_PROFIT_USD
    }


def detect_trend(prices: List[float], lookback: int = 168) -> str:
    """Detect market trend."""
    if len(prices) < lookback:
        return 'sideways'

    recent = prices[-lookback:]
    start = recent[0]
    end = recent[-1]
    price_change = (end - start) / start
    price_range = (max(recent) - min(recent)) / min(recent)

    if price_change > 0.05 and price_range > 0.10:
        return 'uptrend'
    elif price_change < -0.05 and price_range > 0.10:
        return 'downtrend'
    else:
        return 'sideways'


def check_buy_signal(prices: List[float], current_price: float,
                     min_dip: float = 0.02, max_dip: float = 0.03) -> Optional[Dict]:
    """AMR buy signal from config.json settings."""
    if len(prices) < 168:
        return None

    trend = detect_trend(prices, lookback=168)

    if trend == 'downtrend':
        return None

    ma_24h = sum(prices[-24:]) / 24
    deviation_from_ma = (current_price - ma_24h) / ma_24h

    if -max_dip <= deviation_from_ma <= -min_dip:
        return {
            'entry': current_price,
            'trend': trend,
            'deviation_pct': deviation_from_ma * 100
        }

    return None


def backtest_symbol(symbol: str,
                   profit_target_pct: float = 1.7,
                   stop_loss_pct: float = 1.0,
                   max_hold_hours: int = 72,
                   max_hours: int = 4320) -> Optional[Dict]:
    """Backtest with actual fee structure."""

    data = load_price_data(symbol)
    if not data:
        return None

    prices, timestamps = data

    if len(prices) < 200:
        return None

    if len(prices) > max_hours:
        prices = prices[-max_hours:]
        timestamps = timestamps[-max_hours:]

    trades = []
    position = None

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
                    'trend': signal['trend']
                }

        # EXIT
        elif position:
            hours_held = i - position['entry_idx']
            exit_price = None
            exit_reason = None

            # Max hold time
            if hours_held >= max_hold_hours:
                exit_price = current_price
                exit_reason = 'max_hold'

            # Stop loss
            elif current_price <= position['stop']:
                exit_price = current_price
                exit_reason = 'stop_loss'

            # Profit target
            elif current_price >= position['target']:
                exit_price = current_price
                exit_reason = 'target'

            if exit_price:
                pnl = calculate_net_profit(position['entry_price'], exit_price, POSITION_SIZE_USD)

                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': timestamps[i],
                    'gross_pnl_pct': pnl['gross_pnl_pct'],
                    'net_pnl_pct': pnl['net_pnl_pct'],
                    'net_pnl_usd': pnl['net_pnl_usd'],
                    'exit_fee_usd': pnl['exit_fee_usd'],
                    'tax_usd': pnl['tax_usd'],
                    'profitable': pnl['profitable'],
                    'exit_reason': exit_reason,
                    'trend': position['trend'],
                    'hours_held': hours_held
                })
                position = None

    # Close remaining position
    if position:
        final_price = prices[-1]
        hours_held = len(prices) - 1 - position['entry_idx']
        pnl = calculate_net_profit(position['entry_price'], final_price, POSITION_SIZE_USD)

        trades.append({
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': timestamps[-1],
            'gross_pnl_pct': pnl['gross_pnl_pct'],
            'net_pnl_pct': pnl['net_pnl_pct'],
            'net_pnl_usd': pnl['net_pnl_usd'],
            'exit_fee_usd': pnl['exit_fee_usd'],
            'tax_usd': pnl['tax_usd'],
            'profitable': pnl['profitable'],
            'exit_reason': 'eod',
            'trend': position['trend'],
            'hours_held': hours_held
        })

    if not trades:
        return None

    # Calculate stats
    total_trades = len(trades)
    profitable_trades = [t for t in trades if t['profitable']]

    wins = [t for t in trades if t['net_pnl_usd'] > 0]
    losses = [t for t in trades if t['net_pnl_usd'] <= 0]

    target_exits = [t for t in trades if t['exit_reason'] == 'target']
    stop_exits = [t for t in trades if t['exit_reason'] == 'stop_loss']

    total_net_pnl = sum(t['net_pnl_usd'] for t in trades)
    total_fees = sum(t['exit_fee_usd'] for t in trades)
    total_taxes = sum(t['tax_usd'] for t in trades)

    profitability_rate = (len(profitable_trades) / total_trades) * 100
    win_rate = (len(wins) / total_trades) * 100

    avg_net_pnl_usd = total_net_pnl / total_trades
    avg_net_pnl_pct = sum(t['net_pnl_pct'] for t in trades) / total_trades

    avg_win_usd = sum(t['net_pnl_usd'] for t in wins) / len(wins) if wins else 0
    avg_loss_usd = sum(t['net_pnl_usd'] for t in losses) / len(losses) if losses else 0

    # By trend
    uptrend = [t for t in trades if t['trend'] == 'uptrend']
    sideways = [t for t in trades if t['trend'] == 'sideways']

    uptrend_profitable = [t for t in uptrend if t['profitable']]
    sideways_profitable = [t for t in sideways if t['profitable']]

    uptrend_pr = (len(uptrend_profitable) / len(uptrend)) * 100 if uptrend else 0
    sideways_pr = (len(sideways_profitable) / len(sideways)) * 100 if sideways else 0

    avg_hold_hours = sum(t['hours_held'] for t in trades) / total_trades

    return {
        'symbol': symbol,
        'total_trades': total_trades,

        # Profitability ($2+ threshold)
        'profitable_trades': len(profitable_trades),
        'profitability_rate': profitability_rate,

        # Win/loss
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,

        # Net performance
        'total_net_pnl_usd': total_net_pnl,
        'avg_net_pnl_usd': avg_net_pnl_usd,
        'avg_net_pnl_pct': avg_net_pnl_pct,
        'avg_win_usd': avg_win_usd,
        'avg_loss_usd': avg_loss_usd,

        # Costs
        'total_fees_usd': total_fees,
        'total_taxes_usd': total_taxes,

        # Exit reasons
        'target_exits': len(target_exits),
        'stop_exits': len(stop_exits),
        'target_exit_rate': (len(target_exits) / total_trades) * 100,

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


def main():
    """Run backtest with actual fee structure."""

    with open('config.json', 'r') as f:
        config = json.load(f)

    amr_config = config.get('adaptive_mean_reversion', {})
    market_rotation = config.get('market_rotation', {})

    profit_target = amr_config.get('profit_target_percentage', 1.7)
    stop_loss = amr_config.get('stop_loss_percentage', 1.0)
    max_hold = amr_config.get('max_hold_hours', 72)

    print(f"\n{'='*120}")
    print(f"BACKTEST WITH ACTUAL FEE STRUCTURE")
    print(f"{'='*120}")
    print(f"Coinbase Taker Fee: {TAKER_FEE*100}%")
    print(f"Federal Tax Rate: {TAX_RATE*100}%")
    print(f"Min Profit Threshold: ${MIN_NET_PROFIT_USD} = {MIN_GAIN_PCT*100:.2f}% gain")
    print(f"\nStrategy Settings:")
    print(f"  Position Size: ${POSITION_SIZE_USD}")
    print(f"  Profit Target: {profit_target}%")
    print(f"  Stop Loss: {stop_loss}%")
    print(f"  Max Hold: {max_hold} hours")
    print(f"  Period: 180 days")
    print(f"{'='*120}\n")

    symbols = [w['symbol'] for w in config['wallets'] if w.get('enabled', False)]

    print(f"Backtesting {len(symbols)} symbols...\n")

    results = []

    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        result = backtest_symbol(
            symbol,
            profit_target_pct=profit_target,
            stop_loss_pct=stop_loss,
            max_hold_hours=max_hold
        )

        if result:
            results.append(result)
            emoji = "✅" if result['profitability_rate'] >= 50 else "⚠️" if result['profitability_rate'] >= 40 else "❌"
            print(f"{emoji} {result['total_trades']} trades, {result['profitability_rate']:.1f}% profitable "
                  f"({result['win_rate']:.1f}% WR), ${result['total_net_pnl_usd']:+.2f}")
        else:
            print("⚠️  No data")

    if not results:
        print("\n⚠️  No results")
        return

    # Sort by profitability rate
    results.sort(key=lambda x: x['profitability_rate'], reverse=True)

    # Summary table
    print(f"\n{'='*120}")
    print(f"RESULTS - {MIN_GAIN_PCT*100:.2f}% GAIN THRESHOLD (${MIN_NET_PROFIT_USD}+ profit)")
    print(f"{'='*120}")
    print(f"{'Symbol':<10} {'Trades':<8} {'Profitable':<12} {'Win Rate':<10} {'Net P/L':<12} {'Avg EV':<12} {'Target %':<10}")
    print(f"{'-'*120}")

    for r in results:
        emoji = "✅" if r['profitability_rate'] >= 50 else "⚠️" if r['profitability_rate'] >= 40 else "❌"
        print(f"{emoji} {r['symbol']:<8} {r['total_trades']:<8} {r['profitability_rate']:>6.1f}%     "
              f"{r['win_rate']:>6.1f}%    ${r['total_net_pnl_usd']:>+9.2f}  ${r['avg_net_pnl_usd']:>+8.2f}   "
              f"{r['target_exit_rate']:>6.1f}%")

    # Overall
    total_trades = sum(r['total_trades'] for r in results)
    total_profitable = sum(r['profitable_trades'] for r in results)
    overall_pr = (total_profitable / total_trades) * 100 if total_trades else 0
    total_net = sum(r['total_net_pnl_usd'] for r in results)
    avg_ev = total_net / total_trades if total_trades else 0

    print(f"{'-'*120}")
    print(f"OVERALL: {total_trades} trades | {overall_pr:.1f}% profitable | ${total_net:+.2f} total | ${avg_ev:+.2f} avg EV")
    print(f"{'='*120}\n")

    # Verdict
    print(f"VERDICT:")
    if overall_pr >= 55:
        print(f"✅ PROFITABLE: {overall_pr:.1f}% of trades meet ${MIN_NET_PROFIT_USD}+ threshold")
    elif overall_pr >= 45:
        print(f"⚠️  MARGINAL: {overall_pr:.1f}% profitability")
    else:
        print(f"❌ NOT VIABLE: Only {overall_pr:.1f}% profitable")
    print()

    # Save
    output = {
        'backtest_date': datetime.now().isoformat(),
        'fee_structure': {
            'taker_fee_pct': TAKER_FEE * 100,
            'tax_rate_pct': TAX_RATE * 100,
            'min_gain_for_profit_pct': MIN_GAIN_PCT * 100
        },
        'results': results,
        'overall': {
            'total_trades': total_trades,
            'profitable_trades': total_profitable,
            'profitability_rate': overall_pr,
            'total_net_pnl_usd': total_net
        }
    }

    with open('backtest_actual_fees_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"📊 Results saved to backtest_actual_fees_results.json\n")


if __name__ == '__main__':
    main()
