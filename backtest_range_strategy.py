#!/usr/bin/env python3
"""
Backtest script for Range-Based Support Zone Trading Strategy

This script backtests the range support strategy over the last 2 weeks by:
1. Walking through historical price data hour by hour
2. Checking for buy signals at each point
3. Simulating trades with proper entry/exit logic
4. Calculating win rate, profit/loss, and other metrics
"""

import json
from datetime import datetime
from utils.range_support_strategy import (
    check_range_support_buy_signal,
    calculate_zone_based_targets
)
from utils.file_helpers import get_property_values_from_crypto_file


class Trade:
    """Represents a single trade"""
    def __init__(self, entry_time, entry_price, stop_loss, profit_target, zone_strength):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.zone_strength = zone_strength
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.profit_loss_pct = 0
        self.profit_loss_usd = 0
        self.duration_hours = 0
        self.status = 'open'

    def close(self, exit_time, exit_price, exit_reason):
        """Close the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = 'closed'

        # Calculate profit/loss
        self.profit_loss_pct = ((exit_price - self.entry_price) / self.entry_price) * 100

        # Assume $1000 position size for simplicity
        position_size = 1000
        self.profit_loss_usd = (exit_price - self.entry_price) / self.entry_price * position_size

        # Calculate duration
        self.duration_hours = exit_time - self.entry_time

    def __repr__(self):
        if self.status == 'open':
            return f"Trade(OPEN @ ${self.entry_price:.4f})"
        return f"Trade({self.exit_reason} @ ${self.exit_price:.4f}, P/L: {self.profit_loss_pct:+.2f}%)"


def backtest_symbol(symbol, lookback_hours=336, backtest_period_hours=336):
    """
    Backtest the range strategy for a single symbol

    Args:
        symbol: Trading pair (e.g., 'BTC-USD')
        lookback_hours: Hours of history to use for strategy (default 336 = 14 days)
        backtest_period_hours: Hours to backtest over (default 336 = 14 days)

    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {symbol}")
    print(f"{'='*80}\n")

    # Load all available price data
    all_prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=4380)

    if not all_prices or len(all_prices) < lookback_hours + backtest_period_hours:
        print(f"✗ Insufficient data for {symbol}")
        print(f"  Need: {lookback_hours + backtest_period_hours} hours")
        print(f"  Have: {len(all_prices) if all_prices else 0} hours")
        return None

    print(f"✓ Loaded {len(all_prices)} hours of price data")
    print(f"✓ Strategy lookback window: {lookback_hours} hours ({lookback_hours//24} days)")
    print(f"✓ Backtest period: {backtest_period_hours} hours ({backtest_period_hours//24} days)")

    # Strategy parameters (moderate configuration)
    strategy_params = {
        'min_touches': 2,
        'zone_tolerance_percentage': 3.0,
        'entry_tolerance_percentage': 1.5,
        'extrema_order': 5,
        'lookback_window': lookback_hours
    }

    trades = []
    current_trade = None

    # Start backtest from the point where we have enough history
    start_index = len(all_prices) - backtest_period_hours

    print(f"\nSimulating trading from hour {start_index} to {len(all_prices)}...")
    print("-" * 80)

    for i in range(start_index, len(all_prices)):
        current_hour = i - start_index
        current_price = all_prices[i]

        # Get historical data up to this point for strategy
        historical_prices = all_prices[:i]

        # Check if we're in a trade
        if current_trade:
            # Check exit conditions

            # 1. Stop loss hit
            if current_price <= current_trade.stop_loss:
                current_trade.close(current_hour, current_price, 'STOP_LOSS')
                trades.append(current_trade)
                current_trade = None
                continue

            # 2. Profit target hit
            if current_price >= current_trade.profit_target:
                current_trade.close(current_hour, current_price, 'PROFIT_TARGET')
                trades.append(current_trade)
                current_trade = None
                continue

            # 3. Time-based exit (max 7 days = 168 hours)
            if current_hour - current_trade.entry_time >= 168:
                current_trade.close(current_hour, current_price, 'TIME_LIMIT')
                trades.append(current_trade)
                current_trade = None
                continue

        else:
            # Not in a trade - check for buy signal
            signal = check_range_support_buy_signal(
                prices=historical_prices,
                current_price=current_price,
                **strategy_params
            )

            if signal['signal'] == 'buy':
                # Calculate entry/stop/target
                targets = calculate_zone_based_targets(signal['zone'])

                # Enter trade
                current_trade = Trade(
                    entry_time=current_hour,
                    entry_price=targets['entry_price'],
                    stop_loss=targets['stop_loss'],
                    profit_target=targets['profit_target'],
                    zone_strength=signal['zone_strength']
                )

                # Optional: Print trade entry for debugging
                # print(f"  Hour {current_hour}: BUY @ ${targets['entry_price']:.4f} (Zone: {signal['zone_strength']} touches)")

    # Close any open trade at the end of backtest period
    if current_trade:
        final_price = all_prices[-1]
        current_trade.close(backtest_period_hours - 1, final_price, 'BACKTEST_END')
        trades.append(current_trade)

    # Calculate statistics
    if not trades:
        print("\n✗ No trades generated during backtest period")
        return {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0,
            'total_profit_usd': 0,
            'avg_profit_pct': 0,
            'trades': []
        }

    winning_trades = [t for t in trades if t.profit_loss_pct > 0]
    losing_trades = [t for t in trades if t.profit_loss_pct <= 0]

    total_profit_usd = sum(t.profit_loss_usd for t in trades)
    avg_profit_pct = sum(t.profit_loss_pct for t in trades) / len(trades)
    avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0
    avg_duration_hours = sum(t.duration_hours for t in trades) / len(trades)

    win_rate = (len(winning_trades) / len(trades)) * 100

    # Print results
    print(f"\nBACKTEST RESULTS:")
    print(f"{'='*80}")
    print(f"Total Trades: {len(trades)}")
    print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"Losing Trades: {len(losing_trades)} ({100-win_rate:.1f}%)")
    print(f"\nPerformance:")
    print(f"  Total P/L (on $1000/trade): ${total_profit_usd:+.2f}")
    print(f"  Average Profit/Trade: {avg_profit_pct:+.2f}%")
    print(f"  Average Win: {avg_win_pct:+.2f}%")
    print(f"  Average Loss: {avg_loss_pct:+.2f}%")
    print(f"  Average Trade Duration: {avg_duration_hours:.1f} hours ({avg_duration_hours/24:.1f} days)")

    # Exit reason breakdown
    exit_reasons = {}
    for trade in trades:
        exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1

    print(f"\nExit Reasons:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} trades ({count/len(trades)*100:.1f}%)")

    # Show individual trades
    print(f"\nIndividual Trades:")
    print(f"{'-'*80}")
    for idx, trade in enumerate(trades, 1):
        print(f"{idx}. Entry: ${trade.entry_price:.4f} | Exit: ${trade.exit_price:.4f} | "
              f"P/L: {trade.profit_loss_pct:+.2f}% | Duration: {trade.duration_hours}h | "
              f"Reason: {trade.exit_reason}")

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'total_profit_usd': total_profit_usd,
        'avg_profit_pct': avg_profit_pct,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'avg_duration_hours': avg_duration_hours,
        'exit_reasons': exit_reasons,
        'trades': trades
    }


def main():
    """
    Main backtest function - runs backtest on all enabled wallets
    """
    print("=" * 80)
    print("RANGE SUPPORT STRATEGY - BACKTEST (LAST 2 WEEKS)")
    print("=" * 80)
    print()

    # Load config to get enabled wallets
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    print(f"Backtesting {len(enabled_symbols)} enabled wallets: {', '.join(enabled_symbols)}")
    print(f"Period: Last 2 weeks (336 hours)")
    print(f"Strategy lookback: 2 weeks (336 hours)")
    print(f"Position size: $1,000 per trade")
    print()

    all_results = {}

    for symbol in enabled_symbols:
        result = backtest_symbol(
            symbol=symbol,
            lookback_hours=336,  # 14 days for strategy
            backtest_period_hours=336  # 14 days backtest
        )
        if result:
            all_results[symbol] = result

    # Overall summary
    if not all_results:
        print("\n✗ No backtest results available")
        return

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - ALL WALLETS")
    print("=" * 80)

    total_trades_all = sum(r['total_trades'] for r in all_results.values())
    total_winning_all = sum(r['winning_trades'] for r in all_results.values())
    total_losing_all = sum(r['losing_trades'] for r in all_results.values())
    total_profit_all = sum(r['total_profit_usd'] for r in all_results.values())

    overall_win_rate = (total_winning_all / total_trades_all * 100) if total_trades_all > 0 else 0

    print(f"\nAcross all {len(all_results)} wallets:")
    print(f"  Total Trades: {total_trades_all}")
    print(f"  Winning Trades: {total_winning_all} ({overall_win_rate:.1f}%)")
    print(f"  Losing Trades: {total_losing_all} ({100-overall_win_rate:.1f}%)")
    print(f"  Total P/L: ${total_profit_all:+.2f}")

    print(f"\nPer-Wallet Performance:")
    for symbol, result in sorted(all_results.items(), key=lambda x: x[1]['total_profit_usd'], reverse=True):
        print(f"  {symbol}: {result['total_trades']} trades, "
              f"{result['win_rate']:.1f}% win rate, "
              f"${result['total_profit_usd']:+.2f} P/L")

    # Best and worst performers
    if all_results:
        best_symbol = max(all_results.items(), key=lambda x: x[1]['total_profit_usd'])
        worst_symbol = min(all_results.items(), key=lambda x: x[1]['total_profit_usd'])

        print(f"\nBest Performer: {best_symbol[0]} (${best_symbol[1]['total_profit_usd']:+.2f})")
        print(f"Worst Performer: {worst_symbol[0]} (${worst_symbol[1]['total_profit_usd']:+.2f})")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print("\nNote: This is a simplified backtest using $1,000 per trade.")
    print("Real results may vary based on:")
    print("  - Actual position sizing")
    print("  - Exchange fees (not included)")
    print("  - Taxes (not included)")
    print("  - Slippage and execution delays")
    print("  - Different exit strategies (trailing stops, etc.)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
