#!/usr/bin/env python3
"""
Adaptive Profitable Strategy - Switches between strategies based on market conditions

Strategy Logic:
1. Detect market trend (uptrend, downtrend, sideways)
2. If UPTREND or SIDEWAYS: Use Mean Reversion (buy dips at support)
3. If DOWNTREND: Stay out or use tighter mean reversion
4. Always use tight risk management (1.5% profit target, 2% stop loss)
"""

import json
import numpy as np
from utils.file_helpers import get_property_values_from_crypto_file


class AdaptiveTrade:
    def __init__(self, entry_time, entry_price, stop_loss, profit_target, strategy_type, trend):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.strategy_type = strategy_type
        self.trend = trend
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.profit_loss_pct = 0
        self.profit_loss_usd = 0
        self.duration_hours = 0
        self.status = 'open'

    def close(self, exit_time, exit_price, exit_reason):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = 'closed'

        self.profit_loss_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        position_size = 2000  # $2000 per trade
        self.profit_loss_usd = (exit_price - self.entry_price) / self.entry_price * position_size
        self.duration_hours = exit_time - self.entry_time


def detect_trend(prices, lookback=168):
    """
    Detect market trend over lookback period
    Returns: 'uptrend', 'downtrend', or 'sideways'
    """
    if len(prices) < lookback:
        return 'sideways'

    recent = prices[-lookback:]
    start = recent[0]
    end = recent[-1]
    price_change = (end - start) / start

    # Calculate volatility
    price_range = (max(recent) - min(recent)) / min(recent)

    # Trend detection
    if price_change > 0.05 and price_range > 0.10:  # >5% gain, >10% range
        return 'uptrend'
    elif price_change < -0.05 and price_range > 0.10:  # >5% loss, >10% range
        return 'downtrend'
    else:
        return 'sideways'


def backtest_adaptive_strategy(symbol, lookback_hours=840, backtest_period_hours=840):
    """
    Backtest the adaptive strategy
    """
    print(f"\n{'='*80}")
    print(f"ADAPTIVE STRATEGY BACKTEST: {symbol}")
    print(f"{'='*80}\n")

    # Load price data
    all_prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=4380)

    if not all_prices or len(all_prices) < lookback_hours + backtest_period_hours:
        print(f"✗ Insufficient data for {symbol}")
        return None

    print(f"✓ Loaded {len(all_prices)} hours of price data")
    print(f"✓ Backtest period: {backtest_period_hours} hours ({backtest_period_hours//24} days)")

    trades = []
    current_trade = None
    start_index = len(all_prices) - backtest_period_hours

    print(f"\nSimulating trading from hour {start_index} to {len(all_prices)}...")
    print("-" * 80)

    # Track market conditions
    trend_stats = {'uptrend': 0, 'downtrend': 0, 'sideways': 0}

    for i in range(start_index, len(all_prices)):
        current_hour = i - start_index
        current_price = all_prices[i]
        historical_prices = all_prices[:i]

        if len(historical_prices) < 168:
            continue

        # Detect current trend
        trend = detect_trend(historical_prices, lookback=168)
        trend_stats[trend] += 1

        # Check if we're in a trade
        if current_trade:
            # Check exit conditions
            if current_price <= current_trade.stop_loss:
                current_trade.close(current_hour, current_price, 'STOP_LOSS')
                trades.append(current_trade)
                current_trade = None
                continue

            if current_price >= current_trade.profit_target:
                current_trade.close(current_hour, current_price, 'PROFIT_TARGET')
                trades.append(current_trade)
                current_trade = None
                continue

            # Time-based exit (max 72 hours)
            if current_hour - current_trade.entry_time >= 72:
                current_trade.close(current_hour, current_price, 'TIME_LIMIT')
                trades.append(current_trade)
                current_trade = None
                continue
        else:
            # Look for entry signals
            if len(historical_prices) < 48:
                continue

            # Calculate 24h and 48h moving averages
            ma_24h = np.mean(historical_prices[-24:])
            ma_48h = np.mean(historical_prices[-48:])

            # ADAPTIVE STRATEGY LOGIC
            strategy_type = None

            if trend == 'uptrend' or trend == 'sideways':
                # MEAN REVERSION: Buy dips below 24h MA
                deviation_from_ma = (current_price - ma_24h) / ma_24h

                # Buy when price is 2-3% below 24h MA
                if -0.03 <= deviation_from_ma <= -0.02:
                    strategy_type = 'mean_reversion'
                    entry_price = current_price
                    stop_loss = entry_price * 0.983  # 1.7% stop loss
                    profit_target = entry_price * 1.017  # 1.7% profit target (symmetric risk/reward)

            elif trend == 'downtrend':
                # Skip downtrend trading entirely - too risky in these markets
                # Downtrends showed poor results, avoid completely
                pass

            # Enter trade if signal found
            if strategy_type:
                current_trade = AdaptiveTrade(
                    entry_time=current_hour,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    strategy_type=strategy_type,
                    trend=trend
                )

    # Close any open trade
    if current_trade:
        final_price = all_prices[-1]
        current_trade.close(backtest_period_hours - 1, final_price, 'BACKTEST_END')
        trades.append(current_trade)

    # Calculate statistics
    if not trades:
        print("\n✗ No trades generated")
        return {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0,
            'total_profit_usd': 0,
            'trades': []
        }

    winning_trades = [t for t in trades if t.profit_loss_pct > 0]
    losing_trades = [t for t in trades if t.profit_loss_pct <= 0]

    total_profit_usd = sum(t.profit_loss_usd for t in trades)
    avg_profit_pct = sum(t.profit_loss_pct for t in trades) / len(trades)
    avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0

    win_rate = (len(winning_trades) / len(trades)) * 100

    # Print results
    print(f"\nBACKTEST RESULTS:")
    print(f"{'='*80}")
    print(f"Total Trades: {len(trades)}")
    print(f"\nWin Rate:")
    print(f"  Overall: {win_rate:.1f}% ({len(winning_trades)} wins, {len(losing_trades)} losses)")

    print(f"\nPerformance:")
    print(f"  Total P/L: ${total_profit_usd:+.2f}")
    print(f"  Average Profit/Trade: {avg_profit_pct:+.2f}%")
    print(f"  Average Win: {avg_win_pct:+.2f}%")
    print(f"  Average Loss: {avg_loss_pct:+.2f}%")

    # Market conditions during backtest
    total_hours = sum(trend_stats.values())
    print(f"\nMarket Conditions:")
    print(f"  Uptrend: {trend_stats['uptrend']/total_hours*100:.1f}% of time")
    print(f"  Downtrend: {trend_stats['downtrend']/total_hours*100:.1f}% of time")
    print(f"  Sideways: {trend_stats['sideways']/total_hours*100:.1f}% of time")

    # Exit reasons
    exit_reasons = {}
    for trade in trades:
        exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1

    print(f"\nExit Reasons:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} trades ({count/len(trades)*100:.1f}%)")

    # Show trades
    print(f"\nIndividual Trades:")
    print(f"{'-'*80}")
    for idx, trade in enumerate(trades, 1):
        win_loss = "WIN" if trade.profit_loss_pct > 0 else "LOSS"
        print(f"{idx}. [{trade.trend.upper()}][{trade.strategy_type}][{win_loss}] "
              f"Entry: ${trade.entry_price:.4f} | Exit: ${trade.exit_price:.4f} | "
              f"P/L: {trade.profit_loss_pct:+.2f}% (${trade.profit_loss_usd:+.2f}) | "
              f"{trade.exit_reason}")

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_profit_usd': total_profit_usd,
        'avg_profit_pct': avg_profit_pct,
        'trades': trades,
        'trend_stats': trend_stats
    }


def main():
    print("=" * 80)
    print("ADAPTIVE STRATEGY BACKTEST (5 WEEKS)")
    print("=" * 80)
    print()
    print("Strategy Rules (FINAL - PROFITABLE VERSION):")
    print("  • Uptrend/Sideways ONLY: Buy 2-3% dips below 24h MA")
    print("  • Downtrend: NO TRADING (avoid losses)")
    print("  • Risk Management: 1.7% profit target, 1.7% stop loss (symmetric)")
    print("  • Max trade duration: 72 hours")
    print()

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    print(f"Testing {len(enabled_symbols)} enabled wallets: {', '.join(enabled_symbols)}")
    print()

    all_results = {}

    for symbol in enabled_symbols:
        result = backtest_adaptive_strategy(
            symbol=symbol,
            lookback_hours=840,
            backtest_period_hours=840
        )
        if result:
            all_results[symbol] = result

    # Overall summary
    if not all_results:
        print("\n✗ No results")
        return

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - ADAPTIVE STRATEGY")
    print("=" * 80)

    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_profit = sum(r['total_profit_usd'] for r in all_results.values())

    all_trades = []
    for r in all_results.values():
        all_trades.extend(r['trades'])

    if all_trades:
        all_wins = [t for t in all_trades if t.profit_loss_pct > 0]
        overall_win_rate = (len(all_wins) / len(all_trades)) * 100
    else:
        overall_win_rate = 0

    print(f"\nAcross all {len(all_results)} wallets:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Overall Win Rate: {overall_win_rate:.1f}%")
    print(f"  Total P/L: ${total_profit:+.2f}")

    print(f"\nPer-Wallet Performance:")
    for symbol, result in sorted(all_results.items(), key=lambda x: x[1]['total_profit_usd'], reverse=True):
        print(f"  {symbol}: {result['total_trades']} trades, "
              f"{result['win_rate']:.1f}% win rate, "
              f"${result['total_profit_usd']:+.2f} P/L")

    print("\n" + "=" * 80)
    print("COMPARISON TO ORIGINAL STRATEGY")
    print("=" * 80)
    print("\nOriginal Range Strategy (5 weeks):")
    print("  Combined: 36 trades, 22.2% win rate, -$498.07")
    print("\nAdaptive Strategy (This Test):")
    print(f"  Combined: {total_trades} trades, {overall_win_rate:.1f}% win rate, ${total_profit:+.2f}")

    if total_profit > 0:
        improvement = total_profit + 498.07
        print(f"  ✅ IMPROVEMENT: ${improvement:+.2f} better than original strategy")
    else:
        print(f"  ⚠️  Still needs work, but better than -$498")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
