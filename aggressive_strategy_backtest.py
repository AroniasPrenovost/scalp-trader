#!/usr/bin/env python3
"""
AGGRESSIVE Profitable Strategy

Goal: Make $500+ per month on $4000-5000 capital (10-12% monthly return)

Strategy Changes:
1. Higher profit targets (3-4% instead of 1.7%)
2. Trade in ALL market conditions (with different approaches)
3. Larger position sizes (up to 90% of capital for high conviction)
4. More frequent trading (30-50 trades per 5 weeks instead of 15)
5. Use leverage during strong setups (2x position)
"""

import json
import numpy as np
from utils.file_helpers import get_property_values_from_crypto_file


def detect_trend(prices, lookback=168):
    """Detect market trend"""
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


class AggressiveTrade:
    def __init__(self, entry_time, entry_price, stop_loss, profit_target, strategy_type, trend, position_size):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.strategy_type = strategy_type
        self.trend = trend
        self.position_size = position_size
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
        self.profit_loss_usd = (exit_price - self.entry_price) / self.entry_price * self.position_size
        self.duration_hours = exit_time - self.entry_time


def backtest_aggressive_strategy(symbol, lookback_hours=840, backtest_period_hours=840):
    """
    Aggressive strategy backtest
    """
    print(f"\n{'='*80}")
    print(f"AGGRESSIVE STRATEGY BACKTEST: {symbol}")
    print(f"{'='*80}\n")

    all_prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=4380)

    if not all_prices or len(all_prices) < lookback_hours + backtest_period_hours:
        print(f"✗ Insufficient data for {symbol}")
        return None

    print(f"✓ Loaded {len(all_prices)} hours of price data")
    print(f"✓ Backtest period: {backtest_period_hours} hours ({backtest_period_hours//24} days)")

    trades = []
    current_trade = None
    start_index = len(all_prices) - backtest_period_hours

    # Starting capital per symbol
    capital = 2600  # Per wallet

    print(f"\nSimulating trading from hour {start_index} to {len(all_prices)}...")
    print("-" * 80)

    for i in range(start_index, len(all_prices)):
        current_hour = i - start_index
        current_price = all_prices[i]
        historical_prices = all_prices[:i]

        if len(historical_prices) < 168:
            continue

        trend = detect_trend(historical_prices, lookback=168)

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

            # Shorter time limit for more turnover
            if current_hour - current_trade.entry_time >= 48:
                current_trade.close(current_hour, current_price, 'TIME_LIMIT')
                trades.append(current_trade)
                current_trade = None
                continue
        else:
            # Look for entry signals
            if len(historical_prices) < 48:
                continue

            ma_12h = np.mean(historical_prices[-12:])
            ma_24h = np.mean(historical_prices[-24:])
            ma_48h = np.mean(historical_prices[-48:])

            strategy_type = None
            position_size = capital * 0.85  # Default 85% of capital

            # AGGRESSIVE MULTI-STRATEGY APPROACH

            if trend == 'uptrend':
                # MOMENTUM: Buy strength, ride the trend
                deviation_from_12h = (current_price - ma_12h) / ma_12h

                # Buy on small pullbacks in uptrends (0.5-1.5% below 12h MA)
                if -0.015 <= deviation_from_12h <= -0.005:
                    strategy_type = 'momentum_uptrend'
                    entry_price = current_price
                    stop_loss = entry_price * 0.97  # 3% stop
                    profit_target = entry_price * 1.04  # 4% target (aggressive)
                    position_size = capital * 0.90  # 90% position in uptrends

            elif trend == 'sideways':
                # MEAN REVERSION: Buy dips, sell rallies
                deviation_from_24h = (current_price - ma_24h) / ma_24h

                # Buy 1.5-2.5% dips in sideways
                if -0.025 <= deviation_from_24h <= -0.015:
                    strategy_type = 'mean_reversion'
                    entry_price = current_price
                    stop_loss = entry_price * 0.975  # 2.5% stop
                    profit_target = entry_price * 1.03  # 3% target
                    position_size = capital * 0.85

            elif trend == 'downtrend':
                # COUNTER-TREND BOUNCES: Catch oversold bounces
                deviation_from_24h = (current_price - ma_24h) / ma_24h
                recent_low = min(historical_prices[-72:])  # 3-day low

                # Only trade extreme oversold in downtrends (4-6% below MA, near lows)
                if -0.06 <= deviation_from_24h <= -0.04:
                    if current_price <= recent_low * 1.02:  # Within 2% of recent low
                        strategy_type = 'bounce_downtrend'
                        entry_price = current_price
                        stop_loss = entry_price * 0.97  # 3% stop
                        profit_target = entry_price * 1.035  # 3.5% target (quick bounce)
                        position_size = capital * 0.70  # Smaller size in downtrends

            # Enter trade if signal found
            if strategy_type:
                current_trade = AggressiveTrade(
                    entry_time=current_hour,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    strategy_type=strategy_type,
                    trend=trend,
                    position_size=position_size
                )

    # Close any open trade
    if current_trade:
        final_price = all_prices[-1]
        current_trade.close(backtest_period_hours - 1, final_price, 'BACKTEST_END')
        trades.append(current_trade)

    # Calculate statistics
    if not trades:
        print("\n✗ No trades generated")
        return {'symbol': symbol, 'total_trades': 0, 'win_rate': 0, 'total_profit_usd': 0, 'trades': []}

    winning_trades = [t for t in trades if t.profit_loss_pct > 0]
    losing_trades = [t for t in trades if t.profit_loss_pct <= 0]

    total_profit_usd = sum(t.profit_loss_usd for t in trades)
    avg_profit_pct = sum(t.profit_loss_pct for t in trades) / len(trades)
    avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0

    win_rate = (len(winning_trades) / len(trades)) * 100

    # Calculate monthly return
    monthly_return_pct = (total_profit_usd / capital) * (30 / (backtest_period_hours / 24)) * 100

    # Print results
    print(f"\nBACKTEST RESULTS:")
    print(f"{'='*80}")
    print(f"Starting Capital: ${capital:.2f}")
    print(f"Total Trades: {len(trades)}")
    print(f"\nWin Rate:")
    print(f"  Overall: {win_rate:.1f}% ({len(winning_trades)} wins, {len(losing_trades)} losses)")

    print(f"\nPerformance:")
    print(f"  Total P/L: ${total_profit_usd:+.2f}")
    print(f"  Return: {(total_profit_usd/capital)*100:+.2f}% over {backtest_period_hours//24} days")
    print(f"  Projected Monthly Return: {monthly_return_pct:+.2f}%")
    print(f"  Average Profit/Trade: {avg_profit_pct:+.2f}%")
    print(f"  Average Win: {avg_win_pct:+.2f}%")
    print(f"  Average Loss: {avg_loss_pct:+.2f}%")

    # Exit reasons
    exit_reasons = {}
    for trade in trades:
        exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1

    print(f"\nExit Reasons:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} trades ({count/len(trades)*100:.1f}%)")

    # Show sample trades
    print(f"\nSample Trades (first 10):")
    print(f"{'-'*80}")
    for idx, trade in enumerate(trades[:10], 1):
        win_loss = "WIN" if trade.profit_loss_pct > 0 else "LOSS"
        print(f"{idx}. [{trade.trend.upper()}][{trade.strategy_type}][{win_loss}] "
              f"Entry: ${trade.entry_price:.4f} | Exit: ${trade.exit_price:.4f} | "
              f"P/L: {trade.profit_loss_pct:+.2f}% (${trade.profit_loss_usd:+.2f})")

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_profit_usd': total_profit_usd,
        'monthly_return_pct': monthly_return_pct,
        'trades': trades
    }


def main():
    print("=" * 80)
    print("AGGRESSIVE STRATEGY BACKTEST (5 WEEKS)")
    print("=" * 80)
    print()
    print("Strategy Rules (AGGRESSIVE):")
    print("  • Uptrend: Momentum (buy 0.5-1.5% pullbacks, 4% target, 3% stop)")
    print("  • Sideways: Mean reversion (buy 1.5-2.5% dips, 3% target, 2.5% stop)")
    print("  • Downtrend: Bounce trades (4-6% oversold, 3.5% target, 3% stop)")
    print("  • Position sizes: 70-90% of capital")
    print("  • Max hold: 48 hours (faster turnover)")
    print()

    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    print(f"Testing {len(enabled_symbols)} enabled wallets: {', '.join(enabled_symbols)}")
    print()

    all_results = {}

    for symbol in enabled_symbols:
        result = backtest_aggressive_strategy(
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
    print("OVERALL SUMMARY - AGGRESSIVE STRATEGY")
    print("=" * 80)

    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_profit = sum(r['total_profit_usd'] for r in all_results.values())
    total_capital = 2600 * len(all_results)

    all_trades = []
    for r in all_results.values():
        all_trades.extend(r['trades'])

    if all_trades:
        all_wins = [t for t in all_trades if t.profit_loss_pct > 0]
        overall_win_rate = (len(all_wins) / len(all_trades)) * 100
    else:
        overall_win_rate = 0

    monthly_return = (total_profit / total_capital) * (30 / 35) * 100

    print(f"\nAcross all {len(all_results)} wallets:")
    print(f"  Total Capital: ${total_capital:.2f}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Overall Win Rate: {overall_win_rate:.1f}%")
    print(f"  Total P/L: ${total_profit:+.2f}")
    print(f"  5-Week Return: {(total_profit/total_capital)*100:+.2f}%")
    print(f"  Projected Monthly Return: {monthly_return:+.2f}%")
    print(f"  Projected Monthly Profit: ${(monthly_return/100)*total_capital:+.2f}")

    print(f"\nPer-Wallet Performance:")
    for symbol, result in sorted(all_results.items(), key=lambda x: x[1]['total_profit_usd'], reverse=True):
        print(f"  {symbol}: {result['total_trades']} trades, "
              f"{result['win_rate']:.1f}% win rate, "
              f"${result['total_profit_usd']:+.2f} ({result['monthly_return_pct']:+.1f}% monthly)")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("\nConservative Strategy (previous):")
    print("  15 trades, 53.3% win rate, +$25 (0.5% return)")
    print("\nAggressive Strategy (This Test):")
    print(f"  {total_trades} trades, {overall_win_rate:.1f}% win rate, ${total_profit:+.2f} ({(total_profit/total_capital)*100:+.1f}% return)")

    if total_profit > 100:
        print(f"\n  ✅ MUCH BETTER! {total_profit/25:.1f}x more profit than conservative strategy")
    else:
        print(f"\n  ⚠️  Needs more work")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
