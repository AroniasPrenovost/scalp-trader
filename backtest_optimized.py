#!/usr/bin/env python3
"""
Optimized Backtest - Focus on What Works

Only includes improvements that increase win rate without restricting trade frequency:
1. ATR-based stop losses (1.5x ATR minimum)
2. Volatility-aware stop widening (MAX of 1.5xATR or 1.5x volatility range)
3. Lower profit targets (1.5% instead of 2.5%)
4. Very basic resistance check (only within 1% of peak - extreme cases only)
"""

import json
import numpy as np
from utils.range_support_strategy import (
    check_range_support_buy_signal,
    calculate_zone_based_targets
)
from utils.file_helpers import get_property_values_from_crypto_file
from utils.matplotlib import calculate_rsi


def calculate_atr(prices, period=24):
    """Calculate 24-hour ATR"""
    if len(prices) < period + 1:
        return None

    highs = []
    lows = []

    for i in range(len(prices) - period, len(prices)):
        price = prices[i]
        highs.append(price * 1.002)
        lows.append(price * 0.998)

    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prices[len(prices) - period + i - 1]),
            abs(lows[i] - prices[len(prices) - period + i - 1])
        )
        true_ranges.append(tr)

    return np.mean(true_ranges) if true_ranges else None


def calculate_volatility_range(prices, period=24):
    """Calculate price volatility range over period"""
    if len(prices) < period:
        return None
    recent = prices[-period:]
    return max(recent) - min(recent)


def is_at_peak(current_price, prices, threshold=0.01):
    """Check if price is within 1% of recent peak (only extreme cases)"""
    if len(prices) < 50:
        return False

    recent_high = max(prices[-50:])
    distance_from_high = (recent_high - current_price) / current_price

    return distance_from_high < threshold


def detect_sideways_market(prices, period=168):
    """Detect sideways market (for tracking only)"""
    if len(prices) < period:
        return False

    recent = prices[-period:]
    price_range = (max(recent) - min(recent)) / min(recent)

    return price_range < 0.10


class OptimizedTrade:
    """Trade with optimized improvements"""
    def __init__(self, entry_time, entry_price, stop_loss, profit_target,
                 atr, volatility_range, sideways_market):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.atr = atr
        self.volatility_range = volatility_range
        self.sideways_market = sideways_market
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

        self.profit_loss_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        position_size = 2000
        self.profit_loss_usd = (exit_price - self.entry_price) / self.entry_price * position_size
        self.duration_hours = exit_time - self.entry_time

    def __repr__(self):
        marker = "⚠" if self.sideways_market else "✓"
        if self.status == 'open':
            return f"Trade[{marker}](OPEN @ ${self.entry_price:.4f})"
        return f"Trade[{marker}]({self.exit_reason} @ ${self.exit_price:.4f}, P/L: {self.profit_loss_pct:+.2f}%)"


def backtest_optimized(symbol, lookback_hours=336, backtest_period_hours=336):
    """
    Backtest with optimized improvements only
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZED STRATEGY BACKTEST: {symbol}")
    print(f"{'='*80}\n")

    # Load all available price data
    all_prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=4380)

    if not all_prices or len(all_prices) < lookback_hours + backtest_period_hours:
        print(f"✗ Insufficient data for {symbol}")
        return None

    print(f"✓ Loaded {len(all_prices)} hours of price data")
    print(f"✓ Optimized improvements: ATR stops, volatility-aware, 1.5% targets, peak avoidance")
    print(f"✓ Backtest period: {backtest_period_hours} hours ({backtest_period_hours//24} days)")

    # Strategy parameters
    strategy_params = {
        'min_touches': 2,
        'zone_tolerance_percentage': 3.0,
        'entry_tolerance_percentage': 1.5,
        'extrema_order': 5,
        'lookback_window': lookback_hours
    }

    trades = []
    current_trade = None
    rejected_at_peak = 0

    start_index = len(all_prices) - backtest_period_hours

    print(f"\nSimulating trading from hour {start_index} to {len(all_prices)}...")
    print("-" * 80)

    for i in range(start_index, len(all_prices)):
        current_hour = i - start_index
        current_price = all_prices[i]
        historical_prices = all_prices[:i]

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

            if current_hour - current_trade.entry_time >= 168:
                current_trade.close(current_hour, current_price, 'TIME_LIMIT')
                trades.append(current_trade)
                current_trade = None
                continue
        else:
            # Check for buy signal with range strategy
            range_signal = check_range_support_buy_signal(
                prices=historical_prices,
                current_price=current_price,
                **strategy_params
            )

            if range_signal['signal'] == 'buy' and range_signal['zone']:
                # Only reject if literally at peak (within 1%)
                if is_at_peak(current_price, historical_prices):
                    rejected_at_peak += 1
                    continue

                # Calculate ATR and volatility range
                atr = calculate_atr(historical_prices, period=24)
                volatility_range = calculate_volatility_range(historical_prices, period=24)

                if atr is None or volatility_range is None:
                    continue

                # Detect sideways market (tracking only)
                sideways = detect_sideways_market(historical_prices)

                # Calculate improved stop loss (1.0x ATR for tighter risk control)
                # This gives us better risk/reward with 1.5% profit targets
                atr_stop_distance = 1.0 * atr
                volatility_stop_distance = 1.0 * volatility_range
                stop_distance = max(atr_stop_distance, volatility_stop_distance)

                improved_stop_loss = current_price - stop_distance

                # Lower profit target (1.5% instead of 2.5%)
                improved_profit_target = current_price * 1.015

                current_trade = OptimizedTrade(
                    entry_time=current_hour,
                    entry_price=current_price,
                    stop_loss=improved_stop_loss,
                    profit_target=improved_profit_target,
                    atr=atr,
                    volatility_range=volatility_range,
                    sideways_market=sideways
                )

    # Close any open trade
    if current_trade:
        final_price = all_prices[-1]
        current_trade.close(backtest_period_hours - 1, final_price, 'BACKTEST_END')
        trades.append(current_trade)

    # Calculate statistics
    if not trades:
        print("\n✗ No trades generated")
        print(f"  Rejected at peak: {rejected_at_peak}")
        return {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0,
            'total_profit_usd': 0,
            'trades': []
        }

    # Separate by market condition
    sideways_trades = [t for t in trades if t.sideways_market]
    trending_trades = [t for t in trades if not t.sideways_market]

    winning_trades = [t for t in trades if t.profit_loss_pct > 0]
    losing_trades = [t for t in trades if t.profit_loss_pct <= 0]

    total_profit_usd = sum(t.profit_loss_usd for t in trades)
    avg_profit_pct = sum(t.profit_loss_pct for t in trades) / len(trades)
    avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0

    win_rate = (len(winning_trades) / len(trades)) * 100

    # Sideways vs trending performance
    if sideways_trades:
        sideways_wins = [t for t in sideways_trades if t.profit_loss_pct > 0]
        sideways_win_rate = (len(sideways_wins) / len(sideways_trades)) * 100
    else:
        sideways_win_rate = 0

    if trending_trades:
        trending_wins = [t for t in trending_trades if t.profit_loss_pct > 0]
        trending_win_rate = (len(trending_wins) / len(trending_trades)) * 100
    else:
        trending_win_rate = 0

    # Print results
    print(f"\nBACKTEST RESULTS:")
    print(f"{'='*80}")
    print(f"Total Trades: {len(trades)}")
    print(f"  - In Trending Markets: {len(trending_trades)} trades ({trending_win_rate:.1f}% win rate)")
    print(f"  - In Sideways Markets: {len(sideways_trades)} trades ({sideways_win_rate:.1f}% win rate)")
    print(f"\nWin Rate:")
    print(f"  - Overall: {win_rate:.1f}% ({len(winning_trades)} wins, {len(losing_trades)} losses)")

    print(f"\nPerformance:")
    print(f"  Total P/L: ${total_profit_usd:+.2f}")
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

    print(f"\nFilters:")
    print(f"  Rejected at peak (within 1%): {rejected_at_peak}")

    # Show trades
    print(f"\nIndividual Trades:")
    print(f"{'-'*80}")
    for idx, trade in enumerate(trades, 1):
        market_type = "SIDEWAYS" if trade.sideways_market else "TRENDING"
        win_loss = "WIN" if trade.profit_loss_pct > 0 else "LOSS"
        print(f"{idx}. [{market_type}][{win_loss}] Entry: ${trade.entry_price:.4f} | Exit: ${trade.exit_price:.4f} | "
              f"P/L: {trade.profit_loss_pct:+.2f}% (${trade.profit_loss_usd:+.2f}) | "
              f"{trade.exit_reason} | Stop: ${trade.stop_loss:.4f} | Target: ${trade.profit_target:.4f}")

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'trending_trades': len(trending_trades),
        'sideways_trades': len(sideways_trades),
        'win_rate': win_rate,
        'trending_win_rate': trending_win_rate,
        'sideways_win_rate': sideways_win_rate,
        'total_profit_usd': total_profit_usd,
        'avg_profit_pct': avg_profit_pct,
        'trades': trades
    }


def main():
    """Run optimized strategy backtest"""
    print("=" * 80)
    print("OPTIMIZED STRATEGY BACKTEST (PAST 2 WEEKS)")
    print("=" * 80)
    print()
    print("Testing ONLY the most impactful improvements:")
    print("  1. ✅ ATR-based stop losses (1.0x ATR - balanced risk/reward)")
    print("  2. ✅ Volatility-aware stops (MAX of 1.0xATR or 1.0x volatility)")
    print("  3. ✅ Lower profit targets (1.5% instead of 2.5%)")
    print("  4. ✅ Peak avoidance (only reject if within 1% of literal peak)")
    print("  5. ✅ Sideways market tracking (awareness, no position changes)")
    print()
    print("  ❌ Removed: Multi-timeframe alignment (too restrictive)")
    print("  ❌ Removed: Near resistance check (counter to range trading)")
    print("  ❌ Removed: Position size reduction (let it trade)")
    print()

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    print(f"Testing {len(enabled_symbols)} enabled wallets: {', '.join(enabled_symbols)}")
    print()

    all_results = {}

    for symbol in enabled_symbols:
        result = backtest_optimized(
            symbol=symbol,
            lookback_hours=336,
            backtest_period_hours=336
        )
        if result:
            all_results[symbol] = result

    # Overall summary
    if not all_results:
        print("\n✗ No results")
        return

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - OPTIMIZED STRATEGY")
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
    print("\nOriginal (Basic Range Strategy):")
    print("  ETH: 6 trades, 66.7% win rate, +$152.45")
    print("  XRP: 5 trades, 60.0% win rate, +$100.40")
    print("  Combined: 11 trades, 63.6% win rate, +$252.85")
    print("\nOptimized Strategy (This Test):")
    print(f"  Combined: {total_trades} trades, {overall_win_rate:.1f}% win rate, ${total_profit:+.2f}")

    if total_profit > 252.85:
        improvement = total_profit - 252.85
        improvement_pct = (total_profit / 252.85 - 1) * 100
        print(f"  ✅ IMPROVEMENT: ${improvement:+.2f} ({improvement_pct:+.1f}%)")
    else:
        decline = total_profit - 252.85
        decline_pct = (total_profit / 252.85 - 1) * 100
        print(f"  ⚠️  CHANGE: ${decline:+.2f} ({decline_pct:+.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
