#!/usr/bin/env python3
"""
Backtest with ALL New Improvements (Past 2 Weeks)

Tests the new improvements:
1. ATR-based stop losses (1.5x ATR minimum)
2. Volatility-aware stop widening (MAX of 1.5xATR or 1.5x volatility range)
3. Entry timing - wait for pullbacks (not near resistance)
4. Multi-timeframe support/resistance (3+ timeframe alignment)
5. Sideways market position reduction (50% smaller)
6. Lower profit targets (1.5% instead of 2.5%)
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
        # Use a small range around each hourly close as high/low
        price = prices[i]
        highs.append(price * 1.002)  # Simulate +0.2% wick
        lows.append(price * 0.998)   # Simulate -0.2% wick

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


def is_near_resistance(current_price, prices, minor_threshold=0.05, major_threshold=0.08):
    """Check if price is too close to resistance (within 5% minor, 8% major) - RELAXED"""
    if len(prices) < 100:
        return False

    # Find recent highs (resistance levels)
    recent_high = max(prices[-30:])
    major_high = max(prices[-90:]) if len(prices) >= 90 else recent_high

    # Check if we're within danger zones
    minor_resistance_distance = (recent_high - current_price) / current_price
    major_resistance_distance = (major_high - current_price) / current_price

    if minor_resistance_distance < minor_threshold:
        return True, "near_minor_resistance"
    if major_resistance_distance < major_threshold:
        return True, "near_major_resistance"

    return False, None


def is_near_support(current_price, support_level, tolerance=0.04):
    """Check if price is within 4% above support (good entry zone) - RELAXED"""
    distance_above = (current_price - support_level) / support_level
    return 0 <= distance_above <= tolerance


def check_multi_timeframe_alignment(prices, support_level, timeframes=[14*24, 30*24, 90*24]):
    """Check if support level aligns across multiple timeframes (need 2+) - RELAXED"""
    aligned_count = 0

    for tf_hours in timeframes:
        if len(prices) < tf_hours:
            continue

        tf_prices = prices[-tf_hours:]
        tf_low = min(tf_prices)

        # Check if support level is within 5% of this timeframe's low
        if abs(support_level - tf_low) / tf_low < 0.05:
            aligned_count += 1

    return aligned_count >= 2  # Relaxed from 3 to 2


def detect_sideways_market(prices, period=168):
    """Detect sideways market (low trend strength)"""
    if len(prices) < period:
        return False

    recent = prices[-period:]
    price_range = (max(recent) - min(recent)) / min(recent)

    # Sideways if range < 10% over past week
    return price_range < 0.10


class ImprovedTrade:
    """Trade with all new improvements"""
    def __init__(self, entry_time, entry_price, stop_loss, profit_target,
                 atr, volatility_range, sideways_market, position_size):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.atr = atr
        self.volatility_range = volatility_range
        self.sideways_market = sideways_market
        self.position_size = position_size
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
        self.profit_loss_usd = (exit_price - self.entry_price) / self.entry_price * self.position_size
        self.duration_hours = exit_time - self.entry_time

    def __repr__(self):
        marker = "⚠" if self.sideways_market else "✓"
        if self.status == 'open':
            return f"Trade[{marker}](OPEN @ ${self.entry_price:.4f}, size=${self.position_size})"
        return f"Trade[{marker}]({self.exit_reason} @ ${self.exit_price:.4f}, P/L: {self.profit_loss_pct:+.2f}%)"


def backtest_with_improvements(symbol, lookback_hours=336, backtest_period_hours=336):
    """
    Backtest with all new improvements
    """
    print(f"\n{'='*80}")
    print(f"IMPROVED STRATEGY BACKTEST: {symbol}")
    print(f"{'='*80}\n")

    # Load all available price data
    all_prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=4380)

    if not all_prices or len(all_prices) < lookback_hours + backtest_period_hours:
        print(f"✗ Insufficient data for {symbol}")
        return None

    print(f"✓ Loaded {len(all_prices)} hours of price data")
    print(f"✓ Testing improvements: ATR stops, volatility-aware, entry timing, multi-TF, sideways detection")
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
    rejected_entries = {'near_resistance': 0, 'sideways_market': 0, 'no_mtf_alignment': 0}

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
                # NEW: Check entry timing - reject if near resistance
                near_resistance, resistance_type = is_near_resistance(current_price, historical_prices)
                if near_resistance:
                    rejected_entries[resistance_type] = rejected_entries.get(resistance_type, 0) + 1
                    continue

                # NEW: Check if near support (within 2% above)
                support_level = range_signal['zone']['zone_price_avg']
                if not is_near_support(current_price, support_level):
                    continue

                # NEW: Check multi-timeframe alignment
                if not check_multi_timeframe_alignment(historical_prices, support_level):
                    rejected_entries['no_mtf_alignment'] += 1
                    continue

                # Calculate ATR and volatility range
                atr = calculate_atr(historical_prices, period=24)
                volatility_range = calculate_volatility_range(historical_prices, period=24)

                if atr is None or volatility_range is None:
                    continue

                # NEW: Detect sideways market
                sideways = detect_sideways_market(historical_prices)

                # Calculate improved stop loss (MAX of 1.5xATR or 1.5x volatility range)
                atr_stop_distance = 1.5 * atr
                volatility_stop_distance = 1.5 * volatility_range
                stop_distance = max(atr_stop_distance, volatility_stop_distance)

                improved_stop_loss = current_price - stop_distance

                # NEW: Lower profit target (1.5% instead of 2.5%)
                improved_profit_target = current_price * 1.015

                # NEW: Reduce position size by 50% in sideways markets
                base_position_size = 2000
                position_size = base_position_size * 0.5 if sideways else base_position_size

                if sideways:
                    rejected_entries['sideways_market'] += 1
                    # Still take the trade but with reduced size

                current_trade = ImprovedTrade(
                    entry_time=current_hour,
                    entry_price=current_price,
                    stop_loss=improved_stop_loss,
                    profit_target=improved_profit_target,
                    atr=atr,
                    volatility_range=volatility_range,
                    sideways_market=sideways,
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
        print(f"\nRejected entries:")
        for reason, count in rejected_entries.items():
            print(f"  {reason}: {count}")
        return {
            'symbol': symbol,
            'total_trades': 0,
            'trending_trades': 0,
            'sideways_trades': 0,
            'win_rate': 0,
            'total_profit_usd': 0,
            'avg_profit_pct': 0,
            'trades': [],
            'rejected_entries': rejected_entries
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

    # Print results
    print(f"\nBACKTEST RESULTS:")
    print(f"{'='*80}")
    print(f"Total Trades: {len(trades)}")
    print(f"  - In Trending Markets: {len(trending_trades)} trades")
    print(f"  - In Sideways Markets: {len(sideways_trades)} trades (50% position size)")
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

    print(f"\nRejected Entry Signals:")
    for reason, count in sorted(rejected_entries.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")

    # Show trades
    print(f"\nIndividual Trades:")
    print(f"{'-'*80}")
    for idx, trade in enumerate(trades, 1):
        market_type = "SIDEWAYS" if trade.sideways_market else "TRENDING"
        print(f"{idx}. [{market_type}] Entry: ${trade.entry_price:.4f} | Exit: ${trade.exit_price:.4f} | "
              f"P/L: {trade.profit_loss_pct:+.2f}% (${trade.profit_loss_usd:+.2f}) | "
              f"{trade.exit_reason} | Stop: ${trade.stop_loss:.4f} | Target: ${trade.profit_target:.4f}")

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'trending_trades': len(trending_trades),
        'sideways_trades': len(sideways_trades),
        'win_rate': win_rate,
        'total_profit_usd': total_profit_usd,
        'avg_profit_pct': avg_profit_pct,
        'trades': trades,
        'rejected_entries': rejected_entries
    }


def main():
    """Run improved strategy backtest"""
    print("=" * 80)
    print("IMPROVED STRATEGY BACKTEST (PAST 2 WEEKS)")
    print("=" * 80)
    print()
    print("Testing ALL new improvements (BALANCED PARAMETERS):")
    print("  1. ATR-based stop losses (1.5x ATR minimum)")
    print("  2. Volatility-aware stop widening (MAX of 1.5xATR or 1.5x volatility)")
    print("  3. Entry timing - avoid resistance (5% minor, 8% major) - RELAXED")
    print("  4. Entry near support (within 4% above) - RELAXED")
    print("  5. Multi-timeframe S/R alignment (2+ timeframes) - RELAXED")
    print("  6. Sideways market position reduction (50% smaller)")
    print("  7. Lower profit targets (1.5% instead of 2.5%)")
    print()

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    print(f"Testing {len(enabled_symbols)} enabled wallets: {', '.join(enabled_symbols)}")
    print()

    all_results = {}

    for symbol in enabled_symbols:
        result = backtest_with_improvements(
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
    print("OVERALL SUMMARY - IMPROVED STRATEGY")
    print("=" * 80)

    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_profit = sum(r['total_profit_usd'] for r in all_results.values())
    total_trending = sum(r['trending_trades'] for r in all_results.values())
    total_sideways = sum(r['sideways_trades'] for r in all_results.values())

    print(f"\nAcross all {len(all_results)} wallets:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Trending Market Trades: {total_trending}")
    print(f"  Sideways Market Trades: {total_sideways} (reduced position size)")
    print(f"  Total P/L: ${total_profit:+.2f}")

    print(f"\nPer-Wallet Performance:")
    for symbol, result in sorted(all_results.items(), key=lambda x: x[1]['total_profit_usd'], reverse=True):
        print(f"  {symbol}: {result['total_trades']} trades, "
              f"{result['win_rate']:.1f}% win rate, "
              f"${result['total_profit_usd']:+.2f} P/L")

    print("\n" + "=" * 80)
    print("COMPARISON TO PREVIOUS STRATEGY")
    print("=" * 80)
    print("\nPrevious (Basic Range Strategy):")
    print("  ETH: 6 trades, 66.7% win rate, +$152.45")
    print("  XRP: 5 trades, 60.0% win rate, +$100.40")
    print("  Combined: 11 trades, 63.6% win rate, +$252.85")
    print("\nWith All Improvements (This Test):")
    print(f"  Combined: {total_trades} trades, ${total_profit:+.2f}")
    print(f"  Improvement: ${total_profit - 252.85:+.2f} ({((total_profit/252.85 - 1) * 100):+.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
