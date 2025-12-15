#!/usr/bin/env python3
"""
Backtest with AI + Range Strategy Confluence Simulation

This script simulates how the AI would have performed over the last 2 weeks
if it had the range strategy information integrated (which it now does).

We simulate AI decisions based on:
1. Range strategy signals (support zones)
2. Technical indicators (RSI, price position)
3. Confluence logic (both must align for HIGH confidence)
"""

import json
import numpy as np
from utils.range_support_strategy import (
    check_range_support_buy_signal,
    calculate_zone_based_targets
)
from utils.file_helpers import get_property_values_from_crypto_file
from utils.matplotlib import calculate_rsi


class SimulatedTrade:
    """Represents a simulated trade with AI + Range confluence"""
    def __init__(self, entry_time, entry_price, stop_loss, profit_target,
                 range_strength, ai_confidence, has_confluence):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.range_strength = range_strength
        self.ai_confidence = ai_confidence
        self.has_confluence = has_confluence
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

        # Position size based on confidence and confluence
        if self.has_confluence and self.ai_confidence == 'high':
            position_size = 2000  # Full position for ETH/XRP with $2,600 capital
        elif self.ai_confidence == 'high':
            position_size = 1500  # Reduced without confluence
        else:
            position_size = 1000  # Conservative

        self.profit_loss_usd = (exit_price - self.entry_price) / self.entry_price * position_size
        self.duration_hours = exit_time - self.entry_time

    def __repr__(self):
        conf_str = f"{self.ai_confidence.upper()}"
        if self.has_confluence:
            conf_str += "+CONFLUENCE"
        if self.status == 'open':
            return f"Trade(OPEN @ ${self.entry_price:.4f}, {conf_str})"
        return f"Trade({self.exit_reason} @ ${self.exit_price:.4f}, P/L: {self.profit_loss_pct:+.2f}%, {conf_str})"


def simulate_ai_decision(prices, current_price, current_index, range_signal):
    """
    Simulate what the AI would decide based on:
    - Range strategy signal
    - RSI
    - Price trend
    - Confluence

    Returns: (should_trade: bool, confidence: str, has_confluence: bool)
    """
    # Calculate RSI
    if len(prices) < 15:
        rsi = 50
    else:
        rsi_values = calculate_rsi(prices, period=14)
        rsi = rsi_values[-1] if rsi_values[-1] is not None else 50

    # Check if price is in uptrend (simple: current > 20-period MA)
    if len(prices) >= 20:
        ma20 = np.mean(prices[-20:])
        is_uptrend = current_price > ma20
    else:
        is_uptrend = True

    # Determine AI confidence based on conditions
    has_range_signal = (range_signal['signal'] == 'buy')
    range_strength = range_signal.get('zone_strength', 0)

    # HIGH confidence criteria (simulated):
    # 1. Range strategy confirms (in support zone)
    # 2. RSI oversold or neutral (< 70)
    # 3. Uptrend confirmed
    # 4. Strong support zone (3+ touches)

    if has_range_signal and range_strength >= 3 and rsi < 70 and is_uptrend:
        confidence = 'high'
        has_confluence = True
        should_trade = True
    elif has_range_signal and rsi < 70:
        confidence = 'high'
        has_confluence = False  # Range signal but weaker zone
        should_trade = True
    elif is_uptrend and rsi < 60 and range_strength >= 2:
        confidence = 'medium'
        has_confluence = False
        should_trade = False  # Don't trade on medium confidence
    else:
        confidence = 'low'
        has_confluence = False
        should_trade = False

    return should_trade, confidence, has_confluence


def backtest_with_ai_simulation(symbol, lookback_hours=336, backtest_period_hours=336):
    """
    Backtest with AI + Range strategy confluence simulation
    """
    print(f"\n{'='*80}")
    print(f"AI + RANGE CONFLUENCE BACKTEST: {symbol}")
    print(f"{'='*80}\n")

    # Load all available price data
    all_prices = get_property_values_from_crypto_file('coinbase-data', symbol, 'price', max_age_hours=4380)

    if not all_prices or len(all_prices) < lookback_hours + backtest_period_hours:
        print(f"✗ Insufficient data for {symbol}")
        return None

    print(f"✓ Loaded {len(all_prices)} hours of price data")
    print(f"✓ Simulating AI + Range strategy confluence")
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

            # Simulate AI decision
            should_trade, confidence, has_confluence = simulate_ai_decision(
                historical_prices, current_price, i, range_signal
            )

            if should_trade:
                # Calculate entry/stop/target from range strategy
                if range_signal['zone']:
                    targets = calculate_zone_based_targets(range_signal['zone'])

                    current_trade = SimulatedTrade(
                        entry_time=current_hour,
                        entry_price=targets['entry_price'],
                        stop_loss=targets['stop_loss'],
                        profit_target=targets['profit_target'],
                        range_strength=range_signal['zone_strength'],
                        ai_confidence=confidence,
                        has_confluence=has_confluence
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

    # Separate by confluence
    confluence_trades = [t for t in trades if t.has_confluence]
    non_confluence_trades = [t for t in trades if not t.has_confluence]

    winning_trades = [t for t in trades if t.profit_loss_pct > 0]
    losing_trades = [t for t in trades if t.profit_loss_pct <= 0]

    total_profit_usd = sum(t.profit_loss_usd for t in trades)
    avg_profit_pct = sum(t.profit_loss_pct for t in trades) / len(trades)
    avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0

    win_rate = (len(winning_trades) / len(trades)) * 100

    # Confluence-specific stats
    if confluence_trades:
        confluence_wins = [t for t in confluence_trades if t.profit_loss_pct > 0]
        confluence_win_rate = (len(confluence_wins) / len(confluence_trades)) * 100
        confluence_profit = sum(t.profit_loss_usd for t in confluence_trades)
    else:
        confluence_win_rate = 0
        confluence_profit = 0

    # Print results
    print(f"\nBACKTEST RESULTS:")
    print(f"{'='*80}")
    print(f"Total Trades: {len(trades)}")
    print(f"  - With Confluence: {len(confluence_trades)} trades")
    print(f"  - Without Confluence: {len(non_confluence_trades)} trades")
    print(f"\nWin Rate:")
    print(f"  - Overall: {win_rate:.1f}% ({len(winning_trades)} wins, {len(losing_trades)} losses)")
    if confluence_trades:
        print(f"  - With Confluence: {confluence_win_rate:.1f}% ({len(confluence_wins)}/{len(confluence_trades)} trades)")

    print(f"\nPerformance:")
    print(f"  Total P/L: ${total_profit_usd:+.2f}")
    if confluence_trades:
        print(f"  P/L from Confluence trades: ${confluence_profit:+.2f}")
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

    # Show trades
    print(f"\nIndividual Trades:")
    print(f"{'-'*80}")
    for idx, trade in enumerate(trades, 1):
        conf_marker = "✓" if trade.has_confluence else " "
        print(f"{idx}. [{conf_marker}] Entry: ${trade.entry_price:.4f} | Exit: ${trade.exit_price:.4f} | "
              f"P/L: {trade.profit_loss_pct:+.2f}% | {trade.exit_reason} | "
              f"Conf: {trade.ai_confidence.upper()}{'+RANGE' if trade.has_confluence else ''}")

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'confluence_trades': len(confluence_trades),
        'win_rate': win_rate,
        'confluence_win_rate': confluence_win_rate,
        'total_profit_usd': total_profit_usd,
        'confluence_profit_usd': confluence_profit,
        'avg_profit_pct': avg_profit_pct,
        'trades': trades
    }


def main():
    """Run AI + Range confluence backtest"""
    print("=" * 80)
    print("AI + RANGE STRATEGY CONFLUENCE BACKTEST (LAST 2 WEEKS)")
    print("=" * 80)
    print()
    print("This simulates how the AI performs WITH range strategy integration.")
    print("Trades are only taken when AI confidence is HIGH.")
    print("Position size is larger ($2,000) for confluence trades.")
    print()

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

    print(f"Testing {len(enabled_symbols)} enabled wallets: {', '.join(enabled_symbols)}")
    print()

    all_results = {}

    for symbol in enabled_symbols:
        result = backtest_with_ai_simulation(
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
    print("OVERALL SUMMARY - AI + RANGE CONFLUENCE")
    print("=" * 80)

    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_confluence = sum(r['confluence_trades'] for r in all_results.values())
    total_profit = sum(r['total_profit_usd'] for r in all_results.values())
    confluence_profit = sum(r['confluence_profit_usd'] for r in all_results.values())

    print(f"\nAcross all {len(all_results)} wallets:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Confluence Trades: {total_confluence} ({total_confluence/total_trades*100:.1f}%)")
    print(f"  Total P/L: ${total_profit:+.2f}")
    print(f"  P/L from Confluence: ${confluence_profit:+.2f}")

    print(f"\nPer-Wallet Performance:")
    for symbol, result in sorted(all_results.items(), key=lambda x: x[1]['total_profit_usd'], reverse=True):
        print(f"  {symbol}: {result['total_trades']} trades, "
              f"{result['win_rate']:.1f}% win rate, "
              f"${result['total_profit_usd']:+.2f} P/L, "
              f"Confluence: {result['confluence_win_rate']:.1f}%")

    print("\n" + "=" * 80)
    print("COMPARISON TO ORIGINAL BACKTEST")
    print("=" * 80)
    print("\nOriginal (Range Strategy Only):")
    print("  ETH: 6 trades, 66.7% win rate, +$152.45")
    print("  XRP: 5 trades, 60.0% win rate, +$100.40")
    print("  Combined: 11 trades, 63.6% win rate, +$252.85")
    print("\nWith AI Confluence (This Test):")
    print(f"  Combined: {total_trades} trades, ${total_profit:+.2f}")
    print(f"  Confluence trades had: ${confluence_profit:+.2f} profit")

    print("\n" + "=" * 80)
    print("\nNote: Results may vary due to AI simulation simplifications.")
    print("Real AI uses GPT-4 with full market context and learning.")
    print("This simulation uses basic indicators (RSI, MA, support zones).")
    print("=" * 80)


if __name__ == "__main__":
    main()
