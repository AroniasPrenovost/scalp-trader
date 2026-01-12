#!/usr/bin/env python3
"""
Backtest Final Strategy - Data-Driven Mean Reversion with Intelligent Rotation

Tests the complete strategy:
- Quantitative scoring (0-100) with historical pattern validation
- Two-stage validation: Quant ≥75 + AI HIGH confidence
- Single position with full capital (~$4,379)
- Fast rotation to best opportunity
- Profit target: 1.5-2.5% gross (targeting 1.5%+ NET after fees/taxes)
- Stop loss: 0.75% maximum
- Market rotation across 8 assets
"""

import json
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from setup_scorer import (
    calculate_setup_score,
    get_crypto_data_from_file,
    calculate_rsi,
    calculate_ma,
    calculate_volatility
)


class Trade:
    """Represents a simulated trade"""
    def __init__(self, symbol, entry_idx, entry_time, entry_price, stop_loss, profit_target,
                 score, setup_type, confidence='high'):
        self.symbol = symbol
        self.entry_idx = entry_idx
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.score = score
        self.setup_type = setup_type
        self.confidence = confidence

        self.exit_idx = None
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.gross_pct = 0
        self.gross_usd = 0
        self.net_usd = 0
        self.duration_hours = 0
        self.status = 'open'

    def close(self, exit_idx, exit_time, exit_price, exit_reason, position_size_usd=4379):
        """Close the trade and calculate P&L"""
        self.exit_idx = exit_idx
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = 'closed'

        # Gross profit/loss percentage
        self.gross_pct = ((exit_price - self.entry_price) / self.entry_price) * 100

        # Gross profit/loss in USD
        self.gross_usd = (self.gross_pct / 100) * position_size_usd

        # Fees: 0.25% taker fee * 2 (buy + sell) = 0.5% total
        fee_pct = 0.5
        fees_usd = (fee_pct / 100) * position_size_usd

        # Net profit/loss after fees and taxes
        if self.gross_pct > 0:
            # Profitable trade: subtract fees and taxes (37% on profit)
            tax_rate = 0.37
            net_before_tax = self.gross_usd - fees_usd
            taxes_usd = net_before_tax * tax_rate
            self.net_usd = net_before_tax - taxes_usd
        else:
            # Losing trade: just subtract fees (no taxes on losses)
            self.net_usd = self.gross_usd - fees_usd

        # Duration in hours
        self.duration_hours = exit_idx - self.entry_idx


def simulate_ai_confidence(score, rsi, price_vs_ma_pct, volume_score, setup_type):
    """
    Simulate AI confidence based on scoring components.

    Pattern recognition showed mean_reversion has 76.9% win rate.
    This is the PRIMARY pattern to trade.

    In reality, AI analyzes 6 timeframe charts. Here we simulate based on:
    - Quantitative score
    - Setup type (prioritize mean_reversion)
    - RSI position (30-50 for mean reversion, avoid extremes)
    - Price vs MA (mean reversion signal)
    - Volume confirmation

    Returns: 'high', 'medium', or 'low'
    """
    if score < 75:
        return 'low'  # Don't trade if quant score too low

    # PRIORITY: Mean reversion setup (proven 76.9% win rate)
    # Requirements: Price <-1.5% below MA (any dip), RSI 30-50, volume confirmation
    has_mean_reversion = (
        price_vs_ma_pct is not None and
        price_vs_ma_pct < -1.5 and  # ANY price below -1.5% MA
        rsi is not None and
        30 <= rsi <= 50
    )

    # Volume confirmation
    has_volume = volume_score >= 65

    # HIGH confidence: Mean reversion setup with all conditions met
    if setup_type == 'mean_reversion' and has_mean_reversion and has_volume and score >= 75:
        return 'high'

    # MEDIUM confidence: Mean reversion without perfect conditions
    elif setup_type == 'mean_reversion' and price_vs_ma_pct is not None and price_vs_ma_pct < -1.0:
        return 'medium'

    # LOW confidence: Everything else (don't trade oversold_bounce, it has 33% win rate)
    else:
        return 'low'


def backtest_strategy(symbols, start_date, end_date, starting_capital=4609,
                     position_size_pct=95, max_hold_hours=3):
    """
    Backtest the final strategy across multiple assets with market rotation.

    Args:
        symbols: List of symbols to trade
        start_date: Start date for backtest (timestamp or index)
        end_date: End date for backtest (timestamp or index)
        starting_capital: Starting capital in USD
        position_size_pct: Percentage of capital per trade (default 95%)
        max_hold_hours: Maximum hours to hold position
    """

    print(f"\n{'='*100}")
    print(f"FINAL STRATEGY BACKTEST")
    print(f"{'='*100}")
    print(f"Assets: {', '.join(symbols)}")
    print(f"Starting Capital: ${starting_capital:,.2f}")
    print(f"Position Size: {position_size_pct}% (${starting_capital * (position_size_pct/100):,.2f} per trade)")
    print(f"Max Concurrent Positions: 1 (single best opportunity)")
    print(f"Max Hold Time: {max_hold_hours} hours")
    print(f"Strategy: PURE QUANTITATIVE SCALPING (No AI)")
    print(f"Philosophy: HIGH VELOCITY - Many $4-5 wins > Few large wins")
    print(f"Entry Criteria: Score ≥75 + Mean Reversion (price <-1.5% below MA, RSI 30-50)")
    print(f"Profit Target: 0.6% gross (~$4-5 NET per trade)")
    print(f"Stop Loss: 0.6% (1:1 risk/reward, realistic for hourly data)")
    print(f"Math: Need >50% win rate for profitability with 1:1 R/R")
    print(f"{'='*100}\n")

    # Position sizing
    position_size_usd = starting_capital * (position_size_pct / 100)

    # Load all asset data
    all_data = {}
    for symbol in symbols:
        data = get_crypto_data_from_file('coinbase-data', symbol, max_age_hours=4380)
        if data and len(data) >= 50:
            all_data[symbol] = data
            print(f"✓ Loaded {len(data)} hours of data for {symbol}")
        else:
            print(f"✗ Insufficient data for {symbol}")

    if not all_data:
        print("\n❌ No valid data found. Exiting.")
        return

    print(f"\nBacktesting across {len(all_data)} assets...\n")

    # Track all trades
    completed_trades = []
    active_trade = None

    # Determine backtest window (use shortest dataset)
    min_length = min(len(data) for data in all_data.values())
    start_idx = max(50, int(start_date) if isinstance(start_date, (int, float)) else 50)
    end_idx = min(min_length, int(end_date) if isinstance(end_date, (int, float)) else min_length)

    print(f"Backtest window: index {start_idx} to {end_idx} ({end_idx - start_idx} hours)\n")

    # Iterate through each hour
    for current_idx in range(start_idx, end_idx):

        # MANAGE ACTIVE POSITION FIRST
        if active_trade:
            symbol = active_trade.symbol
            data = all_data[symbol]
            current_price = float(data[current_idx]['price'])

            # Check exit conditions
            # 1. Profit target hit
            profit_pct = ((current_price - active_trade.entry_price) / active_trade.entry_price) * 100
            if current_price >= active_trade.profit_target:
                active_trade.close(current_idx, data[current_idx]['timestamp'],
                                 current_price, 'profit_target', position_size_usd)
                completed_trades.append(active_trade)
                active_trade = None
                continue

            # 2. Stop loss hit
            elif current_price <= active_trade.stop_loss:
                active_trade.close(current_idx, data[current_idx]['timestamp'],
                                 current_price, 'stop_loss', position_size_usd)
                completed_trades.append(active_trade)
                active_trade = None
                continue

            # 3. Max hold time exceeded
            elif (current_idx - active_trade.entry_idx) >= max_hold_hours:
                active_trade.close(current_idx, data[current_idx]['timestamp'],
                                 current_price, 'max_hold', position_size_usd)
                completed_trades.append(active_trade)
                active_trade = None
                continue

        # SCAN FOR NEW OPPORTUNITY (only if no active position)
        if not active_trade:
            opportunities = []

            for symbol in all_data.keys():
                data = all_data[symbol]
                if current_idx >= len(data):
                    continue

                # Extract data up to current point
                prices = [float(entry['price']) for entry in data[:current_idx]]
                volumes = [float(entry.get('volume_24h', 0)) for entry in data[:current_idx]]
                current_price = float(data[current_idx]['price'])
                current_volume = float(data[current_idx].get('volume_24h', 0))

                if len(prices) < 50:
                    continue

                # Calculate scoring components
                rsi = calculate_rsi(prices)
                ma_24h = calculate_ma(prices, period=24)
                volatility = calculate_volatility(prices, lookback=24)

                # Calculate price vs MA
                price_vs_ma_pct = None
                if ma_24h and ma_24h > 0:
                    price_vs_ma_pct = ((current_price - ma_24h) / ma_24h) * 100

                # Score this setup
                score_result = calculate_setup_score(symbol, 'coinbase-data', 'coingecko-global-volume')

                if score_result['score'] == 0:
                    continue

                score = score_result['score']
                setup_type = score_result['setup_type']

                # PURE QUANTITATIVE: No AI simulation, just score + setup type
                # Entry criteria: Score ≥75 AND setup is mean_reversion
                if score >= 75 and setup_type == 'mean_reversion':
                    opportunities.append({
                        'symbol': symbol,
                        'score': score,
                        'price': current_price,
                        'rsi': rsi,
                        'price_vs_ma_pct': price_vs_ma_pct,
                        'setup_type': setup_type,
                        'confidence': 'quantitative'  # No AI, pure quant
                    })

            # Enter best opportunity
            if opportunities:
                # Sort by score (highest first)
                opportunities.sort(key=lambda x: x['score'], reverse=True)
                best = opportunities[0]

                # Calculate entry parameters
                entry_price = best['price']

                # Profit target: 0.6% gross = ~$4-5 NET after fees/taxes
                # Small but achievable, high velocity
                profit_target_pct = 0.6
                profit_target = entry_price * (1 + profit_target_pct / 100)

                # Stop loss: 0.6% (EQUAL to target for 1:1 risk/reward, realistic for hourly data)
                # With 50%+ win rate, 1:1 risk/reward is profitable
                stop_loss_pct = 0.6
                stop_loss = entry_price * (1 - stop_loss_pct / 100)

                # Create trade
                symbol = best['symbol']
                data = all_data[symbol]
                active_trade = Trade(
                    symbol=symbol,
                    entry_idx=current_idx,
                    entry_time=data[current_idx]['timestamp'],
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    score=best['score'],
                    setup_type=best['setup_type'],
                    confidence=best['confidence']
                )

    # Close any remaining active trade at end of backtest
    if active_trade:
        symbol = active_trade.symbol
        data = all_data[symbol]
        final_price = float(data[end_idx - 1]['price'])
        active_trade.close(end_idx - 1, data[end_idx - 1]['timestamp'],
                          final_price, 'backtest_end', position_size_usd)
        completed_trades.append(active_trade)

    # RESULTS ANALYSIS
    print(f"\n{'='*100}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*100}\n")

    if not completed_trades:
        print("No trades executed during backtest period.\n")
        return

    # Calculate metrics
    total_trades = len(completed_trades)
    winners = [t for t in completed_trades if t.net_usd > 0]
    losers = [t for t in completed_trades if t.net_usd <= 0]

    win_count = len(winners)
    loss_count = len(losers)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    total_gross_profit = sum(t.gross_usd for t in completed_trades)
    total_net_profit = sum(t.net_usd for t in completed_trades)

    avg_winner = sum(t.net_usd for t in winners) / len(winners) if winners else 0
    avg_loser = sum(t.net_usd for t in losers) / len(losers) if losers else 0

    expected_value = total_net_profit / total_trades if total_trades > 0 else 0

    final_capital = starting_capital + total_net_profit
    total_return_pct = (total_net_profit / starting_capital * 100) if starting_capital > 0 else 0

    # Drawdown analysis
    cumulative = starting_capital
    peak = starting_capital
    max_drawdown = 0

    for trade in completed_trades:
        cumulative += trade.net_usd
        if cumulative > peak:
            peak = cumulative
        drawdown = ((peak - cumulative) / peak * 100) if peak > 0 else 0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Print summary
    print(f"Total Trades: {total_trades}")
    print(f"Winners: {win_count} ({win_rate:.1f}%)")
    print(f"Losers: {loss_count} ({100-win_rate:.1f}%)")
    print(f"\nGross P&L: ${total_gross_profit:+,.2f}")
    print(f"Net P&L (after fees & taxes): ${total_net_profit:+,.2f}")
    print(f"\nAverage Winner: ${avg_winner:+.2f}")
    print(f"Average Loser: ${avg_loser:+.2f}")
    print(f"Expected Value per Trade: ${expected_value:+.2f}")
    print(f"\nStarting Capital: ${starting_capital:,.2f}")
    print(f"Ending Capital: ${final_capital:,.2f}")
    print(f"Total Return: {total_return_pct:+.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")

    # Exit reason breakdown
    print(f"\n{'='*100}")
    print(f"EXIT REASON BREAKDOWN")
    print(f"{'='*100}\n")

    exit_reasons = {}
    for trade in completed_trades:
        reason = trade.exit_reason
        if reason not in exit_reasons:
            exit_reasons[reason] = []
        exit_reasons[reason].append(trade)

    for reason, trades in exit_reasons.items():
        count = len(trades)
        pct = count / total_trades * 100
        avg_profit = sum(t.net_usd for t in trades) / count
        print(f"{reason.replace('_', ' ').title():<20} {count:>4} ({pct:>5.1f}%)  Avg: ${avg_profit:+.2f}")

    # Setup type breakdown
    print(f"\n{'='*100}")
    print(f"SETUP TYPE PERFORMANCE")
    print(f"{'='*100}\n")

    setup_types = {}
    for trade in completed_trades:
        setup = trade.setup_type
        if setup not in setup_types:
            setup_types[setup] = []
        setup_types[setup].append(trade)

    for setup, trades in setup_types.items():
        count = len(trades)
        wins = sum(1 for t in trades if t.net_usd > 0)
        win_rate_setup = (wins / count * 100) if count > 0 else 0
        avg_profit = sum(t.net_usd for t in trades) / count
        print(f"{setup.replace('_', ' ').title():<25} {count:>4} trades  |  Win Rate: {win_rate_setup:>5.1f}%  |  Avg: ${avg_profit:+.2f}")

    # Best/worst trades
    print(f"\n{'='*100}")
    print(f"TOP 5 WINNERS")
    print(f"{'='*100}\n")

    top_winners = sorted(completed_trades, key=lambda t: t.net_usd, reverse=True)[:5]
    for i, trade in enumerate(top_winners, 1):
        print(f"{i}. {trade.symbol:<10} ${trade.net_usd:+8.2f}  ({trade.gross_pct:+.2f}%)  |  {trade.setup_type.replace('_', ' ').title():<20} Score: {trade.score:.1f}")

    print(f"\n{'='*100}")
    print(f"TOP 5 LOSERS")
    print(f"{'='*100}\n")

    top_losers = sorted(completed_trades, key=lambda t: t.net_usd)[:5]
    for i, trade in enumerate(top_losers, 1):
        print(f"{i}. {trade.symbol:<10} ${trade.net_usd:+8.2f}  ({trade.gross_pct:+.2f}%)  |  {trade.exit_reason.replace('_', ' ').title():<15} Score: {trade.score:.1f}")

    print(f"\n{'='*100}\n")

    # Verdict
    print(f"STRATEGY VERDICT:")
    if win_rate >= 65 and expected_value >= 15 and total_return_pct > 0:
        print(f"✅ STRATEGY VALIDATED")
        print(f"   - Win rate ≥65%: {'✓' if win_rate >= 65 else '✗'} ({win_rate:.1f}%)")
        print(f"   - EV ≥$15/trade: {'✓' if expected_value >= 15 else '✗'} (${expected_value:.2f})")
        print(f"   - Positive returns: ✓ ({total_return_pct:+.2f}%)")
        print(f"\n🚀 READY FOR LIVE TRADING\n")
    else:
        print(f"⚠️  STRATEGY NEEDS ADJUSTMENT")
        print(f"   - Win rate ≥65%: {'✓' if win_rate >= 65 else '✗'} ({win_rate:.1f}%)")
        print(f"   - EV ≥$15/trade: {'✓' if expected_value >= 15 else '✗'} (${expected_value:.2f})")
        print(f"   - Positive returns: {'✓' if total_return_pct > 0 else '✗'} ({total_return_pct:+.2f}%)")
        print(f"\n⏸  REVIEW BEFORE LIVE TRADING\n")


def main():
    """Run backtest"""
    # Load config to get enabled assets
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_symbols = [
        wallet['symbol']
        for wallet in config['wallets']
        if wallet.get('enabled', False) and wallet.get('ready_to_trade', False)
    ]

    print(f"Found {len(enabled_symbols)} enabled assets: {', '.join(enabled_symbols)}\n")

    # Backtest last 180 days (approx 4,320 hours)
    # Use last 4,320 hours of data, skip first 100 hours for indicator warmup
    backtest_strategy(
        symbols=enabled_symbols,
        start_date=100,  # Start at index 100 (after indicator warmup)
        end_date=4320,   # End at index 4320 (180 days * 24 hours)
        starting_capital=4609,
        position_size_pct=95,
        max_hold_hours=12
    )


if __name__ == '__main__':
    main()
