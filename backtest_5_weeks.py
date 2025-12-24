#!/usr/bin/env python3
"""
5-Week Backtest - Original Range Strategy
"""

import json
from backtest_range_strategy import backtest_symbol

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

enabled_symbols = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]

print('='*80)
print('5-WEEK BACKTEST - ORIGINAL RANGE STRATEGY')
print('='*80)
print(f"\nTesting {len(enabled_symbols)} wallets: {', '.join(enabled_symbols)}")
print()

all_results = {}

for symbol in enabled_symbols:
    # 5 weeks = 35 days = 840 hours
    result = backtest_symbol(
        symbol=symbol,
        lookback_hours=840,  # 5 weeks for strategy lookback
        backtest_period_hours=840  # 5 weeks backtest period
    )
    if result:
        all_results[symbol] = result

# Summary
if all_results:
    print('\n' + '='*80)
    print('5-WEEK SUMMARY')
    print('='*80)

    total_trades = sum(r['total_trades'] for r in all_results.values())
    total_profit = sum(r['total_profit_usd'] for r in all_results.values())

    all_trades = []
    for r in all_results.values():
        all_trades.extend(r['trades'])

    if all_trades:
        wins = [t for t in all_trades if t.profit_loss_pct > 0]
        overall_win_rate = (len(wins) / len(all_trades)) * 100
    else:
        overall_win_rate = 0

    print(f'\nTotal Trades: {total_trades}')
    print(f'Overall Win Rate: {overall_win_rate:.1f}%')
    print(f'Total P/L: ${total_profit:+.2f}')
    print()

    for symbol, result in sorted(all_results.items(), key=lambda x: x[1]['total_profit_usd'], reverse=True):
        print(f"{symbol}: {result['total_trades']} trades, {result['win_rate']:.1f}% win rate, ${result['total_profit_usd']:+.2f}")

print('\n' + '='*80)
