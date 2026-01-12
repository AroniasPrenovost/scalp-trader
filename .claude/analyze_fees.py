import json

with open('/Users/arons_stuff/Documents/scalp-scripts/transactions/data.json', 'r') as f:
    trades = json.load(f)

print(f"Total trades: {len(trades)}\n")

# Analyze last 20 trades
recent = trades[-20:]

total_net = 0
total_fees = 0
wins = 0
losses = 0

print("="*80)
print(f"{'Symbol':<12} {'Net Profit':<15} {'Fees':<15} {'Net %':<10}")
print("="*80)

for t in recent:
    symbol = t.get('symbol', 'N/A')
    net_profit = t.get('net_profit_usd', 0)
    fees = t.get('exchange_fee_usd', 0)
    net_pct = t.get('net_profit_percentage', 0)

    total_net += net_profit
    total_fees += fees

    if net_profit > 0:
        wins += 1
    else:
        losses += 1

    print(f"{symbol:<12} ${net_profit:>12.2f} ${fees:>12.2f} {net_pct:>8.2f}%")

print("="*80)
print(f"\nTotals:")
print(f"Net Profit: ${total_net:.2f}")
print(f"Total Fees: ${total_fees:.2f}")
print(f"Win Rate: {wins}/{wins+losses} ({100*wins/(wins+losses):.1f}%)")
print(f"Avg Fee per Trade: ${total_fees/len(recent):.2f}")
print(f"Avg Fee %: {100*total_fees/len(recent)/3000:.3f}% (assuming ~$3k positions)")
