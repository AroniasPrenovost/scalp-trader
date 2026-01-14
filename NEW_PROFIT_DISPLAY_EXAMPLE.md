# New Profitability Display - Example Output

## Before (Old Display)
```
current_market_value: $4508.69
  (298.22 shares × $15.12/share)
total_cost_basis: $4500.13
  (Original purchase price + entry fees)
gross_profit (before exit costs): $8.56
  ($4508.69 - $4500.13)
exit_exchange_fee: $27.05
  (0.6% taker fee on $4508.69)
unrealized_capital_gain: $8.56
  ($4508.69 - $4500.13)
capital_gains_tax_owed: $3.17
  (37% tax rate on $8.56 gain)
NET_PROFIT (take-home): $-21.66
  Formula: Current Value - Cost Basis - Exit Fee - Taxes
  $4508.69 - $4500.13 - $27.05 - $3.17
NET_PROFIT %: -0.4813%
  ($-21.66 ÷ $4500.13 × 100)
```

## After (New Atomic Calculator Display)
```
================================================================================
📊 PROFITABILITY BREAKDOWN - AVAX-USD
================================================================================

POSITION DETAILS:
  Shares: 298.21947357
  Entry Price: $15.0000
  Current Price: $15.1200
  Price Change: +0.80%

ENTRY COSTS:
  Entry Value: $4473.29
    (298.21947357 shares × $15.0000)
  Entry Fee (0.6%): $26.84
  Total Cost Basis: $4500.13

CURRENT VALUE:
  Market Value: $4508.69
    (298.21947357 shares × $15.1200)

EXIT COSTS (if sold now):
  Exit Fee (0.6%): $27.05
  Proceeds After Fee: $4481.64

PROFIT (before tax):
  Gross Profit: $-18.49
    ($4481.64 proceeds - $4500.13 cost)

TAXES:
  Capital Gain: $-18.49
  Tax Owed: $0.00 (no tax on losses)

================================================================================
💰 NET PROFIT (Your Take-Home): $-18.49 (-0.41%)
================================================================================

Formula: Market Value - Cost Basis - Exit Fee - Taxes
  $4508.69 - $4500.13 - $27.05 - $0.00 = $-18.49

```

## Key Improvements

### 1. **Visual Hierarchy**
- Clear sections with headers
- Color-coded profits (green) and losses (red)
- Box borders for emphasis on NET PROFIT

### 2. **Detailed Breakdown**
Shows every step:
- ✅ Position details (shares, prices, % change)
- ✅ Entry costs (value + fee = cost basis)
- ✅ Current value (shares × current price)
- ✅ Exit costs (fee + proceeds after fee)
- ✅ Gross profit (before tax)
- ✅ Taxes (capital gain + rate + tax owed)
- ✅ **NET PROFIT** (final take-home)

### 3. **Atomic Calculation**
Uses `calculate_net_profit_from_price_move()` from `utils/profit_calculator.py`:
- Single source of truth
- Consistent across entire codebase
- Accounts for all costs: entry fee, exit fee, taxes
- Validates against test scenarios

### 4. **Clear Formula**
Shows the exact calculation:
```
Market Value - Cost Basis - Exit Fee - Taxes = Net Profit
$4508.69 - $4500.13 - $27.05 - $0.00 = $-18.49
```

### 5. **Real-World Example**
With $4,500 capital and 0.8% gross price move:
- **Old understanding:** "0.8% move = profit!"
- **New reality:** "0.8% move = -$18.49 loss (-0.41%)"

This is why we updated targets to 2.5% gross moves!

## What Shows for Profitable Positions

Example with 2.5% price move ($15.00 → $15.375):

```
================================================================================
📊 PROFITABILITY BREAKDOWN - AVAX-USD
================================================================================

POSITION DETAILS:
  Shares: 298.21947357
  Entry Price: $15.0000
  Current Price: $15.3750
  Price Change: +2.50%

ENTRY COSTS:
  Entry Value: $4473.29
    (298.21947357 shares × $15.0000)
  Entry Fee (0.6%): $26.84
  Total Cost Basis: $4500.13

CURRENT VALUE:
  Market Value: $4584.99
    (298.21947357 shares × $15.3750)

EXIT COSTS (if sold now):
  Exit Fee (0.6%): $27.51
  Proceeds After Fee: $4557.48

PROFIT (before tax):
  Gross Profit: $57.35
    ($4557.48 proceeds - $4500.13 cost)

TAXES:
  Capital Gain: $57.35
  Tax Rate: 37%
  Tax Owed: $21.22

================================================================================
💰 NET PROFIT (Your Take-Home): $36.13 (+0.80%)
================================================================================

Formula: Market Value - Cost Basis - Exit Fee - Taxes
  $4584.99 - $4500.13 - $27.51 - $21.22 = $36.13
```

✅ **2.5% gross move = $36.13 net profit (0.80%)** - This is realistic and profitable!

## Benefits

1. **Transparency** - Every dollar is accounted for
2. **Accuracy** - Uses atomic calculator (single source of truth)
3. **Clarity** - Easy to understand where profits/losses come from
4. **Consistency** - Same calculation logic everywhere in codebase
5. **Reality Check** - Shows true take-home after ALL costs

## Color Coding

- **Green** - Positive values (profits, gains)
- **Red** - Negative values (losses)
- **Yellow** - Important totals (cost basis)
- **Cyan** - Formulas and explanations
- **Bold** - Section headers and final NET PROFIT

---

**This display makes it crystal clear whether a position is actually profitable or not!**
