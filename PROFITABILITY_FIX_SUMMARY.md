# Profitability Calculation Fix - Summary

## Issues Fixed

### 1. **Insufficient Funds Error** ✅
**Problem:** When trying to buy $4,500 worth of shares, the system calculated shares without accounting for entry fees, causing the total order (subtotal + fees) to exceed available balance.

**Solution:** Updated share calculation in `index.py` (lines 1582-1607) to properly account for entry fees:
```python
# Before: shares = buy_amount / current_price
# After:
fee_multiplier = 1 + (coinbase_spot_taker_fee / 100)
subtotal_amount = buy_amount / fee_multiplier
shares = subtotal_amount / current_price
```

**Result:** Total cost including fees now stays within the configured capital limit.

---

### 2. **Unrealistic Profit Targets** ✅
**Problem:** Strategy was targeting 0.8% gross price moves, which after accounting for:
- Entry fee: 0.6%
- Exit fee: 0.6%
- Capital gains tax: 37%

...resulted in **NET LOSSES** instead of profits!

**Example:**
- $4,500 capital, $15/share entry
- +0.8% gross price move → **-$18.49 NET profit (-0.40%)** ❌

**Solution:** Created atomic profit calculator and updated all targets to be NET profit after all costs.

---

## New Profitability System

### **Atomic Profit Calculator** (`utils/profit_calculator.py`)

Single source of truth for all profit calculations. Key functions:

1. **`calculate_required_shares_for_capital()`** - How many shares to buy given capital (accounting for entry fees)

2. **`calculate_net_profit_from_price_move()`** - Calculate actual take-home profit after all costs:
   - Entry fees
   - Exit fees
   - Capital gains taxes (only on profits)

3. **`calculate_required_price_for_target_profit()`** - What price move is needed to achieve a target NET profit

4. **`calculate_breakeven_price()`** - Price needed to break even (NET = $0)

### **Validated Results** (from profit_calculator.py test run):

With $4,500 capital, $15/share, 0.6% fees, 37% tax:

| Gross Price Move | Net Profit $ | Net Profit % |
|------------------|--------------|--------------|
| +0.8% | -$18.11 | -0.40% ❌ |
| +1.5% | +$8.20 | +0.18% ✅ |
| +2.5% | +$36.21 | +0.80% ✅ |

**Breakeven:** +1.21% gross price move
**For 0.8% NET profit:** +2.49% gross price move required

---

## Updated Strategy Targets

### **Momentum Scalping Strategy** (`utils/momentum_scalping_strategy.py`)

**Old Targets (GROSS):**
- Support Bounce: 0.8% gross move
- Breakout: 0.8% gross move
- Consolidation Break: 0.6% gross move
- Stop Loss: 0.4% (all)

**New Targets (NET):**
- Support Bounce: **2.5% gross move** → 0.8% NET profit
- Breakout: **2.5% gross move** → 0.8% NET profit
- Consolidation Break: **2.2% gross move** → 0.6% NET profit
- Stop Loss: 0.4% (unchanged)

All targets now use `calculate_required_price_for_target_profit()` with real-time:
- Entry fee % (from Coinbase API)
- Exit fee % (from Coinbase API)
- Tax rate % (from .env)

---

## Changes Made

### 1. **Created `utils/profit_calculator.py`**
- Atomic profitability calculation functions
- Self-validating with test scenarios
- Single source of truth for all profit math

### 2. **Updated `utils/momentum_scalping_strategy.py`**
- Import profit calculator
- Pass fees/taxes to all signal functions
- Calculate realistic profit targets using atomic function
- Updated docstrings to clarify NET vs GROSS

### 3. **Updated `utils/opportunity_scorer.py`**
- Accept fees/taxes as parameters
- Pass them through to momentum strategy
- Ensures all opportunities score with realistic targets

### 4. **Updated `index.py`**
- Fixed share calculation to account for entry fees (lines 1582-1607)
- Pass real-time fees and taxes to opportunity scorer (all find_best_opportunity calls)
- Fees from Coinbase API: `coinbase_spot_taker_fee`
- Taxes from .env: `federal_tax_rate`

---

## Data Flow

```
.env (FEDERAL_TAX_RATE)
    ↓
Coinbase API (taker_fee_rate)
    ↓
index.py (federal_tax_rate, coinbase_spot_taker_fee)
    ↓
find_best_opportunity(entry_fee_pct, exit_fee_pct, tax_rate_pct)
    ↓
score_opportunity(entry_fee_pct, exit_fee_pct, tax_rate_pct)
    ↓
check_scalp_entry_signal(entry_fee_pct, exit_fee_pct, tax_rate_pct)
    ↓
calculate_required_price_for_target_profit() [ATOMIC]
    ↓
profit_target = $XX.XX (realistic NET target)
```

---

## Validation

Run the profit calculator test:
```bash
./venv/bin/python3 utils/profit_calculator.py
```

Expected output shows:
- Share calculations accounting for fees
- Net profit for various price moves
- Breakeven price (+1.21%)
- Required price for 0.8% NET profit (+2.49%)

---

## Key Takeaways

1. ✅ **Share calculations now account for entry fees** - No more insufficient funds errors
2. ✅ **All profit targets are NET** - After fees and taxes, not gross price moves
3. ✅ **Single atomic calculator** - One source of truth for all profitability math
4. ✅ **Real-time fees/taxes** - Uses actual Coinbase fees and configured tax rate
5. ✅ **Realistic expectations** - 0.8% NET profit requires ~2.5% gross price move

---

## Before vs After

### Before:
- ❌ Insufficient funds errors on buy orders
- ❌ 0.8% gross target = -0.40% NET (loss!)
- ❌ Strategy thought it was profitable when it wasn't
- ❌ Multiple inconsistent profit calculations

### After:
- ✅ Buy orders stay within capital limits
- ✅ 0.8% NET target = actual $36 profit on $4,500 trade
- ✅ Strategy only enters when truly profitable
- ✅ Single atomic profit calculator everywhere

---

## Testing Recommendations

1. Monitor next buy order to confirm no insufficient funds error
2. Watch first completed trade to verify NET profit matches prediction
3. Compare opportunity report targets to actual sell prices
4. Validate that trades are now consistently profitable

The system is now mathematically sound and accounts for all real-world costs!
