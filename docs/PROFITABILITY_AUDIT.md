# Profitability Calculation Audit

## Current State Analysis

### 1. **Momentum Scalping Strategy** (`utils/momentum_scalping_strategy.py`)

**Profit Targets (Lines 303-507):**
- Support Bounce: `profit_target = entry_price * 1.008` (0.8% target)
- Breakout: `profit_target = entry_price * 1.008` (0.8% target)
- Consolidation Break: `profit_target = entry_price * 1.006` (0.6% target)
- Stop Loss (all): `stop_loss = entry_price * 0.996` (0.4% stop)

**Issue:** These are GROSS targets - they don't account for:
- Entry fee (taker fee on buy)
- Exit fee (taker fee on sell)
- Capital gains tax

### 2. **Opportunity Scorer** (`utils/opportunity_scorer.py`)

**Net Profit Calculation (Lines 580-594):**
```python
# Fees: 2 * exchange_fee_percentage (buy + sell)
total_fees_pct = 2 * exchange_fee_percentage

# Net profit = gross - fees - (taxes on profit portion)
if gross_pct > 0:
    # On profitable trade: subtract fees and taxes on the profit
    net_profit_pct = gross_pct - total_fees_pct - (tax_rate_percentage / 100 * gross_pct)
else:
    # On losing trade: just subtract fees (no taxes on losses)
    net_profit_pct = gross_pct - total_fees_pct

# Convert to USD
net_profit_usd = (net_profit_pct / 100) * trading_capital_usd
```

**Issue:** This calculation is simplified and doesn't match actual Coinbase fee mechanics:
- Fees are charged on the ORDER VALUE, not as a percentage subtraction
- The formula treats fees as direct percentage deductions
- Tax calculation is oversimplified

### 3. **Index.py - Buy Order Execution** (Lines 1582-1661)

**NEW - Share Calculation with Fees (Lines 1582-1607):**
```python
# We want: (subtotal + fee) = buy_amount
# Where: fee = (fee_rate/100) * subtotal
# So: subtotal * (1 + fee_rate/100) = buy_amount

fee_multiplier = 1 + (coinbase_spot_taker_fee / 100)
subtotal_amount = buy_amount / fee_multiplier
shares_calculation = subtotal_amount / current_price
```

**Status:** ✅ **CORRECT** - This was just fixed to account for entry fees.

### 4. **Index.py - Sell Profit Calculation** (Lines 1733-1810)

**Current Formula:**
```python
# STEP 1: Current market value
current_position_value_usd = current_price * number_of_shares

# STEP 2: Cost basis (includes entry fees)
total_cost_basis_usd = entry_position_value_after_fees

# STEP 3: Gross profit (before exit costs)
gross_profit_before_exit_costs = current_position_value_usd - total_cost_basis_usd

# STEP 4: Exit fee
exit_exchange_fee_usd = calculate_exchange_fee(current_price, number_of_shares, coinbase_spot_taker_fee)
# Formula: (fee_rate / 100) * current_price * number_of_shares

# STEP 5: Capital gain (for taxes)
unrealized_gain_usd = current_position_value_usd - total_cost_basis_usd

# STEP 6: Taxes on gains
capital_gains_tax_usd = (federal_tax_rate / 100) * unrealized_gain_usd

# STEP 7: NET PROFIT
net_profit_after_all_costs_usd = current_position_value_usd - total_cost_basis_usd - exit_exchange_fee_usd - capital_gains_tax_usd

# STEP 8: Percentage return
net_profit_percentage = (net_profit_after_all_costs_usd / total_cost_basis_usd) * 100
```

**Status:** ✅ **CORRECT** - This properly calculates net profit accounting for entry fees (in cost basis), exit fees, and taxes.

---

## THE CORE PROBLEM

### **Disconnect Between Strategy Targets and Reality**

The momentum scalping strategy sets targets like:
- Target: +0.8% profit
- Stop: -0.4% loss

But these are **gross price movements**, not net profit after fees and taxes.

### **Example with $4500 capital:**

**Scenario: 0.8% Gross Price Move**

1. **BUY:**
   - Capital available: $4,500
   - Taker fee: 0.6% (Coinbase Advanced tier)
   - Subtotal: $4,500 / 1.006 = $4,473.29
   - Entry fee: $26.84
   - **Total cost: $4,500.13**
   - Price: $15.00/share
   - Shares: 298.22

2. **SELL at +0.8% price:**
   - Exit price: $15.12 ($15.00 * 1.008)
   - Gross value: 298.22 * $15.12 = $4,508.69
   - Exit fee: $4,508.69 * 0.006 = $27.05
   - **Net proceeds: $4,481.64**

3. **TAXES:**
   - Capital gain: $4,481.64 - $4,500.13 = -$18.49 (LOSS!)
   - No taxes on losses

4. **FINAL NET PROFIT:**
   - **-$18.49 (-0.41%)**

### **The 0.8% target actually results in a LOSS!**

---

## REQUIRED NET PROFIT CALCULATION

To break even after all costs, we need:

```
Net Proceeds = Cost Basis

Where:
- Cost Basis = Buy Subtotal + Entry Fee
- Net Proceeds = Sell Subtotal - Exit Fee
- Sell Subtotal = (Price * Shares) - Exit Fee
- Exit Fee = (Sell Subtotal * fee_rate)
- Tax = federal_tax_rate * (Net Proceeds - Cost Basis) if positive

Breakeven Price Move = ?
```

### **Breakeven Analysis:**

With 0.6% taker fees (buy + sell) and 37% tax rate:
- Entry fee: 0.6%
- Exit fee: 0.6%
- Total fees: 1.2%
- Tax on profit: 37% of (gross - fees)

**Minimum profitable price move (NET > $0):**
- Need to cover: entry fee (0.6%) + exit fee (0.6%) = 1.2% minimum
- For net profit after taxes, need approximately **1.9% gross price move**

So:
- 0.6% target = **GUARANTEED LOSS**
- 0.8% target = **GUARANTEED LOSS**
- 1.9% target = **BREAKEVEN**
- 2.5%+ target = **PROFITABLE**

---

## RECOMMENDATIONS

### 1. **Update Momentum Scalping Strategy Targets**

Change `utils/momentum_scalping_strategy.py` to set **NET** profit targets:

```python
# For $4500 capital @ 0.6% fees @ 37% tax:
# To achieve +0.8% NET profit, need ~2.7% gross price move

# Support Bounce & Breakout:
stop_loss = entry_price * 0.996  # Keep 0.4% stop (reasonable)
profit_target = entry_price * 1.027  # 2.7% gross for 0.8% net

# Consolidation Break:
stop_loss = entry_price * 0.996  # 0.4% stop
profit_target = entry_price * 1.024  # 2.4% gross for 0.6% net
```

### 2. **Create Atomic Profitability Function**

Create a single source of truth function:

```python
def calculate_net_profit_from_price_move(
    entry_price: float,
    exit_price: float,
    shares: float,
    entry_fee_pct: float,  # 0.6
    exit_fee_pct: float,   # 0.6
    tax_rate_pct: float    # 37
) -> Dict:
    """
    Calculate net profit accounting for all costs.

    Returns: {
        'gross_profit_usd': float,
        'entry_fee_usd': float,
        'exit_fee_usd': float,
        'capital_gain_usd': float,
        'tax_usd': float,
        'net_profit_usd': float,
        'net_profit_pct': float
    }
    """
```

### 3. **Use This Function Everywhere**

- Opportunity scorer
- Trade execution logic
- Sell decision logic
- Backtesting
- Reporting

---

## IMMEDIATE ACTION ITEMS

1. ✅ Fix share calculation to account for entry fees (DONE)
2. ⚠️ Update momentum scalping strategy targets to NET profit targets
3. ⚠️ Create atomic profitability calculation function
4. ⚠️ Update opportunity scorer to use atomic function
5. ⚠️ Update sell logic to use atomic function
6. ⚠️ Add validation: reject trades that can't be profitable

