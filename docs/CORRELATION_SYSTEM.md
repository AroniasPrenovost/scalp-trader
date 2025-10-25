# Multi-Asset Correlation System

## Overview

This sophisticated correlation system manages portfolio-level risk across BTC, SOL, and ETH by analyzing their correlations, relative performance, and implementing intelligent position sizing and trade filtering.

---

## Architecture

### **Core Components**

1. **`utils/correlation_manager.py`** - Main correlation engine
2. **`utils/portfolio_dashboard.py`** - Risk monitoring and visualization
3. **`index.py`** - Integration with main trading loop
4. **`utils/openai_analysis.py`** - Enhanced with BTC context for altcoins
5. **`config.json`** - Correlation settings and limits

---

## Key Features

### **1. BTC as Market Leader**
- **BTC analyzed first** in every iteration
- **BTC sentiment stored** and used as context for altcoin trades
- **BTC charts included** in altcoin analysis (30d, 14d, 72h)

### **2. Correlation-Based Trade Filtering**

**Rules Applied:**
- ❌ **BTC Bearish** → Block all SOL/ETH BUY signals
- ⚠️ **BTC Sideways** → Only allow SOL/ETH BUY if strong outperformance
- ✅ **BTC Bullish** → Green light for aligned altcoin trades

**Example Console Output:**
```
🚫 BTC bearish backdrop - blocking SOL-USD BUY
STATUS: Correlation filter blocked trade - BTC trend overrides altcoin signals
```

### **3. Relative Strength Analysis**

Calculates how each altcoin performs vs BTC over 7 days:

```python
SOL: +12.3% | BTC: +8.5%
Outperformance: +3.8% (mild_outperformer)
```

**Categories:**
- `strong_outperformer`: +5% or more vs BTC
- `mild_outperformer`: 0% to +5% vs BTC
- `mild_underperformer`: 0% to -5% vs BTC
- `strong_underperformer`: -5% or worse vs BTC

### **4. Confidence Adjustment**

LLM confidence is dynamically adjusted based on BTC alignment:

```
BTC bullish + SOL bullish → No adjustment (aligned)
BTC sideways + SOL bullish → Reduce by 1 level
BTC bearish + SOL bullish → Force "no_trade"
```

**Example:**
```
⚖️  Confidence adjusted: medium → high (correlation analysis)
```

### **5. Position Limit Management**

**Default:** Max 2 correlated long positions simultaneously

```
📊 Portfolio State: 2 positions | $1,450.00 exposure
   Correlation Risk: $1,740.00
   ✓ BTC trend: bullish | Portfolio exposure: 2/2 positions
```

### **6. Correlation-Adjusted Position Sizing**

Position sizes scale down as you add more correlated positions:

- **1st position:** 100% of base size
- **2nd position:** 75% of base size
- **3rd position:** 50% of base size

**Example:**
```
⚖️  Position size adjusted for correlation: $750.00 → $562.50
   (scaling down due to 1 existing correlated positions)
```

### **7. Visual Correlation Analysis**

For SOL/ETH trades, the LLM receives:
- **3 BTC charts** (30d, 14d, 72h) BEFORE altcoin charts
- **Correlation prompt** with mandatory rules
- **Visual comparison** instructions

**LLM Instructions (excerpt):**
```
BTC CORRELATION ANALYSIS - CRITICAL MARKET CONTEXT:
You are analyzing SOL-USD (Solana), which is HIGHLY CORRELATED with Bitcoin.

MANDATORY CORRELATION RULES:
1. BTC BEARISH TREND = NO BUY FOR SOL-USD
2. BTC SIDEWAYS = SELECTIVE BUYING (only if strong setup)
3. BTC BULLISH = GREEN LIGHT

Compare charts visually:
- Are they moving in sync or diverging?
- Is SOL respecting BTC support/resistance proportionally?
```

---

## Configuration

### **`config.json` Settings:**

```json
"correlation_settings": {
  "enabled": true,
  "max_correlated_long_positions": 2,
  "max_correlated_short_positions": 2,
  "btc_trend_weight": 0.5,
  "correlation_lookback_hours": 168,
  "strong_outperformance_threshold": 5.0,
  "weak_underperformance_threshold": -5.0,
  "require_btc_bullish_for_altcoins": true,
  "allow_altcoins_in_btc_sideways_if_strong": true,
  "correlation_position_size_scaling": true,
  "confidence_boost_on_alignment": true,
  "min_correlation_for_risk_adjustment": 0.6
}
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_correlated_long_positions` | 2 | Max simultaneous long positions |
| `correlation_lookback_hours` | 168 | 7 days for relative strength calc |
| `strong_outperformance_threshold` | 5.0% | Threshold for "strong outperformer" |
| `weak_underperformance_threshold` | -5.0% | Threshold for "strong underperformer" |
| `require_btc_bullish_for_altcoins` | true | Enforce BTC trend filter |
| `correlation_position_size_scaling` | true | Reduce size as positions added |

---

## Workflow

### **Main Loop Execution:**

```
1. Sort assets → BTC first
2. Analyze BTC
   ├─ Store BTC sentiment
   ├─ Store BTC chart paths (30d, 14d, 72h)
   └─ Store BTC price history

3. Analyze SOL
   ├─ Build btc_context dict
   │  ├─ BTC sentiment
   │  ├─ BTC chart paths
   │  └─ BTC price metrics (7d%, 24h%)
   ├─ Pass btc_context to OpenAI analysis
   │  ├─ LLM sees BTC charts FIRST
   │  ├─ LLM applies correlation rules
   │  └─ Returns correlation-aware analysis
   ├─ Calculate relative strength vs BTC
   ├─ Apply portfolio filters
   │  ├─ Check BTC trend
   │  ├─ Check position limits
   │  └─ Adjust confidence
   └─ Scale position size if needed

4. Analyze ETH (same as SOL)

5. Display Portfolio Dashboard
   ├─ Current positions
   ├─ BTC market context
   ├─ Correlation matrix
   └─ Risk alerts
```

---

## Portfolio Dashboard

**End of each iteration, console displays:**

```
================================================================================
                         PORTFOLIO RISK DASHBOARD
================================================================================

📊 CURRENT POSITIONS:
--------------------------------------------------------------------------------
   LONG SOL-USD
      Entry: $142.50 | Shares: 5.2632
      Value: $750.00 | Opened: 2025-10-19T14:32:15

   Total Positions: 1
   Total Exposure: $750.00
   Correlation-Adjusted Risk: $750.00

🔷 BTC MARKET CONTEXT:
--------------------------------------------------------------------------------
   Trend: BULLISH
   Confidence: HIGH
   Recommendation: BUY
   Support: $95,000.00 | Resistance: $105,000.00
   Volume Trend: increasing

📈 CORRELATION ANALYSIS:
--------------------------------------------------------------------------------
   Pairwise Correlations:
      BTC ↔ SOL: +0.823
      BTC ↔ ETH: +0.879
      SOL ↔ ETH: +0.791
      Average:   +0.831

   Interpretation: Very high correlation - assets moving in lockstep

   Relative Strength vs BTC (7-day):
      SOL:
         Performance: +12.30% (BTC: +8.50%)
         Outperformance: +3.80%
         Category: mild_outperformer
      ETH:
         Performance: +9.20% (BTC: +8.50%)
         Outperformance: +0.70%
         Category: mild_outperformer

⚠️  RISK ALERTS:
--------------------------------------------------------------------------------
   ✅ No risk alerts at this time

================================================================================
```

---

## Trade Examples

### **Scenario 1: BTC Bullish + SOL Bullish (APPROVED)**

```
[ BTC-USD ]
✓ BTC sentiment stored: bullish

[ SOL-USD ]
📊 BTC context prepared for SOL-USD correlation analysis
   BTC trend: bullish | 7d: +8.50% | 24h: +2.30%

📈 Relative Strength vs BTC:
   SOL-USD: +12.30% | BTC: +8.50%
   Outperformance: +3.80% (mild_outperformer)

AI Strategy: Buy at $142.00, Target profit 5.5%
Market Trend: bullish | Confidence: high
Trade Recommendation: buy

📊 Portfolio State: 0 positions | $0.00 exposure
   Correlation Risk: $0.00
   ✓ BTC trend: bullish | ✓ Relative strength: mild_outperformer | ✓ Portfolio exposure: 0/2 positions

STATUS: Looking to BUY at $142.00 (Confidence: high)
```

**Result:** ✅ Trade executed

---

### **Scenario 2: BTC Bearish + SOL Bullish (BLOCKED)**

```
[ BTC-USD ]
✓ BTC sentiment stored: bearish

[ SOL-USD ]
📊 BTC context prepared for SOL-USD correlation analysis
   BTC trend: bearish | 7d: -6.20% | 24h: -3.10%

📈 Relative Strength vs BTC:
   SOL-USD: -2.50% | BTC: -6.20%
   Outperformance: +3.70% (mild_outperformer)

AI Strategy: Buy at $138.00, Target profit 4.2%
Market Trend: bullish | Confidence: high
Trade Recommendation: buy

📊 Portfolio State: 0 positions | $0.00 exposure
   Correlation Risk: $0.00
   🚫 BTC bearish backdrop - blocking SOL-USD BUY

STATUS: Correlation filter blocked trade - BTC bearish backdrop overrides SOL technicals
```

**Result:** ❌ Trade blocked by correlation filter

---

### **Scenario 3: BTC Sideways + SOL Strong Outperformer (APPROVED)**

```
[ BTC-USD ]
✓ BTC sentiment stored: sideways

[ SOL-USD ]
📊 BTC context prepared for SOL-USD correlation analysis
   BTC trend: sideways | 7d: +1.20% | 24h: -0.30%

📈 Relative Strength vs BTC:
   SOL-USD: +7.80% | BTC: +1.20%
   Outperformance: +6.60% (strong_outperformer)

AI Strategy: Buy at $145.00, Target profit 6.0%
Market Trend: bullish | Confidence: high
Trade Recommendation: buy

📊 Portfolio State: 0 positions | $0.00 exposure
   ✓ SOL outperforming BTC consolidation - selective buy allowed

STATUS: Looking to BUY at $145.00 (Confidence: high)
```

**Result:** ✅ Trade executed (strong relative strength override)

---

### **Scenario 4: Position Limit Reached (BLOCKED)**

```
[ ETH-USD ]
📊 BTC context prepared for ETH-USD correlation analysis
   BTC trend: bullish | 7d: +8.50% | 24h: +2.30%

AI Strategy: Buy at $3,420.00, Target profit 5.2%
Market Trend: bullish | Confidence: high
Trade Recommendation: buy

📊 Portfolio State: 2 positions | $1,450.00 exposure
   Correlation Risk: $1,740.00
   🚫 Max correlated longs reached (2/2)

STATUS: Correlation filter blocked trade - 🚫 Max correlated longs reached (2/2)
```

**Result:** ❌ Trade blocked (position limit)

---

## Benefits

### **Risk Management**
- ✅ Prevents over-concentration in correlated assets
- ✅ Avoids buying altcoins in BTC downtrends
- ✅ Scales position size based on portfolio exposure
- ✅ Enforces position limits

### **Performance Optimization**
- ✅ Identifies relative strength opportunities
- ✅ Boosts confidence when assets aligned with BTC
- ✅ Reduces confidence when diverging from BTC
- ✅ Visual comparison via LLM for better decisions

### **Transparency**
- ✅ Full correlation metrics displayed each iteration
- ✅ Clear reasoning for blocked trades
- ✅ Risk alerts for over-exposure
- ✅ Real-time portfolio state tracking

---

## Monitoring & Logging

### **Event Logging** (Future Enhancement)

Can be enabled via `portfolio_dashboard.log_correlation_event()`:

```python
from utils.portfolio_dashboard import log_correlation_event

log_correlation_event(
    event_type='trade_blocked',
    symbol='SOL-USD',
    details={
        'reason': 'btc_bearish',
        'btc_trend': 'bearish',
        'asset_trend': 'bullish'
    }
)
```

Events stored in `correlation_events.json` for analysis.

---

## Disabling Correlation System

To disable (revert to independent asset trading):

```json
"correlation_settings": {
  "enabled": false,
  ...
}
```

System will skip all correlation logic and trade assets independently.

---

## Advanced Tuning

### **More Aggressive (allow more risk):**
```json
"max_correlated_long_positions": 3,  // Allow 3 simultaneous longs
"require_btc_bullish_for_altcoins": false,  // Don't enforce BTC trend
"correlation_position_size_scaling": false  // Don't scale down sizes
```

### **More Conservative (reduce risk):**
```json
"max_correlated_long_positions": 1,  // Only 1 position at a time
"strong_outperformance_threshold": 7.0,  // Require stronger outperformance
"min_correlation_for_risk_adjustment": 0.5  // More sensitive to correlation
```

---

## File Reference

| File | Purpose |
|------|---------|
| `utils/correlation_manager.py` | Core correlation logic (350 lines) |
| `utils/portfolio_dashboard.py` | Risk visualization (200 lines) |
| `index.py` | Main loop integration (lines 185-910) |
| `utils/openai_analysis.py` | BTC context in prompts (lines 157-258, 426-442) |
| `config.json` | Correlation settings (lines 46-59) |
| `CORRELATION_SYSTEM.md` | This documentation |

---

## Summary

This correlation system transforms your trading bot from a **multi-asset independent trader** into a **portfolio-aware risk manager** that:

1. **Respects BTC as market leader**
2. **Filters trades based on correlation**
3. **Adjusts position sizes dynamically**
4. **Identifies relative strength opportunities**
5. **Provides comprehensive risk monitoring**

**Net effect:** Smarter trade selection, better risk management, and reduced drawdown during BTC downtrends.
