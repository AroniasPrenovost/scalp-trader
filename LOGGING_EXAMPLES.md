# Market Rotation Logging Examples

This document shows exactly what you'll see when running the bot with market rotation enabled.

## Scenario 1: No Active Positions - Scanning for Best Opportunity

```
====================================================================================================
🔍 MARKET ROTATION: Scanning 6 assets for best opportunity...
💰 NO ACTIVE POSITIONS - Capital ready to deploy
====================================================================================================

====================================================================================================
OPPORTUNITY SCANNER - Market Rotation Analysis
====================================================================================================

Rank   Symbol       Score    Signal       Strategy                  Trend        Status
----------------------------------------------------------------------------------------------------
→ #1   SOL-USD      78.5     BUY          Range Support             Sideways     ✅ Ready (high)     ⭐ SELECTED
  #2   ETH-USD      65.2     BUY          Adaptive Mean Reversion   Uptrend      ✅ Ready (medium)
  #3   BTC-USD      52.3     BUY          Ai Analysis               Uptrend      ✅ Ready (medium)
  #4   XRP-USD      -        NO_SIGNAL    -                         Sideways     No Setup
  #5   DOGE-USD     -        NO_SIGNAL    -                         Downtrend    No Setup
  #6   LINK-USD     -        NO_SIGNAL    -                         Sideways     Waiting

====================================================================================================
🎯 BEST OPPORTUNITY: SOL-USD
====================================================================================================
Strategy: Range Support
Score: 78.5/100
Confidence: HIGH
Trend: Sideways
Entry: $142.35
Stop Loss: $139.80
Profit Target: $145.90
Risk/Reward: 1:2.54

Reasoning: Price $142.35 in support zone (avg $142.10, 3 touches)
====================================================================================================

✅ TRADING NOW: SOL-USD
   Score: 78.5/100 | Strategy: Range Support
   Entry: $142.3500 | Stop: $139.8000 | Target: $145.9000
   Risk/Reward: 1:2.54

====================================================================================================
  🎯 SELECTED OPPORTUNITY: SOL-USD - Evaluating Entry
====================================================================================================

Starting Capital:    $3,000.00
Current Value:       $3,000.00
...

--- AI STRATEGY ---
buy_at: $142.35, stop_loss: $139.80, target_profit_%: 2.5%
current_price: $142.40, support: $142.10, resistance: $145.50
market_trend: sideways, confidence: high

--- RANGE SUPPORT STRATEGY CHECK ---
✓ RANGE SIGNAL: BUY
  Support zone: $141.85 - $142.35 (avg: $142.10)
  Zone strength: 3 touches
  Current price: $142.40
  Distance from zone avg: +0.21%
  Price $142.4000 in support zone (avg $142.1000, 3 touches)

====================================================================================================
  ✓ ALL BUY CONDITIONS MET - EXECUTING BUY
====================================================================================================
Range Strategy: ✓ In support zone ($142.10, 3 touches)
AI Analysis: ✓ BUY recommendation with HIGH confidence
Market Price: $142.40 (AI target: $142.35)

Using buy amount: $2850.0 (from LLM analysis)
Calculated shares to buy: 20 whole shares ($2850.0 / $142.40)
Placing LIMIT buy order at $142.3500 (AI target: $142.3500, buffer: 0.0%)
✓ Stored buy screenshot and original AI analysis in ledger

────────────────────────────────────────────────────────────────────────────────────────────────
  👁  MONITORING: BTC-USD (tracking for future opportunities)
────────────────────────────────────────────────────────────────────────────────────────────────

Starting Capital:    $3,000.00
Current Value:       $3,000.00
...

⏭  Skipping trade analysis for BTC-USD - Best opportunity is SOL-USD

[Same for ETH, XRP, DOGE, LINK...]

====================================================================================================
  📋 ITERATION SUMMARY
====================================================================================================
  ✅ Ready to Trade: SOL-USD
  💰 Capital Status: READY ($3,000 available)
  ⏰ Next scan in 5 minutes
====================================================================================================
```

---

## Scenario 2: Active Position Being Managed, Monitoring Others

```
====================================================================================================
🔍 MARKET ROTATION: Scanning 6 assets for best opportunity...
📊 ACTIVE POSITION(S): SOL-USD (managing these)
🔎 MONITORING: BTC-USD, ETH-USD, XRP-USD, DOGE-USD, LINK-USD (scanning for next opportunity)
====================================================================================================

[Opportunity scanner runs in background]

🎯 BEST NEXT OPPORTUNITY: ETH-USD
   Score: 82.5/100 | Strategy: Adaptive Mean Reversion
   ⏸  Waiting for active position(s) to close: SOL-USD
   → Will trade ETH-USD immediately after exit

====================================================================================================
  🔥 ACTIVE TRADE: SOL-USD - Managing Open Position
====================================================================================================

Starting Capital:    $3,000.00
Current Value:       $2,975.12
Gross Profit:        -$24.88
Percentage Gain:     -0.83%
Exchange Fees:       $35.00
Taxes:               $0.00
Net Profit:          -$59.88

📊 Monitoring other assets in background for next opportunity after exit

--- OPEN POSITION ---
entry_price: $142.35
current_price: $141.80
Current profit: -$59.88 (-1.99%)
Stop loss: $139.80 | Profit target: 2.50%

--- POSITION STATUS ---
Entry price: $142.35
Current price: $141.80
Current profit: -$59.88 (-1.99%)
Stop loss: $139.80 | Profit target: 2.50%

────────────────────────────────────────────────────────────────────────────────────────────────
  👁  MONITORING: BTC-USD (tracking for future opportunities)
────────────────────────────────────────────────────────────────────────────────────────────────

Starting Capital:    $3,000.00
Current Value:       $3,000.00
...

⏭  Skipping trade analysis for BTC-USD - Best opportunity is SOL-USD

[ETH, XRP, DOGE, LINK also show as MONITORING and get skipped]

====================================================================================================
  📋 ITERATION SUMMARY
====================================================================================================
  🔥 ACTIVE POSITION(S): SOL-USD
  💰 Capital Status: DEPLOYED (managing position)
  🎯 Next Opportunity Queued: ETH-USD (score: 82.5)
  ⏭  Will trade immediately after current position exits
  ⏰ Next scan in 5 minutes
====================================================================================================
```

---

## Scenario 3: Position Hits Profit Target, Immediately Rotates to Next

```
====================================================================================================
🔍 MARKET ROTATION: Scanning 6 assets for best opportunity...
📊 ACTIVE POSITION(S): SOL-USD (managing these)
🔎 MONITORING: BTC-USD, ETH-USD, XRP-USD, DOGE-USD, LINK-USD (scanning for next opportunity)
====================================================================================================

🎯 BEST NEXT OPPORTUNITY: ETH-USD
   Score: 82.5/100 | Strategy: Adaptive Mean Reversion
   ⏸  Waiting for active position(s) to close: SOL-USD
   → Will trade ETH-USD immediately after exit

====================================================================================================
  🔥 ACTIVE TRADE: SOL-USD - Managing Open Position
====================================================================================================

Current profit: $86.42 (2.88%)
Stop loss: $139.80 | Profit target: 2.50%

~ POTENTIAL SELL OPPORTUNITY (profit % target reached) ~
[Executes market sell order]
✓ Trade closed: +$86.42 profit

====================================================================================================
  📋 ITERATION SUMMARY
====================================================================================================
  🔥 ACTIVE POSITION(S): None (just exited SOL-USD)
  💰 Capital Status: READY ($3,086 available)
  🎯 Next Opportunity: ETH-USD (will trade on next iteration)
  ⏰ Next scan in 5 minutes
====================================================================================================

[5 minutes later - NEXT ITERATION]

====================================================================================================
🔍 MARKET ROTATION: Scanning 6 assets for best opportunity...
💰 NO ACTIVE POSITIONS - Capital ready to deploy
====================================================================================================

✅ TRADING NOW: ETH-USD
   Score: 82.5/100 | Strategy: Adaptive Mean Reversion
   Entry: $3,245.50 | Stop: $3,190.25 | Target: $3,300.75

====================================================================================================
  🎯 SELECTED OPPORTUNITY: ETH-USD - Evaluating Entry
====================================================================================================

[Executes buy order for ETH-USD with full capital]
```

---

## Scenario 4: No Strong Opportunities - Waiting

```
====================================================================================================
🔍 MARKET ROTATION: Scanning 6 assets for best opportunity...
💰 NO ACTIVE POSITIONS - Capital ready to deploy
====================================================================================================

[All scores below 50]

⚠️  Best opportunity XRP-USD has score 42.3 below minimum 50
   Skipping all NEW trades this iteration - waiting for better setups

====================================================================================================
  📋 ITERATION SUMMARY
====================================================================================================
  ⏸  No Strong Opportunities Currently
  💰 Capital Status: IDLE (waiting for quality setup)
  🔎 Monitoring 6 assets: BTC-USD, ETH-USD, SOL-USD, XRP-USD, DOGE-USD, LINK-USD
  ⏰ Next scan in 5 minutes
====================================================================================================
```

---

## Key Logging Indicators

### Asset Status Indicators
- **🔥 ACTIVE TRADE** = This asset has an open position being managed
- **🎯 SELECTED OPPORTUNITY** = This asset is the best opportunity, evaluating entry
- **👁 MONITORING** = This asset is being tracked but not traded this iteration
- **⏭ Skipping** = Asset analysis skipped (not the best opportunity)

### Capital Status
- **💰 READY ($3,000 available)** = No positions, ready to deploy
- **💰 DEPLOYED (managing position)** = Capital in active trade
- **💰 IDLE (waiting for quality setup)** = No positions and no good opportunities

### Opportunity Status
- **✅ TRADING NOW** = Entering this trade right now
- **🎯 Next Opportunity Queued** = This will be traded after current position exits
- **⏸ Waiting for active position(s) to close** = Good opportunity but capital tied up
- **⏸ No Strong Opportunities Currently** = All scores below minimum threshold

### Visual Separators
- `====` (double lines) = Active trade or selected opportunity (important)
- `────` (single lines) = Monitoring only (informational)

This enhanced logging ensures you always know:
1. Which asset is actively trading
2. What the bot is monitoring in the background
3. What opportunity is queued next
4. Why decisions are being made
