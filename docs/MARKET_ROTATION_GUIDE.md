# Market Rotation Strategy Guide

## Overview

The **Market Rotation System** enables your trading bot to actively scan 4-11 crypto assets and automatically deploy your full capital into the **single best trading opportunity** at any moment. Instead of dividing capital across multiple mediocre setups, you concentrate firepower on the highest-quality trade.

## Key Concepts

### 1. Active Capital Rotation
- **Old approach**: Monitor 3 coins, trade them all simultaneously with divided capital
- **New approach**: Monitor 6-11 coins, find the BEST setup, trade only that one with full capital
- **Result**: Higher win rate by only taking the cream of the crop

### 2. Opportunity Scoring
Every enabled asset gets scored 0-100 based on:
- **Strategy match** (50 points base): Does it fit range support or mean reversion?
- **Setup strength** (0-20 points): How strong is the signal? (zone touches, dip depth, etc.)
- **Market trend** (0-20 points): Uptrend > Sideways > Downtrend
- **Risk/Reward** (0-10 points): Bonus for 2:1 or better R/R ratio
- **Penalties**: Downtrends (-30), low volatility (-20)

### 3. Single Position at a Time
- Only the **#1 ranked opportunity** gets traded
- All other coins are **skipped** (unless they have an open position to manage)
- When you exit, the system **immediately reassesses** all coins and finds the next best opportunity

## Configuration

### Enable Market Rotation

In `config.json`:

```json
{
  "market_rotation": {
    "enabled": true,
    "mode": "single_best_opportunity",
    "scan_all_enabled_assets": true,
    "total_trading_capital_usd": 2500,
    "min_score_for_entry": 50,
    "print_opportunity_report": true
  }
}
```

**Settings explained:**
- `enabled`: Turn market rotation on/off
- `mode`: "single_best_opportunity" = only trade the best one
- `total_trading_capital_usd`: Your full trading capital per trade
- `min_score_for_entry`: Skip trades if best score is below this (50 = reasonable threshold)
- `print_opportunity_report`: Show detailed report of all opportunities each iteration

### Add Coins to Monitor

Add top crypto coins to the `wallets` array:

```json
{
  "wallets": [
    {
      "title": "BTC",
      "symbol": "BTC-USD",
      "coingecko_id": "bitcoin",
      "enabled": true,              // Must be true to scan
      "ready_to_trade": false,       // Set to true when you want to trade (start with false)
      "starting_capital_usd": 2500,
      "enable_chart_snapshot": false
    },
    {
      "title": "ETH",
      "symbol": "ETH-USD",
      "coingecko_id": "ethereum",
      "enabled": true,
      "ready_to_trade": false,
      "starting_capital_usd": 2500,
      "enable_chart_snapshot": false
    }
    // ... add more coins (SOL, XRP, DOGE, ADA, AVAX, DOT, MATIC, LINK, UNI)
  ]
}
```

**Recommended top coins:**
1. BTC-USD (Bitcoin) - Highest liquidity
2. ETH-USD (Ethereum) - Very liquid, good volatility
3. SOL-USD (Solana) - Excellent for trading
4. XRP-USD (Ripple) - High volume
5. DOGE-USD (Dogecoin) - High retail volume
6. ADA-USD (Cardano) - Good liquidity
7. LINK-USD (Chainlink) - Consistent volume
8. AVAX-USD (Avalanche) - Good volatility
9. DOT-USD (Polkadot) - Decent liquidity
10. MATIC-USD (Polygon) - Good for scalping
11. UNI-USD (Uniswap) - DeFi leader

## Strategies Supported

The opportunity scorer evaluates three strategies for each coin:

### 1. Range Support Strategy
**Best for:** Sideways/ranging markets

Detects support zones with multiple price bounces (2-5 touches) and triggers buy when price revisits the zone.

**Scoring:**
- Base: 50 points
- +5 per zone touch (up to +20)
- +15 for sideways markets
- +10 for proximity to zone

**When it triggers:**
- Price is within 1.5% of a support zone
- Zone has 2+ touches
- Not in strong downtrend

### 2. Adaptive Mean Reversion
**Best for:** Trending markets with pullbacks

Buys dips 2-3% below 24h moving average in uptrending markets.

**Scoring:**
- Base: 45 points
- +20 for uptrend
- +15 for optimal dip depth (2-3%)

**When it triggers:**
- Price is 2-3% below 24h MA
- Market is uptrend or sideways
- Not in downtrend

## How It Works

### Every 5 minutes, the bot:

1. **Scans all enabled assets**
   - Loads price data for each coin
   - Calculates volatility, trend, support zones
   - Runs all 3 strategy checks

2. **Scores each opportunity**
   - Assigns 0-100 score based on setup quality
   - Identifies which strategy fits best
   - Marks which coins have open positions

3. **Selects the best one**
   - Sorts by score (highest first)
   - Picks #1 if score ≥ min_score_for_entry
   - Skips trading if no good setups

4. **Executes trade logic**
   - Only processes the selected coin for BUY signals
   - Skips all other coins (unless managing existing positions)
   - Manages SELL logic for any open positions

5. **After exit, reassess immediately**
   - Once you sell, the next iteration finds the new best opportunity
   - Capital rotates to the next high-quality setup

## Example Output

```
====================================================================================================
🔍 MARKET ROTATION: Scanning 6 assets for best opportunity...
====================================================================================================

====================================================================================================
OPPORTUNITY SCANNER - Market Rotation Analysis
====================================================================================================

Rank   Symbol       Score    Signal       Strategy                  Trend        Status
----------------------------------------------------------------------------------------------------
→ #1   SOL-USD      78.5     BUY          Range Support             Sideways     ✅ Ready (high)     ⭐ SELECTED
  #2   ETH-USD      65.2     BUY          Adaptive Mean Reversion   Uptrend      ✅ Ready (medium)
  #3   XRP-USD      -        NO_SIGNAL    -                         Sideways     No Setup
  #4   BTC-USD      -        NO_SIGNAL    -                         Uptrend      Waiting
  #5   DOGE-USD     -        NO_SIGNAL    -                         Downtrend    No Setup
  #6   LINK-USD     -        NO_SIGNAL    -                         Sideways     Position Open

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

✓ Trading SOL-USD (score: 78.5, strategy: range_support)

==================================================
    SOL-USD
==================================================

Starting Capital:    $2,500.00
Current Value:       $2,500.00
...

⏭  Skipping BTC-USD - not the best opportunity (trading SOL-USD instead)
⏭  Skipping ETH-USD - not the best opportunity (trading SOL-USD instead)
⏭  Skipping XRP-USD - not the best opportunity (trading SOL-USD instead)
...
```

## Setup Instructions

### Step 1: Add Coins to Config

1. Open `config.json`
2. Add/enable the coins you want to monitor (see list above)
3. Set `enabled: true` for 6-11 top coins
4. Set `ready_to_trade: false` initially (test mode)
5. Update `starting_capital_usd: 2500` for each

### Step 2: Backfill Historical Data

For each new coin you add, run the backfill script:

```bash
python backfill_historical_data.py
```

This downloads hourly price/volume data for all enabled coins.

### Step 3: Test the Opportunity Scorer

```bash
python test_opportunity_scorer.py
```

This will:
- Score all enabled coins
- Show which strategy fits each one
- Display the best opportunity
- Verify the rotation logic works

### Step 4: Enable Market Rotation

In `config.json`:

```json
{
  "market_rotation": {
    "enabled": true,
    "min_score_for_entry": 50
  }
}
```

### Step 5: Monitor in Dry-Run Mode

Run the main bot with `ready_to_trade: false` for all coins:

```bash
python index.py
```

Watch the opportunity reports for a few days to verify:
- ✅ It correctly identifies the best setups
- ✅ It skips poor opportunities
- ✅ The scoring makes sense

### Step 6: Enable Live Trading

When confident, set `ready_to_trade: true` for coins you want to trade:

```json
{
  "title": "SOL",
  "symbol": "SOL-USD",
  "enabled": true,
  "ready_to_trade": true,  // ← Enable live trading
  ...
}
```

**Important:** Only set `ready_to_trade: true` for 1-2 coins initially to test the rotation.

## How Capital Gets Deployed

### Scenario 1: First Trade
- Bot scans: BTC, ETH, SOL, XRP, DOGE, LINK
- Best opportunity: **SOL** (score 78.5, range support at $142)
- **Action**: Buy SOL with full $2,500
- **Skip**: BTC, ETH, XRP, DOGE, LINK

### Scenario 2: While in SOL Position
- Bot scans again
- SOL: Open position (managing it)
- ETH: Great setup (score 82.5)
- **Action**: Continue managing SOL, ignore ETH
- **Rationale**: Capital is tied up, can't take new trades

### Scenario 3: Exit SOL, Find Next Best
- SOL hits profit target, sells
- Bot immediately scans all coins
- Best opportunity: **ETH** (score 82.5)
- **Action**: Buy ETH with full $2,500
- **Skip**: All others

### Scenario 4: No Good Setups
- Bot scans all coins
- Best score: **XRP** (35.2, below min 50)
- **Action**: Skip all trades, wait for better setup
- **Rationale**: Don't force trades when quality is low

## Best Practices

### 1. Start Conservative
- Begin with `min_score_for_entry: 60` (higher threshold)
- Only trade when setups are excellent
- Lower to 50 once you gain confidence

### 2. Monitor 6-11 Coins
- Too few (1-3): Not enough opportunities
- Too many (15+): Spreads analysis too thin
- Sweet spot: 6-11 top liquid coins

### 3. Let Positions Breathe
- Don't set `cooldown_hours_after_sell: 0`
- Use 3-5 hours cooldown to avoid revenge trading
- Lets you reassess with a clear head

### 4. Trust the Scoring
- If score is low, skip it
- Don't override the system
- The math is designed to filter bad trades

### 5. Review Performance Weekly
- Which coins produce the best setups?
- Are you missing opportunities due to min_score being too high?
- Adjust enabled coins based on market conditions

## Troubleshooting

### "No tradeable opportunities found"
**Cause:** All coins either have positions open or no valid setups

**Fix:**
- Wait for market conditions to improve
- Check if `min_score_for_entry` is too high
- Verify strategies are enabled in config

### "Skipping all trades - best score below minimum"
**Cause:** Best setup score < `min_score_for_entry`

**Fix:**
- This is GOOD - system is protecting you from mediocre trades
- If happens too often, lower min_score to 45-50

### "Error scoring {symbol}: Insufficient data"
**Cause:** Missing historical price data for that coin

**Fix:**
```bash
python backfill_historical_data.py
```

### Multiple coins showing "Position Open"
**Cause:** You have leftover positions from before market rotation was enabled

**Fix:**
- Let the bot manage and exit those positions
- Once sold, only 1 position will be open at a time going forward

## Performance Metrics

Track these to measure success:

1. **Opportunity Utilization**
   - How often does the bot find a trade? (target: 60-80% of iterations)
   - Too high (95%+): min_score too low, taking bad trades
   - Too low (20%): min_score too high, missing good trades

2. **Win Rate by Strategy**
   - Which strategy produces the best results?
   - Support bounce: typically 55-65% win rate
   - Breakout: typically 50-60% win rate
   - Consolidation break: typically 50-60% win rate

3. **Capital Efficiency**
   - Old: $2500 / 3 coins = $833 per trade
   - New: $2500 / 1 coin = full firepower
   - Result: Larger wins, faster compounding

## Summary

The Market Rotation System transforms your bot from a multi-coin passive monitor into an **active opportunity hunter**. By concentrating capital on the single best setup at any moment, you:

✅ Maximize capital efficiency
✅ Improve win rate (only take A+ setups)
✅ Stay actively deployed (rotate between 6-11 coins)
✅ Avoid mediocre trades
✅ Compound gains faster

Configure it, backfill data, test it, then let it run. Your money will always be working for you in the best available opportunity.
