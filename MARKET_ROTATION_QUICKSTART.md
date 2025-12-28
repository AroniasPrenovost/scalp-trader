# Market Rotation Quick Start

## What Changed?

Your bot now **actively rotates capital** between the best opportunities instead of trading all coins simultaneously.

### Before:
- Monitor 3 coins (BTC, ETH, XRP)
- Trade all 3 at once with divided capital ($1,000 each)
- Miss opportunities in other coins
- Capital sits idle when no setup on your 3 coins

### After:
- Monitor 6-11 top coins
- **Trade only the BEST one** with full capital ($3,000)
- Skip all others (unless managing an open position)
- After exit, immediately find the next best opportunity
- **Money always working in the highest-quality setup**
- **Clear logging shows exactly which asset is actively trading vs monitoring**

## Quick Setup (5 minutes)

### 1. Enable More Coins

The config now includes 11 top coins. Enable the ones you want to monitor:

In `config.json`, set `enabled: true` for:
- ✅ BTC-USD (already enabled)
- ✅ ETH-USD (already enabled)
- ✅ SOL-USD (already enabled)
- ✅ XRP-USD (already enabled)
- ✅ DOGE-USD (already enabled)
- ✅ LINK-USD (already enabled)
- ⚪ ADA-USD (optional)
- ⚪ AVAX-USD (optional)
- ⚪ DOT-USD (optional)
- ⚪ MATIC-USD (optional)
- ⚪ UNI-USD (optional)

**Recommendation:** Start with 6 (the ones already enabled), add more later.

### 2. Backfill Historical Data

For any NEW coins you enable, run:

```bash
python backfill_historical_data.py
```

This downloads hourly price history for analysis.

### 3. Market Rotation is Already Enabled

Check `config.json`:

```json
{
  "market_rotation": {
    "enabled": true,
    "total_trading_capital_usd": 3000,
    "min_opportunity_score": 50
  }
}
```

### 4. Test It

```bash
python test_opportunity_scorer.py
```

You'll see:
- All coins ranked by opportunity score
- Which strategy fits each one
- The single best trade highlighted

### 5. Run in Dry-Run Mode

Keep `ready_to_trade: false` for all coins initially:

```bash
python index.py
```

Watch the output - you'll see:
```
🔍 MARKET ROTATION: Scanning 6 assets for best opportunity...
✓ Trading SOL-USD (score: 78.5, strategy: range_support)
⏭  Skipping BTC-USD - not the best opportunity
⏭  Skipping ETH-USD - not the best opportunity
...
```

### 6. Enable Live Trading (When Ready)

Set `ready_to_trade: true` for the coins you want to trade:

```json
{
  "symbol": "XRP-USD",
  "enabled": true,
  "ready_to_trade": true  // ← Enable this
}
```

**Start with 1-2 coins**, test for a few days, then enable more.

## How It Works (Simple Explanation)

Every 5 minutes:

1. **Scan** all enabled coins
2. **Score** each one (0-100) based on:
   - Is there a buy signal? (range support, mean reversion, AI)
   - How strong is the signal?
   - What's the market trend? (uptrend > sideways > downtrend)
   - Risk/reward ratio
3. **Pick** the highest-scoring one
4. **Trade** only that one, skip all others
5. **After exit**, repeat (find next best opportunity)

## Key Settings

### Minimum Score Threshold
```json
"min_opportunity_score": 50
```
- Below 50 = skip (setup not strong enough)
- 50-70 = decent setup
- 70-85 = very strong setup
- 85+ = excellent setup

**Adjust this based on your risk tolerance:**
- Conservative: 60 (only trade great setups)
- Balanced: 50 (trade good setups)
- Aggressive: 40 (trade more often, accept weaker setups)

### Total Trading Capital
```json
"total_trading_capital_usd": 3000
```
- This is how much you deploy per trade ($3,000)
- Update this to match your actual capital if different

## What to Expect

### Enhanced Logging
Every iteration now clearly shows:
- **🔥 ACTIVE TRADE**: Which asset has an open position
- **🎯 SELECTED OPPORTUNITY**: Which asset is being evaluated for entry
- **👁 MONITORING**: Which assets are being tracked for future opportunities
- **📋 ITERATION SUMMARY**: Status of capital, active positions, and next queued opportunity

Example output:
```
====================================================================================================
🔍 MARKET ROTATION: Scanning 6 assets for best opportunity...
📊 ACTIVE POSITION(S): XRP-USD (managing these)
🔎 MONITORING: BTC-USD, ETH-USD, SOL-USD, DOGE-USD, LINK-USD (scanning for next opportunity)
====================================================================================================

====================================================================================================
  🔥 ACTIVE TRADE: XRP-USD - Managing Open Position
====================================================================================================
  Monitoring other assets in background for next opportunity after exit

────────────────────────────────────────────────────────────────────────────────────────────────
  👁  MONITORING: BTC-USD (tracking for future opportunities)
────────────────────────────────────────────────────────────────────────────────────────────────
  ⏭  Skipping trade analysis for BTC-USD - Best opportunity is XRP-USD

====================================================================================================
  📋 ITERATION SUMMARY
====================================================================================================
  🔥 ACTIVE POSITION(S): XRP-USD
  💰 Capital Status: DEPLOYED (managing position)
  🎯 Next Opportunity Queued: SOL-USD (score: 78.5)
  ⏭  Will trade immediately after current position exits
  ⏰ Next scan in 5 minutes
====================================================================================================
```

### First Few Hours
- Bot shows opportunity reports every 5 min
- Clear indication of which asset is actively trading
- Other assets continue being monitored in background
- Summary shows queued opportunity for immediate rotation after exit

### After First Trade Entry
- **Active position clearly labeled** with 🔥 ACTIVE TRADE
- Other assets show as 👁 MONITORING (analysis continues)
- Summary shows next best opportunity queued
- When you exit, immediately jumps to the queued opportunity

### Daily/Weekly
- Rotate between 3-5 different coins based on best setups
- Always know exactly which asset is deployed vs monitoring
- Capital efficiency: Full $3,000 per trade

## Troubleshooting

**Q: "Skipping all trades - best score below minimum"**
A: Good! No strong setups right now. The system is protecting you.

**Q: "No tradeable opportunities - all have open positions"**
A: You have multiple open positions from before. Once they close, rotation will work normally.

**Q: Bot keeps trading the same coin**
A: That coin keeps having the best setup. This is expected. Rotation happens when other coins get better scores.

**Q: Error: "Insufficient data for {symbol}"**
A: Run `python backfill_historical_data.py` to download historical data.

## Files Created

- **`utils/opportunity_scorer.py`** - Core scoring logic
- **`test_opportunity_scorer.py`** - Test script
- **`MARKET_ROTATION_GUIDE.md`** - Full documentation (read this for details)
- **`config.json`** - Updated with 11 coins + rotation settings

## Next Steps

1. ✅ Run `test_opportunity_scorer.py` to verify it works
2. ✅ Run `index.py` in dry-run mode (ready_to_trade: false)
3. ✅ Watch the opportunity reports for a few hours
4. ✅ Enable live trading for 1-2 coins
5. ✅ Monitor performance for a week
6. ✅ Adjust `min_opportunity_score` if needed
7. ✅ Enable more coins once comfortable

## Summary

You now have an **active market rotation system** that:
- 🔍 Scans 6-11 top crypto coins
- 📊 Scores each opportunity (0-100)
- 🎯 Trades only the BEST one
- 💰 Deploys full capital ($3,000) per trade
- 🔄 Rotates to next best opportunity after exit
- ⏭️  Skips mediocre setups
- 🔥 **Clear logging shows active trades vs monitoring**
- 📋 **Iteration summary shows what's happening at a glance**

Your money is always working for you in the highest-quality opportunity available, and you always know exactly what the bot is doing.

**Read `MARKET_ROTATION_GUIDE.md` for full details.**
