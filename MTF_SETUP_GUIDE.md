# MTF Momentum Breakout Strategy - Setup Guide

## Complete Workflow

### Step 1: Backfill Historical Data (ONE TIME)

Your MTF strategy requires **at least 200 days** of 5-minute candle data to calculate the 200-day moving average.

```bash
python3 backfill_coinbase_candles.py --days 210
```

**What this does:**
- Fetches 210 days of 5-minute candles from Coinbase for all 8 enabled coins
- Saves to `coinbase-data/{SYMBOL}.json` files
- Takes ~5-10 minutes (processes in chunks of 25 hours due to API limits)
- Merges with any existing data (won't duplicate)

**Expected output:**
- ~60,480 candles per coin (210 days × 288 5-min candles/day)
- Total: ~483,840 candles across all 8 coins

### Step 2: Test the Opportunity Scorer

After backfill completes, verify the scorer works:

```bash
python3 test_mtf_scorer.py
```

**What this does:**
- Scans all 8 enabled coins (BTC, ETH, SOL, AVAX, LINK, ADA, LTC, ATOM)
- Runs MTF strategy check on each one
- Converts signals to 0-100 scores
- Shows you the top 1-2 opportunities

**Expected output:**
- Table showing all 8 coins ranked by score
- Details on any coins with signals (score ≥ 75)
- Clear indication if no opportunities exist right now

### Step 3: Run in Perpetuity

Once data is backfilled and tested, start the bot:

```bash
python3 index.py
```

**What this does (continuous loop, runs forever):**

**Every 5 minutes:**
1. **Collect new candles** - Fetches latest 5-min candle for each coin
2. **Scan for opportunities** - Scores all 8 coins using MTF strategy
3. **Select best 1-2** - Picks highest-scoring opportunities (≥75 score)
4. **Execute trades** - Buys if `ready_to_trade: true` and criteria met
5. **Manage positions** - Monitors stop-loss, profit targets, trailing stops
6. **Repeat** - Loop continues indefinitely

**Data management:**
- Keeps last 210 days of data (`max_hours: 5040`)
- Automatically prunes older data
- Continuously appends new 5-min candles

## System Requirements

### Data Requirements

| Requirement | Amount | Why |
|-------------|--------|-----|
| **Minimum 5-min candles** | 300 candles | 25 hours for 4H aggregation |
| **Minimum 4H candles** | 30 candles | 5 days for BB/RSI calculations |
| **Minimum daily candles** | 200 candles | 200 days for 200-MA filter |
| **Recommended backfill** | 210 days | 10-day buffer above minimum |

### Strategy Logic (per coin, every 5 minutes)

1. **Load data** - Uses `config.json` → `data_retention.max_hours` (currently 5040 = 210 days)
2. **Aggregate to 4H candles** (210 days ÷ 4H = ~1,260 candles)
3. **Aggregate to daily candles** (210 candles)
4. **Check MTF signal:**
   - Daily: Price > 200-MA? ✓
   - 4H: Bollinger Band squeeze? ✓
   - 4H: Breakout above upper BB? ✓
   - Volume: 2x average? ✓
   - RSI: 50-70? ✓
   - MACD: Positive & increasing? ✓
5. **Score 0-100** based on confidence, R/R, profit potential
6. **Return signal** if score ≥ 75

**Note:** The scorer automatically reads `max_hours` from `config.json`, so if you change the data retention period, the strategy will automatically use that amount of historical data.

## Your Config Settings

**Enabled coins:** 8 (BTC, ETH, SOL, AVAX, LINK, ADA, LTC, ATOM)

**Market rotation:**
- `min_score_for_entry`: 75 (only trade strong signals)
- `max_concurrent_orders`: 2 (max 2 positions at once)
- `capital_per_position`: $2,250
- `total_trading_capital_usd`: $4,500

**MTF strategy:**
- `target_profit_pct`: 7.5% (gross, ~5% net after fees/taxes)
- `atr_stop_multiplier`: 2.0 (stop-loss at 2× ATR below entry)
- `max_concurrent_positions`: 2

**Data retention:**
- `max_hours`: 5040 (210 days)
- Collects new candles every 5 minutes
- Auto-cleans data older than 210 days

## Testing & Validation

### Before Going Live

1. **Backfill data** (as above)
2. **Run test script** - Verify signals appear correctly
3. **Keep `ready_to_trade: false`** for all coins initially
4. **Run index.py in dry-run mode** - Watch for a few hours/days
5. **Review opportunity reports** - Ensure scoring makes sense
6. **Enable 1-2 coins** - Set `ready_to_trade: true` for BTC/ETH only
7. **Monitor first few trades** - Validate live execution
8. **Scale up gradually** - Enable more coins once confident

### What to Expect

**In dry-run mode (ready_to_trade: false):**
- Opportunity scanner runs every 5 minutes
- Shows which coins have signals
- Reports would-be entries/exits
- No actual trades placed

**In live mode (ready_to_trade: true):**
- Same scanning, but executes real trades
- Max 2 positions at once
- Each position: $2,250 (50% of $4,500 capital)
- Targets: 7.5% gross profit (~5% net)
- Stops: 2× ATR below entry (typically 3-6%)

## Troubleshooting

### "Insufficient data for 200-MA"
**Cause:** Not enough historical candles
**Fix:** Run `python3 backfill_coinbase_candles.py --days 210`

### "No opportunities found"
**Cause:** No coins currently have MTF breakout signals
**Fix:** This is normal - MTF signals are selective. Wait for better market conditions.

### "All opportunities scored below 75"
**Cause:** Signals exist but confidence/quality is low
**Fix:** System is protecting you from weak trades. Can lower `min_score_for_entry` to 65-70 if desired.

### "Error fetching price"
**Cause:** Coinbase API connection issue
**Fix:** Check `.env` has valid `COINBASE_API_KEY` and `COINBASE_API_SECRET`

## File Structure

```
coinbase-data/
  BTC-USD.json          # 210 days of 5-min candles
  ETH-USD.json
  SOL-USD.json
  AVAX-USD.json
  LINK-USD.json
  ADA-USD.json
  LTC-USD.json
  ATOM-USD.json

utils/
  mtf_momentum_breakout_strategy.py   # Core strategy logic
  mtf_opportunity_scorer.py           # Scoring & ranking system
  technical_indicators.py             # BB, MACD, ATR, etc.

config.json                           # Bot configuration
index.py                              # Main bot (runs perpetually)
backfill_coinbase_candles.py          # Historical data loader
test_mtf_scorer.py                    # Test/validation script
```

## Summary

**To get started:**
```bash
# 1. Backfill data (one time, ~10 minutes)
python3 backfill_coinbase_candles.py --days 210

# 2. Test the scorer (verify it works)
python3 test_mtf_scorer.py

# 3. Run the bot (perpetually)
python3 index.py
```

The system is now **fully focused** on the MTF Momentum Breakout strategy with no legacy code. It will:
- Continuously collect new 5-min candles
- Scan all 8 coins every 5 minutes
- Only trade when strong MTF signals appear (score ≥ 75)
- Target 5-15% swing moves with proper risk management
