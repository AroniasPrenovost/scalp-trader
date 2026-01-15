# Simple Momentum Strategy - Implementation Plan

## What Changed

### 1. Data Collection (NOW 1-MINUTE INTERVALS)

**Before:**
- Checked API every 20 seconds
- Saved data every 1 hour (3600 seconds)
- Threw away 99% of data

**After:**
- Checks API every 20 seconds (unchanged)
- **Saves data every 1 minute (60 seconds)** ✅
- Keeps 3x more data points per hour (60 vs 1)

**Config change in `config.json`:**
```json
"interval_seconds": 60  // was 3600
```

### 2. New Strategy (SIMPLE MOMENTUM + VOLUME)

Created: `utils/simple_momentum_strategy.py`

**Entry Logic:**
1. ✅ 5-min momentum > 15-min momentum (acceleration)
2. ✅ 5-min momentum > 0.3% (meaningful move)
3. ✅ Recent 3-min momentum > 0% (confirming direction)
4. ✅ Volume > 1.5x average (confirmation)

**Exit Logic:**
1. Hit 0.7% NET target
2. Hit 0.5% GROSS stop
3. 5-min momentum turns negative (reversal)
4. Max hold 3 hours

**Example Entry:**
```
15-min momentum: +0.4% (baseline trend)
5-min momentum:  +0.7% (accelerating!)
3-min momentum:  +0.3% (confirming)
Volume: 2.1x average (strong confirmation)
→ BUY
```

## Next Steps

### Phase 1: Collect 1-Minute Data (1-2 Days)

**Step 1:** Run your existing `index.py` with the new config:
```bash
python3 index.py
```

Your app will now save data **every 1 minute** instead of every hour.

**Step 2:** Let it run for **24-48 hours** to collect data

You'll see in the console:
```
✓ Appended 1 new data point for 8 cryptos (next append in 1 minute)
```
Instead of:
```
✓ Appended 1 new data point for 8 cryptos (next append in 1 hour)
```

**Step 3:** Check data quality after 1 hour:
```bash
python3 -c "
import json
with open('coinbase-data/LINK-USD.json', 'r') as f:
    data = json.load(f)
    print(f'Total data points: {len(data)}')
    print(f'Expected per hour: ~60')
    print(f'Data looks good!' if len(data) >= 50 else 'Need more time...')
"
```

### Phase 2: Backtest with 1-Minute Data (After 1-2 Days)

Once you have 24-48 hours of 1-minute data, I'll create:

1. **Backtest script:** `backtest/backtest_simple_momentum.py`
   - Uses 1-minute data from `coinbase-data/`
   - Tests simple momentum strategy
   - Reports win rate, P/L, etc.

2. **Compare results:**
   - Hourly data strategies: 18-35% win rate ❌
   - 1-minute momentum strategy: ??? (hopefully 50%+) 🤞

### Phase 3: Add Order Book Logic (Later)

After momentum strategy is validated, we can add:
- Bid-ask spread filter
- Buy/sell pressure signals
- Order book depth analysis

But let's prove momentum works first.

## Why This Should Work Better

**Previous strategies failed because:**
- ❌ Hourly data too coarse (can't see 1-hour movements)
- ❌ Entry signals had no acceleration detection
- ❌ Entries on pullbacks that kept falling

**This strategy fixes it:**
- ✅ 1-minute data shows actual price action
- ✅ Acceleration detection (5-min > 15-min = momentum building)
- ✅ Recent confirmation (3-min positive = not fading)
- ✅ Volume confirmation (not just price noise)
- ✅ Early exit on reversal (don't wait for stop loss)

## Data Storage Impact

**Before (hourly):**
- 24 data points per day per symbol
- 192 data points per day (8 symbols)
- ~10 KB per day

**After (1-minute):**
- 1,440 data points per day per symbol
- 11,520 data points per day (8 symbols)
- ~500 KB per day

**Storage for 180 days:**
- Was: ~2 MB
- Now: ~90 MB (still tiny)

## Testing Readiness Checklist

Before running backtest, ensure:
- [ ] Collected 24+ hours of 1-minute data
- [ ] At least 1,400 data points per symbol (close to 1,440)
- [ ] No large gaps in timestamps (script didn't crash)
- [ ] All 8 enabled symbols have data

Check with:
```bash
python3 -c "
import json, os
symbols = ['LINK-USD', 'AVAX-USD', 'SOL-USD', 'XRP-USD', 'UNI-USD', 'NEAR-USD', 'AAVE-USD', 'FIL-USD']
for symbol in symbols:
    with open(f'coinbase-data/{symbol}.json', 'r') as f:
        data = json.load(f)
        hours = len(data) / 60
        status = '✅' if hours >= 24 else '🔴'
        print(f'{status} {symbol}: {len(data):,} points ({hours:.1f} hours)')
"
```

## What to Watch For

While collecting data, monitor:
1. **Console output** - Should say "next append in 1 minute"
2. **File sizes** - `coinbase-data/*.json` should grow ~20 KB/hour per symbol
3. **No crashes** - Script should run continuously for 24-48 hours

If script crashes, just restart it. Data appends to existing files, so no data loss.

## Questions?

- **Q: Can I backtest with partial data (e.g., 12 hours)?**
  - A: Yes, but need at least 12 hours for meaningful results

- **Q: Should I stop my current trading bot?**
  - A: YES! Don't trade real money while testing. Let it collect data only.

- **Q: What if I already have hourly data?**
  - A: Keep it. New 1-minute data will append to same files. Backtest will handle mixed data.

- **Q: How do I know if 1-minute is better than hourly?**
  - A: Run backtest with both datasets and compare win rates.
