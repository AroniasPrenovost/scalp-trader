# 5-Minute Candle Data Collection Guide

## Overview

Your trading bot now collects 5-minute candles from Coinbase Advanced API instead of 30-second ticker snapshots. This gives you:
- Better correlation calculations (consistent intervals)
- Lower API usage (5 min vs 30 sec = 90% reduction)
- Perfect alignment with your momentum divergence strategy

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. BACKFILL (One-time or as needed)                        │
│     backfill_coinbase_candles.py --days 90                  │
│     → Populates /coinbase-data/*.json with historical data  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. LIVE COLLECTION (Continuous via index.py)               │
│     Every 5 minutes, index.py fetches latest candle         │
│     → Appends to same /coinbase-data/*.json files           │
│     → Auto-deduplicates (skips existing timestamps)         │
│     → Auto-cleanup (keeps only max_hours from config)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. STRATEGY EXECUTION (index.py uses this data)            │
│     Your momentum divergence strategy reads candles         │
│     → Calculates correlations over LOOKBACK_WINDOW          │
│     → Identifies divergence opportunities                   │
│     → Executes trades                                       │
└─────────────────────────────────────────────────────────────┘
```

## Current Data Status

| Asset     | Candles | Coverage | Ready |
|-----------|---------|----------|-------|
| BTC-USD   | 25,927  | 100%     | ✅    |
| ETH-USD   | 25,928  | 100%     | ✅    |
| SOL-USD   | 25,926  | 100%     | ✅    |
| AVAX-USD  | 25,770  | 99.4%    | ✅    |
| LINK-USD  | 25,926  | 100%     | ✅    |
| ALGO-USD  | 24,655  | 95.1%    | ✅    |
| XRP-USD   | 25,925  | 100%     | ✅    |
| DOGE-USD  | 25,924  | 100%     | ✅    |

**Total:** ~206,000 5-minute candles (90 days of historical data)

## How It Works in index.py

### Before (30-second ticker snapshots):
```python
# Old approach - ticker API every 30 seconds
coinbase_data = coinbase_client.get_products()['products']
# Stored: timestamp, price, volume_24h, etc.
```

### Now (5-minute candles):
```python
# New approach - candle API every 5 minutes
from utils.candle_helpers import fetch_latest_5min_candle, candle_to_data_entry

for product_id in enabled_wallets:
    # Fetch most recent completed 5-minute candle
    candle = fetch_latest_5min_candle(coinbase_client, product_id)

    # Transform to same format as before
    data_entry = candle_to_data_entry(candle, product_id)
    # → {timestamp, product_id, price, volume_24h}

    # Append to /coinbase-data/{product_id}.json
    append_crypto_data_to_file(coinbase_data_directory, product_id, data_entry)
```

**Key changes in index.py (lines 283-319):**
1. Import candle helpers (line 13)
2. Fetch candles instead of ticker data (lines 297-308)
3. Check for duplicates before appending (line 311-313)
4. Same file format, same storage location

## Configuration

**In config.json:**
```json
{
  "data_retention": {
    "max_hours": 4380,        // Keep 6 months (auto-cleanup)
    "interval_seconds": 300   // 5 minutes = 300 seconds
  }
}
```

**Your strategy settings:**
```json
{
  "momentum_divergence": {
    "lookback_window": 6      // 6 candles = 30 minutes
  }
}
```

## Commands

### Backfill Historical Data

```bash
# All enabled assets, 7 days (recommended minimum)
python3 backfill_coinbase_candles.py --days 7

# All enabled assets, 90 days (recommended for robust testing)
python3 backfill_coinbase_candles.py --days 90

# Single asset
python3 backfill_coinbase_candles.py BTC-USD --days 30

# Custom period
python3 backfill_coinbase_candles.py --days 180
```

### Run Live Trading (with auto-collection)

```bash
# index.py now automatically collects candles every 5 minutes
python3 index.py

# Or with nohup for background
nohup python3 index.py > trading_bot.log 2>&1 &
```

### Test Integration

```bash
# Verify candle collection works
python3 test_candle_collection.py
```

## Data Format

Each `/coinbase-data/{ASSET}.json` file contains:

```json
[
  {
    "timestamp": 1768690800.0,
    "product_id": "BTC-USD",
    "price": "95095.97",
    "volume_24h": "10.84506839"
  },
  {
    "timestamp": 1768691100.0,
    "product_id": "BTC-USD",
    "price": "95120.50",
    "volume_24h": "10.92341234"
  }
  // ... ~26,000 more candles
]
```

**Timestamps:** Unix timestamps in seconds (5-minute intervals)
**Price:** Close price of the 5-minute candle
**Volume:** Volume in base currency (BTC for BTC-USD)

## Benefits

### For Your Strategy

1. **Reliable Correlations:**
   - Consistent 5-min intervals (not variable ticker timing)
   - LOOKBACK_WINDOW=6 → exactly 30 minutes of data

2. **Lower API Usage:**
   - Before: 288 calls/day (every 5 min) × 8 assets = 2,304 calls
   - With tickers: 1,728 calls/day (every 30 sec) × 8 assets = 13,824 calls
   - **Reduction: 84% fewer API calls**

3. **Better Backtesting:**
   - 90 days × 288 candles/day = 25,920 candles per asset
   - See multiple market cycles (bull, bear, ranging)
   - Statistically significant correlation calculations

## Troubleshooting

### "No candles returned"
- Asset might be delisted or disabled
- Check Coinbase status: https://status.coinbase.com/

### "Duplicate candle"
- Normal! index.py skips candles that already exist
- Prevents data corruption from overlapping runs

### "File not found"
- Run backfill first: `python3 backfill_coinbase_candles.py`
- Or wait for index.py to collect first candle (5 min)

### Missing candles (coverage < 95%)
- Some gaps are normal (Coinbase API limits, network issues)
- Re-run backfill to fill gaps: `python3 backfill_coinbase_candles.py {ASSET} --days 90`

## Next Steps

1. ✅ **Backfill complete** - 90 days of data loaded
2. ✅ **index.py updated** - Collecting 5-min candles automatically
3. **Run your bot:** `python3 index.py`
4. **Monitor:** Check logs for "Collected X new 5-min candles"
5. **Backtest:** Use the historical data to optimize strategy parameters

## Files Changed

- ✅ `utils/candle_helpers.py` - New candle fetching functions
- ✅ `index.py` (lines 13, 283-319) - Integrated candle collection
- ✅ `backfill_coinbase_candles.py` - Historical data backfill script
- ✅ `test_candle_collection.py` - Integration test script

**No breaking changes:** Same file format, same storage location (`/coinbase-data`)
