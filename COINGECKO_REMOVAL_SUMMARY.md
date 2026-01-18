# CoinGecko Removal Summary

## What Was Removed

All CoinGecko-related code and configuration has been removed since we now use Coinbase Advanced API exclusively for 5-minute candle data.

### Files Deleted

- ✅ `backfill_historical_data.py` - Old CoinGecko backfill script
- ✅ `coingecko-global-volume/` - Old CoinGecko data directory (was empty)
- ✅ `__pycache__/backfill_historical_data.cpython-313.pyc` - Python cache

### Files Updated

**config.json:**
- ❌ Removed all `coingecko_id` fields from wallet configurations
- ✅ Wallets now only have: `title`, `symbol`, `enabled`, `ready_to_trade`, `starting_capital_usd`, `comment`

**example-config.json:**
- ❌ Removed all `coingecko_id` fields

**.env:**
- ❌ Removed `COINGECKO_API_KEY=...`

**.env.sample:**
- ❌ Removed `COINGECKO_API_KEY=...`

**setup_scorer.py:**
- ❌ Removed unused `coingecko_dir='coingecko-global-volume'` parameter from `calculate_setup_score()`
- ✅ Now: `calculate_setup_score(product_id, coinbase_dir='coinbase-data')`

**backtest/backtest_final_strategy.py:**
- ❌ Removed `'coingecko-global-volume'` argument from function call
- ✅ Now: `calculate_setup_score(symbol, 'coinbase-data')`

**analyze_historical_moves.py:**
- ⚠️ Added deprecation notice (references old coingecko-global-volume directory)
- ⚠️ Kept for reference only

**analyze_scalping_opportunities.py:**
- ⚠️ Added deprecation notice (references old coingecko-global-volume directory)
- ⚠️ Kept for reference only

## What We're Using Now

**Single data source:** Coinbase Advanced API

### Data Collection
```python
# Every 5 minutes in index.py
from utils.candle_helpers import fetch_latest_5min_candle, candle_to_data_entry

for product_id in enabled_wallets:
    candle = fetch_latest_5min_candle(coinbase_client, product_id)
    data_entry = candle_to_data_entry(candle, product_id)
    append_crypto_data_to_file('coinbase-data', product_id, data_entry)
```

### Historical Backfill
```bash
# Backfill 90 days of 5-minute candles
python3 backfill_coinbase_candles.py --days 90
```

## Why We Removed CoinGecko

1. **Redundant:** CoinGecko only provided hourly data (much worse than 5-minute candles)
2. **Extra API calls:** Unnecessary API usage and costs
3. **Complexity:** Maintaining two data sources was overcomplicated
4. **Better alternative:** Coinbase Advanced API gives us 5-minute candles directly

## Benefits

1. ✅ **Simpler codebase:** Single data source
2. ✅ **Better data quality:** 5-minute candles vs hourly snapshots
3. ✅ **Fewer API dependencies:** Only Coinbase (which we already use for trading)
4. ✅ **Lower costs:** No CoinGecko API subscription needed
5. ✅ **Cleaner config:** Less fields to manage

## Before vs After

### Before (CoinGecko)
```json
{
  "wallets": [
    {
      "title": "BTC",
      "symbol": "BTC-USD",
      "coingecko_id": "bitcoin",  ❌ REMOVED
      "enabled": true,
      ...
    }
  ]
}
```

### After (Coinbase Only)
```json
{
  "wallets": [
    {
      "title": "BTC",
      "symbol": "BTC-USD",
      "enabled": true,
      ...
    }
  ]
}
```

## Migration Notes

**No action required!** Your existing `/coinbase-data` files already contain 5-minute candle data from the Coinbase backfill script.

**If you had old CoinGecko data:**
- It was stored in `coingecko-global-volume/` (now deleted)
- Replace it with: `python3 backfill_coinbase_candles.py --days 90`

## Validation

✅ `config.json` is valid JSON
✅ All Python files compile successfully
✅ Integration test passed (8/8 assets)
✅ Data collection working with 5-minute candles

## Files You Can Now Ignore

The following files reference the old CoinGecko directory but are **deprecated** and kept only for reference:

- `analyze_historical_moves.py` ⚠️
- `analyze_scalping_opportunities.py` ⚠️

Use the backtest scripts in `/backtest` instead.
