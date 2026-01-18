# API Cleanup Summary

## What Was Removed

**Eliminated the redundant `get_products()` call:**

### Before (Lines 273-278):
```python
coinbase_data = coinbase_client.get_products()['products']
coinbase_data_dictionary = convert_products_to_dicts(coinbase_data)
coinbase_data_dictionary_all = coinbase_data_dictionary
coinbase_data_dictionary = [coin for coin in coinbase_data_dictionary if coin['product_id'] in enabled_wallets]
```

### After:
```python
# Removed entirely - no longer needed
```

## Why This Works

**We now use:**
1. **For data collection:** `fetch_latest_5min_candle()` - Gets 5-minute candles
2. **For trading prices:** `get_asset_price()` - Gets real-time price at moment of trade
3. **For display prices:** `get_asset_price()` - Gets current price for showing "distance from entry"

**We iterate through:** `enabled_wallets` from config (line 261) instead of `coinbase_data_dictionary`

## Changes Made

| Line | Before | After |
|------|--------|-------|
| 273-278 | `get_products()` call | Removed |
| 282 | `for coin in coinbase_data_dictionary:` | `for product_id in enabled_wallets:` |
| 476 | `next((c for c in coinbase_data_dictionary...` | `get_asset_price(coinbase_client, ...)` |
| 498 | `next((c for c in coinbase_data_dictionary...` | `get_asset_price(coinbase_client, ...)` |
| 512 | `next((c for c in coinbase_data_dictionary...` | `get_asset_price(coinbase_client, ...)` |
| 547 | `for coin in coinbase_data_dictionary:` | `for symbol in enabled_wallets:` |

## Benefits

1. **Fewer API calls:** Eliminated one `get_products()` call per 5-minute interval
2. **Cleaner code:** Single source of truth for prices (`get_asset_price()`)
3. **More accurate:** Real-time prices instead of stale snapshot prices
4. **Single data source:** Only candles for historical data

## API Call Reduction

**Per 5-minute interval:**
- Before: `get_products()` + `get_candles()` × 8 assets + `get_asset_price()` when trading
- After: `get_candles()` × 8 assets + `get_asset_price()` when needed

**Savings:** 1 fewer call every 5 minutes = 288 fewer calls per day

