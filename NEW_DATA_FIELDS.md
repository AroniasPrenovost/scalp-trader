# New Data Fields Now Being Stored

## What Changed

Updated `index.py` to store **8 fields** instead of **4 fields** from Coinbase API.

### Before (4 fields)
```python
{
    'timestamp': 1737841234.567,
    'product_id': 'BTC-USD',
    'price': '95234.50',
    'volume_24h': '12345678.90'
}
```

### After (8 fields) ✅
```python
{
    'timestamp': 1737841234.567,
    'product_id': 'BTC-USD',
    'price': '95234.50',
    'volume_24h': '12345678.90',
    # NEW: Built-in momentum from Coinbase
    'price_percentage_change_24h': '2.5',      # +2.5% in 24h
    'volume_percentage_change_24h': '150.0',   # Volume up 150%
    # NEW: Trading status flags
    'trading_disabled': False,
    'cancel_only': False,
    'post_only': False,
    'is_disabled': False
}
```

## Why These Fields Matter

### 1. `price_percentage_change_24h` 🔥

**What it is:** 24-hour price momentum already calculated by Coinbase

**Why it's valuable:**
- You were calculating momentum manually from historical data
- Coinbase gives it to you for FREE in every API call
- Can use as a pre-filter: "Only trade if 24h momentum > 1%"

**Example use:**
```python
if coin['price_percentage_change_24h'] > 1.0:
    # Strong 24h trend, check for acceleration
```

### 2. `volume_percentage_change_24h` 🔥

**What it is:** 24-hour volume change already calculated by Coinbase

**Why it's valuable:**
- Shows if volume is spiking (breakout) or dying (consolidation)
- Can detect volume acceleration without historical lookback
- Volume spike + price momentum = strong signal

**Example use:**
```python
if coin['volume_percentage_change_24h'] > 100:
    # Volume doubled in 24h = something happening
```

### 3. Trading Status Flags 🚫

**What they are:** Tells you if trading is restricted

**Why they're valuable:**
- `trading_disabled`: Market halted (news event, technical issue)
- `cancel_only`: Can't enter new positions (liquidity issue)
- `post_only`: Only maker orders allowed (high volatility)
- `is_disabled`: Product completely disabled

**Example use:**
```python
if coin['trading_disabled'] or coin['is_disabled']:
    continue  # Skip this coin, can't trade it
```

## How This Improves Strategy

### Old Momentum Strategy
```python
# Had to calculate from historical data
momentum_5min = (prices[-1] - prices[-6]) / prices[-6] * 100
momentum_15min = (prices[-1] - prices[-16]) / prices[-16] * 100
```

### New Momentum Strategy (Enhanced)
```python
# Use Coinbase's 24h momentum as pre-filter
if price_percentage_change_24h < 0.5:
    return 'no_signal'  # Not enough 24h movement

# Then calculate short-term acceleration
momentum_5min = ...  # Still calculate this
momentum_15min = ...  # Still calculate this

# Add volume confirmation
if volume_percentage_change_24h < 50:
    return 'no_signal'  # Volume not spiking
```

## Strategy Enhancement Ideas

### Filter 1: Strong 24h Trend Required
```python
# Only trade coins with strong 24h momentum
if price_percentage_change_24h < 1.0:
    return 'no_signal'
```

### Filter 2: Volume Acceleration
```python
# Only trade when volume is spiking
if volume_percentage_change_24h < 75:
    return 'no_signal'  # Volume must be 75%+ higher
```

### Filter 3: Market Health Check
```python
# Skip disabled/restricted markets
if trading_disabled or cancel_only or is_disabled:
    return 'no_signal'
```

## Data Example

With 1-minute collection, you'll now see:
```json
[
  {
    "timestamp": 1737841260.0,
    "product_id": "LINK-USD",
    "price": "23.45",
    "volume_24h": "12500000.00",
    "price_percentage_change_24h": "3.2",
    "volume_percentage_change_24h": "125.5",
    "trading_disabled": false,
    "cancel_only": false,
    "post_only": false,
    "is_disabled": false
  },
  {
    "timestamp": 1737841320.0,
    "product_id": "LINK-USD",
    "price": "23.47",
    "volume_24h": "12550000.00",
    "price_percentage_change_24h": "3.3",
    "volume_percentage_change_24h": "127.0",
    "trading_disabled": false,
    "cancel_only": false,
    "post_only": false,
    "is_disabled": false
  }
]
```

Every minute, you get fresh momentum and volume acceleration data from Coinbase!

## Performance Impact

**Storage increase:**
- Was: 4 fields × 8 bytes = ~32 bytes per entry
- Now: 8 fields × 8 bytes = ~64 bytes per entry
- **2x storage** (still tiny: ~1 MB/day per symbol)

**Value increase:**
- Built-in momentum calculation ✅
- Built-in volume acceleration ✅
- Market health monitoring ✅
- Can add 2-3 more filters to improve win rate ✅

## Next: Enhanced Strategy

Once you have 24-48 hours of this new data, I can create an **enhanced momentum strategy** that uses:

1. ✅ 24h momentum filter (from `price_percentage_change_24h`)
2. ✅ Volume spike filter (from `volume_percentage_change_24h`)
3. ✅ 5-min vs 15-min acceleration (calculated from historical)
4. ✅ Market health check (from trading status flags)

This should push win rate from ~35% (old strategies) to hopefully 55%+.

## Sources

- [Coinbase Advanced Trade API - List Products](https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/products/list-products)
- [Coinbase Developer Platform](https://www.coinbase.com/developer-platform/products/advanced-trade-api)
