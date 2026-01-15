# High-Frequency Data Collection

## The Problem

Your current system:
- ✅ Checks Coinbase API every 20 seconds
- ❌ Only saves data to disk every **1 hour**
- ❌ Throws away 99% of the data (179 out of 180 checks per hour)

For scalping/swing trading, **hourly data is too coarse**. You can't see:
- Intra-hour price swings
- Buy/sell pressure changes
- Bid-ask spread fluctuations
- Actual entry/exit points

## The Solution

The new `collect_high_frequency_data.py` script:
- ✅ Collects data every **30 seconds** (customizable)
- ✅ Stores **ALL available Coinbase fields** (not just price/volume)
- ✅ Saves to separate directory: `coinbase-data-hf/`
- ✅ Auto-cleans old data (keeps 7 days by default)

### Data Fields Collected

**Standard fields:**
- `timestamp` - Unix timestamp
- `product_id` - Trading pair (e.g., BTC-USD)
- `price` - Last trade price

**New high-value fields:**
- `best_bid` - Current best bid
- `best_ask` - Current best ask
- `bid_ask_spread` - Spread in dollars
- `bid_ask_spread_pct` - Spread as % of price
- `buy_pressure_pct` - % of recent trades that were buys
- `sell_pressure_pct` - % of recent trades that were sells
- `recent_trades` - Last 10 trades with price, size, side, time
- `num_recent_trades` - Count of recent trades

### Why These Fields Matter

1. **Bid-ask spread** - Wide spread = low liquidity, risky entry
2. **Buy/sell pressure** - Shows which side is dominant
3. **Recent trades** - See actual executed prices and sizes
4. **30-second granularity** - Catch momentum shifts that hourly data misses

## How to Use

### Step 1: Start Collecting Data

```bash
python3 collect_high_frequency_data.py
```

This will:
- Connect to Coinbase API
- Fetch ticker data every 30 seconds for all enabled symbols
- Save to `coinbase-data-hf/{SYMBOL}.json`
- Print stats every 5 minutes
- Auto-cleanup old data every hour

### Step 2: Let It Run for 1-2 Days

You need at least **24 hours** of high-frequency data for meaningful backtests.

The script shows collection status:
- 🔴 Need more data (< 6 hours)
- 🟡 Getting close (6-24 hours)
- ✅ Ready for backtesting (24+ hours)

### Step 3: Backtest with High-Frequency Data

Once you have 1-2 days of data, modify your backtest scripts to use `coinbase-data-hf/` instead of `coinbase-data/`.

## Example Data Structure

```json
{
  "timestamp": 1737841234.567,
  "product_id": "BTC-USD",
  "price": "95234.50",
  "best_bid": "95233.00",
  "best_ask": "95236.00",
  "bid_ask_spread": "3.00",
  "bid_ask_spread_pct": "0.003",
  "buy_pressure_pct": "65.5",
  "sell_pressure_pct": "34.5",
  "recent_trades": [
    {
      "trade_id": "abc123",
      "price": "95234.50",
      "size": "0.0523",
      "side": "BUY",
      "time": "2026-01-15T12:00:34.000Z"
    }
  ],
  "num_recent_trades": 10
}
```

## Data Retention

- **Default:** 7 days of high-frequency data
- **Storage:** ~1-2 MB per symbol per day at 30-second intervals
- **Total:** ~10-20 MB for 8 symbols over 7 days

To change retention period, edit:
```python
MAX_AGE_DAYS = 7  # Change this value
```

## Configuration

Edit these variables in `collect_high_frequency_data.py`:

```python
COLLECTION_INTERVAL_SECONDS = 30  # How often to collect (default 30s)
DATA_DIRECTORY = 'coinbase-data-hf'  # Where to store data
MAX_AGE_DAYS = 7  # How long to keep data
```

## Running in Background

### Option 1: Screen/tmux (Linux/Mac)
```bash
screen -S hf-collector
python3 collect_high_frequency_data.py
# Press Ctrl+A, then D to detach
# Reattach with: screen -r hf-collector
```

### Option 2: nohup (Unix)
```bash
nohup python3 collect_high_frequency_data.py > hf-collector.log 2>&1 &
```

### Option 3: System Service (Linux)
Create `/etc/systemd/system/hf-collector.service`:
```ini
[Unit]
Description=High-Frequency Crypto Data Collector
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/Users/arons_stuff/Documents/scalp-scripts
ExecStart=/usr/bin/python3 collect_high_frequency_data.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable hf-collector
sudo systemctl start hf-collector
sudo systemctl status hf-collector
```

## Next Steps

1. ✅ Run collector for 1-2 days
2. ⏳ Modify backtest scripts to use high-frequency data
3. ⏳ Test with new strategies that use bid-ask spread and buy/sell pressure
4. ⏳ Compare results to hourly data backtests

## Why This Will Help

**With hourly data:**
- ❌ 150 trades in 180 days
- ❌ 18-25% win rate
- ❌ Can't see intra-hour movements

**With 30-second data:**
- ✅ See actual price action between hours
- ✅ Detect momentum shifts faster
- ✅ Use bid-ask spread to filter bad entries
- ✅ Use buy/sell pressure to gauge market sentiment
- ✅ Proper backtesting for 1-2 hour scalping strategies

## Monitoring

The script prints stats every 5 minutes showing:
- Number of data points collected per symbol
- Duration of data collection
- Average collection interval
- Status for backtesting readiness

Press `Ctrl+C` to stop and see final stats.
