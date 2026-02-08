# Scalp-Scripts: MTF Momentum Breakout Trading Bot

An automated cryptocurrency trading bot that uses Multi-Timeframe (MTF) technical analysis to identify and execute momentum breakout trades on Coinbase. The bot scans multiple assets, ranks opportunities using a proprietary scoring system, and manages positions with ATR-based stops and profit targets.

## Overview

**Scalp-Scripts** automates the entire trading lifecycle:
- Continuously collects 5-minute candle data for 8 cryptocurrencies
- Analyzes market conditions using Multi-Timeframe Momentum Breakout Strategy
- Scores opportunities across all assets and selects the best 1-2 trades
- Executes buy/sell orders automatically when high-quality signals appear
- Manages positions with ATR-based stop-loss and profit targets
- Tracks all costs: exchange fees, taxes, net profits
- Sends email notifications for all trading activity

## Key Features

### 📊 Multi-Timeframe Momentum Breakout Strategy
- **200-Day MA Filter**: Only considers trades when price is above long-term trend (bullish bias)
- **4-Hour Bollinger Band Squeeze**: Identifies compression periods that precede explosive moves
- **Volume Confirmation**: Requires 2x average volume to confirm breakout validity
- **Multi-Indicator Confluence**: RSI (50-70), MACD positive & increasing, ATR expansion
- **Market Rotation**: Scans 8 cryptocurrencies every 5 minutes, ranks by opportunity score
- **Selective Entry**: Only trades signals scoring ≥75 (high confidence, favorable risk/reward)
- **5-Minute Candles**: High-granularity data for precise entry/exit timing

### 💰 Professional Risk Management
- **ATR-Based Stops**: Dynamic stop-loss calculated at 2× ATR below entry price (adapts to volatility)
- **Trailing Stops**: Locks in profits as position moves favorably (moves up, never down)
- **Profit Targets**: 7.5% gross target (~5% net after fees/taxes)
- **Position Sizing**: Fixed $2,250 per position (50% of $4,500 capital)
- **Max Concurrent Positions**: Limited to 2 simultaneous trades (prevents overexposure)
- **Tax Calculation**: Factors in federal tax rates on realized capital gains
- **Exchange Fee Accounting**: Calculates Coinbase taker fees on both buy and sell sides
- **Wallet Metrics Dashboard**: Tracks starting capital, current value, percentage gain/loss, gross vs. net profit

### 📊 Comprehensive Order Management
- **Local Ledger Tracking**: Maintains per-symbol JSON files tracking complete order history
- **Order State Detection**:
  - `none` - No active position
  - `buy` - Currently holding position
  - `sell` - Just exited position
  - `placeholder` - Pending order confirmation
- **Buy Event Documentation**: Captures screenshots at time of purchase for later analysis
- **Complete Transaction Records**: Stores detailed data for every trade:
  - Entry/exit prices and timestamps
  - Time held position
  - Profit/loss calculations
  - Tax and fee amounts
  - Buy event screenshot paths

### 📈 Data Collection & Analysis
- **5-Minute Candle Collection**: Appends latest candle to per-crypto JSON files every 5 minutes
- **Data Retention**: Automatically maintains 210 days of historical data (5,040 hours)
- **Candle Properties**: timestamp, product_id, close price, 24h volume
- **Historical Data Backfilling**: Coinbase Advanced API integration for populating up to 210 days of 5-minute candles
- **Smart Data Merging**: Timestamp-based deduplication prevents duplicates when merging backfilled and real-time data
- **Automated Cleanup**: Removes data older than retention window to maintain consistent dataset size

### 🔔 Notifications & Monitoring
- **Email Notifications** (via Mailjet):
  - Buy order placement alerts with entry price and opportunity score
  - Sell order execution alerts with profit percentage and hold time
  - Error/crash notifications
  - Graceful degradation (continues after repeated errors)
- **Console Logging**: Real-time status updates including opportunity scores, signals, entry/exit prices
- **Opportunity Reports**: Every scan cycle shows ranked list of all 8 coins with scores and signal details

## Prerequisites

- **Python 3.7+**
- **Coinbase Account** with API credentials (trading permissions enabled)
- **Mailjet Account** (optional, for email notifications)

## Installation

### Option 1: Quick Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AroniasPrenovost/scalp-trader.git
   cd scalp-trader
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables (see [Environment Setup](#environment-setup) below)

4. Configure your trading parameters (see [Configuration](#configuration) below)

5. Run the bot:
   ```bash
   python3 index.py
   ```

### Option 2: Using Virtual Environment (Recommended)

Using a virtual environment isolates your project dependencies and is the recommended approach.

1. Navigate to your project directory:
   ```bash
   cd ~/scalp-scripts
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

   **Note**: On Windows, use `venv\Scripts\activate` instead.

   To deactivate the virtual environment later, simply run:
   ```bash
   deactivate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up your environment variables (see [Environment Setup](#environment-setup) below)

6. Configure your trading parameters (see [Configuration](#configuration) below)

7. Run the bot:
   ```bash
   python3 index.py
   ```

The bot will run continuously, monitoring your specified assets and executing trades based on market analysis.

## Environment Setup

Create a `.env` file in the root directory with the following variables. You can copy `.env.sample` if available:

```bash
# Coinbase API Credentials
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here

# Tax Configuration
FEDERAL_TAX_RATE=15                # Your federal tax rate on capital gains (e.g., 15 for 15%)

# Email Notifications (Mailjet) - Optional
MAILJET_API_KEY=your_mailjet_api_key
MAILJET_SECRET_KEY=your_mailjet_secret_key
MAILJET_FROM_EMAIL=bot@example.com
MAILJET_FROM_NAME=Trading Bot
MAILJET_TO_EMAIL=your@email.com
MAILJET_TO_NAME=Your Name
```

### Getting Your API Keys

- **Coinbase API**: https://www.coinbase.com/settings/api (enable trading permissions)
- **Mailjet API**: https://app.mailjet.com/account/api_keys (optional)

## Configuration

### Basic Configuration

Copy `example-config.json` to `config.json` and customize your settings:

```bash
cp example-config.json config.json
```

### Configuration Parameters

#### MTF Strategy Settings

```json
{
  "mtf_strategy": {
    "target_profit_pct": 7.5,
    "atr_stop_multiplier": 2.0,
    "trailing_stop_activation_pct": 3.0,
    "trailing_stop_distance_pct": 1.5,
    "max_concurrent_positions": 2
  },
  "market_rotation": {
    "enabled": true,
    "min_score_for_entry": 75,
    "max_concurrent_orders": 2,
    "capital_per_position": 2250,
    "total_trading_capital_usd": 4500
  }
}
```

**MTF Strategy:**
- **`target_profit_pct`**: Target gross profit percentage (7.5% = ~5% net after fees/taxes)
- **`atr_stop_multiplier`**: Stop-loss distance in ATR units below entry (2.0 = 2× ATR)
- **`trailing_stop_activation_pct`**: Profit level to activate trailing stop (3.0%)
- **`trailing_stop_distance_pct`**: Distance trailing stop trails below peak (1.5%)
- **`max_concurrent_positions`**: Maximum simultaneous open positions (2)

**Market Rotation:**
- **`min_score_for_entry`**: Minimum opportunity score required to enter trade (75)
- **`max_concurrent_orders`**: Maximum positions to hold at once (2)
- **`capital_per_position`**: USD allocated per trade ($2,250)
- **`total_trading_capital_usd`**: Total capital available for trading ($4,500)

#### Data Management Settings

```json
{
  "data_retention": {
    "max_hours": 5040,
    "interval_seconds": 300
  }
}
```

- **`max_hours`**: Maximum hours of candle data to retain (5,040 hours = 210 days). Required for 200-day MA calculation. Default: `5040`
- **`interval_seconds`**: How often to collect new 5-minute candles in seconds (300 seconds = 5 minutes). Default: `300`

#### Wallet/Asset Configuration

```json
{
  "wallets": [
    {
      "title": "BTC: Large cap, high liquidity",
      "symbol": "BTC-USD",
      "enabled": true,
      "ready_to_trade": false,
      "starting_capital_usd": 2250
    },
    {
      "title": "ETH: Second largest, DeFi leader",
      "symbol": "ETH-USD",
      "enabled": true,
      "ready_to_trade": false,
      "starting_capital_usd": 2250
    }
  ]
}
```

Each asset in the `wallets` array supports the following parameters:

- **`title`**: Friendly description for the asset (for reference only)
- **`symbol`**: The Coinbase trading pair symbol (e.g., `"BTC-USD"`, `"ETH-USD"`, `"SOL-USD"`)
- **`enabled`**: Whether to scan this asset for opportunities (`true`) or skip it (`false`)
- **`ready_to_trade`**: Whether to execute actual trades (`true`) or run in simulation mode (`false`)
  - **IMPORTANT**: Start with `false` to test the bot in simulation mode before risking real money
- **`starting_capital_usd`**: Initial capital allocated for this asset (used for performance metrics calculation)

**Note**: With market rotation enabled, the bot scans all enabled assets but only trades the best 1-2 opportunities.

## Quick Start Guide

### Step 1: Backfill Historical Data (Required - One Time Only)

The MTF strategy **requires at least 200 days** of 5-minute candle data to calculate the 200-day moving average filter.

```bash
python3 backfill_coinbase_candles.py --days 210
```

**What this does:**
- Fetches 210 days of 5-minute candles from Coinbase for all enabled coins
- Saves to `coinbase-data/{SYMBOL}.json` files
- Takes ~5-10 minutes (processes in chunks due to API limits)
- Merges with any existing data (won't duplicate)
- **Expected**: ~60,480 candles per coin (210 days × 288 candles/day)

### Step 2: Test the Strategy (Optional but Recommended)

Verify the MTF strategy works with your data:

```bash
python3 test_mtf_scorer.py
```

**What this does:**
- Scans all enabled coins in your config
- Runs MTF strategy check on each one
- Converts signals to 0-100 opportunity scores
- Shows ranked list of opportunities

### Step 3: Run the Bot

Start the bot in continuous mode:

```bash
python3 index.py
```

**What happens (continuous loop every 5 minutes):**
1. Collects latest 5-min candle for each enabled coin
2. Scans all coins and scores opportunities using MTF strategy
3. Selects best 1-2 opportunities (score ≥75)
4. Executes trades if `ready_to_trade: true` and criteria met
5. Manages positions (stop-loss, profit targets, trailing stops)
6. Repeat indefinitely

### How the Backfilling Works

The backfilling script (`backfill_coinbase_candles.py`) performs the following operations:

1. **Data Fetching**:
   - Fetches 5-minute candles directly from Coinbase Advanced Trade API
   - Uses the `/candles` endpoint with `FIVE_MINUTE` granularity
   - Automatically chunks requests into 25-hour periods (max 300 candles per request)

2. **Data Format**:
   - Each candle contains: `timestamp`, `product_id`, `price` (close), `volume_24h` (base currency)
   - Matches the exact format used by the live collection in `index.py`
   - Compatible with all existing strategies and backtests

3. **Smart Merging**:
   - Loads any existing data from `coinbase-data/{SYMBOL}.json`
   - Merges new candles with existing data using timestamp-based deduplication
   - Sorts all data points chronologically by timestamp
   - Only adds new candles that don't already exist (no overwrites)

### Data Granularity

**5-minute candles** - Perfect for multi-timeframe momentum strategies:

| Time Period | Candles | Storage Size |
|-------------|---------|--------------|
| **1 day** | 288 | ~50 KB |
| **1 week** | 2,016 | ~350 KB |
| **1 month** | 8,640 | ~1.5 MB |
| **210 days** | 60,480 | ~10 MB |

**Benefits:**
- Aggregates to 4H candles for Bollinger Band analysis
- Aggregates to daily candles for 200-day MA filter
- Precise entry/exit timing on 5-min resolution
- Captures intra-hour volatility expansions

### Running the Backfill

**Backfill all enabled assets (210 days required for MTF strategy):**
```bash
python3 backfill_coinbase_candles.py --days 210
```

**Backfill specific asset:**
```bash
python3 backfill_coinbase_candles.py BTC-USD --days 210
```

**Note**: The MTF strategy requires at least 200 days for the 200-MA filter. 210 days provides a 10-day buffer.

### Example Output

```
======================================================================
Coinbase Advanced API - Historical 5-Minute Candle Backfill
======================================================================
✓ Coinbase client initialized

📅 Backfill period: 210 days
📊 Granularity: 5-minute candles
📈 Expected candles per asset: ~60,480

Found 8 enabled wallet(s) to backfill:
  - BTC-USD
  - ETH-USD
  - SOL-USD
  ...

=== Backfilling 5-minute candles for BTC-USD ===
  Requesting 210 days of data
  From: 2025-07-05 to 2026-01-31

  Chunk 1: 2025-07-05 00:00 to 2025-07-06 01:00 (25.0 hours)
  ✓ Fetched 300 candles
  ...

  Total candles fetched: 60,384
  Found 0 existing data points
  ✓ Added 60,384 new data points, total: 60,384
✓ Saved 60,384 data points to coinbase-data/BTC-USD.json

  ✓ Backfill complete for BTC-USD
  📊 Data coverage: 60,384/60,480 candles (99.8%)
```

### Data Structure

Each candle is stored in the same format as live collection:

```json
{
  "timestamp": 1768690800.0,
  "product_id": "BTC-USD",
  "price": "95095.97",
  "volume_24h": "10.84506839"
}
```

### Best Practices

- **Always use 210 days** - Required for the 200-day MA filter in MTF strategy
- **Re-run if needed** - The script automatically deduplicates, safe to run multiple times
- **Run before going live** - Populate historical data before enabling `ready_to_trade`
- **One-time operation** - After backfill, `index.py` automatically maintains the data

### Continuous Data Collection

Once backfilled, `index.py` seamlessly continues the data collection:
- Fetches latest 5-minute candle every 5 minutes
- Appends to same `coinbase-data/*.json` files
- Auto-deduplicates to prevent overlaps
- Auto-prunes data older than 210 days

**Result:** Continuous, up-to-date 210-day dataset for strategy analysis

## How It Works

### Main Trading Loop

The bot runs continuously in a loop (every 5 minutes):

1. **Data Collection**
   - Fetches latest 5-minute candle from Coinbase API for each enabled coin
   - Appends candle data to per-symbol JSON files in `coinbase-data/`
   - Maintains 210-day rolling window (auto-prunes older data)

2. **Order State Management**
   - Checks last order status for each symbol (`none`, `buy`, `sell`, or `placeholder`)
   - If `placeholder` (pending), polls Coinbase for final order status
   - Updates position tracking and wallet metrics

3. **Opportunity Scanning** (every cycle)
   - For each enabled coin:
     - Loads 210 days of 5-min candles
     - Aggregates to 4-hour and daily timeframes
     - Runs MTF Momentum Breakout Strategy checks
     - Calculates 0-100 opportunity score
   - Ranks all coins by score
   - Identifies best 1-2 opportunities (score ≥75)

4. **Entry Logic** (if fewer than 2 open positions)
   - If top opportunity scores ≥75:
     - Verify price above 200-day MA
     - Confirm BB squeeze and breakout
     - Check volume is 2× average
     - Validate RSI in 50-70 range
     - **If all conditions met:**
       - Calculate position size ($2,250)
       - Calculate ATR-based stop-loss (2× ATR below entry)
       - Place market buy order on Coinbase
       - Save order to ledger with entry price, stop, target

5. **Exit Logic** (if holding positions)
   - **For each open position:**
     - Calculate current profit/loss
     - Update trailing stop if profit ≥ activation threshold
     - **Exit triggers:**
       - Price hits ATR stop-loss → Sell (limit losses)
       - Price hits trailing stop → Sell (lock in profits)
       - Profit ≥ 7.5% target → Sell (take profits)
   - **After sell:**
     - Save transaction record to `transactions/data.json`
     - Clear position state
     - Free up capital for next opportunity

6. **Data Cleanup**
   - Remove candle data older than 210 days
   - Optionally prune old screenshots

### MTF Momentum Breakout Strategy

The bot uses a **Multi-Timeframe Momentum Breakout** approach:

#### Entry Requirements (ALL must be true)
1. **Daily Timeframe**: Price > 200-day MA (bullish trend filter)
2. **4-Hour Bollinger Bands**: Recent squeeze (bandwidth in lower 20th percentile)
3. **4-Hour Breakout**: Price broke above upper BB in last 3 candles
4. **Volume Confirmation**: Current volume ≥ 2× average volume
5. **RSI**: Between 50-70 (momentum without overbought)
6. **MACD**: Positive and increasing (trend strength)
7. **Opportunity Score**: ≥75 (high confidence, favorable R/R ratio)

#### Exit Signals

**Stop Loss:**
- Price drops below (entry - 2× ATR)
- Protects against major drawdowns
- Adapts to asset volatility

**Trailing Stop:**
- Activates when profit ≥ 3%
- Trails 1.5% below peak profit
- Moves up only (never down)
- Locks in gains as position moves favorably

**Profit Target:**
- Triggers when profit ≥ 7.5% gross
- Approximately 5% net after fees and taxes

#### Market Rotation
- Scans all 8 enabled coins simultaneously
- Only trades the highest-scoring opportunities
- Maximum 2 concurrent positions (prevents overexposure)
- Each position: $2,250 (50% of $4,500 capital)
- Dynamically shifts to best opportunities across market

## File Structure

```
/scalp-scripts/
├── index.py                              # Main entry point - core trading loop
├── backfill_coinbase_candles.py          # Coinbase 5-minute candle backfill script
├── test_mtf_scorer.py                    # Test/validation script for MTF strategy
├── config.json                           # Active configuration (user-specific)
├── example-config.json                   # Template configuration
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── MTF_SETUP_GUIDE.md                    # Detailed MTF strategy setup guide
├── .env                                  # Environment variables (create from .env.sample)
│
├── utils/                                # Utility modules
│   ├── coinbase.py                       # Coinbase API wrapper & order management
│   ├── mtf_momentum_breakout_strategy.py # MTF strategy logic
│   ├── mtf_opportunity_scorer.py         # Opportunity scoring & ranking
│   ├── technical_indicators.py           # BB, MACD, RSI, ATR calculations
│   ├── wallet_helpers.py                 # Wallet metrics & transaction history
│   ├── candle_helpers.py                 # 5-minute candle fetching & formatting
│   ├── file_helpers.py                   # File I/O & data management
│   ├── profit_calculator.py              # Profit/loss calculations
│   ├── price_helpers.py                  # Price calculation utilities
│   ├── time_helpers.py                   # Time formatting utilities
│   └── email.py                          # Email notifications (Mailjet)
│
├── docs/                                 # Documentation files
│
├── coinbase-data/                        # Runtime: 5-minute candle data per symbol
│   ├── BTC-USD.json                      # 210 days of 5-min candles
│   ├── ETH-USD.json
│   ├── SOL-USD.json                      # (auto-generated for all enabled coins)
│   └── ...
│
├── transactions/                         # Runtime: Trade history
│   └── data.json                         # All completed trades with metrics
│
├── screenshots/                          # Runtime: Buy event visual records
│   ├── BTC-USD_chart_buy_20251015120000.png
│   └── (auto-generated)
│
├── active-coinbase-orders/                      # Per-symbol order ledgers (auto-generated)
│   ├── BTC-USD_orders.json               # Position tracking and order history
│   ├── ETH-USD_orders.json
│   └── {SYMBOL}_orders.json
│
└── venv/                                 # Python virtual environment (if using Option 2)
```

## Monitoring & Logs

### Console Output
The bot provides real-time status updates in the console:
- Current 5-minute candle data collection
- Opportunity scanning results (ranked list of all coins with scores)
- MTF strategy signals (200-MA, BB squeeze, breakout, volume, RSI, MACD)
- Entry/exit decisions with reasoning
- Position tracking (profit/loss, trailing stops, ATR stops)
- Order placement and execution confirmations

### Email Notifications
You'll receive email alerts for:
- Buy order placements (with entry price, stop-loss, and opportunity score)
- Sell order executions (with profit percentage, exit reason, and hold time)
- Error conditions or crashes

### Transaction History
All completed trades are logged in `transactions/data.json` with complete details:
- Entry/exit prices and timestamps
- Gross profit, taxes, fees, and net profit
- Time held position
- Exit reason (profit target, stop-loss, trailing stop)
- Buy event screenshot path
- Symbol and trade metadata

### Opportunity Reports
Every 5-minute scan cycle displays:
- Ranked list of all 8 coins with opportunity scores (0-100)
- Signal details for coins with MTF breakout patterns
- Current positions and their profit/loss status
- Available capital for new entries

## Safety & Best Practices

### Start in Simulation Mode
1. **Backfill 210 days** of historical data first (`python3 backfill_coinbase_candles.py --days 210`)
2. **Test the strategy** with `python3 test_mtf_scorer.py` to verify signals appear
3. Set `ready_to_trade: false` for all wallets initially
4. Run `index.py` for several days and monitor:
   - Opportunity scores and rankings
   - Entry/exit signals and reasoning
   - Simulated trade performance
5. Only set `ready_to_trade: true` when confident in the strategy

### Risk Management
- **Never invest more than you can afford to lose**
- Start with 1-2 coins enabled, not all 8
- MTF strategy targets 5-15% swing moves (not scalping)
- Expect 2-5 trades per week across all coins (selective, not frequent)
- Monitor positions daily, especially during first 2 weeks
- ATR-based stops adapt to volatility (typical stop: 3-6%)

### API Security
- Never commit your `.env` file to version control
- Use API keys with trading permissions only (no withdrawal permissions)
- Regularly rotate your API keys
- Monitor your Coinbase account for unauthorized activity

### Strategy Understanding
- The 200-day MA filter keeps you out of downtrends
- BB squeezes are rare (typically 1-3 per month per coin)
- Not all signals score ≥75 (quality over quantity)
- Market rotation means capital shifts to best opportunities
- Trailing stops protect profits during reversals

## Troubleshooting

### Common Issues

**"Insufficient data for 200-MA calculation":**
- **Cause:** Not enough historical candles (need 210 days)
- **Fix:** Run `python3 backfill_coinbase_candles.py --days 210`

**"No opportunities found" or "All scores below 75":**
- **Cause:** No MTF breakout signals currently present
- **Fix:** This is normal and expected. MTF signals are selective (typically 2-5 per week across all 8 coins)
- **Note:** The strategy waits for high-quality setups rather than forcing trades

**Bot not executing trades despite signals:**
- Check that `ready_to_trade: true` for the specific wallet
- Verify `enabled: true` for the wallet
- Ensure fewer than 2 positions currently open (max_concurrent_orders limit)
- Check that you have sufficient balance in your Coinbase account ($2,250+ per position)

**Coinbase API errors:**
- Verify your `COINBASE_API_KEY` and `COINBASE_API_SECRET` in `.env`
- Check that your API key has trading permissions enabled
- Ensure your Coinbase account has sufficient balance

**Email notifications not working:**
- Verify all Mailjet credentials in `.env` (if using email notifications)
- Check your Mailjet account status and sending limits
- Verify email addresses are correct
- Email is optional - bot will continue to work without it

**"Error fetching 5-minute candle":**
- **Cause:** Coinbase API connection issue or rate limiting
- **Fix:** Bot will retry automatically. Check your internet connection and API credentials

## Strategy Customization

### Adjusting Entry Threshold
Modify `min_score_for_entry` in `config.json`:
- **75-80** (default): Conservative, only high-quality setups
- **65-74**: Moderate, more opportunities but lower win rate
- **50-64**: Aggressive, many signals but less selective

**Recommendation:** Start with 75 and only lower if you understand the trade-offs.

### Position Sizing
Current setup: 2 positions × $2,250 = $4,500 total capital

To adjust:
1. Update `capital_per_position` (e.g., $1,000, $5,000)
2. Update `max_concurrent_orders` (1-3 positions)
3. Set `total_trading_capital_usd` = capital_per_position × max_concurrent_orders

**Example (Conservative):**
```json
{
  "market_rotation": {
    "capital_per_position": 1000,
    "max_concurrent_orders": 1,
    "total_trading_capital_usd": 1000
  }
}
```

### Risk/Reward Tuning
Modify in `mtf_strategy` section:
- **`atr_stop_multiplier`**: 1.5 (tighter), 2.0 (default), 2.5 (wider)
- **`target_profit_pct`**: 5.0 (conservative), 7.5 (default), 10.0 (ambitious)
- **`trailing_stop_activation_pct`**: When to start trailing (default: 3.0%)
- **`trailing_stop_distance_pct`**: How far to trail below peak (default: 1.5%)

### Multi-Asset Scanning
The bot scans all `enabled: true` wallets but only trades the best opportunities:
- Enable 8 coins for maximum rotation
- Or focus on 2-3 specific coins
- Market rotation automatically shifts capital to strongest signals

## Dependencies

| Package | Purpose |
|---------|---------|
| `python-dotenv` | Environment variable loading |
| `coinbase-advanced-py` | Coinbase API client |
| `numpy` | Numerical calculations (technical indicators, timeframe aggregation) |
| `matplotlib` | Chart generation and visualization (optional) |
| `mailjet-rest` | Email notification service (optional) |

Install all dependencies with:
```bash
pip install -r requirements.txt
```

**Note:** The bot is fully self-contained with no external API costs beyond Coinbase trading. No AI/ML APIs required.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Disclaimer

**IMPORTANT**: This trading bot is for **educational purposes only**.

- Cryptocurrency trading involves **significant financial risk**
- You should only trade with money you can **afford to lose completely**
- Past performance does not guarantee future results
- The MTF strategy is a mechanical trading system, not financial advice
- No trading strategy is profitable 100% of the time
- Market conditions can change and invalidate historical patterns
- The author is **not responsible** for any financial losses incurred while using this bot
- Always do your own research and consult with financial professionals
- Test thoroughly in simulation mode (minimum 2 weeks) before risking real capital
- **Use at your own risk**

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

## Acknowledgments

- Built with [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs)
- 5-minute candle data from [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs/welcome)
- Technical indicators implementation based on standard TA-Lib formulas
- Email notifications via [Mailjet](https://www.mailjet.com/)
- Inspired by classic momentum breakout strategies (Bollinger Band squeeze, volatility expansion)
