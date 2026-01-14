o

# Scalp-Scripts: AI-Powered Cryptocurrency Trading Bot

An advanced cryptocurrency scalp trading bot that leverages GPT-4o Vision for intelligent market analysis, self-learning from historical trades, and automated execution on the Coinbase exchange. The bot combines multi-timeframe technical analysis with AI-driven decision-making to identify and execute profitable short-term trades while managing risk, taxes, and fees.

## Overview

**Scalp-Scripts** automates the entire trading lifecycle:
- Continuously monitors cryptocurrency prices on Coinbase
- Analyzes market conditions using GPT-4o Vision across multiple timeframes
- Learns from its own trading history to improve decision-making
- Executes buy/sell orders automatically when conditions are favorable
- Manages risk with stop-loss triggers and profit targets
- Tracks all costs: exchange fees, taxes, net profits
- Sends email notifications for all trading activity

## Key Features

### 🤖 AI-Powered Market Analysis (GPT-4o Vision)
- **Multi-Timeframe Chart Analysis**: Generates and analyzes charts across five timeframes
  - **72-hour view**: High-detail analysis for precise entry/exit timing with recent context
  - **7-day view**: Confirms short-term trend momentum and direction
  - **30-day view**: Recent trend analysis and medium-term momentum
  - **90-day view**: Extended trend analysis and quarterly patterns
  - **6-month view**: Provides macro trend context and long-term support/resistance
- **Visual + Numerical Analysis**: Combines chart images with price data for holistic market understanding
- **Technical Indicators**: Calculates and analyzes RSI, Bollinger Bands, Moving Averages, Support/Resistance levels
- **Confidence Scoring**: Provides high/medium/low confidence levels for each trade recommendation
- **Trend Classification**: Identifies bullish, bearish, or sideways market conditions

### 📚 Self-Learning System (LLM Experience-Based Learning)
- **Historical Context Injection**: Provides the AI with past trade performance for symbol-specific learning
- **Trade Screenshot Archive**: Stores visual snapshots at each buy event for pattern recognition
- **Performance Metrics Tracking**:
  - Win rate calculation (percentage of profitable trades)
  - Average profit per trade
  - Cumulative profit tracking
  - Gross vs. net profit analysis (after taxes and fees)
- **Pattern Recognition**: The AI learns:
  - Which entry conditions historically worked best for each asset
  - Optimal hold times for specific cryptocurrencies
  - Support/resistance levels that have been tested
  - Market conditions that preceded losses
- **Automatic Trade Pruning**: Manages context size by keeping only the N most recent trades
- **Continuous Improvement**: Each completed trade adds to the learning dataset

For detailed information about the LLM learning system, see [LLM Experience-Based Learning Guide](docs/LLM_EXPERIENCE_BASED_LEARNING.md).

### 💰 Professional Risk Management
- **Stop Loss Protection**: Automated exit if price drops below calculated stop loss
- **Profit Target Tracking**: Sells automatically when profit percentage reaches target
- **Tax Calculation**: Factors in federal tax rates on realized capital gains
- **Exchange Fee Accounting**: Calculates Coinbase taker fees on both buy and sell sides
- **Dynamic Position Sizing**: AI-driven USD allocation per trade based on:
  - Confidence level of the analysis
  - Current wallet metrics
  - Trade quality assessment
  - Risk discipline (never commits >75% of available capital)
- **Wallet Metrics Dashboard**: Tracks starting capital, current value, percentage gain/loss, gross vs. net profit

### 📊 Comprehensive Order Management
- **Local Ledger Tracking**: Maintains per-symbol JSON files tracking complete order history
- **Order State Detection**:
  - `none` - No active position
  - `buy` - Currently holding position
  - `sell` - Just exited position
  - `placeholder` - Pending order confirmation
- **Buy Event Documentation**: Captures screenshots at time of purchase for later AI analysis
- **Complete Transaction Records**: Stores detailed data for every trade:
  - Entry/exit prices and timestamps
  - Time held position
  - Profit/loss calculations
  - Tax and fee amounts
  - Buy event screenshot paths

### 📈 Data Collection & Analysis
- **Continuous Price Data Collection**: Appends price data to per-crypto JSON files at configurable intervals (default: 1-hour intervals)
- **Data Retention Policies**: Automatically cleans up old data beyond configured retention window (default: 4,380 hours / ~6 months)
- **Multi-Property Tracking**:
  - Current price
  - 24-hour volume
  - Price percentage change (24h)
  - Volume percentage change (24h)
- **Historical Data Backfilling**: CoinGecko integration for populating historical data before bot deployment
- **Smart Data Merging**: Avoids duplicates when merging backfilled and real-time data

### 🔔 Notifications & Monitoring
- **Email Notifications** (via Mailjet):
  - Buy order placement alerts
  - Sell order execution alerts with profit percentage
  - Error/crash notifications
  - Graceful degradation (continues after repeated errors)
- **Console Logging**: Real-time status updates including current analysis stage, prices, confidence levels, market trends
- **Data Visualization**: Generates chart snapshots for current price action and multi-timeframe context

### 🆕 New Listing Detection (Optional)
- Monitors Coinbase for newly listed trading pairs
- Sends email alerts when new cryptocurrencies are listed
- Tracks listing history for future reference

## Prerequisites

- **Python 3.7+**
- **Coinbase Account** with API credentials
- **OpenAI API Key** (for GPT-4o Vision access)
- **Mailjet Account** (for email notifications)
- **CoinGecko API Key** (optional, for historical data backfilling)

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

The bot will run continuously, monitoring your specified assets and executing trades based on AI analysis.

## Environment Setup

Create a `.env` file in the root directory with the following variables. You can copy `.env.sample` if available:

```bash
# Coinbase API Credentials
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here

# Tax Configuration
FEDERAL_TAX_RATE=15                # Your federal tax rate on capital gains (e.g., 15 for 15%)

# Email Notifications (Mailjet)
MAILJET_API_KEY=your_mailjet_api_key
MAILJET_SECRET_KEY=your_mailjet_secret_key
MAILJET_FROM_EMAIL=bot@example.com
MAILJET_FROM_NAME=Trading Bot
MAILJET_TO_EMAIL=your@email.com
MAILJET_TO_NAME=Your Name

# OpenAI API (for GPT-4o Vision)
OPENAI_API_KEY=sk-your_openai_api_key_here

# CoinGecko API (optional, for historical data backfilling)
COINGECKO_API_KEY=your_coingecko_api_key
```

### Getting Your API Keys

- **Coinbase API**: https://www.coinbase.com/settings/api
- **OpenAI API**: https://platform.openai.com/api-keys
- **Mailjet API**: https://app.mailjet.com/account/api_keys
- **CoinGecko API**: https://www.coingecko.com/en/developers/dashboard

## Configuration

### Basic Configuration

Copy `example-config.json` to `config.json` and customize your settings:

```bash
cp example-config.json config.json
```

### Configuration Parameters

#### Global Settings

```json
{
  "min_profit_target_percentage": 3.8,
  "no_trade_refresh_hours": 5,
  "cooldown_hours_after_sell": 6,
  "low_confidence_wait_hours": 3,
  "medium_confidence_wait_hours": 2,
  ...
}
```

- **`min_profit_target_percentage`**: Minimum net profit percentage threshold to trigger a sell signal (after fees and taxes). Default: `3.8`
- **`no_trade_refresh_hours`**: Hours to wait before re-analyzing an asset after the AI determines no trade should be made. Default: `5`
- **`cooldown_hours_after_sell`**: Hours to pause trading after selling a position, allowing the market to settle. Default: `6`
- **`low_confidence_wait_hours`**: Hours to wait before re-analyzing after a low-confidence recommendation. Default: `3`
- **`medium_confidence_wait_hours`**: Hours to wait before re-analyzing after a medium-confidence recommendation. Default: `2`

#### LLM Learning Settings

```json
{
  "llm_learning": {
    "enabled": true,
    "max_historical_trades": 5,
    "include_screenshots": true,
    "prune_old_trades_after": 100
  }
}
```

- **`enabled`**: Enable/disable the LLM learning system. When `true`, the AI learns from past trades. Default: `true`
- **`max_historical_trades`**: Number of recent trades to include in the AI's context for learning. Default: `5`
- **`include_screenshots`**: Whether to send buy event screenshots to the Vision API for visual pattern recognition. Default: `true`
- **`prune_old_trades_after`**: Automatically prune old trades when total trade count exceeds this number. Default: `100`

#### Data Management Settings

```json
{
  "coingecko": {
    "enable_backfilling_historical_data": true
  },
  "data_retention": {
    "max_hours": 4380,
    "interval_seconds": 3600
  }
}
```

- **`enable_backfilling_historical_data`**: Whether to enable historical data backfilling from CoinGecko. Default: `true`
- **`max_hours`**: Maximum hours of price data to retain (4,380 hours = ~6 months). Default: `4380`
- **`interval_seconds`**: How often to save price snapshots in seconds (3,600 seconds = 1 hour). Default: `3600`

#### Wallet/Asset Configuration

```json
{
  "wallets": [
    {
      "title": "BTC: primary, most predictable",
      "symbol": "BTC-USD",
      "coingecko_id": "bitcoin",
      "enabled": true,
      "ready_to_trade": false,
      "starting_capital_usd": 100,
      "enable_ai_analysis": false,
      "enable_chart_snapshot": false
    }
  ]
}
```

Each asset in the `wallets` array supports the following parameters:

- **`title`**: Friendly description for the asset (for reference only)
- **`symbol`**: The Coinbase trading pair symbol (e.g., `"BTC-USD"`, `"ETH-USD"`, `"SOL-USD"`)
- **`coingecko_id`**: The CoinGecko ID for this asset (e.g., `"bitcoin"`, `"ethereum"`, `"solana"`) - used for historical data backfilling
- **`enabled`**: Whether to monitor this asset (`true`) or skip it (`false`)
- **`ready_to_trade`**: Whether to execute actual trades (`true`) or run in simulation mode (`false`)
  - **IMPORTANT**: Start with `false` to test the bot in simulation mode before risking real money
- **`starting_capital_usd`**: Initial capital allocated for this asset (used for performance metrics calculation)
- **`enable_ai_analysis`**: Whether to use AI analysis for this asset (`true`) or skip analysis (`false`)
- **`enable_chart_snapshot`**: Whether to generate chart images for this asset (`true`/`false`)

### Finding CoinGecko IDs

Common CoinGecko IDs:
- Bitcoin (BTC): `bitcoin`
- Ethereum (ETH): `ethereum`
- Cardano (ADA): `cardano`
- Solana (SOL): `solana`
- Polygon (MATIC): `matic-network`

To find other IDs:
- Visit https://api.coingecko.com/api/v3/coins/list
- Or visit the CoinGecko website and check the URL (e.g., `coingecko.com/en/coins/bitcoin` → ID is `"bitcoin"`)

## Historical Data Backfilling

Before the bot has accumulated enough price history naturally, you can backfill historical data from CoinGecko. This is useful for enabling technical analysis and AI decision-making from day one.

### How the Backfilling Function Works

The backfilling script (`backfill_historical_data.py`) performs the following operations:

1. **Data Fetching**:
   - Queries the CoinGecko API for historical market data (prices and volumes)
   - Uses the `/market_chart/range` endpoint to fetch data within a specific time range
   - Automatically chunks large requests into 90-day periods to ensure optimal granularity

2. **Data Transformation**:
   - Converts CoinGecko's data format to match Coinbase's data structure
   - Calculates 24-hour percentage changes for both price and volume
   - Generates data points with: `timestamp`, `product_id`, `price`, `price_percentage_change_24h`, `volume_24h`, `volume_percentage_change_24h`

3. **Smart Merging**:
   - Loads any existing data from `coinbase-data/{SYMBOL}.json`
   - Merges new historical data with existing data using timestamp-based deduplication
   - Sorts all data points chronologically by timestamp
   - Only adds new data points that don't already exist (no overwrites)

4. **Rate Limiting**:
   - Respects CoinGecko API rate limits (30 calls/min for free tier)
   - Adds 2-second delays between API requests to prevent throttling
   - Displays progress for multi-chunk requests

### Data Intervals and Granularity

The data granularity you receive depends on the time period requested, following CoinGecko's automatic interval rules:

| Time Period | Data Interval | Data Points |
|-------------|---------------|-------------|
| **1 day or less** | 5-minute intervals | ~288 per day |
| **1-90 days** | Hourly intervals | ~24 per day |
| **Above 90 days** | Daily intervals | ~1 per day |

**Important**: The backfill script automatically **chunks requests into 90-day periods** to ensure you get **hourly granularity** instead of daily, even for longer time periods. For example:
- Requesting 180 days (6 months) = 2 API requests of 90 days each
- Requesting 365 days (1 year) = 5 API requests of ~73 days each
- Result: ~4,320 hourly data points for 6 months, instead of just ~180 daily points

### How Far Back Can You Backfill?

The backfill time range is controlled by the `data_retention.max_hours` setting in `config.json`:

```json
{
  "data_retention": {
    "max_hours": 4380,        // 4,380 hours = ~6 months (182.5 days)
    "interval_seconds": 3600  // Not used for backfilling (only for live data collection)
  }
}
```

**Limitations**:
- **CoinGecko Free Tier**: Maximum 365 days (1 year) of historical data
- **Bot Default**: 4,380 hours (~6 months / 182.5 days)
- If you request more than 365 days, the script automatically limits to 365 days

**Examples**:
- `"max_hours": 720` → 30 days of data (~720 hourly data points)
- `"max_hours": 2160` → 90 days of data (~2,160 hourly data points)
- `"max_hours": 4380` → 182.5 days of data (~4,380 hourly data points)
- `"max_hours": 8760` → Requested 365 days, limited to 365 days (~8,760 hourly data points)

### Data Structure

Each backfilled data point matches the Coinbase format and includes:

```json
{
  "timestamp": 1697385600.0,           // Unix timestamp (seconds)
  "product_id": "BTC-USD",             // Trading pair symbol
  "price": "28453.21",                 // Current price in USD
  "price_percentage_change_24h": "2.34",  // Percentage change from 24h ago
  "volume_24h": "15234567890.12",      // 24-hour trading volume in USD
  "volume_percentage_change_24h": "-1.23" // Percentage change in volume from 24h ago
}
```

**Percentage Change Calculations**:
- **24-hour lookback**: Compares current value with value from 96 data points ago (96 × 15-min intervals = 24 hours)
- First 24 hours of data will have `"0"` for percentage changes (insufficient historical data)
- Calculated for both price and volume metrics

### Setup

1. Get a CoinGecko API key from https://www.coingecko.com/en/developers/dashboard

2. Add your API key to the `.env` file:
   ```bash
   COINGECKO_API_KEY=your_api_key_here
   ```

3. Enable backfilling in `config.json`:
   ```json
   {
     "coingecko": {
       "enable_backfilling_historical_data": true
     },
     "data_retention": {
       "max_hours": 4380,
       "interval_seconds": 3600
     }
   }
   ```

4. Ensure each asset has a `coingecko_id` in the config:
   ```json
   {
     "wallets": [
       {
         "symbol": "BTC-USD",
         "coingecko_id": "bitcoin",
         "enabled": true
       }
     ]
   }
   ```

### Running the Backfill

Run the backfill script to fetch historical data:

```bash
python3 backfill_historical_data.py
```

The script will display progress information:
```
============================================================
CoinGecko Historical Data Backfill
============================================================

✓ Backfilling is ENABLED

Found 2 enabled asset(s) to backfill:
  - BTC-USD
  - ETH-USD

=== Backfilling data for BTC-USD ===
  Requesting data from 2025-04-01 to 2025-10-16
  Total period: 182 days
  Splitting into 90-day chunks to get hourly granularity (instead of daily)
  Will make 3 API requests
  Chunk 1: 2025-04-01 to 2025-06-30 (90 days)
  Fetching data from 2025-04-01 00:00:00+00:00 to 2025-06-30 23:59:59+00:00...
  ✓ Fetched 2160 data points
  ...
  ✓ Added 4368 new data points, total: 4368
✓ Saved 4368 data points to coinbase-data/BTC-USD.json
  ✓ Backfill complete for BTC-USD
```

### Best Practices

- **Run backfill before starting the bot** for the first time to populate historical data immediately
- **Re-run periodically** if you've had the bot stopped for extended periods to fill gaps
- **Don't worry about duplicates** - the merge function automatically deduplicates by timestamp
- **Start with smaller time periods** (e.g., 30-90 days) when testing to reduce API calls
- **Monitor API rate limits** - the script includes delays, but be mindful of CoinGecko's 30 calls/min limit on free tier

### Troubleshooting

**Script says backfilling is disabled:**
- Check `config.json` and ensure `"enable_backfilling_historical_data": true`

**CoinGecko API key errors:**
- Verify your API key is correctly added to `.env`
- Check that your API key is active at https://www.coingecko.com/en/developers/dashboard

**Rate limit errors:**
- The script includes 2-second delays between requests
- If you still hit limits, the free tier allows 30 calls/min
- Consider upgrading to a paid CoinGecko plan for higher limits

**Missing coingecko_id:**
- Each enabled asset needs a `coingecko_id` field in `config.json`
- Find IDs at https://api.coingecko.com/api/v3/coins/list or on CoinGecko.com URLs

## How It Works

### Main Trading Loop

The bot runs continuously in a loop (default: every 15 minutes):

1. **Data Collection**
   - Fetches current prices from Coinbase API
   - Appends price data to per-symbol JSON files in `coinbase-data/`
   - Calculates 24-hour volume changes

2. **Order State Management**
   - Checks the last order status for each symbol (`none`, `buy`, `sell`, or `placeholder`)
   - If `placeholder` (pending), polls Coinbase for final order status

3. **Market Analysis** (when needed)
   - Loads historical trading context (last N trades for learning)
   - Generates multi-timeframe charts (24-hour, 7-day, 90-day)
   - Calls GPT-4o Vision API with:
     - Chart images (high detail for short-term, lower detail for medium/long-term)
     - Historical trade screenshots (if LLM learning is enabled)
     - Current price/volume data
     - Cost structure (fees, taxes)
   - Receives JSON analysis with: buy price, sell price, confidence level, trade recommendation

4. **Trading Logic**

   **If no position (last order = 'none' or 'sell')**:
   - If AI recommends `buy` AND confidence is `high` AND market trend is `bullish`:
     - If current price ≤ buy-in price:
       - Calculate shares to buy (USD amount ÷ current price, rounded down)
       - Place market buy order on Coinbase
       - Take screenshot of chart at buy moment
       - Save order to ledger

   **If holding position (last order = 'buy')**:
   - Calculate current profit including all costs
   - If current price ≤ stop loss price:
     - Place market sell order (Stop Loss trigger)
   - Else if profit percentage ≥ target percentage:
     - Place market sell order (Take Profit trigger)

   **After sell**:
   - Save complete transaction record with all metrics to `transactions/data.json`
   - Delete analysis file to trigger fresh analysis on next cycle
   - Start cooldown period (no trading for configured hours)

5. **Data Cleanup**
   - Remove price data older than retention window
   - Optionally prune old transaction records

### Trading Strategy

The bot uses a **Technical Analysis + AI Decision-Making** strategy:

#### Entry Signals
- Price reaches or falls below AI-calculated buy price
- RSI indicates oversold conditions
- Price touches lower Bollinger Band
- Market trend is bullish (confirmed across multiple timeframes)
- AI confidence level is HIGH
- Support level identified by multi-timeframe analysis

#### Exit Signals (Take Profit)
- Profit percentage reaches or exceeds target (after fees and taxes)
- Price reaches AI-calculated sell price
- Resistance level reached

#### Exit Signals (Stop Loss)
- Price falls below AI-calculated stop loss
- Immediate exit to prevent further losses

#### Market Filtering
- No buy signals in bearish markets
- No trades during low/medium confidence periods
- Respects cooldown periods after sells
- Position sizing limited to prevent over-allocation

### Learning Component

The LLM analyzes past winning and losing trades to identify:
- Entry conditions that historically worked (RSI levels, Bollinger Band positions, price actions)
- Optimal hold times for each specific asset
- Probability-weighted profit targets
- Risk factors observed in losing trades
- Support/resistance levels that have been repeatedly tested

Each completed trade adds to the learning dataset, improving future decision-making.

## File Structure

```
/scalp-scripts/
├── index.py                              # Main entry point - core trading loop
├── backfill_historical_data.py           # CoinGecko historical data importer
├── config.json                           # Active configuration (user-specific)
├── example-config.json                   # Template configuration
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── .env                                  # Environment variables (create from .env.sample)
│
├── utils/                                # Utility modules
│   ├── coinbase.py                       # Coinbase API wrapper & order management
│   ├── openai_analysis.py                # GPT-4o integration & market analysis
│   ├── trade_context.py                  # LLM learning & historical context builder
│   ├── matplotlib.py                     # Chart generation & visualization
│   ├── price_helpers.py                  # Price calculation utilities
│   ├── file_helpers.py                   # File I/O & data management
│   ├── time_helpers.py                   # Time formatting utilities
│   ├── email.py                          # Email notifications (Mailjet)
│   └── new_coinbase_listings.py          # New listing detection
│
├── docs/
│   └── LLM_EXPERIENCE_BASED_LEARNING.md  # Detailed LLM learning documentation
│
├── coinbase-data/                        # Runtime: Price history per symbol
│   ├── BTC-USD.json                      # Historical price snapshots
│   └── ETH-USD.json                      # (auto-generated)
│
├── analysis/                             # Runtime: Current analysis files
│   ├── BTC_USD_analysis.json             # Current buy/sell recommendations
│   └── ETH_USD_analysis.json             # (auto-generated)
│
├── transactions/                         # Runtime: Trade history
│   └── data.json                         # All completed trades with metrics
│
├── screenshots/                          # Runtime: Buy event visual records
│   ├── BTC-USD_chart_buy_20251015120000.png
│   └── (auto-generated)
│
├── coinbase-orders/                      # Per-symbol order ledgers (auto-generated)
│   ├── BTC-USD_orders.json
│   ├── ETH-USD_orders.json
│   └── {SYMBOL}_orders.json
│
└── venv/                                 # Python virtual environment (if using Option 2)
```

## Monitoring & Logs

### Console Output
The bot provides real-time status updates in the console:
- Current prices and volume changes
- Analysis stage (short-term, medium-term, long-term)
- AI confidence levels and trade recommendations
- Order placement and execution
- Profit/loss calculations
- Error messages and warnings

### Email Notifications
You'll receive email alerts for:
- Buy order placements (with entry price and confidence level)
- Sell order executions (with profit percentage and hold time)
- Error conditions or crashes
- New Coinbase listings (if enabled)

### Transaction History
All completed trades are logged in `transactions/data.json` with complete details:
- Entry/exit prices and timestamps
- Gross profit, taxes, fees, and net profit
- Time held position
- Buy event screenshot path
- Symbol and trade metadata

### Analysis Files
Current AI analysis is cached in `analysis/{SYMBOL}_analysis.json` files, showing:
- Buy-in price and sell price recommendations
- Support and resistance levels
- Confidence level and market trend
- Trade reasoning
- Profit target percentage

## Safety & Best Practices

### Start in Simulation Mode
1. Set `ready_to_trade: false` for all wallets initially
2. Set `enable_ai_analysis: true` to see what the AI would recommend
3. Monitor the console output and email notifications
4. Review the analysis files to understand the AI's decision-making
5. Only set `ready_to_trade: true` when you're confident in the bot's behavior

### Risk Management
- **Never invest more than you can afford to lose**
- Start with small `starting_capital_usd` amounts ($50-100)
- Set conservative `min_profit_target_percentage` (3-5%)
- Monitor the bot regularly, especially in the first few days
- Keep the `cooldown_hours_after_sell` setting to prevent over-trading

### API Security
- Never commit your `.env` file to version control
- Use API keys with trading permissions only (no withdrawal permissions)
- Regularly rotate your API keys
- Monitor your Coinbase account for unauthorized activity

### Backtesting
- Use the historical data backfilling feature to test strategies
- Review past transaction records to understand win/loss patterns
- Adjust configuration based on observed performance

## Troubleshooting

### Common Issues

**Bot not executing trades despite recommendations:**
- Check that `ready_to_trade: true` for the specific wallet
- Verify `enabled: true` for the wallet
- Ensure `enable_ai_analysis: true` to get recommendations
- Check that you have sufficient balance in your Coinbase account

**OpenAI API errors:**
- Verify your `OPENAI_API_KEY` in `.env` is correct
- Check that you have GPT-4o Vision access on your OpenAI account
- Monitor your OpenAI API usage limits and billing

**Coinbase API errors:**
- Verify your `COINBASE_API_KEY` and `COINBASE_API_SECRET` in `.env`
- Check that your API key has trading permissions
- Ensure your Coinbase account has sufficient balance

**Email notifications not working:**
- Verify all Mailjet credentials in `.env`
- Check your Mailjet account status and sending limits
- Verify email addresses are correct

**Missing historical data:**
- Run `python3 backfill_historical_data.py` first
- Verify `COINGECKO_API_KEY` in `.env`
- Check `enable_backfilling_historical_data: true` in config
- Ensure each wallet has a valid `coingecko_id`

## Advanced Features

### Multiple Asset Trading
Configure multiple wallets in `config.json` to trade different cryptocurrencies simultaneously:
```json
{
  "wallets": [
    {
      "symbol": "BTC-USD",
      "enabled": true,
      "ready_to_trade": true,
      ...
    },
    {
      "symbol": "ETH-USD",
      "enabled": true,
      "ready_to_trade": true,
      ...
    }
  ]
}
```

Each asset maintains independent:
- Order ledger
- Transaction history
- Price data
- Analysis cache
- Learning context

### Custom Profit Targets
Adjust the global `min_profit_target_percentage` or let the AI calculate dynamic targets based on:
- Market volatility
- Historical performance
- Current confidence level
- Risk assessment

### Cooldown Periods
Configure different wait times based on analysis outcomes:
- `cooldown_hours_after_sell`: Pause after selling to let market settle
- `low_confidence_wait_hours`: Don't waste API calls on low-confidence signals
- `medium_confidence_wait_hours`: Brief pause before re-analyzing medium confidence
- `no_trade_refresh_hours`: How long to wait after "no trade" recommendation

## Dependencies

| Package | Purpose |
|---------|---------|
| `python-dotenv` | Environment variable loading |
| `coinbase-advanced-py` | Coinbase API client |
| `openai` | GPT-4o Vision API access |
| `requests` | HTTP requests (CoinGecko API) |
| `numpy` | Numerical calculations (technical indicators) |
| `matplotlib` | Chart generation and visualization |
| `mailjet-rest` | Email notification service |

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Disclaimer

**IMPORTANT**: This trading bot is for **educational purposes only**.

- Cryptocurrency trading involves **significant financial risk**
- You should only trade with money you can **afford to lose completely**
- Past performance does not guarantee future results
- The AI's recommendations are not financial advice
- The author is **not responsible** for any financial losses incurred while using this bot
- Always do your own research and consult with financial professionals
- Test thoroughly in simulation mode before risking real capital
- **Use at your own risk**

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

## Acknowledgments

- Built with [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs)
- AI analysis powered by [OpenAI GPT-4o Vision](https://openai.com/index/gpt-4o-vision/)
- Historical data from [CoinGecko API](https://www.coingecko.com/en/api)
- Email notifications via [Mailjet](https://www.mailjet.com/)
