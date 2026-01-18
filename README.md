# Scalp-Scripts: AI-Powered Cryptocurrency Trading Bot

An advanced cryptocurrency scalp trading bot that leverages GPT-4o Vision for intelligent market analysis and automated execution on the Coinbase exchange. The bot combines multi-timeframe technical analysis with AI-driven decision-making to identify and execute profitable short-term trades while managing risk, taxes, and fees.

## Overview

**Scalp-Scripts** automates the entire trading lifecycle:
- Continuously monitors cryptocurrency prices on Coinbase
- Analyzes market conditions using GPT-4o Vision across multiple timeframes
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

### 💰 Professional Risk Management
- **Stop Loss Protection**: Automated exit if price drops below calculated stop loss
- **Profit Target Tracking**: Sells automatically when profit percentage reaches target
- **Tax Calculation**: Factors in federal tax rates on realized capital gains
- **Exchange Fee Accounting**: Calculates Coinbase taker fees on both buy and sell sides
- **Dynamic Position Sizing**: Strategy-driven USD allocation per trade based on:
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
- **Buy Event Documentation**: Captures screenshots at time of purchase for later analysis
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
- **Historical Data Backfilling**: Coinbase Advanced API integration for populating 5-minute candle data
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

# Email Notifications (Mailjet)
MAILJET_API_KEY=your_mailjet_api_key
MAILJET_SECRET_KEY=your_mailjet_secret_key
MAILJET_FROM_EMAIL=bot@example.com
MAILJET_FROM_NAME=Trading Bot
MAILJET_TO_EMAIL=your@email.com
MAILJET_TO_NAME=Your Name

# OpenAI API (for GPT-4o Vision)
OPENAI_API_KEY=sk-your_openai_api_key_here
```

### Getting Your API Keys

- **Coinbase API**: https://www.coinbase.com/settings/api
- **OpenAI API**: https://platform.openai.com/api-keys
- **Mailjet API**: https://app.mailjet.com/account/api_keys

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
- **`no_trade_refresh_hours`**: Hours to wait before re-analyzing an asset after analysis determines no trade should be made. Default: `5`
- **`cooldown_hours_after_sell`**: Hours to pause trading after selling a position, allowing the market to settle. Default: `6`
- **`low_confidence_wait_hours`**: Hours to wait before re-analyzing after a low-confidence recommendation. Default: `3`
- **`medium_confidence_wait_hours`**: Hours to wait before re-analyzing after a medium-confidence recommendation. Default: `2`

#### Data Management Settings

```json
{
  "data_retention": {
    "max_hours": 4380,
    "interval_seconds": 3600
  }
}
```

- **`max_hours`**: Maximum hours of price data to retain (4,380 hours = ~6 months). Default: `4380`
- **`interval_seconds`**: How often to save price snapshots in seconds (3,600 seconds = 1 hour). Default: `3600`

#### Wallet/Asset Configuration

```json
{
  "wallets": [
    {
      "title": "BTC: primary, most predictable",
      "symbol": "BTC-USD",
      "enabled": true,
      "ready_to_trade": false,
      "starting_capital_usd": 100,
      "enable_chart_snapshot": false
    }
  ]
}
```

Each asset in the `wallets` array supports the following parameters:

- **`title`**: Friendly description for the asset (for reference only)
- **`symbol`**: The Coinbase trading pair symbol (e.g., `"BTC-USD"`, `"ETH-USD"`, `"SOL-USD"`)
- **`enabled`**: Whether to monitor this asset (`true`) or skip it (`false`)
- **`ready_to_trade`**: Whether to execute actual trades (`true`) or run in simulation mode (`false`)
  - **IMPORTANT**: Start with `false` to test the bot in simulation mode before risking real money
- **`starting_capital_usd`**: Initial capital allocated for this asset (used for performance metrics calculation)
- **`enable_chart_snapshot`**: Whether to generate chart images for this asset (`true`/`false`)

## Historical Data Backfilling

Before the bot has accumulated enough price history naturally, you can backfill historical 5-minute candle data from Coinbase Advanced API. This provides high-quality data for technical analysis and strategy backtesting from day one.

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

**5-minute candles** - Perfect for momentum and scalping strategies:

| Time Period | Candles | Storage Size |
|-------------|---------|--------------|
| **1 day** | 288 | ~50 KB |
| **1 week** | 2,016 | ~350 KB |
| **1 month** | 8,640 | ~1.5 MB |
| **3 months** | 25,920 | ~4.5 MB |

**Benefits over hourly data:**
- 12x more granular (5-min vs 60-min)
- See intra-hour price swings and momentum shifts
- More accurate entry/exit timing
- Better correlation calculations for divergence strategies

### Running the Backfill

**Backfill all enabled assets (90 days recommended):**
```bash
python3 backfill_coinbase_candles.py --days 90
```

**Backfill specific asset:**
```bash
python3 backfill_coinbase_candles.py BTC-USD --days 30
```

**Custom time period:**
```bash
python3 backfill_coinbase_candles.py --days 7   # 1 week
python3 backfill_coinbase_candles.py --days 180 # 6 months
```

### Example Output

```
======================================================================
Coinbase Advanced API - Historical 5-Minute Candle Backfill
======================================================================
✓ Coinbase client initialized

📅 Backfill period: 90 days
📊 Granularity: 5-minute candles
📈 Expected candles per asset: ~25920

Found 8 enabled wallet(s) to backfill:
  - BTC-USD
  - ETH-USD
  - SOL-USD
  ...

=== Backfilling 5-minute candles for BTC-USD ===
  Requesting 90 days of data
  From: 2025-10-19 to 2026-01-17

  Chunk 1: 2025-10-19 23:38 to 2025-10-21 00:38 (25.0 hours)
  Fetching candles from 2025-10-19 23:38:12+00:00 to 2025-10-21 00:38:12+00:00...
  ✓ Fetched 300 candles
  ✓ Transformed 300 candles
  ...

  Total candles fetched: 25848
  Found 0 existing data points
  ✓ Added 25848 new data points, total: 25848
✓ Saved 25848 data points to coinbase-data/BTC-USD.json

  ✓ Backfill complete for BTC-USD
  📊 Data coverage: 25848/25920 candles (99.7%)
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

- **Start with 7-30 days** when first testing to verify everything works
- **Use 90 days for production** - provides 3 months of market history for robust strategy testing
- **Re-run if needed** - the script automatically deduplicates, safe to run multiple times
- **Run before going live** - populate historical data before enabling `ready_to_trade`

### Integration with index.py

Once backfilled, `index.py` seamlessly continues appending new 5-minute candles:

```python
# index.py runs every 5 minutes (config.json: interval_seconds=300)
# Fetches latest candle and appends to same coinbase-data/*.json files
# Auto-deduplicates to prevent overlaps
```

**Result:** Continuous data collection from historical backfill through live trading

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
   - Generates multi-timeframe charts (24-hour, 7-day, 90-day)
   - Calls GPT-4o Vision API with:
     - Chart images (high detail for short-term, lower detail for medium/long-term)
     - Current price/volume data
     - Cost structure (fees, taxes)
   - Receives JSON analysis with: buy price, sell price, confidence level, trade recommendation

4. **Trading Logic**

   **If no position (last order = 'none' or 'sell')**:
   - If analysis recommends `buy` AND confidence is `high` AND market trend is `bullish`:
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

The bot uses a **Technical Analysis + Strategy-Based Decision-Making** approach:

#### Entry Signals
- Price reaches or falls below calculated buy price
- RSI indicates oversold conditions
- Price touches lower Bollinger Band
- Market trend is bullish (confirmed across multiple timeframes)
- Confidence level is HIGH
- Support level identified by multi-timeframe analysis

#### Exit Signals (Take Profit)
- Profit percentage reaches or exceeds target (after fees and taxes)
- Price reaches calculated sell price
- Resistance level reached

#### Exit Signals (Stop Loss)
- Price falls below calculated stop loss
- Immediate exit to prevent further losses

#### Market Filtering
- No buy signals in bearish markets
- No trades during low/medium confidence periods
- Respects cooldown periods after sells
- Position sizing limited to prevent over-allocation

## File Structure

```
/scalp-scripts/
├── index.py                              # Main entry point - core trading loop
├── backfill_coinbase_candles.py          # Coinbase 5-minute candle backfill script
├── config.json                           # Active configuration (user-specific)
├── example-config.json                   # Template configuration
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── .env                                  # Environment variables (create from .env.sample)
│
├── utils/                                # Utility modules
│   ├── coinbase.py                       # Coinbase API wrapper & order management
│   ├── openai_analysis.py                # GPT-4o integration & market analysis
│   ├── wallet_helpers.py                 # Wallet metrics & transaction history helpers
│   ├── matplotlib.py                     # Chart generation & visualization
│   ├── price_helpers.py                  # Price calculation utilities
│   ├── file_helpers.py                   # File I/O & data management
│   ├── time_helpers.py                   # Time formatting utilities
│   ├── email.py                          # Email notifications (Mailjet)
│   └── new_coinbase_listings.py          # New listing detection
│
├── docs/                                 # Documentation files
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
- Confidence levels and trade recommendations
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
Current market analysis is cached in `analysis/{SYMBOL}_analysis.json` files, showing:
- Buy-in price and sell price recommendations
- Support and resistance levels
- Confidence level and market trend
- Trade reasoning
- Profit target percentage

## Safety & Best Practices

### Start in Simulation Mode
1. Set `ready_to_trade: false` for all wallets initially
2. Monitor the console output and email notifications
3. Review the analysis files to understand the bot's decision-making
4. Only set `ready_to_trade: true` when you're confident in the bot's behavior

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
- Run `python3 backfill_coinbase_candles.py --days 90` first to populate 5-minute candle data
- Verify your Coinbase API credentials are set in `.env`

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

### Custom Profit Targets
Adjust the global `min_profit_target_percentage` or let the system calculate dynamic targets based on:
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
| `requests` | HTTP requests |
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
- The bot's recommendations are not financial advice
- The author is **not responsible** for any financial losses incurred while using this bot
- Always do your own research and consult with financial professionals
- Test thoroughly in simulation mode before risking real capital
- **Use at your own risk**

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

## Acknowledgments

- Built with [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs)
- Market analysis powered by [OpenAI GPT-4o Vision](https://openai.com/index/gpt-4o-vision/)
- Historical 5-minute candle data from [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs/welcome)
- Email notifications via [Mailjet](https://www.mailjet.com/)
