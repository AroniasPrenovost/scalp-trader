# Coinbase Trading Bot

This project is a cryptocurrency trading bot that interacts with the Coinbase API to automate trading based on predefined strategies. The bot is written in Python and uses the Coinbase REST API to fetch market data, place orders, and manage positions.

## Features

- Fetch current asset prices from Coinbase.
- Manage asset positions and account balances.
- Place market orders for buying and selling assets.
- Calculate potential profits, exchange fees, and taxes.
- Monitor market conditions and trigger buy/sell actions based on technical indicators such as SMA, MACD, and Bollinger Bands.
- Send email notifications for order placements using Mailjet.
- Backfill historical price data from CoinGecko API for analysis (15-minute intervals).

## Prerequisites

- Python 3.7+
- Coinbase API Key and Secret
- Mailjet API Key and Secret
- A `.env` file with the following variables:
  - `COINBASE_API_KEY`
  - `COINBASE_API_SECRET`
  - `COINBASE_SPOT_MAKER_FEE`
  - `COINBASE_SPOT_TAKER_FEE`
  - `FEDERAL_TAX_RATE`
  - `MAILJET_API_KEY`
  - `MAILJET_SECRET_KEY`
  - `MAILJET_FROM_EMAIL`
  - `MAILJET_FROM_NAME`
  - `MAILJET_TO_EMAIL`
  - `MAILJET_TO_NAME`
  - `COINGECKO_API_KEY` (optional, only needed for historical data backfilling)

## Installation (1)

1. Clone the repository:

   ```bash
   git clone https://github.com/AroniasPrenovost/scalp-trader/tree/main && cd coinbase-trading-bot
   ```

2. Install the required packages:

   ```bash
   pip install requirements.txt
   ```

3. Copy the `.env.sample` file and create an `.env` file in the root directory. Add your Coinbase and Mailjet API credentials, along with other tax and fee configurations to the `.env` file:

   ```bash
   COINBASE_API_KEY=your_api_key
   COINBASE_API_SECRET=your_api_secret
   COINBASE_SPOT_MAKER_FEE=0.5
   COINBASE_SPOT_TAKER_FEE=0.5
   FEDERAL_TAX_RATE=15
   MAILJET_API_KEY=your_mailjet_api_key
   MAILJET_SECRET_KEY=your_mailjet_secret_key
   MAILJET_FROM_EMAIL=your_email
   MAILJET_FROM_NAME=your_name
   MAILJET_TO_EMAIL=recipient_email
   MAILJET_TO_NAME=recipient_name
   ```

4. Copy the `example-config.json` file and create a `config.json` file to define the assets and trading parameters. See example:

   ```json
   {
     "min_profit_target_percentage": 3.0,
     "no_trade_refresh_hours": 2.0,
     "assets": [
       {
         "title": "MATIC chart",
         "enabled": true,
         "ready_to_trade": false,
         "symbol": "MATIC-USD",
         "buy_amount_usd": 50,
         "enable_snapshot": false
       }
     ]
   }
   ```

   **Config Parameters:**

   **Global Settings:**
   - `min_profit_target_percentage`: Minimum profit percentage threshold to trigger a sell signal (default: 3.0)
   - `no_trade_refresh_hours`: Hours to wait before re-analyzing an asset after determining no trade should be made (default: 2.0)

   **Asset-Specific Settings:**
   - `title`: Friendly name for the asset (for reference)
   - `enabled`: Whether to monitor this asset (`true`/`false`)
   - `ready_to_trade`: Whether to execute actual trades (`true`) or run in simulation mode (`false`)
   - `symbol`: The Coinbase trading pair symbol (e.g., "BTC-USD", "ETH-USD")
   - `buy_amount_usd`: USD amount to spend per trade. The bot calculates whole shares (rounded down) based on current price
   - `enable_snapshot`: Whether to generate chart snapshots for this asset (`true`/`false`)

## Installation (2) (creating a virtual environment in your project directory)

1. Create a virtual environment in your project directory
 ```bash
    cd ~/scalp-scripts
  ```

  ```bash
    python3 -m venv venv
  ```

2. Activate the virtual environment
  ```bash
    source venv/bin/activate
  ```

3. Install your requirements

  ```bash
    pip install -r requirements.txt
  ```

4. Run the trading bot

  ```bash
    python3 index.py
  ```

The bot will continuously monitor the specified assets and execute trades based on the configured strategies.

## Historical Data Backfilling

The bot includes a feature to backfill historical price data from CoinGecko API. This is useful for populating the `coinbase-data/` directory with historical data for analysis before the bot has been running long enough to collect it naturally.

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
       "max_hours": 730,
       "interval_seconds": 900
     }
   }
   ```

4. Ensure each asset has a `coingecko_id` in the config:
   ```json
   {
     "assets": [
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

The script will:
- Fetch 15-minute interval price data for the time period specified in `data_retention.max_hours`
- Transform the data to match the Coinbase format
- Merge with existing data (avoiding duplicates)
- Save to `coinbase-data/{SYMBOL}.json` files

### Notes

- The backfill uses 15-minute intervals
- Data is fetched based on the `max_hours` setting in your config (up to 6 months for free tier)
- The script only runs when `enable_backfilling_historical_data` is set to `true`
- Existing data is preserved - the script only adds missing historical data
- CoinGecko free tier supports up to 6 months of historical data
- CoinGecko API rate limits apply (30 calls/min for free tier) - the script includes delays between requests

### Finding CoinGecko IDs

Common CoinGecko IDs:
- Bitcoin (BTC): `bitcoin`
- Ethereum (ETH): `ethereum`
- Cardano (ADA): `cardano`
- Solana (SOL): `solana`

To find other IDs:
- Visit https://api.coingecko.com/api/v3/coins/list
- Or use the CoinGecko website and check the URL (e.g., coingecko.com/en/coins/bitcoin → ID is "bitcoin")

## Disclaimer

This trading bot is for educational purposes only. Trading cryptocurrencies involves significant risk, and you should only trade with money you can afford to lose. The author is not responsible for any financial losses incurred while using this bot.
