# Coinbase Trading Bot

This project is a cryptocurrency trading bot that interacts with the Coinbase API to automate trading based on predefined strategies. The bot is written in Python and uses the Coinbase REST API to fetch market data, place orders, and manage positions.

## Features

- Fetch current asset prices from Coinbase.
- Manage asset positions and account balances.
- Place market orders for buying and selling assets.
- Calculate potential profits, exchange fees, and taxes.
- Monitor market conditions and trigger buy/sell actions based on technical indicators such as SMA, MACD, and Bollinger Bands.
- Send email notifications for order placements using Mailjet.

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
  - `COINMARKETCAP_API_KEY`
  - `MAILJET_API_KEY`
  - `MAILJET_SECRET_KEY`
  - `MAILJET_FROM_EMAIL`
  - `MAILJET_FROM_NAME`
  - `MAILJET_TO_EMAIL`
  - `MAILJET_TO_NAME`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AroniasPrenovost/scalp-trader/tree/main
   cd coinbase-trading-bot
   ```

2. Install the required packages:

   ```bash
   pip install
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
     "assets": [
       {
         "enabled": true,
         "symbol": "MATIC-USD",
         "shares_to_acquire": 1,
         "target_profit_percentage": 2
       }
     ]
   }
   ```

## Usage

Run the trading bot:

```bash
python index.py
```

The bot will continuously monitor the specified assets and execute trades based on the configured strategies.

## Disclaimer

This trading bot is for educational purposes only. Trading cryptocurrencies involves significant risk, and you should only trade with money you can afford to lose. The author is not responsible for any financial losses incurred while using this bot.
