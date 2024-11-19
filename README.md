# Coinbase Trading Bot

This project is a simple trading bot that interacts with the Coinbase API to automate cryptocurrency trading based on predefined strategies. The bot is written in Python and uses the Coinbase REST API to fetch market data, place orders, and manage positions.

## Features

- Fetch current asset prices from Coinbase.
- Manage asset positions and account balances.
- Place market orders for buying and selling assets.
- Calculate potential profits, exchange fees, and taxes.
- Monitor market conditions and trigger buy/sell actions based on support and resistance levels.

## Prerequisites

- Python 3.7+
- Coinbase API Key and Secret
- A `.env` file with the following variables:
  - `COINBASE_API_KEY`
  - `COINBASE_API_SECRET`
  - `COINBASE_SPOT_MAKER_FEE`
  - `COINBASE_SPOT_TAKER_FEE`
  - `FEDERAL_TAX_RATE`

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

3. Create a `.env` file in the root directory and add your Coinbase API credentials and other configurations:

   ```plaintext
   COINBASE_API_KEY=your_api_key
   COINBASE_API_SECRET=your_api_secret
   COINBASE_SPOT_MAKER_FEE=0.5
   COINBASE_SPOT_TAKER_FEE=0.5
   FEDERAL_TAX_RATE=15
   ```

4. Create a `config.json` file to define the assets and trading parameters:

   ```json
   {
     "assets": [
       {
         "enabled": true,
         "symbol": "BTC-USD",
         "support": 30000,
         "resistance": 35000,
         "buy_limit_1": 31000,
         "sell_limit_1": 34000
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
