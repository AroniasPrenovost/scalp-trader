# Daily Summary Email Feature

## Overview
Your trading bot now automatically sends a comprehensive daily email summary every 24 hours with portfolio performance, market insights, and trading activity.

## What's Included

### 📊 Portfolio Metrics
- Starting capital vs current value
- Total return percentage
- Net profit after fees and taxes
- Gross profit breakdown

### 📈 Visual Charts & Screenshots
1. **Portfolio Value Chart** - 30-day portfolio value trend
2. **Daily P&L Chart** - Bar chart showing daily profits/losses over 30 days
3. **Symbol Performance Chart** - Comparison of returns across BTC, ETH, SOL

### 🎯 Per-Symbol Breakdown
For each symbol (BTC, ETH, SOL):
- Starting capital
- Current value
- Return percentage
- Net profit

### 📅 Daily Activity
- Number of trades executed today
- Win/loss ratio
- Daily P&L
- Win rate percentage

### 📋 Trade Details
- Individual trade listings with entry/exit prices
- Profit/loss for each trade
- Hold time
- Exit trigger (profit target, stop loss, manual)
- Best and worst trades of the day

## Configuration

The daily summary is configured in `config.json`:

```json
{
  "daily_summary": {
    "enabled": true,
    "send_hour": 8
  }
}
```

**Parameters:**
- `enabled` (boolean): Turn daily summary emails on/off
- `send_hour` (integer 0-23): Hour of day to send email (default: 8 AM local time)

## How It Works

1. **Automatic Scheduling**: The bot checks every iteration if it's time to send the daily summary
2. **Data Aggregation**: Collects all transaction data, calculates metrics across all symbols
3. **Chart Generation**: Creates 3 PNG charts showing portfolio trends
4. **Email Delivery**: Sends beautiful HTML email via Mailjet with charts attached

## Files Created

- `utils/daily_summary.py` - Main daily summary logic
- `screenshots/portfolio_value_YYYYMMDD.png` - Portfolio value chart
- `screenshots/daily_pnl_YYYYMMDD.png` - Daily P&L chart
- `screenshots/symbol_performance_YYYYMMDD.png` - Symbol comparison chart
- `screenshots/.daily_summary_sent_YYYYMMDD` - Marker files to prevent duplicate sends

## Manual Testing

To manually trigger a daily summary email (for testing):

```bash
source venv/bin/activate
python -c "from utils.daily_summary import send_daily_summary_email; import json; config = json.load(open('config.json')); send_daily_summary_email(config['wallets'])"
```

Or run the module directly:

```bash
source venv/bin/activate
python -m utils.daily_summary
```

## Customization Options

### Change Send Time
Edit `config.json` to change when the email is sent:
```json
"send_hour": 9  // Change to 9 AM
```

### Disable Daily Emails
Set `enabled` to `false` in `config.json`:
```json
"daily_summary": {
  "enabled": false
}
```

### Adjust Chart Time Ranges
Edit `utils/daily_summary.py`:
- Line ~246: `generate_portfolio_value_chart(wallets_config, days=30)` - Change `days` parameter
- Line ~247: `generate_daily_pnl_chart(wallets_config, days=30)` - Change `days` parameter

## Email Requirements

Make sure your `.env` file has these Mailjet credentials configured:
```
MAILJET_API_KEY=your_api_key
MAILJET_SECRET_KEY=your_secret_key
MAILJET_FROM_EMAIL=your_email
MAILJET_FROM_NAME=Your Name
MAILJET_TO_EMAIL=recipient_email
MAILJET_TO_NAME=Recipient Name
```

## Troubleshooting

### Email Not Sending
1. Check Mailjet credentials in `.env`
2. Verify `enabled: true` in `config.json`
3. Check logs for error messages

### Wrong Send Time
- The `send_hour` uses your **local timezone**, not UTC
- If bot runs hourly, email sends during the first iteration in that hour

### Missing Charts
- Charts are saved to `screenshots/` directory
- Ensure directory has write permissions
- Check for matplotlib errors in logs

## Features

✅ Automatic daily delivery at configurable time
✅ Beautiful HTML email with responsive design
✅ Portfolio performance tracking
✅ Per-symbol breakdown
✅ Visual charts with 30-day trends
✅ Trade-by-trade details
✅ Best/worst trade highlights
✅ Win rate and P&L statistics
✅ Prevents duplicate sends on same day
✅ Graceful error handling

## Example Email Contents

```
Subject: 📈 Trading Summary - November 2, 2025 (0 trades)

PORTFOLIO OVERVIEW
Starting Capital: $3,900.00
Current Value: $3,740.33
Total Return: -4.10%
Net Profit: -$159.67

DAILY ACTIVITY
Trades: 0 (0 wins, 0 losses)
Win Rate: 0.0%
Daily P&L: $0.00

[Charts displayed as images]

PERFORMANCE BY SYMBOL
BTC: -3.12% ($40.56 loss)
ETH: -4.80% ($62.40 loss)
SOL: -4.39% ($57.07 loss)
```

Enjoy your automated daily trading summaries! 🚀
