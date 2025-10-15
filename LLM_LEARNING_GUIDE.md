# LLM Self-Learning Trading System

## Overview

This trading bot now includes a **self-learning system** where the LLM (GPT-4o) analyzes historical trading data to improve future trading decisions. The system provides the AI with:

1. **Historical transaction data** for the specific symbol
2. **Buy event screenshots** showing market conditions when trades were initiated
3. **Performance metrics** including win rate, average profit, and patterns

Over time, the LLM learns from successful and unsuccessful trades to make better entry/exit decisions.

---

## How It Works

### 1. **Context Building**

When analyzing a trading opportunity, the system:
- Loads the last N trades for that specific symbol from `/transactions/data.json`
- Retrieves the screenshot taken at each buy event
- Calculates performance metrics (win rate, average profit, etc.)
- Formats this data into a comprehensive context for the LLM

### 2. **LLM Analysis with Historical Context**

The LLM receives:
- **Current market data** (price, volume, technical indicators)
- **Historical trade performance** (what worked, what didn't)
- **Buy screenshots from past trades** (visual market conditions)
- **Performance patterns** (typical hold times, profit targets that succeeded)

### 3. **Continuous Learning**

With each completed trade:
- The transaction is saved with the buy screenshot path
- On the next analysis, the LLM can review this trade
- The system learns patterns like:
  - "When I bought at the lower Bollinger Band with RSI < 30, the trade was successful"
  - "Trades held for 2-4 hours had better outcomes than quick scalps"
  - "This symbol tends to reverse at $X resistance level"

### 4. **Context Management**

To prevent context overload:
- Only the most recent N trades are included (configurable)
- Old transactions can be automatically pruned (configurable threshold)
- Screenshots use low-detail encoding to save tokens

---

## Configuration

Edit `config.json` to customize the learning behavior:

```json
{
  "llm_learning": {
    "enabled": true,                    // Enable/disable historical context
    "max_historical_trades": 10,        // Number of recent trades to include
    "include_screenshots": true,        // Include buy event screenshots
    "prune_old_trades_after": 50        // Auto-prune when total trades exceed this
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable LLM learning from historical trades |
| `max_historical_trades` | integer | `10` | Maximum number of recent trades to include in analysis |
| `include_screenshots` | boolean | `true` | Include buy event screenshots (uses more tokens) |
| `prune_old_trades_after` | integer | `50` | Automatically prune old trades when total exceeds this number |

---

## Data Structure

### Transaction Record (in `/transactions/data.json`)

```json
{
  "symbol": "BTC-USD",
  "buy_price": 65000.00,
  "sell_price": 67100.00,
  "potential_profit_percentage": 3.23,
  "timestamp": "2025-10-15T14:30:00+00:00",
  "gross_profit": 2100.00,
  "taxes": 777.00,
  "exchange_fees": 156.00,
  "total_profit": 1167.00,
  "time_held_position": "2.5 hours",
  "buy_timestamp": "2025-10-15T12:00:00Z",
  "buy_screenshot_path": "./screenshots/BTC-USD_chart_buy_20251015120000.png"
}
```

### Trading Context Object

The context built for the LLM includes:

```python
{
    'symbol': 'BTC-USD',
    'total_trades': 15,
    'trades_included': 10,
    'performance_summary': {
        'total_profit': 1250.50,
        'average_profit_percentage': 3.12,
        'win_rate': 80.0,
        'profitable_trades': 12,
        'losing_trades': 3
    },
    'trades': [
        {
            'buy_price': 65000.00,
            'sell_price': 67100.00,
            'profit_percentage': 3.23,
            'total_profit': 1167.00,
            'time_held': '2.5 hours',
            'buy_timestamp': '2025-10-15T12:00:00Z',
            'sell_timestamp': '2025-10-15T14:30:00+00:00',
            'screenshot_available': True,
            'screenshot_base64': '<base64_encoded_image>'
        }
        // ... more trades
    ]
}
```

---

## Code Architecture

### New Files

1. **`utils/trade_context.py`**
   - `build_trading_context()` - Builds historical context for LLM
   - `format_context_for_llm()` - Formats context into readable text
   - `get_trade_screenshots_for_vision()` - Prepares screenshots for Vision API
   - `prune_old_transactions()` - Removes old trades to manage data size

### Modified Files

1. **`utils/openai_analysis.py`**
   - Added `trading_context` parameter to `analyze_market_with_openai()`
   - Includes historical context in prompt
   - Adds historical screenshots to Vision API call

2. **`utils/coinbase.py`**
   - Added `buy_screenshot_path` parameter to `save_transaction_record()`
   - Stores screenshot path in transaction records

3. **`utils/matplotlib.py`**
   - Modified `plot_graph()` to return screenshot path

4. **`index.py`**
   - Imports `build_trading_context`
   - Loads LLM learning config
   - Builds context before each analysis
   - Captures and stores screenshot path with buy orders
   - Passes screenshot path to transaction records

---

## Usage Example

### Enabling LLM Learning

```json
{
  "llm_learning": {
    "enabled": true,
    "max_historical_trades": 10,
    "include_screenshots": true,
    "prune_old_trades_after": 50
  }
}
```

### Console Output

When the bot runs with LLM learning enabled:

```
Generating new AI analysis for BTC-USD...
Building historical trading context for BTC-USD...
✓ Loaded 8 historical trades for context
Including 8 historical trade screenshots in analysis
✓ OpenAI analysis completed for BTC-USD
```

### LLM Prompt (excerpt)

The LLM receives context like:

```
Historical Trading Performance for BTC-USD:

Performance Summary (All-Time):
- Total Trades: 15
- Win Rate: 80.0% (12 wins, 3 losses)
- Average Profit: 3.12% per trade
- Total Cumulative Profit: $1250.50

Recent Trade History (Last 10 trades):

1. ✓ Trade from 2025-10-15:
   - Entry: $65000.00 → Exit: $67100.00
   - Profit: 3.23% ($1167.00)
   - Time Held: 2.5 hours
   - Screenshot: Available

2. ✗ Trade from 2025-10-14:
   - Entry: $66500.00 → Exit: $65800.00
   - Profit: -1.05% ($-350.00)
   - Time Held: 1.2 hours
   - Screenshot: Available

...

INSTRUCTIONS: Use this historical data to:
1. Learn from past winning and losing trades
2. Identify patterns in successful entry/exit points
3. Adjust your strategy based on what has worked for this specific symbol
4. Consider the typical hold times and profit targets that have been successful
```

---

## Benefits

### 1. **Symbol-Specific Learning**
Each cryptocurrency has unique patterns. The LLM learns specific behaviors for BTC, ETH, etc.

### 2. **Pattern Recognition**
The LLM identifies:
- Successful entry conditions (RSI levels, Bollinger Bands, support/resistance)
- Optimal hold times
- Price targets that have historically worked

### 3. **Risk Management**
Learn from losing trades:
- What conditions led to stop losses
- When not to trade
- Better stop loss placement

### 4. **Continuous Improvement**
The more trades executed, the better the LLM's recommendations become.

---

## Token Usage Considerations

Including historical context increases token usage:

| Component | Approximate Tokens |
|-----------|-------------------|
| Text-based trade history (10 trades) | ~800-1,200 tokens |
| Single screenshot (low detail) | ~85 tokens |
| 10 screenshots (low detail) | ~850 tokens |
| **Total additional cost** | ~1,650-2,050 tokens |

**Recommendations:**
- Start with `max_historical_trades: 5` to test
- Set `include_screenshots: false` if token costs are a concern
- Monitor OpenAI API costs and adjust accordingly

---

## Manual Context Management

### View Transaction History

```bash
cat transactions/data.json | jq '.[] | select(.symbol == "BTC-USD")'
```

### Prune Old Trades Manually

```python
from utils.trade_context import prune_old_transactions

# Keep only last 30 trades for BTC-USD
prune_old_transactions('BTC-USD', keep_count=30)
```

### Build Context Manually

```python
from utils.trade_context import build_trading_context, format_context_for_llm

context = build_trading_context('BTC-USD', max_trades=10, include_screenshots=False)
print(format_context_for_llm(context))
```

---

## Troubleshooting

### Issue: "No historical trades found"
**Solution:** This is normal for the first few trades. The system will start learning after the first completed trade.

### Issue: High OpenAI API costs
**Solutions:**
- Reduce `max_historical_trades` to 5 or fewer
- Set `include_screenshots: false`
- Consider using `gpt-4o-mini` model (requires code change)

### Issue: Context too large
**Solution:** Reduce `max_historical_trades` and enable automatic pruning with a lower `prune_old_trades_after` value.

### Issue: Screenshots not found
**Solution:** Ensure the `screenshots/` directory exists and has proper permissions. Screenshots are saved during buy events.

---

## Future Enhancements

Potential improvements to the learning system:

1. **Feedback Loop**: Explicitly rate trades and provide feedback to the LLM
2. **Market Condition Tagging**: Tag trades with market conditions (bull/bear/sideways)
3. **Multi-Symbol Learning**: Learn patterns across correlated assets
4. **Performance Benchmarking**: Compare LLM decisions to simple strategies
5. **A/B Testing**: Run parallel strategies and compare results

---

## Technical Details

### Screenshot Storage

Screenshots are saved as:
```
./screenshots/{SYMBOL}_chart_buy_{TIMESTAMP}.png
```

Example: `./screenshots/BTC-USD_chart_buy_20251015120000.png`

### Vision API Integration

Screenshots are:
1. Read from disk
2. Encoded to base64
3. Sent to GPT-4o Vision API with `"detail": "low"` (85 tokens each)
4. Analyzed alongside text-based market data

### Data Retention

- **Transaction records**: Kept indefinitely (unless manually pruned)
- **Screenshots**: Kept indefinitely (manually delete old ones if needed)
- **Price data**: Rolling 24-hour window (configured in main loop)

---

## API Reference

### `build_trading_context(symbol, max_trades=10, include_screenshots=True)`

Builds contextual information from past trades.

**Parameters:**
- `symbol` (str): Trading pair symbol (e.g., 'BTC-USD')
- `max_trades` (int): Maximum recent trades to include
- `include_screenshots` (bool): Include base64-encoded screenshots

**Returns:** Dictionary with historical context

---

### `format_context_for_llm(context)`

Formats the context dictionary into human-readable text.

**Parameters:**
- `context` (dict): Context from `build_trading_context()`

**Returns:** Formatted string for LLM prompt

---

### `prune_old_transactions(symbol, keep_count=50)`

Removes older transactions to manage file size.

**Parameters:**
- `symbol` (str): Trading pair symbol
- `keep_count` (int): Number of recent transactions to keep

**Returns:** Boolean indicating success

---

## License

This feature is part of the main trading bot codebase.

---

## Support

For issues or questions about the LLM learning system:
1. Check console output for errors
2. Verify `config.json` settings
3. Ensure `transactions/data.json` is properly formatted
4. Check that screenshot files exist and are accessible
