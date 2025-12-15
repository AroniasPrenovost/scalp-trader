# Range-Based Support Zone Trading Strategy

## Overview

This strategy identifies **support zones** by finding 2-3+ price bottoms within a similar price range, then triggers buy signals when the current price revisits that zone. This is a proven **mean reversion** and **support zone trading** approach.

## How It Works

### 1. **Find Local Bottoms (Price Minima)**
The algorithm scans historical price data to find local minimums (bottoms) where price reversed upward. This uses scipy's `argrelextrema` function with an `order` parameter:
- `order=5` (default): A point is a bottom if it's lower than 5 points on each side
- Higher order = more significant bottoms, fewer signals
- Lower order = more sensitive, more signals but noisier

### 2. **Group Bottoms into Support Zones**
Once bottoms are identified, they're grouped into zones based on price proximity:
- **Zone Tolerance**: Bottoms within 3% of each other (default) form a zone
- **Minimum Touches**: Need at least 2 touches (default) to form a valid zone
- **Zone Strength**: More touches = stronger support

Example: If price bottomed at $89,000, $89,500, and $90,000 over the past 14 days, these form a support zone around $89,700 (avg).

### 3. **Check if Current Price is in a Zone**
When the current price enters a support zone (within tolerance), a **BUY signal** is triggered:
- Entry: Zone average price
- Stop Loss: Below zone minimum (default 2% below)
- Profit Target: Based on risk/reward ratio (default 2.5:1)

### 4. **Calculate Trade Setup**
The strategy automatically calculates:
- **Entry Price**: Average of all bottoms in the zone
- **Stop Loss**: 2% below the lowest bottom in the zone
- **Profit Target**: Entry + (Risk × 2.5)
- **Risk/Reward Ratio**: Ensures favorable risk/reward (2.5:1 default)

---

## Configuration Parameters

### Strategy Modes

| Mode | Min Touches | Zone Tolerance | Entry Tolerance | Extrema Order | Best For |
|------|-------------|----------------|-----------------|---------------|----------|
| **Conservative** | 3+ | 2.0% | 1.0% | 6 | High confidence, fewer trades |
| **Moderate** | 2+ | 3.0% | 1.5% | 5 | Balanced approach (recommended) |
| **Aggressive** | 2+ | 4.0% | 2.0% | 4 | More opportunities, higher risk |

### Parameter Explanations

- **min_touches**: Minimum bottoms needed to form a zone (2-5)
  - Higher = stronger zones, fewer signals
  - Lower = more signals but weaker zones

- **zone_tolerance_percentage**: Max % difference to group bottoms into same zone (1-5%)
  - 2% = tight zones, very precise support
  - 3% = moderate zones (recommended)
  - 4% = wide zones, more forgiving

- **entry_tolerance_percentage**: % buffer around zone for entry (0.5-2.5%)
  - Allows entries slightly above/below zone boundaries
  - 1.5% recommended for moderate approach

- **extrema_order**: Sensitivity for finding bottoms (4-7)
  - 4 = very sensitive, finds many bottoms
  - 5 = moderate (recommended)
  - 6-7 = conservative, only significant bottoms

- **lookback_window**: How far back to analyze (hours)
  - 336 hours = 14 days (recommended for swing trading)
  - 720 hours = 30 days (for longer-term zones)
  - 168 hours = 7 days (for shorter-term scalping)

---

## Usage

### Testing the Strategy

Run the test script to see how the strategy performs with your current BTC data:

```bash
python test_range_strategy.py
```

This will:
1. Load the last 30 days of BTC price data
2. Test 3 configurations (Conservative, Moderate, Aggressive)
3. Show identified support zones
4. Generate a visualization chart
5. Display trade setups if buy signals are triggered

### Integrating into Your Trading Bot

To add this strategy to your main trading logic in `index.py`:

```python
from utils.range_support_strategy import check_range_support_buy_signal

# In your trading loop, after loading price data:
range_signal = check_range_support_buy_signal(
    prices=coin_prices_LIST,
    current_price=current_price,
    min_touches=2,
    zone_tolerance_percentage=3.0,
    entry_tolerance_percentage=1.5,
    extrema_order=5,
    lookback_window=336  # 14 days
)

if range_signal['signal'] == 'buy':
    print(f"Range strategy: BUY signal triggered!")
    print(f"Reasoning: {range_signal['reasoning']}")
    print(f"Zone strength: {range_signal['zone_strength']} touches")

    # You can combine this with your AI analysis
    # Example: Only buy if BOTH AI and range strategy agree
    if analysis.get('trade_recommendation') == 'buy' and range_signal['signal'] == 'buy':
        # Execute buy order
        pass
```

---

## Example Trade Setup

From the test output (BTC at $89,491):

```
Support Zone Identified:
  Zone Average: $89,694.92
  Zone Range: $88,906.20 - $90,566.65
  Touches: 5 (very strong support)

Trade Setup:
  Entry: $89,694.92
  Stop Loss: $87,128.08 (-2.86%)
  Profit Target: $96,112.02 (+7.15%)
  Risk/Reward Ratio: 2.5:1

Current Price: $89,491.19 (within zone, -0.23% from avg)
Signal: BUY
```

**Interpretation:**
- Price has bounced from $89k-$90k range **5 times** in the past 14 days
- Current price is testing this support zone again
- If you buy at $89,695, you risk 2.86% but target 7.15% gain
- If support fails, stop loss at $87,128 limits downside

---

## Combining with Your AI Analysis

### Best Approach: **Confluence Strategy**

Only trade when **multiple signals align**:

1. **Range Strategy**: Price in support zone ✓
2. **AI Analysis**: HIGH confidence, BULLISH trend ✓
3. **RSI**: Oversold (<30) ✓
4. **Volume**: Increasing at support ✓

Example integration:

```python
# Check range support strategy
range_signal = check_range_support_buy_signal(...)

# Check AI analysis
ai_analysis = analyze_market_with_openai(...)

# Confluence logic
if (range_signal['signal'] == 'buy' and
    range_signal['zone_strength'] >= 3 and  # Strong zone (3+ touches)
    ai_analysis.get('confidence_level') == 'high' and
    ai_analysis.get('market_trend') == 'bullish'):

    print("✓ CONFLUENCE: Range + AI signals aligned - HIGH confidence buy")
    # Execute trade
```

---

## Advantages of This Strategy

1. **Objective Entry Points**: No guessing - price either in zone or not
2. **Risk Management**: Stop loss automatically placed below support
3. **Favorable Risk/Reward**: Always ensures 2.5:1 or better
4. **Multiple Confirmations**: Requires 2-3+ historical tests of support
5. **Complements Technical Analysis**: Works alongside RSI, MA, AI signals

---

## Limitations & Warnings

1. **Support Can Break**: No support level is guaranteed
2. **Choppy Markets**: May generate false signals in sideways/whipsaw markets
3. **Requires Historical Data**: Needs enough data to identify patterns
4. **Not Trend-Following**: This is mean reversion, works best in ranging markets
5. **Backtesting Recommended**: Test thoroughly before trading real money

---

## Recommended Settings

For **crypto swing trading** (your current strategy):

```python
check_range_support_buy_signal(
    prices=coin_prices_LIST,
    current_price=current_price,
    min_touches=2,              # At least 2 historical tests
    max_touches=5,              # Up to 5 touches max
    zone_tolerance_percentage=3.0,  # 3% zone width
    entry_tolerance_percentage=1.5, # 1.5% entry buffer
    extrema_order=5,            # Moderate sensitivity
    lookback_window=336         # 14 days (2 weeks)
)
```

For **more conservative** approach:
- Increase `min_touches` to 3
- Reduce `zone_tolerance_percentage` to 2.0%
- Increase `extrema_order` to 6

For **more aggressive** approach:
- Keep `min_touches` at 2
- Increase `zone_tolerance_percentage` to 4.0%
- Reduce `extrema_order` to 4

---

## Visualization

The test script generates a chart showing:
- Price history
- All identified support zones (shaded bands)
- Touch points (circular markers)
- Current price (red line)
- Entry/Stop Loss/Target levels (if buy signal)

Chart saved to: `screenshots/{SYMBOL}_range-support-strategy.png`

---

## Next Steps

1. **Run the Test**: `python test_range_strategy.py`
2. **Review the Chart**: Check if zones make sense visually
3. **Adjust Parameters**: Tweak based on your risk tolerance
4. **Backtest**: Test on historical data to measure win rate
5. **Integrate**: Add to your main trading loop with confluence logic
6. **Paper Trade**: Test with small amounts before full deployment

---

## Questions?

- **Is this a real trading strategy?** Yes! Support/resistance zone trading is used by professional traders
- **How does it differ from your AI?** AI analyzes patterns holistically; this focuses purely on support zones
- **Should I replace AI with this?** No - combine them for better results (confluence)
- **What markets work best?** Works best in **ranging/consolidating markets** with clear support levels

---

**Remember**: Always test thoroughly and never risk more than you can afford to lose.
