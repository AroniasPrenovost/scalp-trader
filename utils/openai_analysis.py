import os
import json
import time
import base64
from openai import OpenAI
from io import BytesIO

def analyze_market_with_openai(symbol, coin_data, maker_fee_percentage=0, tax_rate_percentage=0, min_profit_target_percentage=3.0, chart_paths=None, trading_context=None, graph_image_path=None, range_percentage_from_min=None, config=None, btc_context=None):
    """
    Analyzes market data using OpenAI's API to determine key support/resistance levels
    and trading recommendations.

    Args:
        symbol: The trading pair symbol (e.g., 'XLM-USD')
        coin_data: Dictionary containing market data including:
            - current_price
            - coin_prices_list
            - current_volume_24h
            - coin_volume_24h_LIST
            - current_volume_percentage_change_24h
            - coin_price_percentage_change_24h_LIST
        maker_fee_percentage: Exchange maker fee as a percentage (e.g., 0.4 for 0.4%) - used for limit orders
        tax_rate_percentage: Federal tax rate as a percentage (e.g., 37 for 37%)
        min_profit_target_percentage: Minimum profit target percentage (e.g., 3.0 for 3%)
        chart_paths: Optional dictionary with paths to multi-timeframe charts {'short_term': path, 'medium_term': path, 'long_term': path}
        trading_context: Optional dictionary containing historical trading context from build_trading_context()
        graph_image_path: DEPRECATED - use chart_paths instead. Optional path to a single graph image
        range_percentage_from_min: Optional volatility metric showing price range from min to max (e.g., 50 = 50% range)
        config: Optional config dictionary for position sizing and other settings

    Returns:
        Dictionary with analysis results or None if API call fails
    """

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        return None

    client = OpenAI(api_key=api_key)

    # Prepare the market data summary
    current_price = coin_data.get('current_price', 0)
    prices = coin_data.get('coin_prices_list', [])
    volumes = coin_data.get('coin_volume_24h_LIST', [])

    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    avg_price = sum(prices) / len(prices) if prices else 0

    # Calculate total cost burden (fees on both buy and sell, plus taxes on profit)
    # Using maker fees since we use limit orders that sit on the order book
    total_fee_percentage = maker_fee_percentage * 2  # Buy fee + Sell fee
    total_cost_burden = total_fee_percentage + tax_rate_percentage

    # Build historical context section if provided
    historical_context_section = ""
    if trading_context and trading_context.get('total_trades', 0) > 0:
        from utils.trade_context import format_context_for_llm
        historical_context_section = f"\n\n{format_context_for_llm(trading_context)}\n"

    # Build the prompt
    timeframe_context = ""
    if chart_paths:
        timeframe_context = """

IMPORTANT - MULTI-TIMEFRAME ANALYSIS FRAMEWORK:
You are being provided with FIVE charts showing different lookback windows for comprehensive analysis.
NOTE: All charts use 1-hour interval data points - the difference is the lookback window (how much historical data is shown).

DATA STRUCTURE:
- Base interval: 1 hour (each data point represents 1 hour of price action)
- 6-month chart: ~4,380 hourly data points (182.5 days × 24 hours)
- 90-day chart: ~2,160 hourly data points (90 days × 24 hours)
- 30-day chart: ~720 hourly data points (30 days × 24 hours)
- 14-day chart: ~336 hourly data points (14 days × 24 hours)
- 72-hour chart: 72 hourly data points (3 days × 24 hours)

6-MONTH CHART (Full Historical View) - Macro Trends & Context:
- Identify major long-term support/resistance levels and trend channels
- Look for head & shoulders, double tops/bottoms, major chart patterns
- Determine cyclical patterns and long-term market structure
- Are we near all-time highs/lows within this 6-month window?
- Assess overall macro trend: strong uptrend, downtrend, or range-bound?
- Identify key psychological price levels that have been tested multiple times
- This provides the MACRO CONTEXT for your trading decision

90-DAY CHART (Extended Trend) - Quarterly Patterns & Validation:
- Identify extended trend patterns and quarterly support/resistance zones
- Validate macro trend continuation or reversal signals from 6-month chart
- Look for broader accumulation/distribution patterns
- Check if recent 30-day moves are part of larger quarterly structure
- Assess sustainability of trends: are higher timeframe levels holding?
- Identify intermediate support/resistance that may not be visible on shorter timeframes
- This bridges the gap between macro context (6mo) and recent action (30d)

30-DAY CHART (Recent Trend) - Medium-Term Momentum:
- Confirm current trend strength and direction over the past month
- Identify key support/resistance zones that are actively relevant
- Look for swing patterns: higher highs/higher lows (uptrend) or lower highs/lower lows (downtrend)
- Check if price is respecting moving averages (MA20, MA50)
- Assess if we're in a pullback within uptrend or a breakout scenario
- Note recent volume patterns: increasing on rallies = bullish, decreasing = bearish
- This validates whether recent momentum aligns with macro trend

14-DAY CHART (Short-term Swing Momentum) - Immediate Trading Context:
- Identify short-term support/resistance levels forming over the past 2 weeks
- Look for recent breakouts, breakdowns, or consolidation patterns
- Check RSI for oversold (<30) or overbought (>70) conditions
- Assess price action quality: clean directional moves vs. choppy/whipsaw
- Note any candlestick patterns at key levels (doji, hammer, engulfing)
- Verify if short-term swing momentum aligns with 30-day and longer-term trends
- This determines if NOW is a good time to enter based on recent swing price action

72-HOUR CHART (Recent View) - Entry/Exit Timing & Context:
- Identify precise entry zone and immediate micro support/resistance over past 3 days
- Look for recent momentum shifts or reversal signals with better context than 24h
- Check if price is bouncing off key technical levels RIGHT NOW
- Assess immediate risk/reward: is current price near support (good entry) or resistance (poor entry)?
- Note volume spikes in last 72 hours that signal buyer/seller interest
- Confirm RSI trends and extreme conditions with more data points for validation
- This fine-tunes your EXACT entry price and timing with better short-term context

MULTI-TIMEFRAME DECISION LOGIC:
✓ All 5 timeframes bullish + volume confirmation = HIGH confidence long candidate
✓ 6mo uptrend + 90d uptrend + 30d uptrend + 14d pullback to support + 72h showing reversal = HIGH confidence entry
✓ 6mo uptrend + 90d uptrend + 30d consolidation near support + 14d breakout + 72h momentum = HIGH confidence entry
✗ Timeframe conflict (e.g., 72h/14d bullish but 30d/90d/6mo bearish) = NO TRADE (wait for alignment)
✗ Price approaching major 6-month or 90-day resistance = NO TRADE or significantly reduce confidence
✗ 6-month downtrend + 90d downtrend + 14d bounce = Counter-trend risk, NO TRADE unless exceptional setup
✗ 30-day trend weak/sideways + mixed signals = NO TRADE (wait for clarity)

Base your primary trading decision on the 14-DAY chart (immediate swing context), but REQUIRE validation from 30-day, 90-day, and 6-month charts. Use 72-hour chart for fine-tuning entry timing with better short-term context.
"""

    # Build volatility context
    volatility_context = ""
    if range_percentage_from_min is not None:
        volatility_context = f"""

VOLATILITY ANALYSIS (CRITICAL):
- Price Range (Low to High): {range_percentage_from_min:.2f}%
- This metric shows the total percentage change from minimum to maximum price in the data window
- INTERPRETATION:
  * < 5%: Very Low Volatility - tight range, potential breakout setup or dead market
  * 5-15%: Low Volatility - normal consolidation, suitable for range trading
  * 15-30%: Moderate Volatility - healthy trending market, ideal for swing trades
  * 30-50%: High Volatility - strong trending or volatile, larger profit potential but higher risk
  * > 50%: Extreme Volatility - caution advised, ensure stop losses are appropriate for the range

VOLATILITY TRADING RULES:
1. If range < 10% AND profit target >= 5%: Reduce confidence (target exceeds typical range movement)
2. If range > 30%: Widen stop losses proportionally to avoid getting stopped out by normal volatility
3. If range > 50%: Only trade with HIGH conviction setups, risk of whipsaw is elevated
4. Profit targets should be realistic relative to observed volatility (don't target 10% profit on 8% range asset)
5. Position sizing: Higher volatility = smaller position size to manage risk"""

    # Build BTC correlation context section
    btc_correlation_context = ""
    if btc_context and symbol in ['SOL-USD', 'ETH-USD']:
        btc_sentiment = btc_context.get('sentiment', {})
        btc_metrics = btc_context.get('price_metrics', {})
        btc_trend = btc_sentiment.get('market_trend', 'N/A')
        btc_confidence = btc_sentiment.get('confidence_level', 'N/A')
        btc_7d_change = btc_metrics.get('change_7d_pct', 0.0)
        btc_24h_change = btc_metrics.get('change_24h_pct', 0.0)

        asset_name = 'Solana' if symbol == 'SOL-USD' else 'Ethereum'
        btc_correlation_context = f"""

BTC CORRELATION ANALYSIS - CRITICAL MARKET CONTEXT:
You are analyzing {symbol} ({asset_name}), which is HIGHLY CORRELATED with Bitcoin (BTC-USD).
Historical correlation coefficient: 0.75-0.90 (very high correlation).
{asset_name} typically has beta of 1.2-1.8 vs BTC (amplifies BTC moves by 20-80%).

CURRENT BTC MARKET STATE (for mandatory correlation assessment):
- BTC Trend: {btc_trend} (confidence: {btc_confidence})
- BTC Price Change (7 days): {btc_7d_change:+.2f}%
- BTC Price Change (24 hours): {btc_24h_change:+.2f}%
- BTC Current Price: ${btc_metrics.get('current_price', 'N/A')}
- BTC Major Support: ${btc_sentiment.get('major_support', 'N/A')}
- BTC Major Resistance: ${btc_sentiment.get('major_resistance', 'N/A')}

NOTE: You will see BTC charts alongside {symbol} charts in the image payload.
Compare the two assets visually:
- Are they moving in sync or diverging?
- Is {symbol} respecting BTC support/resistance levels proportionally?
- Is {symbol} showing relative strength or weakness vs BTC?

MANDATORY CORRELATION RULES (override technical signals if violated):

1. BTC BEARISH TREND = NO BUY FOR {symbol}
   - If BTC trend is "bearish" → trade_recommendation MUST be "no_trade" or "sell"
   - Reasoning: {asset_name} rarely sustains rallies against BTC downtrends
   - Even if {symbol} technicals look bullish, BTC will likely drag it down

2. BTC SIDEWAYS/CONSOLIDATION = SELECTIVE BUYING
   - If BTC trend is "sideways" → Only recommend "buy" if:
     a) {symbol} shows clear relative strength (outperforming BTC on recent timeframes)
     b) {symbol} has strong technical breakout setup (not just minor bounce)
   - Otherwise → "no_trade" (wait for BTC to establish direction)

3. BTC BULLISH TREND = GREEN LIGHT FOR {symbol}
   - If BTC trend is "bullish" and {symbol} technicals align → confidence_level can be "high"
   - If BTC bullish but {symbol} lagging → reduce confidence by 1 level
   - Best setups: BTC bullish + {symbol} breaking out = maximum confidence

4. DIVERGENCE DETECTION (compare charts visually):
   - BTC breaking support but {symbol} holding support = WARNING (likely to follow BTC down soon)
     → Reduce confidence or recommend "no_trade"
   - BTC consolidating but {symbol} breaking out = LEADERSHIP (strong relative strength)
     → Increase confidence if volume confirms
   - BTC at resistance but {symbol} already broke out = DECOUPLING (rare, high conviction)
     → Can maintain high confidence if volume is strong

5. CONFIDENCE ADJUSTMENT BASED ON BTC ALIGNMENT:
   - BTC bullish + {symbol} bullish = NO ADJUSTMENT (alignment is good)
   - BTC sideways + {symbol} bullish = REDUCE confidence by 1 level (caution)
   - BTC bearish + {symbol} bullish = FORCE "no_trade" (fighting the market)
   - BTC bullish + {symbol} bearish = REDUCE confidence (lagging is concerning)

6. VOLUME CORRELATION:
   - If BTC volume is "increasing" and {symbol} volume "stable/decreasing" = weak participation
     → Reduce confidence (not leading, just following weakly)
   - If both have increasing volume = strong confirmation
     → Can maintain or increase confidence

YOUR ANALYSIS WORKFLOW:
Step 1: Analyze {symbol} charts independently (support, resistance, trends, patterns)
Step 2: Review BTC charts provided in the image payload
Step 3: Compare the two assets:
        - Are trends aligned (both bullish, both bearish, or diverging)?
        - Is {symbol} at a similar technical position as BTC (both near support, both breaking out)?
        - Is {symbol} showing relative strength or weakness?
Step 4: Apply correlation rules above to OVERRIDE or ADJUST your initial {symbol} analysis
Step 5: Final recommendation must factor in BTC context

EXAMPLE SCENARIOS:
✓ GOOD TRADE: BTC bullish uptrend + {symbol} bullish breakout above resistance + both showing volume increase
  → confidence_level: "high", trade_recommendation: "buy"

✗ BAD TRADE: BTC bearish breakdown + {symbol} bullish bounce off support
  → confidence_level: "low", trade_recommendation: "no_trade"
  → reasoning: "{asset_name} bounce against BTC downtrend - likely fails"

⚠️ CAUTION TRADE: BTC sideways consolidation + {symbol} bullish breakout + {symbol} volume weak
  → confidence_level: "medium" → MUST be "no_trade" (per schema rules)
  → reasoning: "{asset_name} breakout lacks BTC confirmation and volume"

✓ SELECTIVE TRADE: BTC sideways + {symbol} strong breakout with volume + clear outperformance
  → confidence_level: "high", trade_recommendation: "buy"
  → reasoning: "{asset_name} showing leadership vs BTC - breakout confirmed"

CRITICAL: Your "reasoning" field MUST mention BTC trend alignment/divergence.
Examples:
- "BTC bullish aligns with SOL breakout - strong setup"
- "BTC bearish backdrop overrides SOL technicals - no trade"
- "SOL outperforming BTC consolidation - selective buy"
"""

    prompt = f"""Analyze the following market data for {symbol} and provide a technical analysis with specific trading levels.
{historical_context_section}{timeframe_context}{volatility_context}{btc_correlation_context}

Market Data:
- Current Price: ${current_price}
- Price Range: ${min_price} - ${max_price}
- Average Price: ${avg_price}
- Number of Data Points: {len(prices)}
- Recent Price Trend: {prices[-20:] if len(prices) >= 20 else prices}

VOLUME DATA (Rolling 24h Snapshots - NOTE: This is Coinbase's rolling 24-hour volume):
- Current 24h Volume: {coin_data.get('current_volume_24h', 0)}
- Average 24h Volume (across all snapshots): {sum(volumes) / len(volumes) if volumes else 0}
- Min/Max 24h Volume observed: {min(volumes) if volumes else 0} / {max(volumes) if volumes else 0}

VOLUME ANALYSIS GUIDELINES (Limited by rolling 24h snapshot data):
- Use the volume snapshot chart to assess GENERAL volume trends over time (increasing, decreasing, or stable)
- Current volume above recent average = generally more market activity/interest
- Current volume below recent average = generally less market activity/interest
- LIMITATION: This data shows rolling 24h totals, NOT specific volume at price levels or candles
- Cannot reliably confirm breakout/breakdown volume or pinpoint volume spikes at specific times
- Use volume as CONTEXT for overall market interest, not as primary confirmation for entries
- Include "volume_trend" field: "increasing", "decreasing", or "stable" based on the snapshot chart

Trading Costs (IMPORTANT - Factor these into your recommendations):
- Exchange Maker Fee: {maker_fee_percentage}% per trade ({total_fee_percentage}% total for buy + sell)
  NOTE: Maker fees apply because we use LIMIT ORDERS that sit on the order book
- Tax Rate on Profits: {tax_rate_percentage}%
- Minimum Profitable Trade: Must exceed ~{total_cost_burden:.2f}% to break even after all costs
- Minimum Required Profit Target: {min_profit_target_percentage}% (NET profit after all costs)

Please analyze this data and respond with a JSON object (ONLY valid JSON, no markdown code blocks) containing:
{{
    "major_resistance": <price level>,
    "minor_resistance": <price level>,
    "major_support": <price level>,
    "minor_support": <price level>,
    "buy_in_price": <recommended buy price>,
    "sell_price": <recommended sell price>,
    "profit_target_percentage": <recommended NET profit percentage AFTER all fees and taxes>,
    "stop_loss": <recommended stop loss price>,
    "risk_reward_ratio": <calculated as (sell_price - buy_in_price) / (buy_in_price - stop_loss)>,
    "confidence_level": <"high", "medium", or "low">,
    "market_trend": <"bullish", "bearish", or "sideways">,
    "volume_trend": <"increasing", "decreasing", or "stable" based on the volume snapshot chart>,
    "trade_invalidation_price": <price level where trade thesis is completely invalidated>,
    "reasoning": <brief factual explanation, max 200 characters>,
    "trade_recommendation": <"buy", "sell", "hold", or "no_trade">,
    "buy_amount_usd": <recommended USD amount to spend on this trade>
}}

JSON SCHEMA VALIDATION REQUIREMENTS:
- All price fields: Must be numeric, max 8 decimal places
- profit_target_percentage: Must be >= {min_profit_target_percentage}
- risk_reward_ratio: Must be >= 2.0 for any "buy" recommendation (MANDATORY)
- confidence_level: ENUM only ["high", "medium", "low"]
- trade_recommendation: ENUM only ["buy", "sell", "hold", "no_trade"]
- volume_trend: ENUM only ["increasing", "decreasing", "stable"]
- reasoning: Max 200 characters, factual only (no subjective language like "could", "might", "possibly")
- If confidence_level = "medium" OR "low" → trade_recommendation MUST be "no_trade"
- If trade_recommendation = "buy" → risk_reward_ratio MUST be >= 2.0
- sell_price must be > buy_in_price (no shorting allowed)
- stop_loss must be < buy_in_price
- trade_invalidation_price: Typically below stop_loss, the price where thesis breaks down completely

CRITICAL REQUIREMENTS:
1. RISK/REWARD RATIO: All "buy" recommendations MUST have risk_reward_ratio >= 2.0
   - Calculate: (sell_price - buy_in_price) / (buy_in_price - stop_loss)
   - If ratio < 2.0, you MUST set trade_recommendation to "no_trade"
   - Example: Buy $1.00, Sell $1.06, Stop $0.97 = (0.06/0.03) = 2.0 ratio ✓

2. PROFIT THRESHOLD: If market conditions do NOT support at least {min_profit_target_percentage}% NET profit (after all fees and taxes), set trade_recommendation to "no_trade" and set profit_target_percentage to {min_profit_target_percentage}%.

3. COST CALCULATION TRANSPARENCY: Show your work in reasoning field:
   - Example: "Buy $0.50, sell $0.52 = 4% gross, 2.8% net after costs"
   - Gross price movement must exceed net profit target to cover {total_fee_percentage}% fees + {tax_rate_percentage}% tax

4. POSITION SIZING (REQUIRED): Position size will be automatically calculated based on volatility.
   - VOLATILITY-ADJUSTED SIZING is enabled: Position size scales inversely with market volatility
   - Low volatility (<15%): Up to 75% of capital for HIGH confidence trades
   - Moderate volatility (15-30%): Up to 64% of capital (85% multiplier)
   - High volatility (30-50%): Up to 49% of capital (65% multiplier)
   - Extreme volatility (>50%): Up to 38% of capital (50% multiplier)
   - NEVER commit more than 75% of current_usd value - ALWAYS keep at least 25% in reserve
   - Portfolio metrics are provided for LEARNING CONTEXT only
   - DO NOT let portfolio performance influence your position sizing - maintain SAME disciplined approach
   - Position sizing confidence requirements:
     * HIGH confidence: Volatility-adjusted position (only for exceptional setups meeting ALL criteria)
     * MEDIUM confidence: DO NOT TRADE - wait for higher quality opportunities
     * LOW confidence: DO NOT TRADE - recommend "no_trade" instead
   - The system will automatically calculate the optimal buy_amount_usd based on current volatility
   - You should set buy_amount_usd to 0 in your JSON response - it will be overridden by volatility calculation
   - Prioritize QUALITY over QUANTITY - focus on high-likelihood trades with strong technical confirmation

5. TRADE INVALIDATION: Set trade_invalidation_price to a level where if breached, the entire trade thesis is wrong
   - Typically below stop_loss by a small margin
   - Example: If support is $0.95, stop is $0.94, invalidation might be $0.93

CONFIDENCE LEVEL CRITERIA (OBJECTIVE RUBRIC):

HIGH confidence requires ALL of the following:
✓ All 5 timeframes (72h, 14d, 30d, 90d, 6mo) aligned in same direction
✓ Price at key technical level (major support/resistance, Fibonacci level)
✓ Volume confirms the setup (above average on bullish setups, spike at support)
✓ Risk/reward ratio >= 3.0 (preferably 3.5+)
✓ No major resistance within profit target range on any timeframe
✓ RSI supports direction (not overbought on longs, not oversold on shorts)
✓ Historical data shows similar setups succeeded (if historical context available)
✓ Multiple technical confirmations (e.g., MACD cross + support bounce + volume)

MEDIUM confidence (DO NOT TRADE - per requirement #4):
- Some but not all HIGH confidence criteria met
- Timeframe alignment weak or contradictory
- Risk/reward ratio 2.0-2.5 range
- Volume neutral or unclear
→ Set trade_recommendation to "no_trade"

LOW confidence (DO NOT TRADE - per requirement #4):
- Conflicting signals across timeframes
- Poor risk/reward ratio (< 2.0)
- Low volume, no technical confirmation
- Price in no-man's land (not at key levels)
→ Set trade_recommendation to "no_trade"

RECENCY BIAS WARNING:
- Historical performance reflects PAST market conditions which may differ from current regime
- Weight current technical setup (70%) MORE than historical patterns (30%)
- If current setup contradicts historical patterns, explain the divergence in reasoning
- DO NOT increase position size due to past success - maintain same disciplined approach

Base your analysis on technical indicators, support/resistance levels, and price action."""

    try:
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": f"""You are a technical market analyst providing educational analysis of cryptocurrency price data. Your analysis must be:

1. QUANTITATIVE: Use specific price levels, percentages, and ratios - no vague language
2. OBJECTIVE: Analyze key technical levels (support, resistance, risk/reward ratios) using standard technical analysis methods
3. SYSTEMATIC: Apply consistent technical analysis methodology across all analyses
4. DATA-DRIVEN: Base conclusions on chart patterns, volume analysis, and historical price action
5. FACTUAL: Present technical observations with calculated metrics including {total_fee_percentage}% transaction costs

This analysis is for educational and informational purposes only, not financial advice.
Output ONLY valid JSON with no markdown formatting or explanatory text outside the JSON structure."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Handle multi-timeframe charts or legacy single chart
        has_charts = False

        # Priority: use chart_paths (multi-timeframe) if available
        if chart_paths and isinstance(chart_paths, dict):
            try:
                content_array = [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]

                # Add BTC correlation charts FIRST if analyzing altcoins
                if btc_context and symbol in ['SOL-USD', 'ETH-USD']:
                    btc_charts = btc_context.get('chart_paths', {})
                    btc_timeframes = ['30_day', '14_day', '72_hour']  # 3 key BTC charts

                    for btc_tf in btc_timeframes:
                        if btc_tf in btc_charts and btc_charts[btc_tf] and os.path.exists(btc_charts[btc_tf]):
                            with open(btc_charts[btc_tf], "rb") as image_file:
                                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                                content_array.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                })
                                print(f"  Added BTC {btc_tf} correlation chart with high detail")

                # Add charts in order: 14d (high detail - primary execution), 30d (high - trend), 90d (high - extended trend), 72h (high - timing), 6mo (low - macro context), volume_snapshots (low - context)
                # This prioritizes the most important timeframes while saving tokens on context
                # 14d = immediate swing context, 30d = recent trend, 90d = extended trend, 72h = entry timing, 6mo = big picture, volume_snapshots = volume context
                timeframe_order = ['14_day', '30_day', '90_day', '72_hour', '6_month', 'volume_snapshot']
                detail_levels = {'14_day': 'high', '30_day': 'high', '90_day': 'high', '72_hour': 'high', '6_month': 'low', 'volume_snapshot': 'low'}

                for timeframe in timeframe_order:
                    if timeframe in chart_paths and chart_paths[timeframe] and os.path.exists(chart_paths[timeframe]):
                        with open(chart_paths[timeframe], "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            content_array.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": detail_levels[timeframe]
                                }
                            })
                            chart_type = "volume snapshot chart" if timeframe == 'volume_snapshot' else f"{symbol} {timeframe} chart"
                            print(f"  Added {chart_type} with {detail_levels[timeframe]} detail")

                # Add historical trade screenshots if available
                if trading_context:
                    from utils.trade_context import get_trade_screenshots_for_vision
                    historical_screenshots = get_trade_screenshots_for_vision(trading_context)
                    if historical_screenshots:
                        print(f"  Including {len(historical_screenshots)} historical trade screenshots")
                        content_array.extend(historical_screenshots)

                messages[1]["content"] = content_array
                has_charts = True
            except Exception as e:
                print(f"Warning: Could not load multi-timeframe charts: {e}")

        # Fallback to legacy single graph_image_path for backward compatibility
        elif graph_image_path and os.path.exists(graph_image_path):
            try:
                with open(graph_image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                content_array = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]

                # Add historical trade screenshots if available
                if trading_context:
                    from utils.trade_context import get_trade_screenshots_for_vision
                    historical_screenshots = get_trade_screenshots_for_vision(trading_context)
                    if historical_screenshots:
                        print(f"  Including {len(historical_screenshots)} historical trade screenshots")
                        content_array.extend(historical_screenshots)

                messages[1]["content"] = content_array
                has_charts = True
            except Exception as e:
                print(f"Warning: Could not load graph image: {e}")

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 with vision capabilities
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1000
        )

        # Extract and parse the response
        response_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        analysis_result = json.loads(response_text)

        # Add metadata
        analysis_result['symbol'] = symbol
        analysis_result['analyzed_at'] = time.time()
        analysis_result['model_used'] = response.model

        # Override buy_amount_usd with volatility-adjusted position sizing if enabled
        if config and range_percentage_from_min is not None and trading_context:
            from utils.dynamic_refresh import calculate_volatility_adjusted_position_size

            # Get current USD value from trading context
            wallet_metrics = trading_context.get('wallet_metrics', {})
            current_usd_value = wallet_metrics.get('current_usd', 0)
            starting_capital = wallet_metrics.get('starting_capital_usd', 0)
            confidence_level = analysis_result.get('confidence_level', 'low')

            if current_usd_value > 0:
                adjusted_position = calculate_volatility_adjusted_position_size(
                    range_percentage_from_min=range_percentage_from_min,
                    starting_capital_usd=starting_capital,
                    current_usd_value=current_usd_value,
                    confidence_level=confidence_level,
                    config=config
                )

                # Override the LLM's buy_amount_usd with volatility-adjusted amount
                analysis_result['buy_amount_usd'] = adjusted_position
                analysis_result['position_sizing_method'] = 'volatility_adjusted'
                print(f"✓ Position size adjusted for volatility: ${adjusted_position:.2f}")

        print(f"✓ OpenAI analysis completed for {symbol}")
        return analysis_result

    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse OpenAI response as JSON: {e}")
        print(f"Response text: {response_text}")
        return None
    except Exception as e:
        print(f"ERROR: OpenAI API call failed: {e}")
        return None


def save_analysis_to_file(symbol, analysis_data):
    """
    Saves the analysis data to a local file in the analysis folder.

    Args:
        symbol: The trading pair symbol (e.g., 'XLM-USD')
        analysis_data: Dictionary containing the analysis results

    Returns:
        The file path where the analysis was saved, or None on failure
    """
    try:
        # Create analysis directory if it doesn't exist
        analysis_dir = 'analysis'
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            print(f"Created analysis directory: {analysis_dir}")

        # Create filename based on symbol
        filename = f"{symbol.replace('-', '_')}_analysis.json"
        file_path = os.path.join(analysis_dir, filename)

        # Save the analysis
        with open(file_path, 'w') as f:
            json.dump(analysis_data, f, indent=4)

        print(f"✓ Analysis saved to: {file_path}")
        return file_path

    except Exception as e:
        print(f"ERROR: Failed to save analysis: {e}")
        return None


def load_analysis_from_file(symbol):
    """
    Loads the analysis data from a local file.

    Args:
        symbol: The trading pair symbol (e.g., 'XLM-USD')

    Returns:
        Dictionary containing the analysis data, or None if file doesn't exist or is invalid
    """
    try:
        analysis_dir = 'analysis'
        filename = f"{symbol.replace('-', '_')}_analysis.json"
        file_path = os.path.join(analysis_dir, filename)

        if not os.path.exists(file_path):
            return None

        with open(file_path, 'r') as f:
            file_content = f.read().strip()

            # Check if file is empty
            if not file_content:
                print(f"WARNING: Analysis file for {symbol} exists but is empty: {file_path}")
                return None

            analysis_data = json.loads(file_content)

            # Validate that we got a dictionary with required fields
            if not isinstance(analysis_data, dict):
                print(f"WARNING: Analysis file for {symbol} contains invalid data (not a dictionary)")
                return None

            # Check for required fields
            required_fields = ['buy_in_price', 'profit_target_percentage', 'trade_recommendation']
            missing_fields = [field for field in required_fields if field not in analysis_data]

            if missing_fields:
                print(f"WARNING: Analysis file for {symbol} is missing required fields: {missing_fields}")
                return None

        return analysis_data

    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse analysis file for {symbol} - invalid JSON: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load analysis for {symbol}: {e}")
        return None


def delete_analysis_file(symbol):
    """
    Deletes the AI analysis file for a given symbol.
    This should be called after a sell order is placed to clear the analysis record.

    Args:
        symbol: The trading pair symbol (e.g., 'XLM-USD')

    Returns:
        Boolean indicating whether the file was successfully deleted
    """
    try:
        analysis_dir = 'analysis'
        filename = f"{symbol.replace('-', '_')}_analysis.json"
        file_path = os.path.join(analysis_dir, filename)

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Deleted AI analysis file for {symbol}: {file_path}")
            return True
        else:
            print(f"No analysis file found for {symbol} to delete")
            return False

    except Exception as e:
        print(f"ERROR: Failed to delete analysis file for {symbol}: {e}")
        return False


def should_refresh_analysis(symbol, last_order_type, no_trade_refresh_hours=1, low_confidence_wait_hours=2, medium_confidence_wait_hours=1, coin_data=None, config=None):
    """
    Determines if a new analysis should be performed, considering both time-based
    and dynamic market condition triggers.

    Args:
        symbol: The trading pair symbol
        last_order_type: The type of the last order ('none', 'buy', 'sell', 'placeholder')
        no_trade_refresh_hours: Hours to wait before refreshing a 'no_trade' analysis (default: 1)
        low_confidence_wait_hours: Hours to wait before refreshing a 'low' confidence analysis (default: 2)
        medium_confidence_wait_hours: Hours to wait before refreshing a 'medium' confidence analysis (default: 1)
        coin_data: Optional dict with current price and volume data for dynamic refresh checks
        config: Optional config dict for dynamic refresh settings

    Returns:
        Boolean indicating whether to refresh the analysis
    """
    # Load existing analysis
    existing_analysis = load_analysis_from_file(symbol)

    # If no analysis exists, we need to create one
    if not existing_analysis:
        return True

    # If we're holding a position (last order was 'buy'), keep existing analysis
    # Don't refresh while we're in a position - maintain the original stop loss and profit targets
    # that were used to enter the trade. We're just waiting for sell conditions to be met.
    if last_order_type == 'buy':
        return False

    # For 'placeholder' (pending orders), don't refresh
    if last_order_type == 'placeholder':
        return False

    # Check if the analysis recommended 'no_trade'
    trade_recommendation = existing_analysis.get('trade_recommendation', 'buy')
    if trade_recommendation == 'no_trade':
        # First check if dynamic conditions warrant immediate refresh
        if coin_data and config:
            from utils.dynamic_refresh import should_trigger_dynamic_refresh
            should_refresh, reason = should_trigger_dynamic_refresh(
                symbol=symbol,
                current_price=coin_data.get('current_price'),
                price_history=coin_data.get('coin_prices_list', []),
                volume_history=coin_data.get('coin_volume_24h_LIST', []),
                current_volume=coin_data.get('current_volume_24h'),
                analysis=existing_analysis,
                config=config
            )
            if should_refresh:
                print(f"Dynamic refresh triggered for no_trade analysis: {reason}")
                return True

        # Check how old the analysis is
        analyzed_at = existing_analysis.get('analyzed_at', 0)
        current_time = time.time()
        hours_since_analysis = (current_time - analyzed_at) / 3600

        if hours_since_analysis >= no_trade_refresh_hours:
            print(f"Analysis is {hours_since_analysis:.1f} hours old and recommended no_trade. Refreshing...")
            return True
        else:
            print(f"Analysis recommended no_trade {hours_since_analysis:.1f} hours ago. Will refresh in {no_trade_refresh_hours - hours_since_analysis:.1f} hours.")
            return False

    # Check confidence level and apply wait times
    confidence_level = existing_analysis.get('confidence_level', 'low')
    analyzed_at = existing_analysis.get('analyzed_at', 0)
    current_time = time.time()
    hours_since_analysis = (current_time - analyzed_at) / 3600

    # Low confidence: wait 2 hours before re-analyzing
    if confidence_level == 'low':
        if hours_since_analysis >= low_confidence_wait_hours:
            print(f"Analysis is {hours_since_analysis:.1f} hours old with LOW confidence. Refreshing...")
            return True
        else:
            print(f"Analysis has LOW confidence from {hours_since_analysis:.1f} hours ago. Will refresh in {low_confidence_wait_hours - hours_since_analysis:.1f} hours.")
            return False

    # Medium confidence: wait 1 hour before re-analyzing
    elif confidence_level == 'medium':
        if hours_since_analysis >= medium_confidence_wait_hours:
            print(f"Analysis is {hours_since_analysis:.1f} hours old with MEDIUM confidence. Refreshing...")
            return True
        else:
            print(f"Analysis has MEDIUM confidence from {hours_since_analysis:.1f} hours ago. Will refresh in {medium_confidence_wait_hours - hours_since_analysis:.1f} hours.")
            return False

    # High confidence: proceed immediately, no waiting
    # If we just sold, we should have deleted the analysis file, so we'd be caught
    # by the "not existing_analysis" check above
    # If analysis exists and we have 'none' or 'sell' status, keep using existing analysis
    # Only refresh if explicitly deleted or doesn't exist
    if last_order_type in ['none', 'sell']:
        return False

    # Default: don't refresh
    return False
