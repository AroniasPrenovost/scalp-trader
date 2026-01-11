import os
import json
import time
import base64
from openai import OpenAI
from io import BytesIO

def analyze_market_with_openai(symbol, coin_data, exchange_fee_percentage=0, tax_rate_percentage=0, min_profit_target_percentage=3.0, chart_paths=None, trading_context=None, graph_image_path=None, range_percentage_from_min=None, config=None):
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
        exchange_fee_percentage: Exchange fee as a percentage (e.g., 1.2 for 1.2% taker fee on market orders)
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
    # Using exchange fees (taker fees for market orders)
    total_fee_percentage = exchange_fee_percentage * 2  # Buy fee + Sell fee
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

IMPORTANT - MULTI-TIMEFRAME ANALYSIS FRAMEWORK FOR SCALPING:
You are being provided with SIX charts optimized for SCALPING strategy (targeting 0.8-1.0% NET profit moves).
NOTE: All charts use 1-hour interval data points - the difference is the lookback window (how much historical data is shown).

⚡ CRITICAL SCALPING EXECUTION MODEL:
This strategy uses MARKET ORDERS for immediate execution when current price reaches suggested entry.
- Your buy_in_price should be VERY CLOSE to current price (within 0.3-1.0%) if the setup is good NOW
- DO NOT predict future dips - if current price is NOT at a good technical level, recommend "no_trade"
- Target: 0.8-1.0% NET profit moves that complete within 1-4 hours of execution (QUICK SCALPS)
- The 4-HOUR and 72-HOUR charts are your PRIMARY tools for entry price decisions
- Use 14d/30d/90d/6mo charts ONLY for trend confirmation and major resistance identification

DATA STRUCTURE (ordered by priority for entry decisions):
- 4-hour chart: 4 hourly data points (last 4 hours) ← PRIMARY for micro entry timing
- 72-hour chart: 72 hourly data points (3 days) ← PRIMARY for recent momentum
- 14-day chart: ~336 hourly data points (14 days) ← SECONDARY for swing context
- 30-day chart: ~720 hourly data points (30 days) ← Trend confirmation only
- 90-day chart: ~2,160 hourly data points (90 days) ← Trend confirmation only
- 6-month chart: ~4,380 hourly data points (182.5 days) ← Major support/resistance only

4-HOUR CHART (Micro View) - Immediate Entry Timing ⚡ PRIMARY:
- This shows the MOST RECENT 4 hours of price action (4 hourly candles)
- Identify CURRENT price position relative to micro support/resistance
- Is price bouncing NOW? Breaking out NOW? Or stuck in chop?
- Look for immediate momentum: last 1-2 candles showing reversal or continuation
- This tells you if THIS MOMENT is a good entry point
- If current price is not at a clear technical level on this chart, recommend "no_trade"

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

72-HOUR CHART (Recent View) - Entry/Exit Timing & Context ⚡ PRIMARY:
- Shows past 3 days (72 hours) of hourly price action
- Identify precise entry zone and immediate micro support/resistance
- Look for recent momentum shifts or reversal signals over the past 72 hours
- Check if price is bouncing off key technical levels in this recent window
- Assess immediate risk/reward: is current price near support (good entry) or resistance (poor entry)?
- Note volume spikes in last 72 hours that signal buyer/seller interest
- Confirm RSI trends and extreme conditions with more data points for validation
- This provides short-term context for the 4-hour micro view

MULTI-TIMEFRAME DECISION LOGIC FOR SCALPING:
✓ 4h + 72h show good entry NOW + 14d/30d/90d/6mo trend aligned = HIGH confidence BUY
✓ 4h shows bounce at support + 72h confirms reversal + longer timeframes bullish = HIGH confidence BUY
✓ 4h shows breakout + 72h shows momentum + no major resistance ahead (30d/90d/6mo) = HIGH confidence BUY
✗ 4h shows price mid-range or at resistance = NO TRADE (wait for pullback or breakout)
✗ 72h shows choppy/sideways action = NO TRADE (wait for clear direction)
✗ 4h/72h bullish but 30d/90d/6mo show major resistance nearby (<1% away) = NO TRADE (quick scalps need clear path)
✗ Timeframe conflict (short-term bullish but medium/long-term bearish) = NO TRADE
✗ Current price is >1% away from nearest technical level on 4h/72h charts = NO TRADE (wait for setup)

SCALPING DECISION PRIORITY (0.8-1% TARGETS):
1. Check 4-HOUR chart: Is current price at a clear technical level RIGHT NOW? (bounce, breakout, support test)
2. Check 72-HOUR chart: Does recent momentum support the entry? Any immediate obstacles?
3. Check 14-DAY chart: Is the swing direction aligned?
4. Check 30d/90d/6mo charts: Any major resistance within 1% that would block profit target? (critical for small scalps)
5. If ALL above conditions pass → recommend BUY with buy_in_price near current price
6. If ANY condition fails → recommend "no_trade" and wait for better setup
7. REMEMBER: With 0.8-1% targets, you need VERY CLEAR path to profit - no resistance zones in the way
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

    # Check adaptive mean reversion strategy
    adaptive_strategy_context = ""
    adaptive_signal = None  # Initialize to None for later enforcement check
    try:
        from utils.adaptive_mean_reversion import check_adaptive_buy_signal

        adaptive_signal = check_adaptive_buy_signal(
            prices=prices,
            current_price=current_price
        )

        adaptive_strategy_context = f"""

ADAPTIVE MEAN REVERSION STRATEGY (CORE FOUNDATION - PROVEN PROFITABLE):
- Market Trend: {adaptive_signal['trend'].upper()}
- Strategy Signal: {adaptive_signal['signal'].upper()}
- Deviation from 24h MA: {adaptive_signal['deviation_from_ma']:+.2f}%

{adaptive_signal['reasoning']}

"""

        if adaptive_signal['signal'] == 'buy':
            adaptive_strategy_context += f"""
✅ AMR BUY SIGNAL DETECTED - YOUR ROLE: VALIDATE & ENHANCE

AMR Baseline (53.3% win rate, proven profitable):
- Entry Price: ${adaptive_signal['entry_price']:.4f}
- Stop Loss: ${adaptive_signal['stop_loss']:.4f} (-1.7%)
- Profit Target: ${adaptive_signal['profit_target']:.4f} (+1.7%)
- Risk/Reward: 1:1 (symmetric - but you should target 0.8-1% NET for quick scalps)

YOUR TASK AS AI VALIDATOR (FOR 0.8-1% SCALPS):
1. Analyze the multi-timeframe charts - do they CONFIRM or CONTRADICT this AMR signal?
2. Check for ANY resistance within 1% that could block our small profit target (critical for scalping)
3. Validate support levels align with the AMR entry/stop prices
4. Set confidence_level based on chart confirmation:
   - "high": Charts strongly confirm AMR signal + CLEAR 1% PATH (no resistance, clear support)
   - "medium": Charts neutral or mixed signals (some support, minor concerns)
   - "low": Charts show warning signs (resistance within 1%, weak support)

5. You may REFINE profit target to 0.8-1% NET if AMR target is too aggressive for current resistance
6. If charts show resistance within 1% of entry, set confidence to "low" or "no_trade"

IMPORTANT: For 0.8-1% targets, you need CRYSTAL CLEAR path to profit. Be very selective.

"""
        else:
            adaptive_strategy_context += f"""
⚠ NO AMR SIGNAL - DOWNTREND OR NO SETUP

AMR Status: {adaptive_signal['reasoning']}

YOUR TASK AS AI VALIDATOR:
- If market trend is DOWNTREND: Set trade_recommendation to "no_trade" (enforce downtrend filter)
- If no AMR signal but you see EXCEPTIONAL chart setup: You may recommend "buy" but set confidence to "medium" (not "high")
- In most cases: Set trade_recommendation to "no_trade" and explain why in reasoning

DO NOT override downtrend filter unless you see exceptional multi-timeframe bullish reversal confirmation.

"""
    except Exception as e:
        # If adaptive strategy fails, continue without it
        adaptive_signal = None
        adaptive_strategy_context = f"\n⚠ Adaptive strategy failed to load: {str(e)}\n"

    # Calculate ATR (Average True Range) for volatility-based stop loss
    # Using recent 24 data points (24 hours of data) for ATR calculation
    atr_period = min(24, len(prices) - 1)
    if atr_period > 1:
        true_ranges = []
        for i in range(len(prices) - atr_period, len(prices)):
            if i > 0:
                true_ranges.append(abs(prices[i] - prices[i-1]))
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0
    else:
        atr = 0

    atr_percentage = (atr / current_price * 100) if current_price > 0 and atr > 0 else 0

    prompt = f"""Analyze the following market data for {symbol} and provide a technical analysis with specific trading levels.
{historical_context_section}{timeframe_context}{volatility_context}{adaptive_strategy_context}

Market Data:
- Current Price: ${current_price}
- Price Range: ${min_price} - ${max_price}
- Average Price: ${avg_price}
- Number of Data Points: {len(prices)}
- Recent Price Trend: {prices[-20:] if len(prices) >= 20 else prices}
- ATR (Average True Range): ${atr:.4f} ({atr_percentage:.2f}% of current price)

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
- Exchange Taker Fee: {exchange_fee_percentage}% per trade ({total_fee_percentage}% total for buy + sell)
  NOTE: Taker fees apply because we use MARKET ORDERS for guaranteed execution
- Tax Rate on Profits: {tax_rate_percentage}%
- Minimum Profitable Trade: Must exceed ~{total_cost_burden:.2f}% to break even after all costs
- TARGET PROFIT RANGE: 0.8-1.0% NET profit after all costs (QUICK SCALPING STRATEGY)
- To achieve 0.8% NET, you need ~1.5% GROSS price movement (accounting for fees + taxes)

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
- profit_target_percentage: MUST be 0.8-1.0% for quick scalping strategy (no higher, no lower)
- risk_reward_ratio: Must be >= 1.5 for any "buy" recommendation (relaxed for tight scalps)
- confidence_level: ENUM only ["high", "medium", "low"]
- trade_recommendation: ENUM only ["buy", "sell", "hold", "no_trade"]
- volume_trend: ENUM only ["increasing", "decreasing", "stable"]
- reasoning: Max 200 characters, factual only (no subjective language like "could", "might", "possibly")
- If confidence_level = "medium" OR "low" → trade_recommendation MUST be "no_trade"
- If trade_recommendation = "buy" → risk_reward_ratio MUST be >= 1.5 (for 0.8-1% scalps)
- sell_price must be > buy_in_price (no shorting allowed)
- stop_loss must be < buy_in_price
- trade_invalidation_price: Typically below stop_loss, the price where thesis breaks down completely

CRITICAL REQUIREMENTS:
1. RISK/REWARD RATIO: All "buy" recommendations MUST have risk_reward_ratio >= 1.5 (relaxed for tight scalps)
   - Calculate: (sell_price - buy_in_price) / (buy_in_price - stop_loss)
   - If ratio < 1.5, you MUST set trade_recommendation to "no_trade"
   - Example for 0.8% scalp: Buy $100, Sell $101.50 (1.5% gross), Stop $99.00 = (1.50/1.00) = 1.5 ratio ✓
   - IMPORTANT: Target 0.8-1.0% NET profit, which requires ~1.3-1.8% GROSS price movement

2. COST CALCULATION TRANSPARENCY: Show your work in reasoning field:
   - Example: "Buy $0.50, sell $0.52 = 4% gross, 2.8% net after costs"
   - Gross price movement must exceed net profit target to cover {total_fee_percentage}% fees + {tax_rate_percentage}% tax

3. POSITION SIZING (REQUIRED): Position size will be automatically calculated based on volatility.
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
   - IMPORTANT: Set buy_amount_usd to a reasonable value (e.g., current_usd * 0.5) as a fallback
   - The system may override your buy_amount_usd with volatility-adjusted calculations
   - Prioritize QUALITY over QUANTITY - focus on high-likelihood trades with strong technical confirmation

4. TRADE INVALIDATION: Set trade_invalidation_price to a level where if breached, the entire trade thesis is wrong
   - Typically below stop_loss by a small margin
   - Example: If support is $0.95, stop is $0.94, invalidation might be $0.93

CONFIDENCE LEVEL CRITERIA FOR 0.8-1% SCALPING (OBJECTIVE RUBRIC):

HIGH confidence requires ALL of the following:
✓ **IMMEDIATE ENTRY SETUP**: 4-hour chart shows current price at clear technical level (bounce, breakout, support/resistance test)
✓ **BUY_IN_PRICE PROXIMITY**: Your recommended buy_in_price must be within 0.3-1.0% of current price
✓ **RECENT MOMENTUM**: 72-hour chart confirms direction and shows clean price action (not choppy)
✓ **TIMEFRAME ALIGNMENT**: All timeframes (4h, 72h, 14d, 30d, 90d, 6mo) aligned in same direction
✓ **CLEAR PATH TO TARGET**: No resistance within 1.5% on 30d/90d/6mo charts (critical for small 0.8-1% targets!)
✓ **MULTI-TIMEFRAME SUPPORT/RESISTANCE**: Key level visible across at least 3 timeframes (e.g., 72h, 14d, 30d all show support at ~same level)
✓ **ENTRY POSITION**: Current price must be within 1% ABOVE support, NOT near resistance (prevents buying at peaks)
✓ **STOP LOSS VALIDATION**: Stop loss below clear technical level, tight (~1%) to maintain R:R for small targets
✓ Volume confirms the setup (above average on bullish setups, spike at support)
✓ Risk/reward ratio >= 1.5 (for 0.8-1% scalps, we need tight stops)
✓ RSI supports direction (not overbought >70 on longs)
✓ Multiple technical confirmations on 4h/72h charts (e.g., support bounce + volume + RSI reversal)

MEDIUM confidence (DO NOT TRADE - per requirement #4):
- Some but not all HIGH confidence criteria met
- Timeframe alignment weak or contradictory
- Risk/reward ratio 1.5-2.0 range
- Volume neutral or unclear
- Any resistance visible within 1.5% of entry (blocks our small target)
→ Set trade_recommendation to "no_trade"

LOW confidence (DO NOT TRADE - per requirement #4):
- Conflicting signals across timeframes
- Poor risk/reward ratio (< 1.5)
- Low volume, no technical confirmation
- Price in no-man's land (not at key levels)
- Resistance within 1% (will block 0.8-1% scalp target)
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

                # Add charts in SCALPING priority order: 4h (PRIMARY), 72h (PRIMARY), 14d, 30d, 90d, 6mo, volume
                # This prioritizes short-term charts for immediate entry decisions
                # 4h = micro entry timing (most critical), 72h = recent momentum, 14d = swing context, 30d/90d = trend, 6mo = major levels
                timeframe_order = ['4h', '72h', '14d', '30d', '90d', '6mo', 'volume_snapshots']
                detail_levels = {'4h': 'high', '72h': 'high', '14d': 'high', '30d': 'high', '90d': 'low', '6mo': 'low', 'volume_snapshots': 'low'}

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
        analysis_result['analysis_price'] = current_price  # Store price at time of analysis
        analysis_result['model_used'] = response.model

        # Store chart paths so they can be referenced later
        if chart_paths:
            analysis_result['chart_paths'] = chart_paths

        # TRACK AMR ALIGNMENT (for metrics and debugging)
        # Store AMR signal data alongside AI analysis for opportunity scorer to use
        if adaptive_signal:
            analysis_result['amr_signal_data'] = {
                'signal': adaptive_signal['signal'],
                'trend': adaptive_signal['trend'],
                'entry_price': adaptive_signal.get('entry_price'),
                'stop_loss': adaptive_signal.get('stop_loss'),
                'profit_target': adaptive_signal.get('profit_target'),
                'deviation_from_ma': adaptive_signal.get('deviation_from_ma'),
                'reasoning': adaptive_signal.get('reasoning')
            }

            # Log alignment between AMR and AI for analysis
            amr_buy = adaptive_signal['signal'] == 'buy'
            ai_buy = analysis_result.get('trade_recommendation') == 'buy'
            ai_conf = analysis_result.get('confidence_level', 'low')

            if amr_buy and ai_buy:
                print(f"✓ AMR + AI ALIGNED: Both recommend BUY (AI confidence: {ai_conf})")
            elif amr_buy and not ai_buy:
                print(f"⚠️  AMR says BUY but AI says {analysis_result.get('trade_recommendation').upper()} (conf: {ai_conf})")
                print(f"   AMR: {adaptive_signal['reasoning'][:100]}")
                print(f"   AI: {analysis_result.get('reasoning', 'No reasoning')[:100]}")
            elif not amr_buy and ai_buy:
                print(f"⚠️  AI says BUY but AMR says {adaptive_signal['signal'].upper()} (trend: {adaptive_signal['trend']})")
                print(f"   Note: Opportunity scorer will reject this (requires AMR buy signal)")
            else:
                print(f"✓ AMR + AI ALIGNED: Both recommend NO TRADE (AMR: {adaptive_signal['trend']}, AI: {ai_conf})")
        else:
            analysis_result['amr_signal_data'] = None
            print(f"⚠️  AMR signal unavailable - AI analysis proceeding without AMR context")

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

        # print(f"✓ Analysis saved to: {file_path}")
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


def adjust_analysis_for_actual_fill(symbol, original_analysis, actual_fill_price, current_price, chart_paths=None, exchange_fee_percentage=0, tax_rate_percentage=0):
    """
    Asks AI to re-analyze and adjust stop loss and profit targets based on actual fill price.
    Called when there's a significant delta (>3%) between AI's recommended price and actual fill.

    Args:
        symbol: The trading pair symbol
        original_analysis: The original AI analysis dictionary
        actual_fill_price: The actual price the order was filled at
        current_price: Current market price
        chart_paths: Dictionary with chart paths (uses 72h, 14d, 30d for context)
        exchange_fee_percentage: Exchange fee percentage (taker fee for market orders)
        tax_rate_percentage: Federal tax rate percentage

    Returns:
        Adjusted analysis dictionary with updated stop_loss, sell_price, and profit_target_percentage
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        return None

    client = OpenAI(api_key=api_key)

    # Extract key data from original analysis
    recommended_buy_price = original_analysis.get('buy_in_price')
    original_stop_loss = original_analysis.get('stop_loss')
    original_sell_price = original_analysis.get('sell_price')
    original_profit_pct = original_analysis.get('profit_target_percentage')

    # Calculate the delta
    fill_delta_pct = ((actual_fill_price - recommended_buy_price) / recommended_buy_price) * 100

    # Calculate original risk/reward percentages for reference
    original_risk_pct = ((recommended_buy_price - original_stop_loss) / recommended_buy_price) * 100
    original_reward_pct = ((original_sell_price - recommended_buy_price) / recommended_buy_price) * 100

    prompt = f"""POST-FILL ADJUSTMENT ANALYSIS for {symbol}

SITUATION:
- Original AI recommended buy price: ${recommended_buy_price}
- Actual fill price: ${actual_fill_price}
- Fill delta: {fill_delta_pct:+.2f}% ({'better' if fill_delta_pct < 0 else 'worse'} than expected)
- Current market price: ${current_price}

ORIGINAL ANALYSIS:
- Stop loss: ${original_stop_loss} ({original_risk_pct:.2f}% risk from recommended entry)
- Sell target: ${original_sell_price} ({original_reward_pct:.2f}% reward from recommended entry)
- Profit target: {original_profit_pct}% NET after fees/taxes
- Risk/Reward ratio: {original_analysis.get('risk_reward_ratio', 'N/A')}
- Confidence: {original_analysis.get('confidence_level', 'N/A')}
- Reasoning: {original_analysis.get('reasoning', 'N/A')}

SUPPORT/RESISTANCE LEVELS (from original analysis):
- Major support: ${original_analysis.get('major_support', 'N/A')}
- Minor support: ${original_analysis.get('minor_support', 'N/A')}
- Major resistance: ${original_analysis.get('major_resistance', 'N/A')}
- Minor resistance: ${original_analysis.get('minor_resistance', 'N/A')}

CHARTS PROVIDED:
- 72-hour chart: Shows immediate market context around the fill
- 14-day chart: Shows current swing structure and key levels
- 30-day chart: Validates support/resistance levels are still relevant

TASK:
Adjust the stop loss and profit targets based on the ACTUAL fill price of ${actual_fill_price}.
Review the charts to ensure adjusted levels respect current market structure.

REQUIREMENTS:
1. Maintain similar risk/reward ratio (~{original_analysis.get('risk_reward_ratio', 3.0)})
2. Ensure stop loss is BELOW actual entry (${actual_fill_price})
3. Respect support/resistance levels visible in the charts
4. Account for trading costs: {exchange_fee_percentage * 2}% fees + {tax_rate_percentage}% tax
5. If fill was better than expected, tighten stop loss proportionally while maintaining R/R
6. If fill was worse than expected, consider if trade thesis is still valid given current chart structure

Respond with ONLY valid JSON (no markdown):
{{
    "adjusted_stop_loss": <new stop loss price below ${actual_fill_price}>,
    "adjusted_sell_price": <new sell target price>,
    "adjusted_profit_target_percentage": <new NET profit % after costs>,
    "adjusted_risk_reward_ratio": <calculated as (sell - entry) / (entry - stop)>,
    "adjustment_reasoning": <brief explanation of changes, max 150 chars>
}}

VALIDATION:
- adjusted_stop_loss MUST be < ${actual_fill_price}
- adjusted_sell_price MUST be > ${actual_fill_price}
- adjusted_risk_reward_ratio MUST be >= 2.0"""

    try:
        # Prepare messages with charts
        messages = [
            {
                "role": "system",
                "content": "You are a technical analyst adjusting trade parameters based on actual fill prices. Output ONLY valid JSON with no markdown formatting."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Add charts if available (72h, 14d, 30d for context)
        if chart_paths and isinstance(chart_paths, dict):
            content_array = [{"type": "text", "text": prompt}]

            # Priority order for post-fill: 4h (immediate), 72h (recent), 14d (swing)
            timeframe_order = ['4h', '72h', '14d']

            for timeframe in timeframe_order:
                if timeframe in chart_paths and chart_paths[timeframe] and os.path.exists(chart_paths[timeframe]):
                    with open(chart_paths[timeframe], "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        content_array.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        })
                        print(f"  Added {timeframe} chart for post-fill adjustment")

            messages[1]["content"] = content_array

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )

        response_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        adjustment_result = json.loads(response_text)

        # Create updated analysis by copying original and updating key fields
        adjusted_analysis = original_analysis.copy()
        adjusted_analysis['stop_loss'] = adjustment_result['adjusted_stop_loss']
        adjusted_analysis['sell_price'] = adjustment_result['adjusted_sell_price']
        adjusted_analysis['profit_target_percentage'] = adjustment_result['adjusted_profit_target_percentage']
        adjusted_analysis['risk_reward_ratio'] = adjustment_result['adjusted_risk_reward_ratio']
        adjusted_analysis['buy_in_price'] = actual_fill_price  # Update to actual fill price

        # Add metadata about the adjustment
        adjusted_analysis['fill_adjustment'] = {
            'original_recommended_price': recommended_buy_price,
            'actual_fill_price': actual_fill_price,
            'fill_delta_percentage': fill_delta_pct,
            'adjustment_reasoning': adjustment_result['adjustment_reasoning'],
            'adjusted_at': time.time()
        }

        print(f"✓ AI post-fill adjustment completed:")
        print(f"  Original stop: ${original_stop_loss:.2f} → Adjusted: ${adjustment_result['adjusted_stop_loss']:.2f}")
        print(f"  Original target: ${original_sell_price:.2f} → Adjusted: ${adjustment_result['adjusted_sell_price']:.2f}")
        print(f"  Reasoning: {adjustment_result['adjustment_reasoning']}")

        return adjusted_analysis

    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse AI adjustment response: {e}")
        print(f"Response text: {response_text}")
        return None
    except Exception as e:
        print(f"ERROR: AI post-fill adjustment failed: {e}")
        return None


def should_refresh_analysis(symbol, last_order_type, no_trade_refresh_hours=1, low_confidence_wait_hours=1, medium_confidence_wait_hours=1, high_confidence_max_age_hours=2, coin_data=None, config=None):
    """
    Determines if a new analysis should be performed, considering both time-based
    and dynamic market condition triggers.

    Args:
        symbol: The trading pair symbol
        last_order_type: The type of the last order ('none', 'buy', 'sell', 'placeholder')
        no_trade_refresh_hours: Hours to wait before refreshing a 'no_trade' analysis (default: 1)
        low_confidence_wait_hours: Hours to wait before refreshing a 'low' confidence analysis (default: 1)
        medium_confidence_wait_hours: Hours to wait before refreshing a 'medium' confidence analysis (default: 1)
        high_confidence_max_age_hours: Hours after which high confidence analyses expire (default: 2)
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
        # Check how old the analysis is first
        analyzed_at = existing_analysis.get('analyzed_at', 0)
        current_time = time.time()
        hours_since_analysis = (current_time - analyzed_at) / 3600

        # Enforce minimum cooldown period before allowing dynamic refresh
        # This prevents constant refreshes immediately after a fresh analysis
        # Dynamic refresh checks RSI extremes, volume spikes, price changes, etc.
        minimum_cooldown_hours = config.get('no_trade_dynamic_refresh_min_cooldown_hours', 2.25) if config else 2.25

        if hours_since_analysis < minimum_cooldown_hours:
            # Too soon after last analysis - skip dynamic checks entirely
            # print(f"Analysis recommended no_trade {hours_since_analysis:.1f} hours ago. Will refresh in {no_trade_refresh_hours - hours_since_analysis:.1f} hours.")
            return False

        # After minimum cooldown, check if dynamic conditions warrant immediate refresh
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

        # Check if time-based refresh is needed
        if hours_since_analysis >= no_trade_refresh_hours:
            print(f"Analysis is {hours_since_analysis:.1f} hours old and recommended no_trade. Refreshing...")
            return True
        else:
            # print(f"Analysis recommended no_trade {hours_since_analysis:.1f} hours ago. Will refresh in {no_trade_refresh_hours - hours_since_analysis:.1f} hours.")
            return False

    # Check confidence level and apply wait times
    confidence_level = existing_analysis.get('confidence_level', 'low')
    analyzed_at = existing_analysis.get('analyzed_at', 0)
    current_time = time.time()
    hours_since_analysis = (current_time - analyzed_at) / 3600

    # Low confidence: wait 1 hour before re-analyzing (faster to catch improving conditions)
    if confidence_level == 'low':
        if hours_since_analysis >= low_confidence_wait_hours:
            print(f"Analysis is {hours_since_analysis:.1f} hours old with LOW confidence. Refreshing...")
            return True
        else:
            # print(f"Analysis has LOW confidence from {hours_since_analysis:.1f} hours ago. Will refresh in {low_confidence_wait_hours - hours_since_analysis:.1f} hours.")
            return False

    # Medium confidence: wait 1 hour before re-analyzing
    elif confidence_level == 'medium':
        if hours_since_analysis >= medium_confidence_wait_hours:
            print(f"Analysis is {hours_since_analysis:.1f} hours old with MEDIUM confidence. Refreshing...")
            return True
        else:
            # print(f"Analysis has MEDIUM confidence from {hours_since_analysis:.1f} hours ago. Will refresh in {medium_confidence_wait_hours - hours_since_analysis:.1f} hours.")
            return False

    # High confidence: Check if it's too old and needs refresh
    # Use the configured max age from the parameter (don't hardcode)
    analyzed_at = existing_analysis.get('analyzed_at', 0)
    current_time = time.time()
    hours_since_analysis = (current_time - analyzed_at) / 3600

    if hours_since_analysis >= high_confidence_max_age_hours:
        print(f"Analysis is {hours_since_analysis:.1f} hours old with HIGH confidence. Refreshing due to age...")
        return True

    # If not too old, keep using existing high confidence analysis
    # If we just sold, we should have deleted the analysis file, so we'd be caught
    # by the "not existing_analysis" check above
    if last_order_type in ['none', 'sell']:
        return False

    # Default: don't refresh
    return False
