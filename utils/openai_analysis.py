import os
import json
import time
import base64
from openai import OpenAI
from io import BytesIO

def analyze_market_with_openai(symbol, coin_data, taker_fee_percentage=0, tax_rate_percentage=0, min_profit_target_percentage=3.0, chart_paths=None, trading_context=None, graph_image_path=None):
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
        taker_fee_percentage: Exchange taker fee as a percentage (e.g., 0.6 for 0.6%)
        tax_rate_percentage: Federal tax rate as a percentage (e.g., 37 for 37%)
        min_profit_target_percentage: Minimum profit target percentage (e.g., 3.0 for 3%)
        chart_paths: Optional dictionary with paths to multi-timeframe charts {'short_term': path, 'medium_term': path, 'long_term': path}
        trading_context: Optional dictionary containing historical trading context from build_trading_context()
        graph_image_path: DEPRECATED - use chart_paths instead. Optional path to a single graph image

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
    total_fee_percentage = taker_fee_percentage * 2  # Buy fee + Sell fee
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
You are being provided with FOUR charts showing different timeframes:

1H CHART (1 hour) - Micro Execution Timing:
- Identify precise entry zone and immediate micro support/resistance
- Look for recent momentum shifts and very short-term patterns
- Check RSI for immediate oversold (<30) or overbought (>70) conditions
- Note if price is bouncing off key technical levels in the last hour
- Assess immediate price action quality (clean moves vs choppy)

4H CHART (4 hours) - Primary Execution Chart:
- Identify precise entry zone (not just single price point)
- Note intraday support/resistance levels forming over last 4 hours
- Check if price is at key Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Assess whether current price offers favorable risk/reward
- Look for candlestick patterns (doji, hammer, engulfing) at key levels
- Verify volume patterns support the directional bias
- This is your PRIMARY execution timeframe

1D CHART (1 day / 24 hours) - Trend Confirmation:
- Confirm trend direction (higher highs/higher lows = uptrend, lower highs/lower lows = downtrend)
- Check if we're in a pullback within uptrend or a breakout scenario
- Verify RSI isn't showing bearish divergence (price makes higher high, RSI makes lower high)
- Ensure volume supports the move (increasing volume on uptrends, decreasing on pullbacks)
- Identify swing highs/lows for position context
- Confirm daily trend aligns with 4H setup

1W CHART (1 week / 7 days) - Macro Context:
- Are we near weekly highs/lows? (extreme = potential mean reversion risk)
- Identify major resistance zones that could cap upside potential
- Check if weekly trend aligns with intended trade direction
- Determine if symbol is in accumulation phase, distribution phase, or trending
- Note weekly support/resistance levels that could impact trade
- Assess overall market structure and position in larger cycle

MULTI-TIMEFRAME DECISION LOGIC:
✓ All 4 timeframes bullish + volume confirmation = HIGH confidence long candidate
✓ Weekly uptrend + daily uptrend + 4H pullback to support + 1H showing reversal = HIGH confidence entry
✓ Weekly uptrend + 4H pullback to key Fibonacci level + 1H momentum shift = HIGH confidence entry
✗ Timeframe conflict (e.g., 4H/1H bullish but 1D/1W bearish) = NO TRADE (wait for alignment)
✗ Price approaching major weekly resistance = NO TRADE or significantly reduce confidence
✗ Weekly downtrend + 4H bounce = Counter-trend risk, NO TRADE unless exceptional setup

Base your primary trading decision on the 4H chart, but REQUIRE validation from 1D and 1W context. Use 1H for fine-tuning entry timing.
"""

    prompt = f"""Analyze the following market data for {symbol} and provide a technical analysis with specific trading levels.
{historical_context_section}{timeframe_context}

Market Data:
- Current Price: ${current_price}
- Price Range: ${min_price} - ${max_price}
- Average Price: ${avg_price}
- Number of Data Points: {len(prices)}
- Recent Price Trend: {prices[-20:] if len(prices) >= 20 else prices}
- Current 24h Volume: {coin_data.get('current_volume_24h', 0)}
- Average 24h Volume: {sum(volumes) / len(volumes) if volumes else 0}
- Min/Max 24h Volume: {min(volumes) if volumes else 0} / {max(volumes) if volumes else 0}
- Recent Volume Trend: {volumes[-20:] if len(volumes) >= 20 else volumes}

VOLUME ANALYSIS REQUIREMENTS:
- Assess if current volume is above or below recent average
- Breakouts/breakdowns on LOW volume = likely fakeout (reduce confidence significantly)
- Support bounces with VOLUME SPIKE = strong reversal signal (increase confidence)
- Sustained moves with increasing volume = healthy trend continuation
- Include "volume_confirmation" field: true if volume supports the trade setup, false otherwise

Trading Costs (IMPORTANT - Factor these into your recommendations):
- Exchange Taker Fee: {taker_fee_percentage}% per trade ({total_fee_percentage}% total for buy + sell)
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
    "volume_confirmation": <true if volume supports trade setup, false otherwise>,
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
- volume_confirmation: Boolean (true/false)
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

4. POSITION SIZING (REQUIRED): Use the Portfolio Status data to determine buy_amount_usd:
   - NEVER commit more than 75% of current_usd value to a single trade - ALWAYS keep at least 25% in reserve
   - Portfolio metrics are provided for LEARNING CONTEXT only - to help you understand what worked/didn't work in past trades
   - DO NOT let portfolio performance (profit/loss) influence your position sizing - maintain the SAME disciplined approach whether up or down
   - Position sizing based SOLELY on THIS trade's quality and confidence:
     * HIGH confidence: 50-75% of current_usd (only for exceptional setups meeting ALL criteria below)
     * MEDIUM confidence: DO NOT TRADE - wait for higher quality opportunities
     * LOW confidence: DO NOT TRADE - recommend "no_trade" instead
   - Prioritize QUALITY over QUANTITY - focus on high-likelihood trades with strong technical confirmation
   - Learn from historical trades (entry/exit timing, market conditions, what worked) but DO NOT increase risk due to past success
   - Maintain strict risk discipline regardless of win streaks or losing streaks

5. TRADE INVALIDATION: Set trade_invalidation_price to a level where if breached, the entire trade thesis is wrong
   - Typically below stop_loss by a small margin
   - Example: If support is $0.95, stop is $0.94, invalidation might be $0.93

CONFIDENCE LEVEL CRITERIA (OBJECTIVE RUBRIC):

HIGH confidence requires ALL of the following:
✓ All 3 timeframes (24h, 7d, 90d) aligned in same direction
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
                "content": f"""You are an expert cryptocurrency technical analyst with 10+ years of experience in swing trading and scalping. Your analysis must be:

1. QUANTITATIVE: Use specific price levels, percentages, and ratios - no vague language
2. RISK-AWARE: Every trade must have defined stop loss and minimum 2:1 reward/risk ratio
3. DISCIPLINED: Reject marginal setups - only trade HIGH conviction opportunities that meet ALL criteria
4. ADAPTIVE: Learn from historical performance but prioritize current technical setup (70% current / 30% historical)
5. COST-CONSCIOUS: Account for {total_fee_percentage}% fees + {tax_rate_percentage}% tax in ALL profit calculations
6. CONSERVATIVE: When in doubt, recommend "no_trade" - protecting capital is the priority

Your reputation depends on accuracy and risk management. Be conservative, precise, and systematic.
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

                # Add charts in order: 4h (high detail - primary execution), 1h (high - timing), 1d (low), 1w (low)
                # This prioritizes the most important timeframes while saving tokens on context
                timeframe_order = ['4h', '1h', '1d', '1w']
                detail_levels = {'4h': 'high', '1h': 'high', '1d': 'low', '1w': 'low'}

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
                            print(f"  Added {timeframe} chart with {detail_levels[timeframe]} detail")

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


def should_refresh_analysis(symbol, last_order_type, no_trade_refresh_hours=1, low_confidence_wait_hours=2, medium_confidence_wait_hours=1):
    """
    Determines if a new analysis should be performed.

    Args:
        symbol: The trading pair symbol
        last_order_type: The type of the last order ('none', 'buy', 'sell', 'placeholder')
        no_trade_refresh_hours: Hours to wait before refreshing a 'no_trade' analysis (default: 1)
        low_confidence_wait_hours: Hours to wait before refreshing a 'low' confidence analysis (default: 2)
        medium_confidence_wait_hours: Hours to wait before refreshing a 'medium' confidence analysis (default: 1)

    Returns:
        Boolean indicating whether to refresh the analysis
    """
    # Load existing analysis
    existing_analysis = load_analysis_from_file(symbol)

    # If no analysis exists, we need to create one
    if not existing_analysis:
        return True

    # If we're holding a position (last order was 'buy'), keep existing analysis
    # Don't refresh while we're in a position
    if last_order_type == 'buy':
        return False

    # For 'placeholder' (pending orders), don't refresh
    if last_order_type == 'placeholder':
        return False

    # Check if the analysis recommended 'no_trade'
    trade_recommendation = existing_analysis.get('trade_recommendation', 'buy')
    if trade_recommendation == 'no_trade':
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
