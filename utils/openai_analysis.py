import os
import json
import time
import base64
from openai import OpenAI
from io import BytesIO

def analyze_market_with_openai(symbol, coin_data, taker_fee_percentage=0, tax_rate_percentage=0, graph_image_path=None):
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
        graph_image_path: Optional path to a graph image for visual analysis

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
    price_changes = coin_data.get('coin_price_percentage_change_24h_LIST', [])

    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    avg_price = sum(prices) / len(prices) if prices else 0

    # Calculate total cost burden (fees on both buy and sell, plus taxes on profit)
    total_fee_percentage = taker_fee_percentage * 2  # Buy fee + Sell fee
    total_cost_burden = total_fee_percentage + tax_rate_percentage

    # Build the prompt
    prompt = f"""Analyze the following market data for {symbol} and provide a technical analysis with specific trading levels.

Market Data:
- Current Price: ${current_price}
- Price Range: ${min_price} - ${max_price}
- Average Price: ${avg_price}
- Number of Data Points: {len(prices)}
- Recent Price Trend: {prices[-20:] if len(prices) >= 20 else prices}
- Current 24h Volume: {coin_data.get('current_volume_24h', 0)}
- 24h Price Change: {coin_data.get('coin_price_percentage_change_24h_LIST', [])[-1] if coin_data.get('coin_price_percentage_change_24h_LIST') else 0}%

Trading Costs (IMPORTANT - Factor these into your recommendations):
- Exchange Taker Fee: {taker_fee_percentage}% per trade ({total_fee_percentage}% total for buy + sell)
- Tax Rate on Profits: {tax_rate_percentage}%
- Minimum Profitable Trade: Must exceed ~{total_cost_burden:.2f}% to break even after all costs

Please analyze this data and respond with a JSON object (ONLY valid JSON, no markdown code blocks) containing:
{{
    "major_resistance": <price level>,
    "minor_resistance": <price level>,
    "major_support": <price level>,
    "minor_support": <price level>,
    "buy_in_price": <recommended buy price>,
    "sell_price": <recommended sell price>,
    "profit_target_percentage": <recommended NET profit percentage AFTER all fees and taxes - must be positive>,
    "stop_loss": <recommended stop loss price>,
    "confidence_level": <"high", "medium", or "low">,
    "market_trend": <"bullish", "bearish", or "sideways">,
    "reasoning": <brief explanation of the analysis>
}}

CRITICAL: The profit_target_percentage MUST account for all trading costs. For example, if you recommend a 3% profit target, the trader should NET 3% profit AFTER paying {total_fee_percentage}% in exchange fees and {tax_rate_percentage}% in taxes. Calculate accordingly - the gross price movement must be larger than the net profit target to cover all costs.

Base your analysis on technical indicators, support/resistance levels, and price action."""

    try:
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert cryptocurrency technical analyst. Provide precise numerical analysis based on the data provided. Always respond with valid JSON only, no markdown formatting."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # If graph image is provided, include it in the analysis
        if graph_image_path and os.path.exists(graph_image_path):
            try:
                with open(graph_image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                messages[1]["content"] = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
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
        Dictionary containing the analysis data, or None if file doesn't exist
    """
    try:
        analysis_dir = 'analysis'
        filename = f"{symbol.replace('-', '_')}_analysis.json"
        file_path = os.path.join(analysis_dir, filename)

        if not os.path.exists(file_path):
            return None

        with open(file_path, 'r') as f:
            analysis_data = json.load(f)

        return analysis_data

    except Exception as e:
        print(f"ERROR: Failed to load analysis: {e}")
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


def should_refresh_analysis(symbol, last_order_type):
    """
    Determines if a new analysis should be performed.

    Args:
        symbol: The trading pair symbol
        last_order_type: The type of the last order ('none', 'buy', 'sell', 'placeholder')

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

    # If we just sold, we should have deleted the analysis file, so we'd be caught
    # by the "not existing_analysis" check above
    # If analysis exists and we have 'none' or 'sell' status, keep using existing analysis
    # Only refresh if explicitly deleted or doesn't exist
    if last_order_type in ['none', 'sell']:
        return False

    # Default: don't refresh
    return False
