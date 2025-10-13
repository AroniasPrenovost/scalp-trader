# utils/llm_analysis.py

import os
import base64
from openai import OpenAI
from typing import Dict, List, Optional
import json

# end boilerplate

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_trading_opportunity(
    screenshot_paths: List[str],
    market_data: Dict,
    tax_rate: float,
    current_price: float
) -> Dict:
    """
    Use OpenAI's GPT-4 Vision to analyze trading charts and market data.

    Args:
        screenshot_paths: List of paths to chart screenshot PNG files
        market_data: Dictionary containing market information (volume, price changes, etc.)
        tax_rate: Federal tax rate as a decimal (e.g., 0.15 for 15%)
        current_price: Current asset price

    Returns:
        Dictionary with trading recommendations:
        {
            'buy_price': float,
            'sell_price': float,
            'continue_waiting_for_setup': bool,
            'ready_to_trade': bool,
            'reasoning': str
        }
    """
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # Prepare image data
    image_contents = []
    for screenshot_path in screenshot_paths:
        base64_image = encode_image_to_base64(screenshot_path)
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })

    # Construct the prompt
    prompt = f"""You are an expert cryptocurrency trading analyst. Analyze the provided price charts and market data to provide trading recommendations.

Market Data:
{json.dumps(market_data, indent=2)}

Current Price: ${current_price}
Tax Rate: {tax_rate * 100}%

Based on the charts and data, provide your analysis in the following JSON format:
{{
    "buy_price": <recommended buy price as float>,
    "sell_price": <recommended sell price (accounting for taxes and fees) as float>,
    "continue_waiting_for_setup": <true if we should wait for a better setup, false otherwise>,
    "ready_to_trade": <true if conditions are favorable for trading, false otherwise>,
    "reasoning": "<brief explanation of your analysis>"
}}

Consider:
1. Price position within trading range
2. Volume trends and patterns
3. Support/resistance levels visible in the charts
4. The sell price must account for the {tax_rate * 100}% tax rate to ensure profitability
5. Risk/reward ratio

Respond ONLY with valid JSON."""

    # Create message content with text and images
    message_content = [{"type": "text", "text": prompt}] + image_contents

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-turbo" depending on your needs
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent analysis
        )

        # Parse the response
        response_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()

        result = json.loads(response_text)

        return result

    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        # Return safe defaults on error
        return {
            'buy_price': 0,
            'sell_price': 0,
            'continue_waiting_for_setup': True,
            'ready_to_trade': False,
            'reasoning': f'Error occurred: {str(e)}'
        }


def analyze_position_management(
    screenshot_paths: List[str],
    buy_price: float,
    current_price: float,
    tax_rate: float,
    exchange_fee_rate: float,
    position_size: float
) -> Dict:
    """
    Use OpenAI's GPT-4 Vision to determine if a position should be held or sold.

    Args:
        screenshot_paths: List of paths to chart screenshot PNG files
        buy_price: Original entry price
        current_price: Current asset price
        tax_rate: Federal tax rate as a decimal (e.g., 0.15 for 15%)
        exchange_fee_rate: Exchange fee rate (e.g., 0.006 for 0.6%)
        position_size: Number of shares/units held

    Returns:
        Dictionary with position management recommendations:
        {
            'action': str ('hold', 'sell', 'adjust_target'),
            'adjusted_sell_price': float or None,
            'urgency': str ('low', 'medium', 'high'),
            'reasoning': str
        }
    """
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # Calculate breakeven price
    buy_cost = buy_price * position_size * (1 + exchange_fee_rate)
    breakeven_price = buy_price * (1 + exchange_fee_rate + tax_rate)

    # Prepare image data
    image_contents = []
    for screenshot_path in screenshot_paths:
        base64_image = encode_image_to_base64(screenshot_path)
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })

    # Construct the prompt
    prompt = f"""You are an expert cryptocurrency trading analyst specializing in position management and risk assessment.

Position Details:
- Entry Price: ${buy_price}
- Current Price: ${current_price}
- Breakeven Price (including fees & taxes): ${breakeven_price:.4f}
- Position Size: {position_size}
- Tax Rate: {tax_rate * 100}%
- Exchange Fee: {exchange_fee_rate * 100}%

Current P&L: ${(current_price - buy_price) * position_size:.2f} ({((current_price - buy_price) / buy_price * 100):.2f}%)

Analyze the price charts to determine if we should:
1. HOLD - Continue holding the position
2. SELL - Exit the position immediately to minimize losses or lock in gains
3. ADJUST_TARGET - Modify the sell target based on new information

Provide your analysis in the following JSON format:
{{
    "action": "<hold, sell, or adjust_target>",
    "adjusted_sell_price": <new recommended sell price as float, or null if not adjusting>,
    "urgency": "<low, medium, or high>",
    "reasoning": "<detailed explanation of your recommendation>"
}}

Consider:
1. Is the price approaching or below breakeven?
2. Are there signs of trend reversal?
3. Is there strong support nearby that might hold?
4. Would it be better to take a small loss now vs. risk a larger loss?
5. Are there signs of continuation that justify holding?

Respond ONLY with valid JSON."""

    # Create message content with text and images
    message_content = [{"type": "text", "text": prompt}] + image_contents

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            max_tokens=800,
            temperature=0.3
        )

        # Parse the response
        response_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()

        result = json.loads(response_text)

        return result

    except Exception as e:
        print(f"Error in position management analysis: {e}")
        # Return safe defaults on error
        return {
            'action': 'hold',
            'adjusted_sell_price': None,
            'urgency': 'low',
            'reasoning': f'Error occurred: {str(e)}'
        }
