import os
import json
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Optional

def load_transaction_history(symbol: str) -> List[Dict]:
    """
    Load transaction history from transactions/data.json filtered by symbol.

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')

    Returns:
        List of transaction records for the specified symbol
    """
    try:
        file_path = os.path.join("transactions", "data.json")

        if not os.path.exists(file_path):
            print(f"No transaction history found at {file_path}")
            return []

        with open(file_path, 'r') as f:
            all_transactions = json.load(f)

        # Filter transactions for this symbol
        symbol_transactions = [t for t in all_transactions if t.get('symbol') == symbol]

        # Sort by timestamp (most recent first)
        symbol_transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return symbol_transactions

    except Exception as e:
        print(f"ERROR: Failed to load transaction history: {e}")
        return []


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string or None if failed
    """
    try:
        if not os.path.exists(image_path):
            return None

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        print(f"Warning: Could not encode image {image_path}: {e}")
        return None


def calculate_portfolio_metrics(symbol: str, starting_capital_usd: float) -> Dict:
    """
    Calculate portfolio performance metrics for a symbol.

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')
        starting_capital_usd: The initial capital allocated to this symbol

    Returns:
        Dictionary containing:
        {
            'starting_capital_usd': float,
            'current_usd': float,
            'percentage_gain': float,
            'total_profit': float
        }
    """
    transactions = load_transaction_history(symbol)

    # Calculate total profit from all completed trades
    total_profit = sum(t.get('total_profit', 0) for t in transactions)

    # Current USD = starting capital + all profits/losses
    current_usd = starting_capital_usd + total_profit

    # Calculate percentage gain
    percentage_gain = ((current_usd - starting_capital_usd) / starting_capital_usd * 100) if starting_capital_usd > 0 else 0

    return {
        'starting_capital_usd': starting_capital_usd,
        'current_usd': current_usd,
        'percentage_gain': percentage_gain,
        'total_profit': total_profit
    }


def build_trading_context(symbol: str, max_trades: int = 10, include_screenshots: bool = True, starting_capital_usd: Optional[float] = None) -> Dict:
    """
    Build contextual information from past trades for LLM analysis.

    This function creates a comprehensive context object containing:
    - Historical trade performance for the symbol
    - Buy event screenshots (if available and enabled)
    - Performance metrics and patterns
    - Portfolio metrics (if starting_capital_usd is provided)

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')
        max_trades: Maximum number of recent trades to include (default: 10)
        include_screenshots: Whether to include base64-encoded screenshots (default: True)
        starting_capital_usd: Initial capital allocated to this symbol (optional)

    Returns:
        Dictionary containing trading context with the following structure:
        {
            'symbol': str,
            'total_trades': int,
            'trades_included': int,
            'performance_summary': {
                'total_profit': float,
                'average_profit_percentage': float,
                'win_rate': float,
                'profitable_trades': int,
                'losing_trades': int
            },
            'portfolio_metrics': {  # Only included if starting_capital_usd is provided
                'starting_capital_usd': float,
                'current_usd': float,
                'percentage_gain': float,
                'total_profit': float
            },
            'trades': [
                {
                    'buy_price': float,
                    'sell_price': float,
                    'profit_percentage': float,
                    'total_profit': float,
                    'time_held': str,
                    'buy_timestamp': str,
                    'sell_timestamp': str,
                    'screenshot_available': bool,
                    'screenshot_base64': str (optional)
                }
            ]
        }
    """
    transactions = load_transaction_history(symbol)

    if not transactions:
        result = {
            'symbol': symbol,
            'total_trades': 0,
            'trades_included': 0,
            'performance_summary': None,
            'trades': []
        }
        # Include portfolio metrics even if no transactions yet
        if starting_capital_usd is not None:
            result['portfolio_metrics'] = calculate_portfolio_metrics(symbol, starting_capital_usd)
        return result

    # Limit to max_trades
    limited_transactions = transactions[:max_trades]

    # Calculate performance metrics
    total_profit = sum(t.get('total_profit', 0) for t in transactions)
    profit_percentages = [t.get('potential_profit_percentage', 0) for t in transactions]
    average_profit_percentage = sum(profit_percentages) / len(profit_percentages) if profit_percentages else 0

    profitable_trades = len([t for t in transactions if t.get('total_profit', 0) > 0])
    losing_trades = len([t for t in transactions if t.get('total_profit', 0) <= 0])
    win_rate = (profitable_trades / len(transactions) * 100) if transactions else 0

    performance_summary = {
        'total_profit': total_profit,
        'average_profit_percentage': average_profit_percentage,
        'win_rate': win_rate,
        'profitable_trades': profitable_trades,
        'losing_trades': losing_trades
    }

    # Build trade records with optional screenshots
    trades = []
    for transaction in limited_transactions:
        trade_record = {
            'buy_price': transaction.get('buy_price'),
            'sell_price': transaction.get('sell_price'),
            'profit_percentage': transaction.get('potential_profit_percentage'),
            'total_profit': transaction.get('total_profit'),
            'time_held': transaction.get('time_held_position'),
            'buy_timestamp': transaction.get('buy_timestamp'),
            'sell_timestamp': transaction.get('timestamp'),
            'screenshot_available': False
        }

        # Include screenshot if available and enabled
        if include_screenshots:
            screenshot_path = transaction.get('buy_screenshot_path')
            if screenshot_path and os.path.exists(screenshot_path):
                encoded_image = encode_image_to_base64(screenshot_path)
                if encoded_image:
                    trade_record['screenshot_available'] = True
                    trade_record['screenshot_base64'] = encoded_image

        trades.append(trade_record)

    result = {
        'symbol': symbol,
        'total_trades': len(transactions),
        'trades_included': len(limited_transactions),
        'performance_summary': performance_summary,
        'trades': trades
    }

    # Include portfolio metrics if starting capital is provided
    if starting_capital_usd is not None:
        result['portfolio_metrics'] = calculate_portfolio_metrics(symbol, starting_capital_usd)

    return result


def format_context_for_llm(context: Dict) -> str:
    """
    Format the trading context into a human-readable string for LLM consumption.

    Args:
        context: The context dictionary from build_trading_context()

    Returns:
        Formatted string suitable for including in LLM prompts
    """
    if context['total_trades'] == 0:
        base_msg = f"No historical trading data available for {context['symbol']}."
        # Include portfolio metrics even if no trades yet
        if 'portfolio_metrics' in context:
            metrics = context['portfolio_metrics']
            base_msg += f"""

Portfolio Status:
- Starting Capital: ${metrics['starting_capital_usd']:.2f}
- Current Value: ${metrics['current_usd']:.2f}
- Total Gain/Loss: ${metrics['total_profit']:.2f} ({metrics['percentage_gain']:.2f}%)
"""
        return base_msg

    summary = context['performance_summary']

    output = f"""
Historical Trading Performance for {context['symbol']}:
"""

    # Add portfolio metrics section if available
    if 'portfolio_metrics' in context:
        metrics = context['portfolio_metrics']
        output += f"""
Portfolio Status:
- Starting Capital: ${metrics['starting_capital_usd']:.2f}
- Current Value: ${metrics['current_usd']:.2f}
- Total Gain/Loss: ${metrics['total_profit']:.2f} ({metrics['percentage_gain']:.2f}%)

"""

    output += f"""
Performance Summary (All-Time):
- Total Trades: {context['total_trades']}
- Win Rate: {summary['win_rate']:.1f}% ({summary['profitable_trades']} wins, {summary['losing_trades']} losses)
- Average Profit: {summary['average_profit_percentage']:.2f}% per trade
- Total Cumulative Profit: ${summary['total_profit']:.2f}

Recent Trade History (Last {context['trades_included']} trades):
"""

    for i, trade in enumerate(context['trades'], 1):
        profit_indicator = "✓" if trade['total_profit'] > 0 else "✗"
        output += f"""
{i}. {profit_indicator} Trade from {trade['buy_timestamp'][:10]}:
   - Entry: ${trade['buy_price']:.6f} → Exit: ${trade['sell_price']:.6f}
   - Profit: {trade['profit_percentage']:.2f}% (${trade['total_profit']:.2f})
   - Time Held: {trade['time_held']}
   - Screenshot: {'Available' if trade['screenshot_available'] else 'Not available'}
"""

    output += """
INSTRUCTIONS: Use this historical data to:
1. Learn from past winning and losing trades
2. Identify patterns in successful entry/exit points
3. Adjust your strategy based on what has worked for this specific symbol
4. Consider the typical hold times and profit targets that have been successful
"""

    return output


def get_trade_screenshots_for_vision(context: Dict) -> List[Dict]:
    """
    Extract screenshot data formatted for OpenAI Vision API.

    Args:
        context: The context dictionary from build_trading_context()

    Returns:
        List of dictionaries formatted for OpenAI Vision API:
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,..."
                }
            }
        ]
    """
    screenshots = []

    for trade in context.get('trades', []):
        if trade.get('screenshot_available') and 'screenshot_base64' in trade:
            screenshots.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{trade['screenshot_base64']}",
                    "detail": "low"  # Use low detail to save tokens
                }
            })

    return screenshots


def prune_old_transactions(symbol: str, keep_count: int = 50) -> bool:
    """
    Remove older transactions for a symbol to manage context size.
    Keeps only the most recent 'keep_count' transactions.

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')
        keep_count: Number of recent transactions to keep (default: 50)

    Returns:
        Boolean indicating success
    """
    try:
        file_path = os.path.join("transactions", "data.json")

        if not os.path.exists(file_path):
            return False

        with open(file_path, 'r') as f:
            all_transactions = json.load(f)

        # Separate transactions for this symbol and others
        symbol_transactions = [t for t in all_transactions if t.get('symbol') == symbol]
        other_transactions = [t for t in all_transactions if t.get('symbol') != symbol]

        # Sort symbol transactions by timestamp (most recent first)
        symbol_transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Keep only the most recent ones
        kept_transactions = symbol_transactions[:keep_count]
        removed_count = len(symbol_transactions) - len(kept_transactions)

        # Combine back together
        new_transactions = other_transactions + kept_transactions

        # Save back to file
        with open(file_path, 'w') as f:
            json.dump(new_transactions, f, indent=4)

        if removed_count > 0:
            print(f"✓ Pruned {removed_count} old transactions for {symbol}, kept {len(kept_transactions)}")

        return True

    except Exception as e:
        print(f"ERROR: Failed to prune transactions: {e}")
        return False
