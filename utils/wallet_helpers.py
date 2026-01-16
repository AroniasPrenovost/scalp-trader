import os
import json
from typing import List, Dict

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


def calculate_wallet_metrics(symbol: str, starting_capital_usd: float) -> Dict:
    """
    Calculate wallet performance metrics for a symbol.

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')
        starting_capital_usd: The initial capital allocated to this symbol

    Returns:
        Dictionary containing:
        {
            'starting_capital_usd': float,
            'current_usd': float,
            'percentage_gain': float,
            'gross_profit': float,
            'taxes': float,
            'exchange_fees': float,
            'total_profit': float
        }
    """
    transactions = load_transaction_history(symbol)

    # Calculate cumulative metrics from all completed trades
    total_profit = sum(t.get('total_profit', 0) for t in transactions)
    gross_profit = sum(t.get('gross_profit', 0) for t in transactions)
    taxes = sum(t.get('taxes', 0) for t in transactions)
    exchange_fees = sum(t.get('exchange_fees', 0) for t in transactions)

    # Current USD = starting capital + all profits/losses
    current_usd = starting_capital_usd + total_profit

    # Calculate percentage gain
    percentage_gain = ((current_usd - starting_capital_usd) / starting_capital_usd * 100) if starting_capital_usd > 0 else 0

    return {
        'starting_capital_usd': starting_capital_usd,
        'current_usd': current_usd,
        'percentage_gain': percentage_gain,
        'gross_profit': gross_profit,
        'taxes': taxes,
        'exchange_fees': exchange_fees,
        'total_profit': total_profit
    }
