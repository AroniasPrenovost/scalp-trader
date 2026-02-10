import os
import json
from typing import List, Dict, Optional

from utils.coinbase import get_last_order_from_local_json_ledger, detect_stored_coinbase_order_type, get_asset_price, get_coinbase_client
from utils.profit_calculator import calculate_net_profit_from_price_move

# Fee and tax rates (should match index.py)
ENTRY_FEE_PCT = 0.25  # taker fee
EXIT_FEE_PCT = 0.25   # taker fee
TAX_RATE_PCT = 24.0   # short-term capital gains


def get_open_position_unrealized_pnl(symbol: str) -> Optional[Dict]:
    """
    Check if there's an open position for this symbol and calculate unrealized P&L.

    Returns:
        Dict with unrealized metrics if position exists, None otherwise
    """
    try:
        last_order = get_last_order_from_local_json_ledger(symbol, verbose=False)
        order_type = detect_stored_coinbase_order_type(last_order)

        # Only proceed if there's an open position (buy order)
        if order_type not in ['buy', 'placeholder']:
            return None

        if not last_order:
            return None

        # Handle both list and dict formats
        order_data = last_order[0] if isinstance(last_order, list) else last_order

        # Get entry price and shares
        entry_price = float(order_data.get('average_filled_price', 0))
        shares = float(order_data.get('filled_size', 0))
        entry_fee = float(order_data.get('total_fees', 0))

        if entry_price <= 0 or shares <= 0:
            return None

        # Get current price
        try:
            coinbase_client = get_coinbase_client()
            current_price = get_asset_price(coinbase_client, symbol)
        except:
            return None

        if current_price is None or current_price <= 0:
            return None

        # Calculate cost basis (entry value + entry fee)
        entry_value = entry_price * shares
        cost_basis = entry_value + entry_fee

        # Calculate unrealized P&L
        profit_calc = calculate_net_profit_from_price_move(
            entry_price=entry_price,
            exit_price=current_price,
            shares=shares,
            entry_fee_pct=ENTRY_FEE_PCT,
            exit_fee_pct=EXIT_FEE_PCT,
            tax_rate_pct=TAX_RATE_PCT,
            cost_basis_usd=cost_basis
        )

        return {
            'has_position': True,
            'entry_price': entry_price,
            'current_price': current_price,
            'shares': shares,
            'cost_basis': cost_basis,
            'gross_profit': profit_calc['gross_profit_usd'],
            'exit_fee': profit_calc['exit_fee_usd'],
            'tax': profit_calc['tax_usd'],
            'net_profit': profit_calc['net_profit_usd'],
            'net_profit_pct': profit_calc['net_profit_pct']
        }

    except Exception as e:
        # Fail silently - just don't include unrealized P&L
        return None


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

    Includes BOTH closed trade history AND unrealized P&L from any open position.

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
            'total_profit': float,
            'has_open_position': bool,
            'unrealized_pnl': float (if open position exists)
        }
    """
    transactions = load_transaction_history(symbol)

    # Calculate cumulative metrics from all completed trades
    closed_total_profit = sum(t.get('total_profit', 0) for t in transactions)
    closed_gross_profit = sum(t.get('gross_profit', 0) for t in transactions)
    closed_taxes = sum(t.get('taxes', 0) for t in transactions)
    closed_exchange_fees = sum(t.get('exchange_fees', 0) for t in transactions)

    # Check for open position and get unrealized P&L
    open_position = get_open_position_unrealized_pnl(symbol)

    if open_position:
        # Include unrealized P&L in totals
        total_profit = closed_total_profit + open_position['net_profit']
        gross_profit = closed_gross_profit + open_position['gross_profit']
        taxes = closed_taxes + open_position['tax']
        exchange_fees = closed_exchange_fees + open_position['exit_fee']
        has_open_position = True
        unrealized_pnl = open_position['net_profit']
    else:
        total_profit = closed_total_profit
        gross_profit = closed_gross_profit
        taxes = closed_taxes
        exchange_fees = closed_exchange_fees
        has_open_position = False
        unrealized_pnl = 0.0

    # Current USD = starting capital + all profits/losses (closed + unrealized)
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
        'total_profit': total_profit,
        'has_open_position': has_open_position,
        'unrealized_pnl': unrealized_pnl
    }
