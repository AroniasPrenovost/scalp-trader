"""
Position Tracker: Tracks peak profits and state for active positions

This module maintains peak profit tracking for positions to enable intelligent
exit strategies based on profit drawdowns from peaks.
"""

import os
import json
from typing import Dict, Optional

POSITION_STATE_FILE = 'position_state.json'

def load_position_state() -> Dict:
    """Load position state from file, return empty dict if doesn't exist"""
    try:
        if os.path.exists(POSITION_STATE_FILE):
            with open(POSITION_STATE_FILE, 'r') as file:
                return json.load(file)
        return {}
    except Exception as e:
        print(f"Warning: Failed to load position state: {e}")
        return {}

def save_position_state(state: Dict) -> None:
    """Save position state to file"""
    try:
        with open(POSITION_STATE_FILE, 'w') as file:
            json.dump(state, file, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save position state: {e}")

def update_peak_profit(symbol: str, current_profit_usd: float, current_profit_pct: float) -> Dict:
    """
    Update peak profit for a symbol if current profit exceeds previous peak

    Args:
        symbol: Trading pair symbol
        current_profit_usd: Current profit in USD
        current_profit_pct: Current profit percentage

    Returns:
        Dict with peak info: {
            'peak_profit_usd': float,
            'peak_profit_pct': float,
            'last_updated': timestamp
        }
    """
    import datetime

    state = load_position_state()

    if symbol not in state:
        state[symbol] = {}

    # Initialize or update peak
    peak_profit_usd = state[symbol].get('peak_profit_usd', current_profit_usd)
    peak_profit_pct = state[symbol].get('peak_profit_pct', current_profit_pct)

    # Update peak if current is higher
    if current_profit_usd > peak_profit_usd:
        peak_profit_usd = current_profit_usd
        peak_profit_pct = current_profit_pct

    state[symbol] = {
        'peak_profit_usd': peak_profit_usd,
        'peak_profit_pct': peak_profit_pct,
        'last_updated': datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    save_position_state(state)
    return state[symbol]

def get_peak_profit(symbol: str) -> Optional[Dict]:
    """
    Get peak profit info for a symbol

    Returns:
        Dict with peak info or None if no position tracked
    """
    state = load_position_state()
    return state.get(symbol)

def clear_position_state(symbol: str) -> None:
    """Clear position state for a symbol (called when position is closed)"""
    state = load_position_state()
    if symbol in state:
        del state[symbol]
        save_position_state(state)

def should_exit_on_downturn(
    symbol: str,
    current_profit_usd: float,
    current_profit_pct: float,
    min_peak_profit_usd: float,
    downturn_threshold_usd: float
) -> tuple[bool, Optional[Dict]]:
    """
    Check if position should exit due to downturn from peak

    Args:
        symbol: Trading pair symbol
        current_profit_usd: Current profit in USD
        current_profit_pct: Current profit percentage
        min_peak_profit_usd: Minimum peak profit required to consider downturn
        downturn_threshold_usd: How much profit must decline from peak to trigger exit

    Returns:
        Tuple of (should_exit: bool, peak_info: Dict or None)
    """
    # Update peak
    peak_info = update_peak_profit(symbol, current_profit_usd, current_profit_pct)

    peak_profit_usd = peak_info['peak_profit_usd']

    # Check if peak is high enough to consider downturn
    if peak_profit_usd < min_peak_profit_usd:
        return False, peak_info

    # Calculate downturn from peak
    downturn_from_peak = peak_profit_usd - current_profit_usd

    # Exit if downturn exceeds threshold
    should_exit = downturn_from_peak >= downturn_threshold_usd

    return should_exit, peak_info
