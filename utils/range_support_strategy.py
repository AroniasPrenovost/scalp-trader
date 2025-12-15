"""
Range-Based Support Zone Trading Strategy

This module implements a strategy that identifies support zones based on multiple
price bottoms and triggers buy signals when price revisits those zones.

Strategy:
- Identify 2-3 recent bottoms within a similar price range (support zone)
- When price returns to that zone, trigger a buy signal
- This is a mean reversion / support zone trading approach
"""

import numpy as np
from scipy.signal import argrelextrema
from utils.price_helpers import calculate_percentage_from_min


def find_local_extrema(prices, order=5):
    """
    Find local minima (bottoms) and maxima (tops) in price data.

    Args:
        prices: List of price values
        order: How many points on each side to use for comparison (default 5)
               Higher = more significant extrema, Lower = more sensitive

    Returns:
        Dictionary with:
        - 'bottoms': List of tuples (index, price) for local minima
        - 'tops': List of tuples (index, price) for local maxima
    """
    if len(prices) < order * 2 + 1:
        return {'bottoms': [], 'tops': []}

    # Convert to numpy array
    price_array = np.array(prices)

    # Find local minima (bottoms)
    local_min_indices = argrelextrema(price_array, np.less_equal, order=order)[0]
    bottoms = [(int(idx), float(price_array[idx])) for idx in local_min_indices]

    # Find local maxima (tops)
    local_max_indices = argrelextrema(price_array, np.greater_equal, order=order)[0]
    tops = [(int(idx), float(price_array[idx])) for idx in local_max_indices]

    return {
        'bottoms': bottoms,
        'tops': tops
    }


def identify_support_zones(bottoms, zone_tolerance_percentage=3.0, min_touches=2, max_touches=5):
    """
    Group bottoms into support zones based on price proximity.

    Args:
        bottoms: List of tuples (index, price) from find_local_extrema
        zone_tolerance_percentage: Max % difference between bottoms to group them (default 3%)
        min_touches: Minimum number of bottoms required to form a valid zone (default 2)
        max_touches: Maximum number of bottoms to consider for a zone (default 5)

    Returns:
        List of support zones, each zone is a dict:
        {
            'zone_price_min': lowest price in the zone,
            'zone_price_max': highest price in the zone,
            'zone_price_avg': average price of all bottoms in zone,
            'touches': number of times price touched this zone,
            'touch_indices': list of indices where price touched the zone,
            'touch_prices': list of actual prices at each touch,
            'last_touch_index': most recent index where zone was touched,
            'strength': zone strength score (more touches = stronger)
        }
    """
    if len(bottoms) < min_touches:
        return []

    # Sort bottoms by price (ascending)
    sorted_bottoms = sorted(bottoms, key=lambda x: x[1])

    zones = []
    used_indices = set()

    # Iterate through sorted bottoms and group them into zones
    for i, (idx, price) in enumerate(sorted_bottoms):
        if idx in used_indices:
            continue

        # Start a new zone
        zone_bottoms = [(idx, price)]
        used_indices.add(idx)

        # Find all bottoms within tolerance of this price
        for j in range(i + 1, len(sorted_bottoms)):
            other_idx, other_price = sorted_bottoms[j]

            if other_idx in used_indices:
                continue

            # Calculate percentage difference from current zone average
            current_zone_avg = sum(p for _, p in zone_bottoms) / len(zone_bottoms)
            pct_diff = abs((other_price - current_zone_avg) / current_zone_avg) * 100

            if pct_diff <= zone_tolerance_percentage and len(zone_bottoms) < max_touches:
                zone_bottoms.append((other_idx, other_price))
                used_indices.add(other_idx)

        # Only create zone if we have enough touches
        if len(zone_bottoms) >= min_touches:
            zone_prices = [p for _, p in zone_bottoms]
            zone_indices = [idx for idx, _ in zone_bottoms]

            zones.append({
                'zone_price_min': min(zone_prices),
                'zone_price_max': max(zone_prices),
                'zone_price_avg': sum(zone_prices) / len(zone_prices),
                'touches': len(zone_bottoms),
                'touch_indices': sorted(zone_indices),
                'touch_prices': [price for _, price in sorted(zone_bottoms, key=lambda x: x[0])],
                'last_touch_index': max(zone_indices),
                'strength': len(zone_bottoms)  # More touches = stronger support
            })

    # Sort zones by strength (descending) and recency (descending)
    zones.sort(key=lambda z: (z['strength'], z['last_touch_index']), reverse=True)

    return zones


def is_price_in_support_zone(current_price, zone, entry_tolerance_percentage=1.5):
    """
    Check if current price is within a support zone with some tolerance.

    Args:
        current_price: Current market price
        zone: Support zone dict from identify_support_zones
        entry_tolerance_percentage: % above zone max to still consider it valid (default 1.5%)

    Returns:
        Boolean indicating if price is in the zone
    """
    # Allow entry slightly below zone min and slightly above zone max
    zone_min_with_tolerance = zone['zone_price_min'] * (1 - entry_tolerance_percentage / 100)
    zone_max_with_tolerance = zone['zone_price_max'] * (1 + entry_tolerance_percentage / 100)

    return zone_min_with_tolerance <= current_price <= zone_max_with_tolerance


def check_range_support_buy_signal(
    prices,
    current_price,
    min_touches=2,
    max_touches=5,
    zone_tolerance_percentage=3.0,
    entry_tolerance_percentage=1.5,
    extrema_order=5,
    lookback_window=None
):
    """
    Main function to check if a range-based support buy signal is present.

    Strategy:
    1. Find recent bottoms (local minima) in price history
    2. Group bottoms into support zones (2-3+ touches at similar price)
    3. Check if current price is in a valid support zone
    4. If yes, return buy signal with zone details

    Args:
        prices: Full list of historical prices (hourly data)
        current_price: Current market price
        min_touches: Minimum bottoms needed to form a zone (default 2)
        max_touches: Maximum bottoms to consider for a zone (default 5)
        zone_tolerance_percentage: Max % difference to group bottoms (default 3%)
        entry_tolerance_percentage: % tolerance for entry around zone (default 1.5%)
        extrema_order: Sensitivity for finding bottoms (default 5)
        lookback_window: Optional - only analyze last N price points (e.g., 336 for 14 days)

    Returns:
        Dictionary with:
        {
            'signal': 'buy' or 'no_signal',
            'zone': support zone dict if signal is 'buy', else None,
            'all_zones': list of all identified zones,
            'current_price': current price,
            'zone_strength': strength of the triggered zone (number of touches),
            'distance_from_zone_avg': % distance from zone average price,
            'reasoning': explanation of the signal
        }
    """
    # Apply lookback window if specified
    if lookback_window and len(prices) > lookback_window:
        analysis_prices = prices[-lookback_window:]
    else:
        analysis_prices = prices

    if len(analysis_prices) < extrema_order * 2 + 1:
        return {
            'signal': 'no_signal',
            'zone': None,
            'all_zones': [],
            'current_price': current_price,
            'zone_strength': 0,
            'distance_from_zone_avg': 0,
            'reasoning': f'Insufficient data: need at least {extrema_order * 2 + 1} price points'
        }

    # Step 1: Find local extrema (bottoms and tops)
    extrema = find_local_extrema(analysis_prices, order=extrema_order)
    bottoms = extrema['bottoms']

    if len(bottoms) < min_touches:
        return {
            'signal': 'no_signal',
            'zone': None,
            'all_zones': [],
            'current_price': current_price,
            'zone_strength': 0,
            'distance_from_zone_avg': 0,
            'reasoning': f'Insufficient bottoms found: {len(bottoms)} (need {min_touches})'
        }

    # Step 2: Group bottoms into support zones
    support_zones = identify_support_zones(
        bottoms,
        zone_tolerance_percentage=zone_tolerance_percentage,
        min_touches=min_touches,
        max_touches=max_touches
    )

    if not support_zones:
        return {
            'signal': 'no_signal',
            'zone': None,
            'all_zones': [],
            'current_price': current_price,
            'zone_strength': 0,
            'distance_from_zone_avg': 0,
            'reasoning': f'No valid support zones found (need {min_touches}+ touches within {zone_tolerance_percentage}%)'
        }

    # Step 3: Check if current price is in any support zone
    for zone in support_zones:
        if is_price_in_support_zone(current_price, zone, entry_tolerance_percentage):
            # Calculate distance from zone average
            distance_pct = ((current_price - zone['zone_price_avg']) / zone['zone_price_avg']) * 100

            return {
                'signal': 'buy',
                'zone': zone,
                'all_zones': support_zones,
                'current_price': current_price,
                'zone_strength': zone['strength'],
                'distance_from_zone_avg': distance_pct,
                'reasoning': f"Price ${current_price:.4f} in support zone (avg ${zone['zone_price_avg']:.4f}, {zone['touches']} touches)"
            }

    # No zone triggered - return details anyway for logging
    strongest_zone = support_zones[0]
    distance_pct = ((current_price - strongest_zone['zone_price_avg']) / strongest_zone['zone_price_avg']) * 100

    return {
        'signal': 'no_signal',
        'zone': None,
        'all_zones': support_zones,
        'current_price': current_price,
        'zone_strength': 0,
        'distance_from_zone_avg': distance_pct,
        'reasoning': f"Price ${current_price:.4f} not in any support zone. Nearest: ${strongest_zone['zone_price_avg']:.4f} ({distance_pct:+.2f}%)"
    }


def calculate_zone_based_targets(zone, risk_reward_ratio=2.5, stop_loss_below_zone_pct=2.0):
    """
    Calculate stop loss and profit targets based on a support zone.

    Args:
        zone: Support zone dict from identify_support_zones
        risk_reward_ratio: Desired risk/reward ratio (default 2.5)
        stop_loss_below_zone_pct: How far below zone min to place stop (default 2%)

    Returns:
        Dictionary with:
        {
            'entry_price': recommended entry at zone average,
            'stop_loss': stop loss price,
            'profit_target': profit target price,
            'risk_amount': $ risk per share,
            'reward_amount': $ reward per share,
            'risk_percentage': % risk from entry,
            'reward_percentage': % reward from entry
        }
    """
    entry = zone['zone_price_avg']
    stop = zone['zone_price_min'] * (1 - stop_loss_below_zone_pct / 100)

    # Calculate risk
    risk = entry - stop

    # Calculate reward based on risk/reward ratio
    reward = risk * risk_reward_ratio
    profit_target = entry + reward

    # Calculate percentages
    risk_pct = (risk / entry) * 100
    reward_pct = (reward / entry) * 100

    return {
        'entry_price': entry,
        'stop_loss': stop,
        'profit_target': profit_target,
        'risk_amount': risk,
        'reward_amount': reward,
        'risk_percentage': risk_pct,
        'reward_percentage': reward_pct
    }
