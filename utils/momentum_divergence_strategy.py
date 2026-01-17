#!/usr/bin/env python3
"""
Momentum Divergence Strategy - Multi-Asset Laggard Detection

Core Concept:
When the crypto market moves, assets don't move in perfect sync. Leaders move first,
laggards follow within 5-30 minutes. This strategy captures laggards by:

1. Tracking all enabled assets (8 cryptos)
2. Calculating a composite "market return" (average of all assets)
3. Identifying assets that lag behind the market move
4. Entering positions on laggards that historically correlate with the market
5. Exiting via stop-loss, take-profit, trailing stop, or time-based exit

Strategy Parameters (from config.json):
- LOOKBACK_WINDOW: 6 bars (30 minutes at 5-min intervals)
- ENTRY_THRESHOLD: 2% market move required
- DIVERGENCE_THRESHOLD: 1% lag from market required
- CORRELATION_MIN: 0.6 minimum correlation to market
- STOP_LOSS: -1%
- TAKE_PROFIT: +2.5%

Fee & Tax Constraints:
- Taker fee: 0.250% (using market orders)
- Maker fee: 0.125% (limit orders, not used)
- Round-trip cost: 0.50% (0.250% × 2)
- Federal tax: 24-37%
- Break-even requirement: ~0.66% move minimum (after fees + taxes)
- Target profitable trade: ≥1.5% move
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from utils.profit_calculator import calculate_required_price_for_target_profit


def calculate_returns(prices: List[float], lookback: int) -> float:
    """
    Calculate percentage return over lookback period.

    Args:
        prices: List of historical prices (most recent last)
        lookback: Number of periods to look back

    Returns:
        Percentage return (e.g., 2.5 for 2.5% gain)
    """
    if len(prices) < lookback + 1:
        return 0.0

    start_price = prices[-(lookback + 1)]
    end_price = prices[-1]

    if start_price == 0:
        return 0.0

    return ((end_price - start_price) / start_price) * 100


def calculate_correlation(asset_returns: List[float], market_returns: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient between asset and market returns.

    Args:
        asset_returns: List of asset returns for each period
        market_returns: List of market returns for each period

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
        return 0.0

    # Convert to numpy arrays
    asset_arr = np.array(asset_returns)
    market_arr = np.array(market_returns)

    # Calculate correlation
    correlation_matrix = np.corrcoef(asset_arr, market_arr)
    correlation = correlation_matrix[0, 1]

    # Handle NaN (happens when one series is constant)
    if np.isnan(correlation):
        return 0.0

    return correlation


def calculate_composite_market_return(all_asset_data: Dict[str, List[float]],
                                     lookback: int,
                                     exclude_symbol: Optional[str] = None) -> float:
    """
    Calculate composite market return (average return across all assets).

    Args:
        all_asset_data: Dict of {symbol: [prices]} for all enabled assets
        lookback: Number of periods to look back
        exclude_symbol: Optional symbol to exclude (when calculating for that asset)

    Returns:
        Average percentage return across all assets
    """
    returns = []

    for symbol, prices in all_asset_data.items():
        if exclude_symbol and symbol == exclude_symbol:
            continue

        if len(prices) >= lookback + 1:
            asset_return = calculate_returns(prices, lookback)
            returns.append(asset_return)

    if not returns:
        return 0.0

    return np.mean(returns)


def calculate_period_returns(prices: List[float], lookback: int) -> List[float]:
    """
    Calculate returns for each period in the lookback window.
    Used for correlation calculations.

    Args:
        prices: List of historical prices
        lookback: Number of periods

    Returns:
        List of period-over-period returns
    """
    if len(prices) < lookback + 1:
        return []

    returns = []
    relevant_prices = prices[-(lookback + 1):]

    for i in range(1, len(relevant_prices)):
        prev_price = relevant_prices[i - 1]
        curr_price = relevant_prices[i]

        if prev_price == 0:
            returns.append(0.0)
        else:
            period_return = ((curr_price - prev_price) / prev_price) * 100
            returns.append(period_return)

    return returns


def calculate_market_period_returns(all_asset_data: Dict[str, List[float]],
                                    lookback: int,
                                    exclude_symbol: Optional[str] = None) -> List[float]:
    """
    Calculate composite market returns for each period in lookback window.

    Args:
        all_asset_data: Dict of {symbol: [prices]} for all enabled assets
        lookback: Number of periods
        exclude_symbol: Optional symbol to exclude

    Returns:
        List of market returns for each period
    """
    # Collect period returns for each asset
    all_period_returns = {}

    for symbol, prices in all_asset_data.items():
        if exclude_symbol and symbol == exclude_symbol:
            continue

        period_returns = calculate_period_returns(prices, lookback)
        if period_returns:
            all_period_returns[symbol] = period_returns

    if not all_period_returns:
        return []

    # Calculate average return for each period
    num_periods = len(next(iter(all_period_returns.values())))
    market_returns = []

    for i in range(num_periods):
        period_values = [returns[i] for returns in all_period_returns.values() if i < len(returns)]
        if period_values:
            market_returns.append(np.mean(period_values))
        else:
            market_returns.append(0.0)

    return market_returns


def check_divergence_signal(symbol: str,
                           asset_prices: List[float],
                           all_asset_data: Dict[str, List[float]],
                           current_price: float,
                           lookback_window: int = 6,
                           entry_threshold: float = 0.02,
                           divergence_threshold: float = 0.01,
                           correlation_min: float = 0.6,
                           stop_loss_pct: float = -0.01,
                           take_profit_pct: float = 0.025,
                           entry_fee_pct: float = 0.6,
                           exit_fee_pct: float = 0.6,
                           tax_rate_pct: float = 37.0) -> Optional[Dict]:
    """
    Check for momentum divergence entry signal.

    Strategy Logic:
    1. Calculate composite market return (average of all assets)
    2. Calculate this asset's return
    3. Calculate divergence = market_return - asset_return
    4. Calculate correlation between asset and market
    5. Generate LONG signal if:
       - Market moved up significantly (>= entry_threshold)
       - Asset lagged behind (divergence >= divergence_threshold)
       - Asset historically follows market (correlation >= correlation_min)
    6. Generate SHORT signal if:
       - Market moved down significantly (<= -entry_threshold)
       - Asset lagged behind (divergence <= -divergence_threshold)
       - Asset historically follows market

    Args:
        symbol: Asset symbol (e.g., 'BTC-USD')
        asset_prices: Historical prices for this asset
        all_asset_data: Dict of {symbol: [prices]} for all enabled assets
        current_price: Current market price
        lookback_window: Number of periods to analyze (default: 6 = 30 min)
        entry_threshold: Minimum market move required (default: 0.02 = 2%)
        divergence_threshold: Minimum divergence required (default: 0.01 = 1%)
        correlation_min: Minimum correlation required (default: 0.6)
        stop_loss_pct: Stop loss percentage (default: -0.01 = -1%)
        take_profit_pct: Take profit percentage (default: 0.025 = 2.5%)
        entry_fee_pct: Entry fee percentage
        exit_fee_pct: Exit fee percentage
        tax_rate_pct: Tax rate percentage

    Returns:
        Dictionary with signal details or None
    """
    # Validate data sufficiency
    if len(asset_prices) < lookback_window + 1:
        return {
            'signal': 'no_signal',
            'reason': f'Insufficient data for {symbol} (need {lookback_window + 1}+ data points)',
            'metrics': {
                'data_points': len(asset_prices)
            }
        }

    # Calculate composite market return (excluding this asset to avoid self-influence)
    composite_market_return = calculate_composite_market_return(
        all_asset_data,
        lookback_window,
        exclude_symbol=symbol
    )

    # Calculate this asset's return
    asset_return = calculate_returns(asset_prices, lookback_window)

    # Calculate divergence
    divergence = composite_market_return - asset_return

    # Calculate correlation
    asset_period_returns = calculate_period_returns(asset_prices, lookback_window)
    market_period_returns = calculate_market_period_returns(
        all_asset_data,
        lookback_window,
        exclude_symbol=symbol
    )

    if not asset_period_returns or not market_period_returns:
        return {
            'signal': 'no_signal',
            'reason': f'Cannot calculate period returns for correlation',
            'metrics': {
                'market_return': composite_market_return,
                'asset_return': asset_return,
                'divergence': divergence
            }
        }

    correlation = calculate_correlation(asset_period_returns, market_period_returns)

    # Calculate confidence score (used for ranking opportunities)
    # Confidence = correlation * (divergence strength / entry threshold)
    # Higher correlation + stronger divergence = higher confidence
    divergence_strength = abs(divergence) / (entry_threshold * 100)  # Normalize to entry threshold
    confidence_score = correlation * divergence_strength

    # =====================================================================
    # SIGNAL 1: LONG (Laggard on Market Up-Move)
    # =====================================================================
    if (composite_market_return >= (entry_threshold * 100) and
        divergence >= (divergence_threshold * 100) and
        correlation >= correlation_min):

        # Determine confidence level
        if confidence_score >= 1.0 and correlation >= 0.8:
            confidence = 'high'
        elif confidence_score >= 0.7 and correlation >= 0.7:
            confidence = 'medium'
        else:
            confidence = 'low'

        entry_price = current_price
        stop_loss = entry_price * (1 + stop_loss_pct)

        # Calculate realistic profit target
        target_calc = calculate_required_price_for_target_profit(
            entry_price=entry_price,
            target_net_profit_pct=take_profit_pct * 100,  # Convert to percentage
            entry_fee_pct=entry_fee_pct,
            exit_fee_pct=exit_fee_pct,
            tax_rate_pct=tax_rate_pct
        )

        profit_target = target_calc['required_sell_price']

        metrics = {
            'market_return': round(composite_market_return, 3),
            'asset_return': round(asset_return, 3),
            'divergence': round(divergence, 3),
            'correlation': round(correlation, 3),
            'confidence_score': round(confidence_score, 3),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'profit_target': round(profit_target, 2),
            'target_net_profit_pct': take_profit_pct * 100,
            'target_gross_move_pct': round(target_calc['price_change_pct'], 2)
        }

        return {
            'signal': 'buy',
            'strategy': 'momentum_divergence_long',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'confidence': confidence,
            'reasoning': (
                f'MOMENTUM DIVERGENCE (LONG): Market moved +{composite_market_return:.2f}% '
                f'while {symbol} only moved +{asset_return:.2f}% (divergence: {divergence:.2f}%). '
                f'Correlation: {correlation:.2f} (strong follower). '
                f'Expecting {symbol} to catch up to market. '
                f'Confidence: {confidence} ({confidence_score:.2f})'
            ),
            'metrics': metrics
        }

    # =====================================================================
    # SIGNAL 2: SHORT (Laggard on Market Down-Move)
    # =====================================================================
    # Note: Shorting not typically available on Coinbase spot, but included for completeness
    # Can be used if trading futures or if Coinbase adds short functionality
    if (composite_market_return <= -(entry_threshold * 100) and
        divergence <= -(divergence_threshold * 100) and
        correlation >= correlation_min):

        return {
            'signal': 'no_signal',  # Changed from 'sell' to 'no_signal' since we can't short spot
            'reason': (
                f'SHORT signal detected but spot trading does not support shorts. '
                f'Market: {composite_market_return:.2f}%, Asset: {asset_return:.2f}%, '
                f'Divergence: {divergence:.2f}%, Correlation: {correlation:.2f}'
            ),
            'metrics': {
                'market_return': round(composite_market_return, 3),
                'asset_return': round(asset_return, 3),
                'divergence': round(divergence, 3),
                'correlation': round(correlation, 3)
            }
        }

    # =====================================================================
    # NO SIGNAL - Waiting for setup
    # =====================================================================
    return {
        'signal': 'no_signal',
        'reason': (
            f'Waiting for divergence setup. '
            f'Market: {composite_market_return:+.2f}%, '
            f'Asset: {asset_return:+.2f}%, '
            f'Divergence: {divergence:+.2f}% (need {divergence_threshold * 100:.1f}%), '
            f'Correlation: {correlation:.2f} (need {correlation_min:.2f})'
        ),
        'metrics': {
            'market_return': round(composite_market_return, 3),
            'asset_return': round(asset_return, 3),
            'divergence': round(divergence, 3),
            'correlation': round(correlation, 3),
            'entry_threshold_met': abs(composite_market_return) >= (entry_threshold * 100),
            'divergence_threshold_met': abs(divergence) >= (divergence_threshold * 100),
            'correlation_threshold_met': correlation >= correlation_min
        }
    }


def get_strategy_info() -> Dict:
    """Return strategy metadata and performance expectations."""
    return {
        'name': 'Momentum Divergence Strategy',
        'version': '1.0',
        'designed_for': 'Multi-asset laggard detection',
        'target_holding_time': '5-60 minutes',
        'target_profit': '2.5%',
        'max_stop_loss': '1.0%',
        'lookback_window': '30 minutes (6 bars at 5-min intervals)',
        'min_market_move': '2%',
        'min_divergence': '1%',
        'min_correlation': '0.6',
        'exit_methods': [
            'Stop Loss: -1%',
            'Take Profit: +2.5%',
            'Trailing Stop: Activates at +1.5%, trails by 0.5%',
            'Time Exit: Max 60 minutes hold'
        ],
        'signal_types': {
            'momentum_divergence_long': {
                'description': 'Buy laggards when market moves up',
                'requirements': [
                    'Market return >= 2%',
                    'Asset lagged by >= 1%',
                    'Correlation >= 0.6'
                ],
                'target': '2.5%',
                'stop': '1.0%'
            }
        }
    }
