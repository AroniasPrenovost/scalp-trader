"""
Multi-Asset Correlation Manager
Tracks BTC dominance, asset correlations, relative strength, and portfolio exposure.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import statistics


class CorrelationManager:
    """Manages multi-asset correlation analysis and portfolio-level risk."""

    def __init__(self, config_path: str = 'config.json'):
        """Initialize correlation manager with configuration."""
        self.config_path = config_path
        self.config = self._load_config()

        # Correlation settings
        self.correlation_settings = self.config.get('correlation_settings', {})
        self.max_correlated_longs = self.correlation_settings.get('max_correlated_long_positions', 2)
        self.max_correlated_shorts = self.correlation_settings.get('max_correlated_short_positions', 2)
        self.btc_trend_weight = self.correlation_settings.get('btc_trend_weight', 0.5)
        self.correlation_lookback_hours = self.correlation_settings.get('correlation_lookback_hours', 168)  # 7 days

        # Relative strength thresholds
        self.strong_outperformance_threshold = self.correlation_settings.get('strong_outperformance_threshold', 5.0)  # %
        self.weak_underperformance_threshold = self.correlation_settings.get('weak_underperformance_threshold', -5.0)  # %

        # Portfolio state cache
        self.btc_sentiment = None
        self.btc_trend = None
        self.asset_sentiments = {}
        self.correlation_scores = {}

    def _load_config(self) -> dict:
        """Load configuration from config.json."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def calculate_correlation(self, prices_a: List[float], prices_b: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient between two price series.

        Args:
            prices_a: List of prices for asset A
            prices_b: List of prices for asset B

        Returns:
            Correlation coefficient between -1 and 1
        """
        if len(prices_a) != len(prices_b) or len(prices_a) < 2:
            return 0.0

        # Calculate returns instead of raw prices for better correlation
        returns_a = [(prices_a[i] - prices_a[i-1]) / prices_a[i-1] * 100
                     for i in range(1, len(prices_a))]
        returns_b = [(prices_b[i] - prices_b[i-1]) / prices_b[i-1] * 100
                     for i in range(1, len(prices_b))]

        if len(returns_a) < 2:
            return 0.0

        # Calculate means
        mean_a = statistics.mean(returns_a)
        mean_b = statistics.mean(returns_b)

        # Calculate correlation
        numerator = sum((returns_a[i] - mean_a) * (returns_b[i] - mean_b)
                       for i in range(len(returns_a)))

        sum_sq_a = sum((r - mean_a) ** 2 for r in returns_a)
        sum_sq_b = sum((r - mean_b) ** 2 for r in returns_b)

        denominator = (sum_sq_a * sum_sq_b) ** 0.5

        if denominator == 0:
            return 0.0

        correlation = numerator / denominator
        return round(correlation, 3)

    def calculate_relative_strength(self,
                                    asset_prices: List[float],
                                    btc_prices: List[float],
                                    lookback_hours: int = 168) -> Dict[str, float]:
        """
        Calculate relative strength of an asset vs BTC.

        Args:
            asset_prices: Price history for the asset
            btc_prices: Price history for BTC
            lookback_hours: How far back to analyze (default 7 days)

        Returns:
            Dict with relative performance metrics
        """
        if len(asset_prices) < lookback_hours or len(btc_prices) < lookback_hours:
            lookback_hours = min(len(asset_prices), len(btc_prices))

        if lookback_hours < 24:
            return {
                'relative_strength': 0.0,
                'asset_change_pct': 0.0,
                'btc_change_pct': 0.0,
                'outperformance': 0.0,
                'strength_category': 'insufficient_data'
            }

        # Get recent data
        recent_asset = asset_prices[-lookback_hours:]
        recent_btc = btc_prices[-lookback_hours:]

        # Calculate percentage changes
        asset_change_pct = ((recent_asset[-1] - recent_asset[0]) / recent_asset[0]) * 100
        btc_change_pct = ((recent_btc[-1] - recent_btc[0]) / recent_btc[0]) * 100

        # Relative strength = how much asset outperformed/underperformed BTC
        outperformance = asset_change_pct - btc_change_pct

        # Categorize strength
        if outperformance >= self.strong_outperformance_threshold:
            strength_category = 'strong_outperformer'
        elif outperformance >= 0:
            strength_category = 'mild_outperformer'
        elif outperformance >= self.weak_underperformance_threshold:
            strength_category = 'mild_underperformer'
        else:
            strength_category = 'strong_underperformer'

        # Calculate relative strength index (RSI-like, but vs BTC)
        relative_strength = 50 + (outperformance * 2)  # Scale to 0-100 range
        relative_strength = max(0, min(100, relative_strength))  # Clamp

        return {
            'relative_strength': round(relative_strength, 2),
            'asset_change_pct': round(asset_change_pct, 2),
            'btc_change_pct': round(btc_change_pct, 2),
            'outperformance': round(outperformance, 2),
            'strength_category': strength_category
        }

    def set_btc_sentiment(self, sentiment: Dict):
        """Store BTC market sentiment for use as portfolio-level context."""
        self.btc_sentiment = sentiment
        self.btc_trend = sentiment.get('market_trend', 'sideways')

    def set_asset_sentiment(self, symbol: str, sentiment: Dict):
        """Store individual asset sentiment."""
        self.asset_sentiments[symbol] = sentiment

    def get_portfolio_state(self, transaction_log_path: str = None) -> Dict:
        """
        Analyze current portfolio state by reading open positions from *_orders.json ledger files.

        Each *_orders.json file contains at most 1 open position at a time.
        When a position is closed (sold), the file is cleared and the completed trade
        is saved to transactions/data.json.

        Returns:
            Dict with portfolio-level metrics
        """
        import glob
        from utils.coinbase import detect_stored_coinbase_order_type, get_last_order_from_local_json_ledger

        # Analyze current positions from order ledger files
        long_positions = []
        short_positions = []
        total_exposure_usd = 0.0

        # Find all *_orders.json files
        order_files = glob.glob('*_orders.json')

        for order_file in order_files:
            # Extract symbol from filename (e.g., "BTC-USD_orders.json" -> "BTC-USD")
            symbol = order_file.replace('_orders.json', '')

            # Get the last order for this symbol
            last_order = get_last_order_from_local_json_ledger(symbol)
            if not last_order:
                continue

            # Check if this is an open position (last order type is 'buy')
            last_order_type = detect_stored_coinbase_order_type(last_order)

            if last_order_type == 'buy':
                # Extract order details from the filled order
                order_data = last_order.get('order', {})
                entry_price = float(order_data.get('average_filled_price', 0.0))
                shares = float(order_data.get('filled_size', 0.0))
                usd_value = float(order_data.get('total_value_after_fees', 0.0))
                entry_time = order_data.get('created_time', '')

                if shares > 0 and usd_value > 0:
                    position = {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'shares': shares,
                        'usd_value': usd_value,
                        'entry_time': entry_time
                    }
                    long_positions.append(position)
                    total_exposure_usd += usd_value

        # Calculate correlation-adjusted risk
        # If holding multiple correlated longs, aggregate risk increases
        correlation_adjusted_risk = total_exposure_usd
        if len(long_positions) > 1:
            # Assume high correlation (0.8) between crypto assets
            # Adjust risk upward to account for lack of diversification
            avg_correlation = 0.8
            correlation_multiplier = 1 + (avg_correlation * (len(long_positions) - 1) * 0.3)
            correlation_adjusted_risk = total_exposure_usd * correlation_multiplier

        return {
            'total_positions': len(long_positions) + len(short_positions),
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_exposure_usd': round(total_exposure_usd, 2),
            'correlation_adjusted_risk': round(correlation_adjusted_risk, 2),
            'long_symbols': [p['symbol'] for p in long_positions],
            'short_symbols': [p['symbol'] for p in short_positions]
        }

    def should_allow_trade(self,
                          symbol: str,
                          trade_recommendation: str,
                          btc_trend: str,
                          portfolio_state: Dict,
                          relative_strength: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Determine if a trade should be allowed based on correlation and portfolio state.

        Args:
            symbol: Trading pair (e.g., 'SOL-USD')
            trade_recommendation: 'buy', 'sell', 'hold', or 'no_trade'
            btc_trend: 'bullish', 'bearish', or 'sideways'
            portfolio_state: Current portfolio state dict
            relative_strength: Optional relative strength metrics

        Returns:
            Tuple of (allow_trade: bool, reason: str)
        """
        # Always allow SELL trades (position exits)
        if trade_recommendation == 'sell':
            return True, "Exit signal - always allowed"

        # Only apply filters to BUY trades
        if trade_recommendation != 'buy':
            return True, "Not a buy signal"

        # Rule 1: BTC Trend Filter for Altcoins
        if symbol in ['SOL-USD', 'ETH-USD']:
            if btc_trend == 'bearish':
                return False, f"🚫 BTC bearish backdrop - blocking {symbol} BUY"

            if btc_trend == 'sideways' and relative_strength:
                # In sideways BTC, only allow strong outperformers
                if relative_strength['strength_category'] not in ['strong_outperformer', 'mild_outperformer']:
                    return False, f"🚫 BTC sideways + {symbol} weak relative strength"

        # Rule 2: Maximum Correlated Long Positions
        current_longs = len(portfolio_state['long_positions'])
        if current_longs >= self.max_correlated_longs:
            # Check if we're already long this symbol
            if symbol not in portfolio_state['long_symbols']:
                return False, f"🚫 Max correlated longs reached ({current_longs}/{self.max_correlated_longs})"

        # Rule 3: Relative Strength Requirement for Altcoins
        if symbol in ['SOL-USD', 'ETH-USD'] and relative_strength:
            if relative_strength['strength_category'] == 'strong_underperformer':
                return False, f"🚫 {symbol} severely underperforming BTC ({relative_strength['outperformance']:.1f}%)"

        # Rule 4: BTC must be analyzed before allowing altcoin trades
        if symbol in ['SOL-USD', 'ETH-USD'] and not self.btc_sentiment:
            return False, f"⚠️ BTC sentiment not yet analyzed - waiting for BTC analysis"

        # All checks passed
        reasons = [f"✓ BTC trend: {btc_trend}"]
        if relative_strength:
            reasons.append(f"✓ Relative strength: {relative_strength['strength_category']}")
        reasons.append(f"✓ Portfolio exposure: {current_longs}/{self.max_correlated_longs} positions")

        return True, " | ".join(reasons)

    def adjust_confidence_for_correlation(self,
                                         symbol: str,
                                         base_confidence: str,
                                         btc_trend: str,
                                         asset_trend: str,
                                         relative_strength: Optional[Dict] = None) -> str:
        """
        Adjust confidence level based on BTC correlation and relative strength.

        Args:
            symbol: Trading pair
            base_confidence: Original confidence from technical analysis
            btc_trend: BTC market trend
            asset_trend: Asset's own market trend
            relative_strength: Relative strength metrics

        Returns:
            Adjusted confidence level ('high', 'medium', 'low')
        """
        confidence_score = {'high': 3, 'medium': 2, 'low': 1}.get(base_confidence, 0)

        if symbol in ['SOL-USD', 'ETH-USD']:
            # Boost confidence if aligned with BTC
            if btc_trend == asset_trend:
                if btc_trend == 'bullish':
                    confidence_score += 1  # BTC bullish + asset bullish = strong signal
                elif btc_trend == 'bearish':
                    confidence_score += 0.5  # Both bearish = consistent downtrend

            # Reduce confidence if diverging from BTC
            if btc_trend == 'bullish' and asset_trend == 'bearish':
                confidence_score -= 1  # Conflicting signals
            elif btc_trend == 'bearish' and asset_trend == 'bullish':
                confidence_score -= 1.5  # Fighting the tide

            # Adjust for relative strength
            if relative_strength:
                if relative_strength['strength_category'] == 'strong_outperformer':
                    confidence_score += 0.5
                elif relative_strength['strength_category'] == 'strong_underperformer':
                    confidence_score -= 0.5

        # Convert score back to confidence level
        if confidence_score >= 3.5:
            return 'high'
        elif confidence_score >= 2:
            return 'medium'
        elif confidence_score >= 1:
            return 'low'
        else:
            return 'low'

    def calculate_correlation_adjusted_position_size(self,
                                                     base_position_size: float,
                                                     portfolio_state: Dict,
                                                     symbol: str) -> float:
        """
        Adjust position size based on existing correlated positions.

        Args:
            base_position_size: Position size from volatility scaling
            portfolio_state: Current portfolio state
            symbol: Trading pair

        Returns:
            Adjusted position size
        """
        current_longs = len(portfolio_state['long_positions'])

        # No adjustment needed if first position
        if current_longs == 0:
            return base_position_size

        # Scale down position size as we add more correlated positions
        # Position 1: 100% of base
        # Position 2: 75% of base
        # Position 3: 50% of base
        reduction_per_position = 0.25
        adjustment_factor = 1.0 - (current_longs * reduction_per_position)
        adjustment_factor = max(0.5, adjustment_factor)  # Never go below 50%

        adjusted_size = base_position_size * adjustment_factor

        return adjusted_size

    def generate_correlation_report(self,
                                   btc_prices: List[float],
                                   sol_prices: List[float],
                                   eth_prices: List[float]) -> Dict:
        """
        Generate comprehensive correlation report across all assets.

        Args:
            btc_prices: BTC price history
            sol_prices: SOL price history
            eth_prices: ETH price history

        Returns:
            Comprehensive correlation metrics
        """
        # Calculate pairwise correlations
        btc_sol_corr = self.calculate_correlation(btc_prices, sol_prices)
        btc_eth_corr = self.calculate_correlation(btc_prices, eth_prices)
        sol_eth_corr = self.calculate_correlation(sol_prices, eth_prices)

        # Calculate relative strengths
        sol_rs = self.calculate_relative_strength(sol_prices, btc_prices, self.correlation_lookback_hours)
        eth_rs = self.calculate_relative_strength(eth_prices, btc_prices, self.correlation_lookback_hours)

        # Portfolio correlation score (avg of all correlations)
        avg_correlation = statistics.mean([abs(btc_sol_corr), abs(btc_eth_corr), abs(sol_eth_corr)])

        return {
            'timestamp': datetime.now().isoformat(),
            'correlations': {
                'BTC-SOL': btc_sol_corr,
                'BTC-ETH': btc_eth_corr,
                'SOL-ETH': sol_eth_corr,
                'average': round(avg_correlation, 3)
            },
            'relative_strength': {
                'SOL': sol_rs,
                'ETH': eth_rs
            },
            'btc_trend': self.btc_trend,
            'interpretation': self._interpret_correlation(avg_correlation)
        }

    def _interpret_correlation(self, avg_correlation: float) -> str:
        """Interpret correlation strength."""
        if avg_correlation >= 0.8:
            return "Very high correlation - assets moving in lockstep"
        elif avg_correlation >= 0.6:
            return "High correlation - diversification limited"
        elif avg_correlation >= 0.4:
            return "Moderate correlation - some diversification benefit"
        elif avg_correlation >= 0.2:
            return "Low correlation - good diversification"
        else:
            return "Very low correlation - strong diversification"


# Utility functions for integration
def load_correlation_manager(config_path: str = 'config.json') -> CorrelationManager:
    """Factory function to create and return correlation manager."""
    return CorrelationManager(config_path)
