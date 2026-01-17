"""
Opportunity Scoring System

Scans all enabled crypto assets and ranks them by trade quality.
Returns the SINGLE best opportunity to trade right now.

This enables active capital rotation: always put money in the best setup,
rather than spreading capital across multiple mediocre opportunities.

STRATEGY: Momentum Scalping (Pattern-Based)
- Support Bounce (39% frequency): Buy dips at support (bottom 30% of range)
  * Target: 0.8% profit, Stop: 0.4% loss
- Breakout (30% frequency): Buy momentum near resistance (top 30% of range)
  * Target: 0.8% profit, Stop: 0.4% loss
- Consolidation Break (25% frequency): Buy breakout after 6h+ consolidation
  * Target: 0.6% profit, Stop: 0.4% loss

Scoring considers:
1. Pattern type (support bounce > breakout > consolidation break)
2. Confidence level (high > medium)
3. Volatility (3-15% sweet spot)
4. Market trend context
5. Risk/reward ratio (typically 2:1 or 1.5:1)
6. Position in range, momentum (1h/3h)
"""

import time
import statistics
from utils.file_helpers import get_property_values_from_crypto_file
from utils.coinbase import get_asset_price, get_last_order_from_local_json_ledger, detect_stored_coinbase_order_type
from utils.momentum_scalping_strategy import check_scalp_entry_signal
from utils.price_helpers import calculate_rsi


# ========================================
# QUANTITATIVE SCORING FUNCTIONS
# Based on historical pattern analysis
# ========================================

def score_rsi_component(prices):
    """Score RSI (20% weight) - Based on pattern recognition"""
    rsi = calculate_rsi(prices)
    if rsi is None:
        return 0

    if rsi < 20:
        return 100
    elif rsi < 30:
        return 85
    elif rsi < 45:
        return 75
    elif rsi < 55:
        return 65
    elif rsi < 70:
        return 50
    elif rsi < 80:
        return 30
    else:
        return 20


def score_price_position_component(current_price, prices):
    """Score price position in 24h range (15% weight)"""
    if len(prices) < 24:
        return 0

    recent_prices = prices[-24:]
    price_min = min(recent_prices)
    price_max = max(recent_prices)

    if price_max == price_min:
        return 50

    position = ((current_price - price_min) / (price_max - price_min)) * 100

    if position < 20:
        return 100
    elif position < 40:
        return 85
    elif position < 60:
        return 65
    elif position < 80:
        return 45
    else:
        return 30


def score_price_vs_ma_component(current_price, prices):
    """
    Score price vs MA(24h) (25% weight)
    CRITICAL: Mean reversion has 76.9% win rate for 2.0%+ targets!
    """
    if len(prices) < 24:
        return 0

    ma_24h = statistics.mean(prices[-24:])
    if ma_24h == 0:
        return 0

    pct_vs_ma = ((current_price - ma_24h) / ma_24h) * 100

    if pct_vs_ma < -3:
        return 100  # Far below MA - strongest mean reversion signal
    elif pct_vs_ma < -1.5:
        return 90  # Below MA - proven 76.9% win rate
    elif pct_vs_ma < -0.5:
        return 75  # Slightly below MA
    elif pct_vs_ma < 0.5:
        return 60  # Near MA
    elif pct_vs_ma < 1.5:
        return 45  # Slightly above MA
    elif pct_vs_ma < 3:
        return 30  # Above MA
    else:
        return 20  # Far above MA


def score_volatility_component(prices):
    """Score volatility (15% weight)"""
    if len(prices) < 24:
        return 0

    recent_prices = prices[-24:]
    price_min = min(recent_prices)
    price_max = max(recent_prices)

    if price_min == 0:
        return 0

    volatility = ((price_max - price_min) / price_min) * 100

    if volatility < 5:
        return 30
    elif volatility < 10:
        return 60
    elif volatility < 15:
        return 85
    elif volatility < 20:
        return 100
    else:
        return 90  # Cap to avoid overly chaotic markets


def score_opportunity(symbol, config, coinbase_client, coin_prices_list, current_price, range_percentage_from_min,
                     entry_fee_pct=0.6, exit_fee_pct=0.6, tax_rate_pct=37.0):
    """
    Score a single asset's trade opportunity quality using Momentum Scalping Strategy.

    Strategy Flow:
    1. Check for scalping patterns (support bounce, breakout, consolidation break)
    2. Validate volatility is in sweet spot (3-15%)
    3. Calculate score based on pattern type and quality

    Args:
        symbol: Asset symbol (e.g. 'BTC-USD')
        config: Full config dict
        coinbase_client: Coinbase client for price checks
        coin_prices_list: Historical price data for this asset
        current_price: Current market price
        range_percentage_from_min: Volatility measure (24h range %)
        entry_fee_pct: Entry fee percentage (default: 0.6%)
        exit_fee_pct: Exit fee percentage (default: 0.6%)
        tax_rate_pct: Tax rate percentage (default: 37%)

    Returns:
        Dictionary with:
        {
            'symbol': 'BTC-USD',
            'score': 85.5,  # 0-100, higher = better opportunity
            'strategy': 'momentum_scalping',
            'strategy_type': 'support_bounce',  # or 'breakout', 'consolidation_break'
            'signal': 'buy' or 'no_signal',
            'confidence': 'high' or 'medium',
            'reasoning': 'SUPPORT BOUNCE: Price at 25% of range...',
            'entry_price': 50000.0,
            'stop_loss': 49800.0,  # 0.4% stop
            'profit_target': 50400.0,  # 0.6-0.8% target
            'risk_reward_ratio': 2.0,
            'can_trade': True,  # False if position already open
            'scalp_metrics': {...}  # Position, momentum, volatility metrics
        }
    """

    # Initialize result
    result = {
        'symbol': symbol,
        'score': 0,
        'strategy': 'momentum_scalping',  # Pattern-based scalping
        'strategy_type': None,  # Will be: 'support_bounce', 'breakout', 'consolidation_break'
        'signal': 'no_signal',
        'confidence': 'low',
        'reasoning': '',
        'entry_price': None,
        'stop_loss': None,
        'profit_target': None,
        'risk_reward_ratio': None,
        'can_trade': True,
        'scalp_metrics': None,  # Momentum, position, volatility metrics
        'analysis_price': current_price  # Current price for delta tracking
    }

    # Check if we already have a position open for this asset
    # Note: We still score the opportunity, but mark it as unavailable for NEW entries
    last_order = get_last_order_from_local_json_ledger(symbol)
    last_order_type = detect_stored_coinbase_order_type(last_order)
    has_open_position = last_order_type in ['placeholder', 'buy']

    if has_open_position:
        result['can_trade'] = False

        # Restore original opportunity data from when position was opened
        if last_order and 'original_analysis' in last_order:
            original = last_order['original_analysis']

            # Restore entry price
            if 'average_filled_price' in last_order:
                result['actual_entry_price'] = float(last_order['average_filled_price'])
            elif 'entry_price' in original:
                result['actual_entry_price'] = float(original['entry_price'])

            # Restore strategy details
            if 'strategy_type' in original:
                result['strategy_type'] = original['strategy_type']
            if 'reasoning' in original:
                result['reasoning'] = original['reasoning']
            if 'confidence_level' in original:
                result['confidence'] = original['confidence_level']
            if 'entry_price' in original:
                result['entry_price'] = float(original['entry_price'])
            if 'stop_loss' in original:
                result['stop_loss'] = float(original['stop_loss'])
            if 'profit_target' in original:
                result['profit_target'] = float(original['profit_target'])

            # Calculate score based on how close we are to target
            if result['actual_entry_price'] and result['profit_target']:
                current_gain_pct = ((current_price - result['actual_entry_price']) / result['actual_entry_price']) * 100
                target_gain_pct = ((result['profit_target'] - result['actual_entry_price']) / result['actual_entry_price']) * 100
                if target_gain_pct > 0:
                    progress = (current_gain_pct / target_gain_pct) * 100
                    # Score based on progress toward target (0-100)
                    result['score'] = max(0, min(100, 50 + progress / 2))  # Center around 50, max 100
                else:
                    result['score'] = 50  # Neutral if no target
            else:
                result['score'] = 50  # Neutral score for active position

            result['signal'] = 'holding'  # Custom signal for active positions
        else:
            # Fallback: try to get entry price from ledger
            if last_order and 'average_filled_price' in last_order:
                result['actual_entry_price'] = float(last_order['average_filled_price'])

            result['reasoning'] = f"Position already open for {symbol} (managing existing trade)"

        # Don't return early - we still want to score it to see if it would be a good opportunity

    # Insufficient data check
    if not coin_prices_list or len(coin_prices_list) < 48:
        result['reasoning'] = f"Insufficient price data ({len(coin_prices_list) if coin_prices_list else 0} points)"
        return result

    # ========================================
    # MOMENTUM SCALPING PATTERN DETECTION
    # ========================================
    # Check for scalping entry signals (support bounce, breakout, consolidation break)
    # Pass symbol for intra-hour momentum acceleration tracking
    # Pass fees and tax rate for realistic profit target calculations
    scalp_signal = check_scalp_entry_signal(
        coin_prices_list,
        current_price,
        symbol=symbol,
        entry_fee_pct=entry_fee_pct,
        exit_fee_pct=exit_fee_pct,
        tax_rate_pct=tax_rate_pct
    )

    if not scalp_signal:
        result['reasoning'] = "Failed to analyze scalping patterns"
        return result

    # Store scalping metrics for display
    result['scalp_metrics'] = scalp_signal.get('metrics', {})
    result['reasoning'] = scalp_signal.get('reason', scalp_signal.get('reasoning', ''))

    # Check if we have a BUY signal
    if scalp_signal['signal'] == 'buy':
        # Valid scalp setup found!
        result['signal'] = 'buy'
        result['strategy_type'] = scalp_signal.get('strategy', 'unknown')
        result['confidence'] = scalp_signal.get('confidence', 'medium')
        result['entry_price'] = scalp_signal.get('entry_price')
        result['stop_loss'] = scalp_signal.get('stop_loss')
        result['profit_target'] = scalp_signal.get('profit_target')

        # Calculate risk/reward ratio
        if result['entry_price'] and result['stop_loss'] and result['profit_target']:
            risk = result['entry_price'] - result['stop_loss']
            reward = result['profit_target'] - result['entry_price']
            if risk > 0:
                result['risk_reward_ratio'] = reward / risk

        # Calculate score based on pattern quality
        # Base score by strategy type (based on historical success rates)
        if result['strategy_type'] == 'support_bounce':
            base_score = 75  # 39% frequency - most common
        elif result['strategy_type'] == 'breakout':
            base_score = 70  # 30% frequency
        elif result['strategy_type'] == 'consolidation_break':
            base_score = 65  # 25% frequency, lower target (0.6%)
        else:
            base_score = 60

        # Bonus for high confidence
        if result['confidence'] == 'high':
            base_score += 15
        elif result['confidence'] == 'medium':
            base_score += 5

        # Bonus for good volatility (from metrics)
        if result['scalp_metrics']:
            volatility = result['scalp_metrics'].get('volatility_24h', 0)
            if 3 <= volatility <= 15:  # Sweet spot
                base_score += 10

        result['score'] = min(100, base_score)
    else:
        # No signal - waiting for setup
        result['signal'] = 'no_signal'
        result['score'] = 0

    return result


def find_best_opportunity(config, coinbase_client, enabled_symbols, interval_seconds, data_retention_hours, min_score=0, return_multiple=False, max_opportunities=5,
                         entry_fee_pct=0.6, exit_fee_pct=0.6, tax_rate_pct=37.0):
    """
    Scan all enabled assets and return the best trade opportunity/opportunities.

    Args:
        config: Full config dict
        coinbase_client: Coinbase API client
        enabled_symbols: List of symbol strings to scan (e.g. ['BTC-USD', 'ETH-USD'])
        interval_seconds: Data collection interval (for calculating volatility window)
        data_retention_hours: Max age of historical data
        min_score: Minimum score threshold to consider (default: 0, no filtering)
        return_multiple: If True, return list of top opportunities (up to max_opportunities)
        max_opportunities: Maximum number of opportunities to return when return_multiple=True
        entry_fee_pct: Entry fee percentage (default: 0.6%)
        exit_fee_pct: Exit fee percentage (default: 0.6%)
        tax_rate_pct: Tax rate percentage (default: 37%)

    Returns:
        If return_multiple=False: Dictionary with best opportunity, or None if no opportunities found
        If return_multiple=True: List of top opportunities (up to max_opportunities), or empty list if none found
    """

    opportunities = []

    for symbol in enabled_symbols:
        try:
            # Get current price
            current_price = get_asset_price(coinbase_client, symbol)

            # Get historical price data
            coin_prices_list = get_property_values_from_crypto_file(
                'coinbase-data',
                symbol,
                'price',
                max_age_hours=data_retention_hours
            )

            if not coin_prices_list or len(coin_prices_list) == 0:
                print(f"  ⚠️  {symbol}: No price data available")
                continue

            # Calculate 24h volatility
            volatility_window_hours = 24
            volatility_data_points = int(volatility_window_hours / (interval_seconds / 3600))
            recent_prices = coin_prices_list[-volatility_data_points:] if len(coin_prices_list) >= volatility_data_points else coin_prices_list

            min_price = min(recent_prices)
            max_price = max(recent_prices)
            range_percentage_from_min = ((max_price - min_price) / min_price) * 100

            # Score this opportunity
            opportunity = score_opportunity(
                symbol=symbol,
                config=config,
                coinbase_client=coinbase_client,
                coin_prices_list=coin_prices_list,
                current_price=current_price,
                range_percentage_from_min=range_percentage_from_min,
                entry_fee_pct=entry_fee_pct,
                exit_fee_pct=exit_fee_pct,
                tax_rate_pct=tax_rate_pct
            )

            opportunities.append(opportunity)

        except Exception as e:
            print(f"  ⚠️  Error scoring {symbol}: {e}")
            continue

    # Filter to only tradeable opportunities (can_trade = True, score >= min_score)
    tradeable = [opp for opp in opportunities if opp['can_trade'] and opp['score'] >= min_score]

    if not tradeable:
        return [] if return_multiple else None

    # Sort by score (highest first)
    tradeable.sort(key=lambda x: x['score'], reverse=True)

    if return_multiple:
        # Return top N opportunities
        return tradeable[:max_opportunities]
    else:
        # Return single best opportunity
        return tradeable[0]


def print_opportunity_report(opportunities_list, best_opportunity=None, racing_opportunities=None, current_prices=None, exchange_fee_percentage=1.2, tax_rate_percentage=37, trading_capital_usd=100, config=None):
    """
    Print a formatted report of all opportunities and highlight the best one.

    Args:
        opportunities_list: List of all scored opportunities
        best_opportunity: The selected best opportunity (optional)
        racing_opportunities: List of racing opportunities being monitored (optional)
        current_prices: Dict mapping symbol to current price (optional)
        exchange_fee_percentage: Exchange fee percentage (default 1.2%)
        tax_rate_percentage: Tax rate percentage (default 37%)
        trading_capital_usd: Trading capital in USD (default 100)
        config: Configuration dictionary (optional)
    """
    # Check if rotation display table logging is enabled
    if config and not config.get('logging', {}).get('rotation_display_table', False):
        return

    if racing_opportunities is None:
        racing_opportunities = []
    if current_prices is None:
        current_prices = {}
    # Sort by score
    sorted_opps = sorted(opportunities_list, key=lambda x: x['score'], reverse=True)

    # Print summary table with new columns
    # Show what capital the profit columns are based on
    gross_header = f"Gross $ (${trading_capital_usd:.0f})"
    net_header = f"Net $ (${trading_capital_usd:.0f})"
    print(f"{'Rank':<6} {'Symbol':<12} {'Score':<8} {'Signal':<12} {'Strategy':<18} {'In Trade':<10} {'Dist Entry':<12} {'Pos':<8} {'Mom 1h':<10} {'Price Δ':<12} {gross_header:<20} \033[1m\033[96m{net_header:<18}\033[0m {'Status':<20}")
    print("-"*180)

    for i, opp in enumerate(sorted_opps, 1):
        rank = f"#{i}"
        symbol = opp['symbol']
        score = f"{opp['score']:.1f}"  # Always show score, even if 0
        signal = opp['signal'].upper() if opp['signal'] != 'holding' else 'HOLDING'

        # Display strategy type based on scalping pattern
        strategy_type = opp.get('strategy_type') or ''
        if strategy_type:
            strategy = strategy_type.replace('_', ' ').title()  # "Support Bounce", "Breakout", "Consolidation Break"
        elif opp['signal'] == 'holding':
            strategy = "Holding Position"  # Active trade
        else:
            strategy = "Scalp Wait"  # Waiting for pattern

        # In Trade indicator (whether we have an open position)
        in_trade_str = "YES" if not opp['can_trade'] else "NO"

        # Distance to Entry (how far current price is from entry price)
        dist_to_entry_str = "-"
        # For active positions, use actual entry price; for opportunities, use target entry price
        entry_price_to_use = opp.get('actual_entry_price') if not opp['can_trade'] else opp.get('entry_price')
        if symbol in current_prices and entry_price_to_use:
            current_price = current_prices[symbol]
            if current_price and entry_price_to_use:
                dist_pct = ((current_price - entry_price_to_use) / entry_price_to_use) * 100
                dist_to_entry_str = f"{dist_pct:+.2f}%"

        # Scalping metrics indicator (position in range %)
        scalp_metrics = opp.get('scalp_metrics', {})
        if scalp_metrics and 'position_in_range' in scalp_metrics:
            position_pct = scalp_metrics['position_in_range']
            ai_indicator = f"{position_pct:.0f}%"  # Show position in range
        else:
            ai_indicator = "-"

        # Momentum 1h
        if scalp_metrics and 'momentum_1h' in scalp_metrics:
            momentum_1h = scalp_metrics['momentum_1h']
            age_str = f"{momentum_1h:+.2f}%"
        else:
            age_str = "-"

        # Price change since entry - ONLY show if we're IN a position
        price_change_str = "-"
        if not opp['can_trade'] and symbol in current_prices and opp.get('actual_entry_price'):
            current_price = current_prices[symbol]
            entry_price = opp['actual_entry_price']
            if current_price and entry_price:
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                # Format without color first to get the base string
                base_str = f"{price_change_pct:+.2f}%"
                # Add color codes
                if price_change_pct > 0:
                    price_change_str = f"\033[92m{base_str}\033[0m"  # GREEN
                elif price_change_pct < 0:
                    price_change_str = f"\033[91m{base_str}\033[0m"  # RED
                else:
                    price_change_str = base_str

        # Gross profit in USD (price change from entry to current, before fees/taxes)
        # ONLY show if we're IN a position
        gross_profit_str = "-"
        if not opp['can_trade'] and symbol in current_prices and opp.get('actual_entry_price'):
            current_price = current_prices[symbol]
            entry_price = opp['actual_entry_price']
            if current_price and entry_price:
                # Gross price change percentage
                gross_pct = ((current_price - entry_price) / entry_price) * 100

                # Convert to USD based on trading capital
                gross_profit_usd = (gross_pct / 100) * trading_capital_usd

                # Format without color first to get the base string
                base_str = f"{gross_profit_usd:+.2f}"
                # Add color codes
                if gross_profit_usd > 0:
                    gross_profit_str = f"\033[92m{base_str}\033[0m"  # GREEN
                elif gross_profit_usd < 0:
                    gross_profit_str = f"\033[91m{base_str}\033[0m"  # RED
                else:
                    gross_profit_str = base_str

        # Net profit delta in USD (accounting for fees and taxes)
        # ONLY show if we're IN a position
        # Use ATOMIC profit calculator for accuracy
        net_profit_str = "-"
        if not opp['can_trade'] and symbol in current_prices and opp.get('actual_entry_price'):
            from utils.profit_calculator import calculate_net_profit_from_price_move

            current_price = current_prices[symbol]
            entry_price = opp['actual_entry_price']
            if current_price and entry_price:
                # Calculate using atomic profit calculator
                # Assume 1 share for percentage-based calculation, then scale to trading capital
                profit_calc = calculate_net_profit_from_price_move(
                    entry_price=entry_price,
                    exit_price=current_price,
                    shares=1.0,  # Use 1 share for percentage calculation
                    entry_fee_pct=exchange_fee_percentage,
                    exit_fee_pct=exchange_fee_percentage,
                    tax_rate_pct=tax_rate_percentage
                )

                # Scale net profit % to actual trading capital
                net_profit_usd = (profit_calc['net_profit_pct'] / 100) * trading_capital_usd

                # Format without color first to get the base string
                base_str = f"{net_profit_usd:+.2f}"
                # Add color codes - brighter colors for better visibility
                if net_profit_usd > 0:
                    net_profit_str = f"\033[1m\033[92m{base_str}\033[0m"  # BOLD BRIGHT GREEN
                elif net_profit_usd < 0:
                    net_profit_str = f"\033[1m\033[91m{base_str}\033[0m"  # BOLD BRIGHT RED
                else:
                    net_profit_str = f"\033[1m{base_str}\033[0m"  # BOLD for neutral

        # Status - Make it crystal clear about trade state
        if not opp['can_trade']:
            # Show progress toward target for active positions
            if opp.get('actual_entry_price') and opp.get('profit_target'):
                current_price_check = current_prices.get(symbol)
                if current_price_check:
                    current_gain_pct = ((current_price_check - opp['actual_entry_price']) / opp['actual_entry_price']) * 100
                    target_gain_pct = ((opp['profit_target'] - opp['actual_entry_price']) / opp['actual_entry_price']) * 100
                    if target_gain_pct > 0:
                        progress_pct = (current_gain_pct / target_gain_pct) * 100
                        status = f"🔵 HOLDING ({progress_pct:.0f}% to target)"
                    else:
                        status = "🔵 IN POSITION"
                else:
                    status = "🔵 IN POSITION"
            else:
                status = "🔵 IN POSITION"
        elif opp['score'] == 0:
            status = "⏸️  No Setup"
        elif opp['signal'] == 'buy':
            # Not in trade, but ready to enter
            status = f"✅ READY - {opp['confidence'].upper()}"
        else:
            status = "⏳ Waiting"

        # Determine status indicator
        indicator = ""
        if not opp['can_trade']:
            # In position - show star
            indicator = "⭐"
        elif any(race_opp['symbol'] == symbol for race_opp in racing_opportunities):
            # In the running - show car
            indicator = "🏎️"
        # else: show nothing (blank)

        # Pad colored strings manually (ANSI codes mess up alignment)
        # Calculate visible length (excluding ANSI codes)
        def visible_len(s):
            import re
            # Remove ANSI escape codes to get visible length
            return len(re.sub(r'\033\[[0-9;]+m', '', s))

        def pad_colored(s, width):
            # Pad string to width, accounting for ANSI codes
            visible = visible_len(s)
            padding_needed = width - visible
            return s + (' ' * padding_needed) if padding_needed > 0 else s

        # Apply padding to colored strings (adjusted for new header widths)
        price_change_padded = pad_colored(price_change_str, 12)
        gross_profit_padded = pad_colored(gross_profit_str, 20)  # Increased from 16 to 20
        net_profit_padded = pad_colored(net_profit_str, 18)  # Increased from 14 to 18

        # Highlight selected opportunity with arrow
        prefix = "→" if best_opportunity and opp['symbol'] == best_opportunity['symbol'] else " "
        print(f"{prefix} {rank:<4} {symbol:<12} {score:<8} {signal:<12} {strategy:<18} {in_trade_str:<10} {dist_to_entry_str:<12} {ai_indicator:<8} {age_str:<10} {price_change_padded} {gross_profit_padded} {net_profit_padded} {status:<20} {indicator}")

    print()

    if best_opportunity:
        print("="*120)
        print(f"🎯 BEST OPPORTUNITY: {best_opportunity['symbol']}")
        print("="*120)

        # Display scalping pattern type
        strategy_type = best_opportunity.get('strategy_type') or ''
        strategy_display = strategy_type.replace('_', ' ').title() if strategy_type else 'Waiting for Pattern'
        print(f"Strategy: Momentum Scalping - {strategy_display}")
        print(f"Score: {best_opportunity['score']:.1f}/100")
        print(f"Confidence: {best_opportunity['confidence'].upper()}")
        print(f"Entry: ${best_opportunity['entry_price']:.4f}")
        print(f"Stop Loss: ${best_opportunity['stop_loss']:.4f}")
        print(f"Profit Target: ${best_opportunity['profit_target']:.4f}")
        if best_opportunity['risk_reward_ratio']:
            print(f"Risk/Reward: 1:{best_opportunity['risk_reward_ratio']:.2f}")

        # Show scalping metrics
        if best_opportunity.get('scalp_metrics'):
            metrics = best_opportunity['scalp_metrics']
            print(f"\nScalping Metrics:")
            print(f"  Position in Range: {metrics.get('position_in_range', 0):.1f}%")
            print(f"  Momentum (1h): {metrics.get('momentum_1h', 0):+.2f}%")
            print(f"  Momentum (3h): {metrics.get('momentum_3h', 0):+.2f}%")
            print(f"  Volatility (24h): {metrics.get('volatility_24h', 0):.2f}%")
            if 'support_level' in metrics:
                print(f"  Support: ${metrics['support_level']:.4f}")
            if 'resistance_level' in metrics:
                print(f"  Resistance: ${metrics['resistance_level']:.4f}")

        print(f"\nReasoning: {best_opportunity['reasoning']}")
        print("="*120)
    else:
        print("⚠️  NO TRADEABLE OPPORTUNITIES FOUND")

    print()
