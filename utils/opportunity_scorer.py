"""
Opportunity Scoring System

Scans all enabled crypto assets and ranks them by trade quality.
Returns the SINGLE best opportunity to trade right now.

This enables active capital rotation: always put money in the best setup,
rather than spreading capital across multiple mediocre opportunities.

Scoring considers:
1. Strategy match (range support, mean reversion, breakout, etc.)
2. Signal strength (how strong is the setup?)
3. Market trend (uptrend > sideways > downtrend)
4. Confidence level (from AI analysis if available)
5. Recent performance (avoid assets that just stopped us out)
"""

import time
from utils.range_support_strategy import check_range_support_buy_signal
from utils.adaptive_mean_reversion import check_adaptive_buy_signal, detect_market_trend
from utils.file_helpers import get_property_values_from_crypto_file
from utils.openai_analysis import load_analysis_from_file
from utils.coinbase import get_asset_price, get_last_order_from_local_json_ledger, detect_stored_coinbase_order_type


def score_opportunity(symbol, config, coinbase_client, coin_prices_list, current_price, range_percentage_from_min):
    """
    Score a single asset's trade opportunity quality.

    Args:
        symbol: Asset symbol (e.g. 'BTC-USD')
        config: Full config dict
        coinbase_client: Coinbase client for price checks
        coin_prices_list: Historical price data for this asset
        current_price: Current market price
        range_percentage_from_min: Volatility measure (24h range %)

    Returns:
        Dictionary with:
        {
            'symbol': 'BTC-USD',
            'score': 85.5,  # 0-100, higher = better opportunity
            'strategy': 'range_support',  # which strategy detected
            'signal': 'buy' or 'no_signal',
            'confidence': 'high',  # from AI or strategy
            'trend': 'uptrend',
            'reasoning': 'Price at strong support zone with 3 touches...',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'profit_target': 52000.0,
            'risk_reward_ratio': 2.0,
            'can_trade': True  # False if position already open or recent stop loss
        }
    """

    # Initialize result
    result = {
        'symbol': symbol,
        'score': 0,
        'strategy': None,
        'signal': 'no_signal',
        'confidence': 'low',
        'trend': 'unknown',
        'reasoning': '',
        'entry_price': None,
        'stop_loss': None,
        'profit_target': None,
        'risk_reward_ratio': None,
        'can_trade': True
    }

    # Check if we already have a position open for this asset
    # Note: We still score the opportunity, but mark it as unavailable for NEW entries
    last_order = get_last_order_from_local_json_ledger(symbol)
    last_order_type = detect_stored_coinbase_order_type(last_order)
    has_open_position = last_order_type in ['placeholder', 'buy']

    if has_open_position:
        result['can_trade'] = False
        result['reasoning'] = f"Position already open for {symbol} (managing existing trade)"
        # Don't return early - we still want to score it to see if it would be a good opportunity

    # Insufficient data check
    if not coin_prices_list or len(coin_prices_list) < 48:
        result['reasoning'] = f"Insufficient price data ({len(coin_prices_list) if coin_prices_list else 0} points)"
        return result

    # Detect market trend
    trend = detect_market_trend(coin_prices_list, lookback=168)
    result['trend'] = trend

    # STRATEGY 1: Range Support Strategy (best for sideways/ranging markets)
    range_strategy_config = config.get('range_support_strategy', {})
    range_strategy_enabled = range_strategy_config.get('enabled', True)

    range_signal = None
    if range_strategy_enabled:
        range_signal = check_range_support_buy_signal(
            prices=coin_prices_list,
            current_price=current_price,
            min_touches=range_strategy_config.get('min_touches', 2),
            zone_tolerance_percentage=range_strategy_config.get('zone_tolerance_percentage', 3.0),
            entry_tolerance_percentage=range_strategy_config.get('entry_tolerance_percentage', 1.5),
            extrema_order=range_strategy_config.get('extrema_order', 5),
            lookback_window=range_strategy_config.get('lookback_window_hours', 336)
        )

    # STRATEGY 2: Adaptive Mean Reversion (best for trending markets with pullbacks)
    adaptive_signal = check_adaptive_buy_signal(coin_prices_list, current_price)

    # STRATEGY 3: AI Analysis (if available and enabled)
    ai_analysis = load_analysis_from_file(symbol)
    ai_signal = None
    if ai_analysis:
        ai_signal = {
            'signal': 'buy' if ai_analysis.get('trade_recommendation') == 'buy' else 'no_signal',
            'confidence': ai_analysis.get('confidence_level', 'low'),
            'entry_price': ai_analysis.get('buy_in_price'),
            'stop_loss': ai_analysis.get('stop_loss'),
            'profit_target': ai_analysis.get('sell_price'),
            'reasoning': ai_analysis.get('reasoning', '')
        }

    # SCORING LOGIC: Evaluate which strategy has the best setup
    best_score = 0
    best_strategy = None
    selected_signal = None

    # Score Range Support Strategy
    if range_signal and range_signal['signal'] == 'buy':
        score = 50  # Base score for valid signal

        # Bonus for strong support zones (more touches = stronger)
        zone_strength = range_signal.get('zone_strength', 0)
        score += min(zone_strength * 5, 20)  # Up to +20 for very strong zones

        # Bonus for sideways/ranging markets (ideal for this strategy)
        if trend == 'sideways':
            score += 15
        elif trend == 'uptrend':
            score += 10  # Still good in uptrends

        # Bonus for proximity to zone (closer = better)
        distance_from_zone = abs(range_signal.get('distance_from_zone_avg', 0))
        if distance_from_zone < 0.5:
            score += 10
        elif distance_from_zone < 1.0:
            score += 5

        if score > best_score:
            best_score = score
            best_strategy = 'range_support'
            selected_signal = range_signal
            zone = range_signal['zone']
            result['entry_price'] = current_price
            result['stop_loss'] = zone['zone_price_min'] * 0.98  # 2% below zone
            result['profit_target'] = current_price * 1.025  # 2.5% profit target
            result['confidence'] = 'high' if zone_strength >= 3 else 'medium'

    # Score Adaptive Mean Reversion Strategy
    if adaptive_signal and adaptive_signal['signal'] == 'buy':
        score = 45  # Base score

        # Bonus for uptrend (ideal market condition)
        if trend == 'uptrend':
            score += 20
        elif trend == 'sideways':
            score += 15

        # Bonus for optimal dip depth (2-3% is sweet spot)
        deviation = abs(adaptive_signal.get('deviation_from_ma', 0))
        if 2.0 <= deviation <= 3.0:
            score += 15
        elif 1.5 <= deviation <= 3.5:
            score += 10

        if score > best_score:
            best_score = score
            best_strategy = 'adaptive_mean_reversion'
            selected_signal = adaptive_signal
            result['entry_price'] = adaptive_signal['entry_price']
            result['stop_loss'] = adaptive_signal['stop_loss']
            result['profit_target'] = adaptive_signal['profit_target']
            result['confidence'] = 'high' if trend == 'uptrend' else 'medium'

    # Score AI Analysis
    if ai_signal and ai_signal['signal'] == 'buy':
        score = 40  # Base score

        # Bonus for high confidence
        if ai_signal['confidence'] == 'high':
            score += 25
        elif ai_signal['confidence'] == 'medium':
            score += 15
        elif ai_signal['confidence'] == 'low':
            score += 5

        # Bonus for favorable trend
        if trend == 'uptrend':
            score += 15
        elif trend == 'sideways':
            score += 10

        if score > best_score:
            best_score = score
            best_strategy = 'ai_analysis'
            selected_signal = ai_signal
            result['entry_price'] = ai_signal['entry_price']
            result['stop_loss'] = ai_signal['stop_loss']
            result['profit_target'] = ai_signal['profit_target']
            result['confidence'] = ai_signal['confidence']

    # Calculate risk/reward ratio if we have targets
    if result['entry_price'] and result['stop_loss'] and result['profit_target']:
        risk = result['entry_price'] - result['stop_loss']
        reward = result['profit_target'] - result['entry_price']
        if risk > 0:
            result['risk_reward_ratio'] = reward / risk

            # Bonus for good risk/reward (2:1 or better)
            if result['risk_reward_ratio'] >= 2.5:
                best_score += 10
            elif result['risk_reward_ratio'] >= 2.0:
                best_score += 5

    # Penalty for downtrends (reduce score significantly)
    if trend == 'downtrend':
        best_score = max(0, best_score - 30)

    # Penalty for low volatility (not enough movement for profit)
    if range_percentage_from_min < 5:
        best_score = max(0, best_score - 20)

    # Final result
    result['score'] = best_score
    result['strategy'] = best_strategy

    if best_strategy and selected_signal:
        result['signal'] = 'buy'
        result['reasoning'] = selected_signal.get('reasoning', f'{best_strategy} detected buy signal')
    else:
        result['signal'] = 'no_signal'
        if trend == 'downtrend':
            result['reasoning'] = f"Downtrend detected - avoiding {symbol}"
        elif range_percentage_from_min < 5:
            result['reasoning'] = f"Low volatility ({range_percentage_from_min:.1f}%) - insufficient movement"
        else:
            result['reasoning'] = f"No strong setup detected for {symbol}"

    return result


def find_best_opportunity(config, coinbase_client, enabled_symbols, interval_seconds, data_retention_hours):
    """
    Scan all enabled assets and return the SINGLE best trade opportunity.

    Args:
        config: Full config dict
        coinbase_client: Coinbase API client
        enabled_symbols: List of symbol strings to scan (e.g. ['BTC-USD', 'ETH-USD'])
        interval_seconds: Data collection interval (for calculating volatility window)
        data_retention_hours: Max age of historical data

    Returns:
        Dictionary with best opportunity (same format as score_opportunity),
        or None if no opportunities found
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
                range_percentage_from_min=range_percentage_from_min
            )

            opportunities.append(opportunity)

        except Exception as e:
            print(f"  ⚠️  Error scoring {symbol}: {e}")
            continue

    # Filter to only tradeable opportunities (score > 0, can_trade = True)
    tradeable = [opp for opp in opportunities if opp['can_trade'] and opp['score'] > 0]

    if not tradeable:
        return None

    # Sort by score (highest first)
    tradeable.sort(key=lambda x: x['score'], reverse=True)

    return tradeable[0]  # Return the best one


def print_opportunity_report(opportunities_list, best_opportunity=None):
    """
    Print a formatted report of all opportunities and highlight the best one.

    Args:
        opportunities_list: List of all scored opportunities
        best_opportunity: The selected best opportunity (optional)
    """
    from utils.time_helpers import print_local_time

    print("\n" + "="*100)
    print("OPPORTUNITY SCANNER - Market Rotation Analysis")
    print("="*100)
    print_local_time()
    print()

    # Sort by score
    sorted_opps = sorted(opportunities_list, key=lambda x: x['score'], reverse=True)

    # Print summary table
    print(f"{'Rank':<6} {'Symbol':<12} {'Score':<8} {'Signal':<12} {'Strategy':<25} {'Trend':<12} {'Status':<20}")
    print("-"*100)

    for i, opp in enumerate(sorted_opps, 1):
        rank = f"#{i}"
        symbol = opp['symbol']
        score = f"{opp['score']:.1f}" if opp['score'] > 0 else "-"
        signal = opp['signal'].upper()
        strategy = (opp['strategy'] or '-').replace('_', ' ').title()
        trend = opp['trend'].title()

        # Status
        if not opp['can_trade']:
            status = "Position Open"
        elif opp['score'] == 0:
            status = "No Setup"
        elif opp['signal'] == 'buy':
            status = f"✅ Ready ({opp['confidence']})"
        else:
            status = "Waiting"

        # Highlight the best opportunity
        if best_opportunity and opp['symbol'] == best_opportunity['symbol']:
            print(f"→ {rank:<4} {symbol:<12} {score:<8} {signal:<12} {strategy:<25} {trend:<12} {status:<20} ⭐ SELECTED")
        else:
            print(f"  {rank:<4} {symbol:<12} {score:<8} {signal:<12} {strategy:<25} {trend:<12} {status:<20}")

    print()

    if best_opportunity:
        print("="*100)
        print(f"🎯 BEST OPPORTUNITY: {best_opportunity['symbol']}")
        print("="*100)
        print(f"Strategy: {best_opportunity['strategy'].replace('_', ' ').title()}")
        print(f"Score: {best_opportunity['score']:.1f}/100")
        print(f"Confidence: {best_opportunity['confidence'].upper()}")
        print(f"Trend: {best_opportunity['trend'].title()}")
        print(f"Entry: ${best_opportunity['entry_price']:.4f}")
        print(f"Stop Loss: ${best_opportunity['stop_loss']:.4f}")
        print(f"Profit Target: ${best_opportunity['profit_target']:.4f}")
        if best_opportunity['risk_reward_ratio']:
            print(f"Risk/Reward: 1:{best_opportunity['risk_reward_ratio']:.2f}")
        print(f"\nReasoning: {best_opportunity['reasoning']}")
        print("="*100)
    else:
        print("⚠️  NO TRADEABLE OPPORTUNITIES FOUND")
        print("   All assets either have positions open or lack strong setups")
        print("="*100)

    print()
