"""
Opportunity Scoring System

Scans all enabled crypto assets and ranks them by trade quality.
Returns the SINGLE best opportunity to trade right now.

This enables active capital rotation: always put money in the best setup,
rather than spreading capital across multiple mediocre opportunities.

UNIFIED STRATEGY: Adaptive AI Strategy
- Core: Adaptive Mean Reversion (53.3% win rate, proven profitable)
- Enhancement: AI validation and confidence adjustment via GPT-4 Vision
- Result: Single consistent strategy that's measurable and profitable

Scoring considers:
1. Adaptive Mean Reversion signal (core requirement)
2. AI confidence level (validates and enhances the setup)
3. Market trend (uptrend > sideways > downtrend)
4. Risk/reward ratio (minimum 2:1, prefer 3:1+)
5. Recent performance (avoid assets that just stopped us out)
"""

import time
from utils.adaptive_mean_reversion import check_adaptive_buy_signal, detect_market_trend
from utils.file_helpers import get_property_values_from_crypto_file
from utils.openai_analysis import load_analysis_from_file
from utils.coinbase import get_asset_price, get_last_order_from_local_json_ledger, detect_stored_coinbase_order_type


def score_opportunity(symbol, config, coinbase_client, coin_prices_list, current_price, range_percentage_from_min):
    """
    Score a single asset's trade opportunity quality using the Unified Adaptive AI Strategy.

    Strategy Flow:
    1. Adaptive Mean Reversion provides core signal (trend, entry, stop, target)
    2. AI Analysis validates and enhances (adjusts confidence, refines prices)
    3. Unified scoring combines both into single measurable strategy

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
            'strategy': 'adaptive_ai',  # unified strategy name
            'signal': 'buy' or 'no_signal',
            'confidence': 'high',  # from AI validation
            'trend': 'uptrend',
            'reasoning': 'AMR: 2.5% dip from MA in uptrend. AI: High confidence, clear support',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'profit_target': 52000.0,
            'risk_reward_ratio': 2.0,
            'can_trade': True,  # False if position already open or recent stop loss
            'amr_signal': {...},  # Raw AMR signal details
            'ai_validation': {...}  # AI analysis details (if available)
        }
    """

    # Initialize result
    result = {
        'symbol': symbol,
        'score': 0,
        'strategy': 'adaptive_ai',  # Always use unified strategy name
        'signal': 'no_signal',
        'confidence': 'low',
        'trend': 'unknown',
        'reasoning': '',
        'entry_price': None,
        'stop_loss': None,
        'profit_target': None,
        'risk_reward_ratio': None,
        'can_trade': True,
        'amr_signal': None,
        'ai_validation': None
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

    # ========================================
    # UNIFIED ADAPTIVE AI STRATEGY
    # ========================================
    # Step 1: Get Adaptive Mean Reversion signal (CORE)
    adaptive_signal = check_adaptive_buy_signal(coin_prices_list, current_price)
    result['amr_signal'] = adaptive_signal

    # Step 2: Get AI Analysis (ENHANCEMENT)
    ai_analysis = load_analysis_from_file(symbol)
    if ai_analysis:
        result['ai_validation'] = {
            'signal': 'buy' if ai_analysis.get('trade_recommendation') == 'buy' else 'no_signal',
            'confidence': ai_analysis.get('confidence_level', 'low'),
            'entry_price': ai_analysis.get('buy_in_price'),
            'stop_loss': ai_analysis.get('stop_loss'),
            'profit_target': ai_analysis.get('sell_price'),
            'reasoning': ai_analysis.get('reasoning', ''),
            'enforced': ai_analysis.get('adaptive_strategy_enforced', False)
        }

    # ========================================
    # UNIFIED SCORING LOGIC
    # ========================================
    score = 0

    # REQUIREMENT: Adaptive Mean Reversion must signal BUY to proceed
    if not adaptive_signal or adaptive_signal['signal'] != 'buy':
        # No AMR signal = no trade
        result['signal'] = 'no_signal'
        result['score'] = 0
        result['reasoning'] = adaptive_signal['reasoning'] if adaptive_signal else "No AMR signal"

        # If downtrend, make it explicit
        if adaptive_signal and adaptive_signal['trend'] == 'downtrend':
            result['reasoning'] = f"AMR: Downtrend detected - {adaptive_signal['reasoning']}"

        return result

    # AMR signal is BUY - start scoring
    score = 60  # Base score for AMR buy signal

    # Use AMR entry/stop/target as defaults
    result['entry_price'] = adaptive_signal['entry_price']
    result['stop_loss'] = adaptive_signal['stop_loss']
    result['profit_target'] = adaptive_signal['profit_target']
    result['signal'] = 'buy'

    # Bonus for trend alignment
    if adaptive_signal['trend'] == 'uptrend':
        score += 10
    elif adaptive_signal['trend'] == 'sideways':
        score += 5

    # Bonus for optimal dip depth (2-3% is AMR sweet spot)
    deviation = abs(adaptive_signal.get('deviation_from_ma', 0))
    if 2.0 <= deviation <= 3.0:
        score += 10
    elif 1.5 <= deviation <= 3.5:
        score += 5

    # AI VALIDATION LAYER
    if result['ai_validation']:
        ai_conf = result['ai_validation']['confidence']
        ai_rec = result['ai_validation']['signal']

        # AI confidence bonus/penalty
        if ai_conf == 'high' and ai_rec == 'buy':
            score += 20
            result['confidence'] = 'high'

            # Use AI-refined prices if available and reasonable
            ai_entry = result['ai_validation']['entry_price']
            ai_stop = result['ai_validation']['stop_loss']
            ai_target = result['ai_validation']['profit_target']

            if ai_entry and ai_stop and ai_target:
                # Only use AI prices if they're within ±2% of AMR prices (prevent wild adjustments)
                entry_diff_pct = abs(ai_entry - result['entry_price']) / result['entry_price'] * 100
                if entry_diff_pct <= 2.0:
                    result['entry_price'] = ai_entry
                    result['stop_loss'] = ai_stop
                    result['profit_target'] = ai_target

        elif ai_conf == 'medium' or ai_rec == 'no_trade':
            score += 5
            result['confidence'] = 'medium'
        elif ai_conf == 'low':
            score -= 10
            result['confidence'] = 'low'
    else:
        # No AI validation available - use AMR confidence
        result['confidence'] = 'high' if adaptive_signal['trend'] == 'uptrend' else 'medium'

    # Calculate risk/reward ratio
    if result['entry_price'] and result['stop_loss'] and result['profit_target']:
        risk = result['entry_price'] - result['stop_loss']
        reward = result['profit_target'] - result['entry_price']
        if risk > 0:
            result['risk_reward_ratio'] = reward / risk

            # Bonus for exceptional risk/reward
            if result['risk_reward_ratio'] >= 3.0:
                score += 10
            elif result['risk_reward_ratio'] >= 2.0:
                score += 5

    # Penalty for low volatility (not enough movement for profit)
    if range_percentage_from_min < 5:
        score = max(0, score - 20)

    # Final score
    result['score'] = min(100, score)  # Cap at 100

    # Build comprehensive reasoning
    amr_reasoning = adaptive_signal.get('reasoning', 'AMR buy signal')
    ai_reasoning = result['ai_validation']['reasoning'] if result['ai_validation'] else 'No AI validation'

    result['reasoning'] = f"AMR: {amr_reasoning[:80]}... | AI: {ai_reasoning[:80]}..." if len(amr_reasoning) > 80 else f"AMR: {amr_reasoning} | AI: {ai_reasoning}"

    return result


def find_best_opportunity(config, coinbase_client, enabled_symbols, interval_seconds, data_retention_hours, min_score=0):
    """
    Scan all enabled assets and return the SINGLE best trade opportunity.

    Args:
        config: Full config dict
        coinbase_client: Coinbase API client
        enabled_symbols: List of symbol strings to scan (e.g. ['BTC-USD', 'ETH-USD'])
        interval_seconds: Data collection interval (for calculating volatility window)
        data_retention_hours: Max age of historical data
        min_score: Minimum score threshold to consider (default: 0, no filtering)

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

    # Filter to only tradeable opportunities (can_trade = True, score >= min_score)
    tradeable = [opp for opp in opportunities if opp['can_trade'] and opp['score'] >= min_score]

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
    print(f"{'Rank':<6} {'Symbol':<12} {'Score':<8} {'Signal':<12} {'Strategy':<18} {'Trend':<12} {'AI':<8} {'Status':<20}")
    print("-"*100)

    for i, opp in enumerate(sorted_opps, 1):
        rank = f"#{i}"
        symbol = opp['symbol']
        score = f"{opp['score']:.1f}" if opp['score'] > 0 else "-"
        signal = opp['signal'].upper()
        strategy = "Adaptive AI"  # Always show unified strategy
        trend = opp['trend'].title()

        # AI validation indicator
        ai_indicator = "✓" if opp.get('ai_validation') else "-"

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
            print(f"→ {rank:<4} {symbol:<12} {score:<8} {signal:<12} {strategy:<18} {trend:<12} {ai_indicator:<8} {status:<20} ⭐ SELECTED")
        else:
            print(f"  {rank:<4} {symbol:<12} {score:<8} {signal:<12} {strategy:<18} {trend:<12} {ai_indicator:<8} {status:<20}")

    print()

    if best_opportunity:
        print("="*100)
        print(f"🎯 BEST OPPORTUNITY: {best_opportunity['symbol']}")
        print("="*100)
        print(f"Strategy: Adaptive AI (AMR + GPT-4 Vision)")
        print(f"Score: {best_opportunity['score']:.1f}/100")
        print(f"Confidence: {best_opportunity['confidence'].upper()}")
        print(f"Trend: {best_opportunity['trend'].title()}")
        print(f"Entry: ${best_opportunity['entry_price']:.4f}")
        print(f"Stop Loss: ${best_opportunity['stop_loss']:.4f}")
        print(f"Profit Target: ${best_opportunity['profit_target']:.4f}")
        if best_opportunity['risk_reward_ratio']:
            print(f"Risk/Reward: 1:{best_opportunity['risk_reward_ratio']:.2f}")

        # Show AMR details
        if best_opportunity.get('amr_signal'):
            amr = best_opportunity['amr_signal']
            print(f"\nAMR Signal: {amr['signal'].upper()} (deviation: {amr.get('deviation_from_ma', 0):.2f}%)")

        # Show AI validation details
        if best_opportunity.get('ai_validation'):
            ai = best_opportunity['ai_validation']
            print(f"AI Validation: {ai['confidence'].upper()} confidence, {ai['signal'].upper()}")

        print(f"\nReasoning: {best_opportunity['reasoning']}")
        print("="*100)
    else:
        print("⚠️  NO TRADEABLE OPPORTUNITIES FOUND")
        print("   All assets either have positions open or lack strong setups")
        print("="*100)

    print()
