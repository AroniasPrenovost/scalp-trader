"""
Opportunity Scoring System

Scans all enabled crypto assets and ranks them by trade quality.
Returns the SINGLE best opportunity to trade right now.

This enables active capital rotation: always put money in the best setup,
rather than spreading capital across multiple mediocre opportunities.

STRATEGY: AI-Only Strategy
- Primary: GPT-4 Vision analysis with multi-timeframe chart analysis
- Bonus: Adaptive Mean Reversion (AMR) alignment (adds confidence when present)
- Requirement: Only trades on HIGH confidence AI signals

Scoring considers:
1. AI confidence level (must be HIGH to trade)
2. AI trade recommendation (must be BUY)
3. Market trend (bullish > sideways > bearish)
4. Risk/reward ratio (minimum 2:1, prefer 3:1+)
5. AMR alignment (bonus points if AMR also signals buy)
6. Volatility (penalty for low volatility markets)
"""

import time
from utils.adaptive_mean_reversion import check_adaptive_buy_signal, detect_market_trend
from utils.file_helpers import get_property_values_from_crypto_file
from utils.openai_analysis import load_analysis_from_file
from utils.coinbase import get_asset_price, get_last_order_from_local_json_ledger, detect_stored_coinbase_order_type


def score_opportunity(symbol, config, coinbase_client, coin_prices_list, current_price, range_percentage_from_min):
    """
    Score a single asset's trade opportunity quality using AI-Only Strategy.

    Strategy Flow:
    1. AI Analysis provides primary signal (must be HIGH confidence BUY)
    2. AMR signal checked for alignment bonus (optional, not required)
    3. Scoring based on AI confidence, trend, risk/reward, and volatility

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
            'strategy': 'adaptive_ai',  # strategy name
            'signal': 'buy' or 'no_signal',
            'confidence': 'high',  # from AI analysis
            'trend': 'uptrend',
            'reasoning': 'AI only: Buy $92000, sell $95000 = 3.26% gross, 2.0% net after costs',
            'entry_price': 50000.0,
            'stop_loss': 49000.0,
            'profit_target': 52000.0,
            'risk_reward_ratio': 2.0,
            'can_trade': True,  # False if position already open or recent stop loss
            'amr_signal': {...},  # Raw AMR signal details (for reference)
            'ai_validation': {...}  # AI analysis details (primary signal)
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
        'ai_validation': None,
        'analysis_age_hours': None,
        'next_refresh_hours': None,
        'analysis_price': None  # Price at time of AI analysis
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
    # AI-ONLY STRATEGY (AMR removed)
    # ========================================
    # Get Adaptive Mean Reversion signal for reference only (not required)
    adaptive_signal = check_adaptive_buy_signal(coin_prices_list, current_price)
    result['amr_signal'] = adaptive_signal

    # Get AI Analysis (PRIMARY signal source)
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

        # Store the price at time of analysis (for price delta calculations)
        # Fallback to buy_in_price for older analyses that don't have analysis_price yet
        result['analysis_price'] = ai_analysis.get('analysis_price') or ai_analysis.get('buy_in_price')

        # Calculate analysis age and next refresh time
        analyzed_at = ai_analysis.get('analyzed_at', 0)
        if analyzed_at > 0:
            current_time = time.time()
            analysis_age_hours = (current_time - analyzed_at) / 3600
            result['analysis_age_hours'] = analysis_age_hours

            # Calculate next refresh based on recommendation/confidence
            trade_recommendation = ai_analysis.get('trade_recommendation', 'buy')
            confidence_level = ai_analysis.get('confidence_level', 'low')

            # Default refresh intervals (should match those in should_refresh_analysis and config.json)
            no_trade_refresh_hours = config.get('no_trade_refresh_hours', 1) if config else 1
            low_confidence_wait_hours = config.get('low_confidence_wait_hours', 1) if config else 1
            medium_confidence_wait_hours = config.get('medium_confidence_wait_hours', 1) if config else 1
            high_confidence_max_age_hours = config.get('high_confidence_max_age_hours', 2) if config else 2

            if trade_recommendation == 'no_trade':
                next_refresh_hours = max(0, no_trade_refresh_hours - analysis_age_hours)
            elif confidence_level == 'low':
                next_refresh_hours = max(0, low_confidence_wait_hours - analysis_age_hours)
            elif confidence_level == 'medium':
                next_refresh_hours = max(0, medium_confidence_wait_hours - analysis_age_hours)
            else:  # high confidence
                next_refresh_hours = max(0, high_confidence_max_age_hours - analysis_age_hours)

            result['next_refresh_hours'] = next_refresh_hours

    # ========================================
    # AI-ONLY SCORING LOGIC
    # ========================================
    score = 0

    # REQUIREMENT: AI must recommend BUY with HIGH confidence
    if not ai_analysis or ai_analysis.get('trade_recommendation') != 'buy':
        result['signal'] = 'no_signal'
        result['score'] = 0
        result['reasoning'] = ai_analysis.get('reasoning', 'No AI analysis available') if ai_analysis else 'No AI analysis available'
        return result

    # AI confidence check - only trade on HIGH confidence
    ai_conf = ai_analysis.get('confidence_level', 'low')
    if ai_conf != 'high':
        result['signal'] = 'no_signal'
        result['score'] = 0
        result['reasoning'] = f"AI confidence too low ({ai_conf}) - {ai_analysis.get('reasoning', '')}"
        return result

    # AI recommends BUY with HIGH confidence - start scoring
    score = 70  # Base score for AI high confidence buy

    # Use AI prices
    result['entry_price'] = ai_analysis.get('buy_in_price')
    result['stop_loss'] = ai_analysis.get('stop_loss')
    result['profit_target'] = ai_analysis.get('sell_price')
    result['signal'] = 'buy'
    result['confidence'] = 'high'

    # Bonus for trend alignment
    market_trend = ai_analysis.get('market_trend', 'sideways')
    if market_trend == 'bullish':
        score += 15
    elif market_trend == 'sideways':
        score += 5

    # AMR bonus (if AMR also agrees, add extra confidence)
    if adaptive_signal and adaptive_signal['signal'] == 'buy':
        score += 10
        result['reasoning'] = f"AI + AMR aligned: {ai_analysis.get('reasoning', '')}"
    else:
        result['reasoning'] = f"AI only: {ai_analysis.get('reasoning', '')}"

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

    return result


def find_best_opportunity(config, coinbase_client, enabled_symbols, interval_seconds, data_retention_hours, min_score=0, return_multiple=False, max_opportunities=5):
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
                range_percentage_from_min=range_percentage_from_min
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


def print_opportunity_report(opportunities_list, best_opportunity=None, racing_opportunities=None, current_prices=None, exchange_fee_percentage=1.2, tax_rate_percentage=37, trading_capital_usd=100):
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
    """
    if racing_opportunities is None:
        racing_opportunities = []
    if current_prices is None:
        current_prices = {}
    from utils.time_helpers import print_local_time

    print("\n" + "="*120)
    print("OPPORTUNITY SCANNER - Market Rotation Analysis")
    print("="*120)
    print_local_time()
    print()

    # Sort by score
    sorted_opps = sorted(opportunities_list, key=lambda x: x['score'], reverse=True)

    # Print summary table with new columns
    print(f"{'Rank':<6} {'Symbol':<12} {'Score':<8} {'Signal':<12} {'Strategy':<18} {'Trend':<12} {'AI':<8} {'Age':<10} {'Expires':<10} {'Price Δ':<12} {'Net Profit $':<14} {'Status':<20}")
    print("-"*146)

    for i, opp in enumerate(sorted_opps, 1):
        rank = f"#{i}"
        symbol = opp['symbol']
        score = f"{opp['score']:.1f}"  # Always show score, even if 0
        signal = opp['signal'].upper()
        strategy = "AI Strategy"  # AI-only strategy
        trend = opp['trend'].title()

        # AI validation indicator
        ai_indicator = "✓" if opp.get('ai_validation') else "-"

        # Analysis age
        age_hours = opp.get('analysis_age_hours')
        if age_hours is not None:
            if age_hours < 1:
                age_str = f"{age_hours * 60:.0f}m"
            else:
                age_str = f"{age_hours:.1f}h"
        else:
            age_str = "-"

        # Next refresh time
        refresh_hours = opp.get('next_refresh_hours')
        if refresh_hours is not None:
            if refresh_hours == 0:
                refresh_str = "now"
            elif refresh_hours < 1:
                refresh_str = f"{refresh_hours * 60:.0f}m"
            else:
                refresh_str = f"{refresh_hours:.1f}h"
        else:
            refresh_str = "-"

        # Price change since analysis
        price_change_str = "-"
        if symbol in current_prices and opp.get('analysis_price'):
            current_price = current_prices[symbol]
            analysis_price = opp['analysis_price']
            if current_price and analysis_price:
                price_change_pct = ((current_price - analysis_price) / analysis_price) * 100
                # Format without color first to get the base string
                base_str = f"{price_change_pct:+.2f}%"
                # Add color codes
                if price_change_pct > 0:
                    price_change_str = f"\033[92m{base_str}\033[0m"  # GREEN
                elif price_change_pct < 0:
                    price_change_str = f"\033[91m{base_str}\033[0m"  # RED
                else:
                    price_change_str = base_str

        # Net profit delta in USD (accounting for fees and taxes)
        net_profit_str = "-"
        if symbol in current_prices and opp.get('analysis_price'):
            current_price = current_prices[symbol]
            analysis_price = opp['analysis_price']
            if current_price and analysis_price:
                # Gross price change percentage
                gross_pct = ((current_price - analysis_price) / analysis_price) * 100

                # Calculate net profit after fees and taxes
                # Fees: 2 * exchange_fee_percentage (buy + sell)
                total_fees_pct = 2 * exchange_fee_percentage

                # Net profit = gross - fees - (taxes on profit portion)
                # Simplified: For small moves, approximate as gross - fees - (tax_rate * gross if positive)
                if gross_pct > 0:
                    # On profitable trade: subtract fees and taxes on the profit
                    net_profit_pct = gross_pct - total_fees_pct - (tax_rate_percentage / 100 * gross_pct)
                else:
                    # On losing trade: just subtract fees (no taxes on losses in this context)
                    net_profit_pct = gross_pct - total_fees_pct

                # Convert to USD based on trading capital
                net_profit_usd = (net_profit_pct / 100) * trading_capital_usd

                # Format without color first to get the base string
                base_str = f"{net_profit_usd:+.2f}"
                # Add color codes
                if net_profit_usd > 0:
                    net_profit_str = f"\033[92m{base_str}\033[0m"  # GREEN
                elif net_profit_usd < 0:
                    net_profit_str = f"\033[91m{base_str}\033[0m"  # RED
                else:
                    net_profit_str = base_str

        # Status
        if not opp['can_trade']:
            status = "Position Open"
        elif opp['score'] == 0:
            status = "No Setup"
        elif opp['signal'] == 'buy':
            status = f"✅ Ready ({opp['confidence']})"
        else:
            status = "Waiting"

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

        # Apply padding to colored strings
        price_change_padded = pad_colored(price_change_str, 12)
        net_profit_padded = pad_colored(net_profit_str, 14)

        # Highlight selected opportunity with arrow
        prefix = "→" if best_opportunity and opp['symbol'] == best_opportunity['symbol'] else " "
        print(f"{prefix} {rank:<4} {symbol:<12} {score:<8} {signal:<12} {strategy:<18} {trend:<12} {ai_indicator:<8} {age_str:<10} {refresh_str:<10} {price_change_padded} {net_profit_padded} {status:<20} {indicator}")

    print()

    if best_opportunity:
        print("="*120)
        print(f"🎯 BEST OPPORTUNITY: {best_opportunity['symbol']}")
        print("="*120)
        print(f"Strategy: AI Strategy (GPT-4 Vision + AMR bonus)")
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
        print("="*120)
    else:
        print("⚠️  NO TRADEABLE OPPORTUNITIES FOUND")
        print("   All assets either have positions open or lack strong setups")
        print("="*120)

    print()
