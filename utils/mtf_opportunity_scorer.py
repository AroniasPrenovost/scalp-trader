"""
MTF Momentum Breakout Opportunity Scorer

Scans all enabled crypto assets and ranks them based on the Multi-Timeframe
Momentum Breakout strategy signals.

Designed to work exclusively with the MTF strategy targeting 5-15% swing moves
on 4-hour timeframe with multi-timeframe confirmation.

Scoring converts MTF strategy signals into a 0-100 score based on:
- Signal presence (buy vs no_signal)
- Confidence level (high > medium > low)
- Net profit potential (after fees/taxes)
- Risk/reward ratio
- Signal strength metrics (RSI, MACD, volume, etc.)
"""

from typing import Dict, List, Optional
from utils.mtf_momentum_breakout_strategy import check_mtf_breakout_signal
from utils.coinbase import get_asset_price


def score_mtf_opportunity(
    symbol: str,
    config: Dict,
    coinbase_client,
    current_price: float = None
) -> Dict:
    """
    Score a single asset based on MTF Momentum Breakout strategy.

    Args:
        symbol: Product ID (e.g., 'BTC-USD')
        config: Full config dictionary
        coinbase_client: Coinbase API client
        current_price: Optional current price (will fetch if not provided)

    Returns:
        Dictionary with:
        {
            'symbol': str,
            'score': float (0-100),
            'signal': 'buy' or 'no_signal',
            'confidence': 'high', 'medium', 'low',
            'strategy': 'mtf_momentum_breakout',
            'entry_price': float or None,
            'stop_loss': float or None,
            'profit_target': float or None,
            'reasoning': str,
            'metrics': dict,
            'has_signal': bool,
            'error': str or None
        }
    """

    mtf_config = config.get('mtf_momentum_breakout', {})
    data_retention_config = config.get('data_retention', {})
    max_age_hours = data_retention_config.get('max_hours', 5040)  # Default 210 days if not specified

    # Get current price if not provided
    if current_price is None:
        try:
            current_price = get_asset_price(coinbase_client, symbol)
        except Exception as e:
            return {
                'symbol': symbol,
                'score': 0,
                'signal': 'no_signal',
                'confidence': 'low',
                'strategy': 'mtf_momentum_breakout',
                'entry_price': None,
                'stop_loss': None,
                'profit_target': None,
                'reasoning': f"Error fetching price: {e}",
                'metrics': {},
                'has_signal': False,
                'error': str(e)
            }

    # Get MTF strategy signal
    try:
        signal_result = check_mtf_breakout_signal(
            symbol=symbol,
            data_directory='coinbase-data',
            current_price=current_price,
            entry_fee_pct=0.50,  # Taker fee
            exit_fee_pct=0.50,   # Taker fee
            tax_rate_pct=24.0,
            atr_stop_multiplier=mtf_config.get('atr_stop_multiplier', 2.0),
            target_profit_pct=mtf_config.get('target_profit_pct', 7.5),
            max_age_hours=max_age_hours
        )
    except Exception as e:
        return {
            'symbol': symbol,
            'score': 0,
            'signal': 'no_signal',
            'confidence': 'low',
            'strategy': 'mtf_momentum_breakout',
            'entry_price': None,
            'stop_loss': None,
            'profit_target': None,
            'reasoning': f"Error checking MTF signal: {e}",
            'metrics': {},
            'has_signal': False,
            'error': str(e)
        }

    # Extract signal details
    signal = signal_result.get('signal', 'no_signal')
    confidence = signal_result.get('confidence', 'low')
    entry_price = signal_result.get('entry_price')
    stop_loss = signal_result.get('stop_loss')
    profit_target = signal_result.get('profit_target')
    reasoning = signal_result.get('reasoning', '')
    metrics = signal_result.get('metrics', {})

    # Calculate score (0-100)
    score = 0

    if signal == 'buy':
        # Base score based on confidence
        if confidence == 'high':
            score = 85
        elif confidence == 'medium':
            score = 70
        else:  # low
            score = 55

        # Bonus for strong metrics
        if metrics:
            # Risk/Reward ratio bonus (up to +10 points)
            rr_ratio = metrics.get('risk_reward_ratio', 0)
            if rr_ratio >= 3.0:
                score += 10
            elif rr_ratio >= 2.5:
                score += 7
            elif rr_ratio >= 2.0:
                score += 5
            elif rr_ratio >= 1.5:
                score += 3

            # Net profit potential bonus (up to +5 points)
            net_profit_pct = metrics.get('net_profit_pct', 0)
            if net_profit_pct >= 5.0:
                score += 5
            elif net_profit_pct >= 4.0:
                score += 4
            elif net_profit_pct >= 3.0:
                score += 3
            elif net_profit_pct >= 2.0:
                score += 2
            elif net_profit_pct >= 1.0:
                score += 1

            # RSI in optimal range bonus (up to +3 points)
            rsi = metrics.get('rsi')
            if rsi is not None:
                if 55 <= rsi <= 65:  # Sweet spot
                    score += 3
                elif 52 <= rsi <= 68:
                    score += 2
                elif 50 <= rsi <= 70:
                    score += 1

            # MACD histogram strength (up to +2 points)
            macd_hist = metrics.get('macd_histogram')
            if macd_hist is not None:
                if macd_hist > 0.015:
                    score += 2
                elif macd_hist > 0.01:
                    score += 1

        # Cap score at 100
        score = min(100, score)

    else:
        # No signal = 0 score
        score = 0

    return {
        'symbol': symbol,
        'score': score,
        'signal': signal,
        'confidence': confidence,
        'strategy': 'mtf_momentum_breakout',
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'reasoning': reasoning,
        'metrics': metrics,
        'has_signal': signal == 'buy',
        'error': None
    }


def find_best_opportunities(
    config: Dict,
    coinbase_client,
    min_score: float = 75,
    max_opportunities: int = 2,
    exclude_symbols: List[str] = None
) -> List[Dict]:
    """
    Scan all enabled assets and return the best MTF opportunities.

    Args:
        config: Full config dictionary
        coinbase_client: Coinbase API client
        min_score: Minimum score to consider (default 75)
        max_opportunities: Maximum number of opportunities to return (default 2)
        exclude_symbols: List of symbols to exclude (e.g., already have positions)

    Returns:
        List of opportunity dicts, sorted by score (highest first)
        Limited to max_opportunities
    """

    if exclude_symbols is None:
        exclude_symbols = []

    wallets = config.get('wallets', [])
    enabled_symbols = [
        w['symbol'] for w in wallets
        if w.get('enabled', False) and w['symbol'] not in exclude_symbols
    ]

    opportunities = []

    for symbol in enabled_symbols:
        opp = score_mtf_opportunity(symbol, config, coinbase_client)
        opportunities.append(opp)

    # Filter by min_score and has_signal
    valid_opportunities = [
        opp for opp in opportunities
        if opp['score'] >= min_score and opp['has_signal']
    ]

    # Sort by score descending
    valid_opportunities.sort(key=lambda x: x['score'], reverse=True)

    # Return top N
    return valid_opportunities[:max_opportunities]


def print_opportunity_report(
    all_opportunities: List[Dict],
    selected_opportunities: List[Dict],
    active_positions: List[str] = None
):
    """
    Print a detailed report of all opportunities and which ones were selected.

    Args:
        all_opportunities: List of all scored opportunities
        selected_opportunities: List of opportunities that were selected for trading
        active_positions: List of symbols with active positions
    """

    if active_positions is None:
        active_positions = []

    selected_symbols = [opp['symbol'] for opp in selected_opportunities]

    print("\n" + "="*100)
    print("MTF MOMENTUM BREAKOUT - OPPORTUNITY SCANNER")
    print("="*100)
    print()

    # Sort all by score
    all_opportunities.sort(key=lambda x: x['score'], reverse=True)

    print(f"{'Rank':<6} {'Symbol':<12} {'Score':<8} {'Signal':<12} {'Confidence':<12} {'Status':<30}")
    print("-" * 100)

    for i, opp in enumerate(all_opportunities, 1):
        symbol = opp['symbol']
        score = opp['score']
        signal = opp['signal'].upper().replace('_', ' ')
        confidence = opp['confidence'].upper()

        # Determine status
        if symbol in active_positions:
            status = "🔥 Position Open"
        elif symbol in selected_symbols:
            status = "⭐ SELECTED FOR ENTRY"
        elif opp['has_signal'] and score > 0:
            status = f"Ready ({confidence.lower()})"
        elif opp.get('error'):
            status = f"Error: {opp['error'][:20]}"
        else:
            status = "No Signal"

        # Format score
        score_str = f"{score:.1f}" if score > 0 else "-"

        # Color coding for selected
        if symbol in selected_symbols:
            print(f"→ #{i:<4} {symbol:<12} {score_str:<8} {signal:<12} {confidence:<12} {status:<30}")
        else:
            print(f"  #{i:<4} {symbol:<12} {score_str:<8} {signal:<12} {confidence:<12} {status:<30}")

    print()

    # Show selected opportunity details
    if selected_opportunities:
        print("="*100)
        print(f"🎯 SELECTED OPPORTUNITIES ({len(selected_opportunities)})")
        print("="*100)

        for opp in selected_opportunities:
            print()
            print(f"Symbol: {opp['symbol']}")
            print(f"Score: {opp['score']:.1f}/100")
            print(f"Confidence: {opp['confidence'].upper()}")
            print(f"Entry: ${opp['entry_price']:.2f}" if opp['entry_price'] else "Entry: N/A")
            print(f"Stop Loss: ${opp['stop_loss']:.2f}" if opp['stop_loss'] else "Stop Loss: N/A")
            print(f"Profit Target: ${opp['profit_target']:.2f}" if opp['profit_target'] else "Target: N/A")

            if opp['metrics']:
                metrics = opp['metrics']
                if 'risk_reward_ratio' in metrics:
                    print(f"Risk/Reward: 1:{metrics['risk_reward_ratio']:.2f}")
                if 'net_profit_pct' in metrics:
                    print(f"Net Profit Potential: {metrics['net_profit_pct']:.2f}%")
                if 'rsi' in metrics:
                    print(f"RSI: {metrics['rsi']:.1f}")
                if 'macd_histogram' in metrics:
                    print(f"MACD Histogram: {metrics['macd_histogram']:.4f}")

            print(f"\nReasoning: {opp['reasoning'][:200]}")
            print()

    else:
        print("="*100)
        print("⚠️  NO OPPORTUNITIES MEET CRITERIA")
        print("="*100)
        print()
        print("All assets either:")
        print("- Have no MTF breakout signal")
        print("- Score below minimum threshold")
        print("- Already have open positions")
        print()

    print("="*100)
    print()


def find_best_opportunity(
    config: Dict,
    coinbase_client,
    min_score: float = 75,
    exclude_symbols: List[str] = None
) -> Optional[Dict]:
    """
    Find the single best MTF opportunity (for backward compatibility).

    Args:
        config: Full config dictionary
        coinbase_client: Coinbase API client
        min_score: Minimum score to consider
        exclude_symbols: List of symbols to exclude

    Returns:
        Single best opportunity dict, or None if no valid opportunities
    """

    opportunities = find_best_opportunities(
        config=config,
        coinbase_client=coinbase_client,
        min_score=min_score,
        max_opportunities=1,
        exclude_symbols=exclude_symbols
    )

    return opportunities[0] if opportunities else None


def score_opportunity(
    symbol: str,
    config: Dict,
    coinbase_client,
    current_price: float = None
) -> Dict:
    """
    Alias for score_mtf_opportunity (for backward compatibility with index.py).
    """
    return score_mtf_opportunity(symbol, config, coinbase_client, current_price)
