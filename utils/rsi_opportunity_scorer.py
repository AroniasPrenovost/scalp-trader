"""
RSI Mean Reversion Opportunity Scorer

Scans configured RSI mean reversion symbols and ranks them based on
RSI oversold conditions. Drop-in replacement for mtf_opportunity_scorer.

Scoring:
  - RSI <= 15: score 95 (deeply oversold)
  - RSI <= 20: score 90 (oversold - primary entry)
  - RSI 20-25: score 75-85 (approaching oversold)
  - RSI > 25: no signal (score 0)

Only scores symbols listed in config.rsi_mean_reversion.symbols.
"""

from typing import Dict, List, Optional
from utils.rsi_mean_reversion_strategy import check_rsi_mean_reversion_signal
from utils.coinbase import get_asset_price


def score_rsi_opportunity(
    symbol: str,
    config: Dict,
    coinbase_client,
    current_price: float = None
) -> Dict:
    """
    Score a single asset based on RSI Mean Reversion strategy.

    Args:
        symbol: Product ID (e.g., 'ATOM-USD')
        config: Full config dictionary
        coinbase_client: Coinbase API client
        current_price: Optional current price (will fetch if not provided)

    Returns:
        Opportunity dictionary with score, signal, and strategy details
    """
    rsi_config = config.get('rsi_mean_reversion', {})
    symbols_config = rsi_config.get('symbols', {})
    rsi_period = rsi_config.get('rsi_period', 14)
    data_retention_config = config.get('data_retention', {})
    max_age_hours = data_retention_config.get('max_hours', 5040)

    error_result = {
        'symbol': symbol,
        'score': 0,
        'signal': 'no_signal',
        'confidence': 'low',
        'strategy': 'rsi_mean_reversion',
        'entry_price': None,
        'stop_loss': None,
        'profit_target': None,
        'reasoning': '',
        'metrics': {},
        'has_signal': False,
        'error': None,
        'risk_reward_ratio': None
    }

    # Check if this symbol is configured for RSI strategy
    if not rsi_config.get('enabled', False) or symbol not in symbols_config:
        return {
            **error_result,
            'reasoning': f"{symbol} not configured for RSI mean reversion strategy"
        }

    symbol_params = symbols_config[symbol]
    symbol_params['rsi_period'] = rsi_period  # Inject global RSI period

    # Get current price if not provided
    if current_price is None:
        try:
            current_price = get_asset_price(coinbase_client, symbol)
        except Exception as e:
            return {
                **error_result,
                'reasoning': f"Error fetching price: {e}",
                'error': str(e)
            }

    # Get RSI strategy signal
    try:
        signal_result = check_rsi_mean_reversion_signal(
            symbol=symbol,
            timeframe_minutes=symbol_params.get('timeframe_minutes', 15),
            config_params=symbol_params,
            data_directory='coinbase-data',
            current_price=current_price,
            max_age_hours=max_age_hours
        )
    except Exception as e:
        return {
            **error_result,
            'reasoning': f"Error checking RSI signal: {e}",
            'error': str(e)
        }

    signal = signal_result.get('signal', 'no_signal')
    confidence = signal_result.get('confidence', 'low')
    entry_price = signal_result.get('entry_price')
    stop_loss = signal_result.get('stop_loss')
    profit_target = signal_result.get('profit_target')
    reasoning = signal_result.get('reasoning', '')
    metrics = signal_result.get('metrics', {})

    # Calculate score
    score = 0
    if signal == 'buy':
        rsi_value = metrics.get('rsi', 50)

        if rsi_value <= 15:
            score = 95
        elif rsi_value <= 18:
            score = 92
        elif rsi_value <= 20:
            score = 90
        elif rsi_value <= 22:
            score = 82
        elif rsi_value <= 25:
            score = 75
        else:
            score = 0

        # Confidence bonus
        if confidence == 'high':
            score = min(100, score + 3)
        elif confidence == 'medium':
            score = min(100, score + 1)

    # Calculate risk/reward ratio
    risk_reward = None
    if entry_price and stop_loss and profit_target:
        risk = entry_price - stop_loss
        reward = profit_target - entry_price
        if risk > 0:
            risk_reward = reward / risk

    return {
        'symbol': symbol,
        'score': score,
        'signal': signal,
        'confidence': confidence,
        'strategy': 'rsi_mean_reversion',
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'profit_target': profit_target,
        'reasoning': reasoning,
        'metrics': metrics,
        'has_signal': signal == 'buy',
        'error': None,
        'risk_reward_ratio': risk_reward
    }


def find_best_opportunities(
    config: Dict,
    coinbase_client,
    min_score: float = 75,
    max_opportunities: int = 2,
    exclude_symbols: List[str] = None
) -> List[Dict]:
    """
    Scan all RSI-configured assets and return the best opportunities.

    Args:
        config: Full config dictionary
        coinbase_client: Coinbase API client
        min_score: Minimum score to consider (default 75)
        max_opportunities: Maximum number of opportunities to return
        exclude_symbols: Symbols to exclude (e.g., already have positions)

    Returns:
        List of opportunity dicts, sorted by score descending
    """
    if exclude_symbols is None:
        exclude_symbols = []

    rsi_config = config.get('rsi_mean_reversion', {})
    symbols_config = rsi_config.get('symbols', {})

    # Only scan symbols configured for RSI strategy
    scan_symbols = [s for s in symbols_config.keys() if s not in exclude_symbols]

    opportunities = []
    for symbol in scan_symbols:
        opp = score_rsi_opportunity(symbol, config, coinbase_client)
        opportunities.append(opp)

    valid = [o for o in opportunities if o['score'] >= min_score and o['has_signal']]
    valid.sort(key=lambda x: x['score'], reverse=True)

    return valid[:max_opportunities]


def find_best_opportunity(
    config: Dict,
    coinbase_client,
    min_score: float = 75,
    exclude_symbols: List[str] = None,
    **kwargs
) -> Optional[Dict]:
    """
    Find the single best RSI opportunity.

    Accepts **kwargs for backward compatibility with index.py calls that pass
    enabled_symbols, interval_seconds, data_retention_hours, entry_fee_pct,
    exit_fee_pct, tax_rate_pct, return_multiple, max_opportunities.

    Args:
        config: Full config dictionary
        coinbase_client: Coinbase API client
        min_score: Minimum score to consider
        exclude_symbols: Symbols to exclude
        **kwargs: Extra params from index.py (absorbed for compatibility)

    Returns:
        Single best opportunity dict, or None if no valid opportunities.
        If return_multiple=True in kwargs, returns a list instead.
    """
    return_multiple = kwargs.get('return_multiple', False)
    max_opportunities = kwargs.get('max_opportunities', 2)
    enabled_symbols = kwargs.get('enabled_symbols')

    # Build exclude list
    if exclude_symbols is None:
        exclude_symbols = []

    # If enabled_symbols is provided, we still only scan RSI-configured symbols
    # but we can use it to further filter
    rsi_config = config.get('rsi_mean_reversion', {})
    symbols_config = rsi_config.get('symbols', {})

    if enabled_symbols:
        # Only scan RSI symbols that are also in the enabled wallets list
        effective_exclude = [s for s in symbols_config.keys()
                           if s not in enabled_symbols or s in exclude_symbols]
    else:
        effective_exclude = exclude_symbols

    if return_multiple:
        return find_best_opportunities(
            config=config,
            coinbase_client=coinbase_client,
            min_score=min_score,
            max_opportunities=max_opportunities,
            exclude_symbols=effective_exclude
        )

    opportunities = find_best_opportunities(
        config=config,
        coinbase_client=coinbase_client,
        min_score=min_score,
        max_opportunities=1,
        exclude_symbols=effective_exclude
    )

    return opportunities[0] if opportunities else None


def score_opportunity(
    symbol: str,
    config: Dict,
    coinbase_client,
    current_price: float = None,
    **kwargs
) -> Dict:
    """
    Score a single opportunity. Alias for score_rsi_opportunity.

    Accepts **kwargs for backward compatibility with index.py calls that pass
    coin_prices_list, range_percentage_from_min, etc.
    """
    return score_rsi_opportunity(symbol, config, coinbase_client, current_price)


def print_opportunity_report(
    all_opportunities,
    best_opportunity=None,
    racing_opportunities=None,
    current_prices=None,
    entry_fee_pct=None,
    tax_rate_pct=None,
    trading_capital=None,
    config=None
):
    """
    Print a report of all RSI opportunities.

    Accepts flexible arguments for backward compatibility with index.py.
    """
    if racing_opportunities is None:
        racing_opportunities = []

    # Handle case where all_opportunities is a list of scored dicts
    selected_symbols = []
    if best_opportunity:
        if isinstance(best_opportunity, dict):
            selected_symbols.append(best_opportunity.get('symbol'))
        elif isinstance(best_opportunity, list):
            selected_symbols.extend([o.get('symbol') for o in best_opportunity])
    for opp in racing_opportunities:
        if isinstance(opp, dict):
            selected_symbols.append(opp.get('symbol'))

    print("\n" + "=" * 100)
    print("RSI MEAN REVERSION - OPPORTUNITY SCANNER")
    print("=" * 100)
    print()

    # Sort all by score
    if isinstance(all_opportunities, list):
        all_opportunities.sort(key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0, reverse=True)

    print(f"{'Rank':<6} {'Symbol':<12} {'Score':<8} {'RSI':<8} {'Signal':<12} {'Confidence':<12} {'Status':<30}")
    print("-" * 100)

    for i, opp in enumerate(all_opportunities, 1):
        if not isinstance(opp, dict):
            continue
        symbol = opp.get('symbol', '?')
        score = opp.get('score', 0)
        signal = opp.get('signal', 'no_signal').upper().replace('_', ' ')
        confidence = opp.get('confidence', 'low').upper()
        rsi = opp.get('metrics', {}).get('rsi')
        rsi_str = f"{rsi:.1f}" if rsi is not None else "-"

        if symbol in selected_symbols:
            status = "SELECTED FOR ENTRY"
        elif opp.get('has_signal') and score > 0:
            status = f"Ready ({confidence.lower()})"
        elif opp.get('error'):
            status = f"Error: {str(opp['error'])[:20]}"
        else:
            status = "No Signal"

        score_str = f"{score:.1f}" if score > 0 else "-"
        prefix = "-> " if symbol in selected_symbols else "   "
        print(f"{prefix}#{i:<4} {symbol:<12} {score_str:<8} {rsi_str:<8} {signal:<12} {confidence:<12} {status:<30}")

    print()

    if best_opportunity and isinstance(best_opportunity, dict) and best_opportunity.get('has_signal'):
        print("=" * 100)
        print(f"SELECTED: {best_opportunity['symbol']}")
        print("=" * 100)
        print(f"  Score: {best_opportunity['score']:.1f}/100")
        print(f"  RSI: {best_opportunity.get('metrics', {}).get('rsi', 0):.1f}")
        print(f"  Entry: ${best_opportunity['entry_price']:.4f}" if best_opportunity.get('entry_price') else "  Entry: N/A")
        print(f"  Stop Loss: ${best_opportunity['stop_loss']:.4f}" if best_opportunity.get('stop_loss') else "  Stop: N/A")
        print(f"  Target: ${best_opportunity['profit_target']:.4f}" if best_opportunity.get('profit_target') else "  Target: N/A")
        print(f"\n  {best_opportunity.get('reasoning', '')[:200]}")
    elif not any(o.get('has_signal') for o in all_opportunities if isinstance(o, dict)):
        print("=" * 100)
        print("NO OPPORTUNITIES - RSI not oversold on any configured symbol")
        print("=" * 100)
        print("  Waiting for RSI to drop below entry threshold")

    print()
    print("=" * 100)
    print()
