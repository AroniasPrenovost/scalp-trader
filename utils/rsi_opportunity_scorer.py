"""
Multi-Strategy Opportunity Scorer

Scans configured symbols across all enabled strategies and ranks them based on
RSI oversold conditions. Supports three strategies:
  - rsi_mean_reversion: Pure RSI oversold entry with trailing stop exits
  - rsi_regime: RSI oversold + EMA regime filter (prevents freefall entries)
  - co_revert: RSI oversold + Bollinger Band filter with fixed profit/stop

Scoring:
  - RSI <= 15: score 95 (deeply oversold)
  - RSI <= 20: score 90 (oversold - primary entry)
  - RSI 20-25: score 75-85 (approaching oversold)
  - RSI > 25: no signal (score 0)
"""

from typing import Dict, List, Optional
from utils.rsi_mean_reversion_strategy import check_rsi_mean_reversion_signal
from utils.rsi_regime_strategy import check_rsi_regime_signal
from utils.co_revert_strategy import check_co_revert_signal
from utils.coinbase import get_asset_price, get_volume_and_fee_summary, get_last_order_from_local_json_ledger, detect_stored_coinbase_order_type
from utils.wallet_helpers import load_transaction_history, calculate_wallet_metrics


# Strategy name -> (config_key, signal_checker_function)
STRATEGY_REGISTRY = {
    'rsi_mean_reversion': ('rsi_mean_reversion', check_rsi_mean_reversion_signal),
    'rsi_regime': ('rsi_regime', check_rsi_regime_signal),
    'co_revert': ('co_revert', check_co_revert_signal),
}


def _get_all_strategy_symbols(config: Dict) -> List[tuple]:
    """
    Get all (symbol, strategy_name) pairs from all enabled strategy configs.

    Returns:
        List of (symbol, strategy_name) tuples
    """
    pairs = []
    for strategy_name, (config_key, _) in STRATEGY_REGISTRY.items():
        strategy_config = config.get(config_key, {})
        if strategy_config.get('enabled', False):
            symbols = strategy_config.get('symbols', {})
            for symbol in symbols:
                pairs.append((symbol, strategy_name))
    return pairs


def score_rsi_opportunity(
    symbol: str,
    config: Dict,
    coinbase_client,
    current_price: float = None
) -> Dict:
    """
    Score a single asset across all configured strategies.

    Checks which strategies this symbol is configured for and returns
    the highest-scoring opportunity.

    Args:
        symbol: Product ID (e.g., 'ATOM-USD')
        config: Full config dictionary
        coinbase_client: Coinbase API client
        current_price: Optional current price (will fetch if not provided)

    Returns:
        Opportunity dictionary with score, signal, and strategy details
    """
    data_retention_config = config.get('data_retention', {})
    max_age_hours = data_retention_config.get('max_hours', 5040)

    error_result = {
        'symbol': symbol,
        'score': 0,
        'signal': 'no_signal',
        'confidence': 'low',
        'strategy': 'none',
        'entry_price': None,
        'stop_loss': None,
        'profit_target': None,
        'reasoning': '',
        'metrics': {},
        'has_signal': False,
        'error': None,
        'risk_reward_ratio': None
    }

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

    best_result = None
    best_score = -1

    # Check all strategies this symbol is configured for
    for strategy_name, (config_key, signal_checker) in STRATEGY_REGISTRY.items():
        strategy_config = config.get(config_key, {})
        if not strategy_config.get('enabled', False):
            continue

        symbols_config = strategy_config.get('symbols', {})
        if symbol not in symbols_config:
            continue

        symbol_params = dict(symbols_config[symbol])  # Copy to avoid mutating config
        symbol_params['rsi_period'] = strategy_config.get('rsi_period', 14)

        try:
            signal_result = signal_checker(
                symbol=symbol,
                timeframe_minutes=symbol_params.get('timeframe_minutes', 15),
                config_params=symbol_params,
                data_directory='coinbase-data',
                current_price=current_price,
                max_age_hours=max_age_hours
            )
        except Exception as e:
            continue

        signal = signal_result.get('signal', 'no_signal')
        confidence = signal_result.get('confidence', 'low')
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
                score = 70  # Valid signal above 25 (some strategies use higher thresholds)

            if confidence == 'high':
                score = min(100, score + 3)
            elif confidence == 'medium':
                score = min(100, score + 1)

        if score > best_score:
            best_score = score
            entry_price = signal_result.get('entry_price')
            stop_loss = signal_result.get('stop_loss')
            profit_target = signal_result.get('profit_target')

            risk_reward = None
            if entry_price and stop_loss and profit_target:
                risk = entry_price - stop_loss
                reward = profit_target - entry_price
                if risk > 0:
                    risk_reward = reward / risk

            best_result = {
                'symbol': symbol,
                'score': score,
                'signal': signal,
                'confidence': confidence,
                'strategy': signal_result.get('strategy', strategy_name),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'profit_target': profit_target,
                'reasoning': signal_result.get('reasoning', ''),
                'metrics': metrics,
                'has_signal': signal == 'buy',
                'error': None,
                'risk_reward_ratio': risk_reward
            }

    if best_result:
        return best_result

    return {
        **error_result,
        'reasoning': f"{symbol} not configured for any enabled strategy"
    }


def find_best_opportunities(
    config: Dict,
    coinbase_client,
    min_score: float = 75,
    max_opportunities: int = 2,
    exclude_symbols: List[str] = None
) -> List[Dict]:
    """
    Scan all strategy-configured assets and return the best opportunities.

    Scans across rsi_mean_reversion, rsi_regime, and co_revert strategies.

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

    # Collect all unique symbols across all strategies
    all_pairs = _get_all_strategy_symbols(config)
    unique_symbols = set(sym for sym, _ in all_pairs if sym not in exclude_symbols)

    opportunities = []
    for symbol in unique_symbols:
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
    Find the single best opportunity across all strategies.

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

    if exclude_symbols is None:
        exclude_symbols = []

    # Get all configured symbols across all strategies
    all_pairs = _get_all_strategy_symbols(config)
    all_configured_symbols = set(sym for sym, _ in all_pairs)

    if enabled_symbols:
        # Only scan strategy symbols that are also in the enabled wallets list
        effective_exclude = [s for s in all_configured_symbols
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
    Score a single opportunity across all strategies. Alias for score_rsi_opportunity.

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
    config=None,
    coinbase_client=None
):
    """
    Print a report of all opportunities across strategies.

    Accepts flexible arguments for backward compatibility with index.py.
    """
    if racing_opportunities is None:
        racing_opportunities = []

    selected_symbols = []
    if best_opportunity:
        if isinstance(best_opportunity, dict):
            selected_symbols.append(best_opportunity.get('symbol'))
        elif isinstance(best_opportunity, list):
            selected_symbols.extend([o.get('symbol') for o in best_opportunity])
    for opp in racing_opportunities:
        if isinstance(opp, dict):
            selected_symbols.append(opp.get('symbol'))

    # Detect active positions by checking order ledger for each symbol
    active_position_symbols = set()
    for opp in all_opportunities:
        if isinstance(opp, dict):
            symbol = opp.get('symbol')
            if symbol:
                last_order = get_last_order_from_local_json_ledger(symbol)
                order_type = detect_stored_coinbase_order_type(last_order)
                if order_type in ['buy', 'placeholder']:
                    active_position_symbols.add(symbol)

    print("\n" + "=" * 150)
    print("MULTI-STRATEGY OPPORTUNITY SCANNER")
    print("=" * 150)

    # Display fee tier and volume if client is available
    if coinbase_client:
        vol_summary = get_volume_and_fee_summary(coinbase_client)
        if vol_summary:
            vol = vol_summary['volume_30d']
            tier = vol_summary['tier']
            maker = vol_summary['maker_fee_pct']
            taker = vol_summary['taker_fee_pct']
            print(f"\nTier: {tier} | Fees: {maker:.2f}% maker / {taker:.2f}% taker | 30d Volume: ${vol:,.0f}")

    print()

    if isinstance(all_opportunities, list):
        all_opportunities.sort(key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0, reverse=True)

    print(f"{'Rank':<6} {'Symbol':<12} {'Strategy':<20} {'Score':<8} {'RSI':<8} {'Gross PnL':<12} {'Net PnL':<12} {'Win Rate':<12} {'Volume':<12} {'Position':<10} {'Status':<16}")
    print("-" * 150)

    for i, opp in enumerate(all_opportunities, 1):
        if not isinstance(opp, dict):
            continue
        symbol = opp.get('symbol', '?')
        score = opp.get('score', 0)
        strategy = opp.get('strategy', 'unknown')
        signal = opp.get('signal', 'no_signal').upper().replace('_', ' ')
        confidence = opp.get('confidence', 'low').upper()
        rsi = opp.get('metrics', {}).get('rsi')
        rsi_str = f"{rsi:.1f}" if rsi is not None else "-"

        # Load historical performance data for this symbol
        transactions = load_transaction_history(symbol)
        trades = len(transactions)
        wins = len([t for t in transactions if t.get('total_profit', 0) > 0])
        losses = trades - wins

        # Calculate metrics using wallet_helpers (use 0 starting capital since we only need gross/net)
        metrics = calculate_wallet_metrics(symbol, 0)
        gross_pnl = metrics.get('gross_profit', 0)
        net_pnl = metrics.get('total_profit', 0)

        # Calculate total trading volume (sum of all trade values)
        # Use 'or 0' to handle explicit null values in JSON
        total_volume = sum((t.get('position_sizing', {}).get('buy_amount_usd') or 0) for t in transactions)

        # Format PnL strings with color indicators
        gross_str = f"${gross_pnl:+.2f}" if trades > 0 else "-"
        net_str = f"${net_pnl:+.2f}" if trades > 0 else "-"
        win_rate_str = f"{wins}/{trades}" if trades > 0 else "-"
        volume_str = f"${total_volume:,.0f}" if total_volume > 0 else "-"

        if symbol in selected_symbols:
            status = "SELECTED"
        elif opp.get('has_signal') and score > 0:
            status = f"Ready ({confidence.lower()})"
        elif opp.get('error'):
            status = f"Err: {str(opp['error'])[:15]}"
        else:
            status = "No Signal"

        # Position indicator
        has_position = symbol in active_position_symbols
        position_str = "* ACTIVE *" if has_position else "-"

        score_str = f"{score:.1f}" if score > 0 else "-"
        prefix = "-> " if symbol in selected_symbols else ("** " if has_position else "   ")
        print(f"{prefix}#{i:<4} {symbol:<12} {strategy:<20} {score_str:<8} {rsi_str:<8} {gross_str:<12} {net_str:<12} {win_rate_str:<12} {volume_str:<12} {position_str:<10} {status:<16}")

    print()

    if best_opportunity and isinstance(best_opportunity, dict) and best_opportunity.get('has_signal'):
        print("=" * 150)
        print(f"SELECTED: {best_opportunity['symbol']} [{best_opportunity.get('strategy', 'unknown')}]")
        print("=" * 150)
        print(f"  Score: {best_opportunity['score']:.1f}/100")
        print(f"  RSI: {best_opportunity.get('metrics', {}).get('rsi', 0):.1f}")
        print(f"  Entry: ${best_opportunity['entry_price']:.4f}" if best_opportunity.get('entry_price') else "  Entry: N/A")
        print(f"  Stop Loss: ${best_opportunity['stop_loss']:.4f}" if best_opportunity.get('stop_loss') else "  Stop: N/A")
        print(f"  Target: ${best_opportunity['profit_target']:.4f}" if best_opportunity.get('profit_target') else "  Target: N/A")
        print(f"\n  {best_opportunity.get('reasoning', '')[:200]}")
    elif not any(o.get('has_signal') for o in all_opportunities if isinstance(o, dict)):
        print("=" * 150)
        print("NO OPPORTUNITIES - No oversold signals on any configured symbol")
        print("=" * 150)
        print("  Waiting for RSI to drop below entry thresholds")

    print()
    print("=" * 150)
    print()
