"""
Market Trend Rotation Filter

Analyzes all enabled assets and ranks them by trend strength.
Only trades assets in uptrend/sideways, avoiding downtrends.

This allows capital to be deployed to the best opportunities while
avoiding assets in bear trends.

Strategy:
1. Check trend for all assets
2. Rank by trend quality (uptrend > sideways > downtrend)
3. Only trade top-ranked assets
4. Redistribute capital to bullish assets
"""

from utils.adaptive_mean_reversion import detect_market_trend
from utils.file_helpers import get_property_values_from_crypto_file
import json


def analyze_all_asset_trends(lookback_hours=168):
    """
    Analyze trend for all enabled assets

    Args:
        lookback_hours: Period to analyze (default 1 week)

    Returns:
        List of dicts sorted by trend strength:
        [{
            'symbol': 'ETH-USD',
            'trend': 'uptrend',
            'trend_score': 2,  # 2=uptrend, 1=sideways, 0=downtrend
            'price_change_pct': 5.2,
            'current_price': 3000.50,
            'should_trade': True
        }, ...]
    """
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    enabled_wallets = [w for w in config['wallets'] if w['enabled']]

    asset_analysis = []

    for wallet in enabled_wallets:
        symbol = wallet['symbol']

        # Get price data
        prices = get_property_values_from_crypto_file(
            'coinbase-data',
            symbol,
            'price',
            max_age_hours=lookback_hours + 24
        )

        if not prices or len(prices) < lookback_hours:
            # Insufficient data
            asset_analysis.append({
                'symbol': symbol,
                'trend': 'unknown',
                'trend_score': -1,
                'price_change_pct': 0,
                'current_price': 0,
                'should_trade': False,
                'reason': 'Insufficient data'
            })
            continue

        current_price = prices[-1]

        # Detect trend
        trend = detect_market_trend(prices, lookback=lookback_hours)

        # Calculate price change over period
        start_price = prices[-lookback_hours]
        price_change_pct = ((current_price - start_price) / start_price) * 100

        # Assign trend score
        if trend == 'uptrend':
            trend_score = 2
            should_trade = True
            reason = f"Uptrend: +{price_change_pct:.1f}% over {lookback_hours//24} days"
        elif trend == 'sideways':
            trend_score = 1
            should_trade = True
            reason = f"Sideways: {price_change_pct:+.1f}% range (tradeable)"
        else:  # downtrend
            trend_score = 0
            should_trade = False
            reason = f"Downtrend: {price_change_pct:.1f}% decline (avoid)"

        asset_analysis.append({
            'symbol': symbol,
            'trend': trend,
            'trend_score': trend_score,
            'price_change_pct': price_change_pct,
            'current_price': current_price,
            'should_trade': should_trade,
            'reason': reason
        })

    # Sort by trend score (best first)
    asset_analysis.sort(key=lambda x: (x['trend_score'], x['price_change_pct']), reverse=True)

    return asset_analysis


def get_tradeable_assets():
    """
    Get list of assets that are currently tradeable (uptrend or sideways)

    Returns:
        List of symbol strings, e.g. ['SOL-USD', 'ETH-USD']
    """
    analysis = analyze_all_asset_trends()
    return [a['symbol'] for a in analysis if a['should_trade']]


def should_trade_asset(symbol):
    """
    Check if a specific asset should be traded right now

    Args:
        symbol: Asset symbol (e.g. 'ETH-USD')

    Returns:
        Boolean
    """
    analysis = analyze_all_asset_trends()

    for asset in analysis:
        if asset['symbol'] == symbol:
            return asset['should_trade']

    return False


def get_market_rotation_summary():
    """
    Get a summary of current market rotation status

    Returns:
        Dict with summary info for logging/display
    """
    analysis = analyze_all_asset_trends()

    tradeable = [a for a in analysis if a['should_trade']]
    uptrend = [a for a in analysis if a['trend'] == 'uptrend']
    sideways = [a for a in analysis if a['trend'] == 'sideways']
    downtrend = [a for a in analysis if a['trend'] == 'downtrend']

    return {
        'total_assets': len(analysis),
        'tradeable_count': len(tradeable),
        'uptrend_count': len(uptrend),
        'sideways_count': len(sideways),
        'downtrend_count': len(downtrend),
        'tradeable_symbols': [a['symbol'] for a in tradeable],
        'best_asset': tradeable[0] if tradeable else None,
        'all_assets': analysis
    }


def print_rotation_summary():
    """
    Print a formatted summary of market rotation status
    """
    summary = get_market_rotation_summary()

    print("="*80)
    print("MARKET ROTATION ANALYSIS")
    print("="*80)
    print()
    print(f"Total Assets Monitored: {summary['total_assets']}")
    print(f"  ✅ Tradeable (Uptrend/Sideways): {summary['tradeable_count']}")
    print(f"  ⚠️  Downtrend (Avoid): {summary['downtrend_count']}")
    print()

    if summary['tradeable_count'] > 0:
        print("📈 TRADEABLE ASSETS (ranked by opportunity):")
        print("-"*80)

        for i, asset in enumerate(summary['all_assets'], 1):
            if asset['should_trade']:
                trend_emoji = "📈" if asset['trend'] == 'uptrend' else "↔️"
                print(f"{i}. {trend_emoji} {asset['symbol']}: {asset['reason']}")

        print()
        print(f"🎯 Focus trading on: {', '.join(summary['tradeable_symbols'])}")
    else:
        print("⚠️  NO TRADEABLE ASSETS - All assets in downtrend")
        print("   Recommendation: Wait for market conditions to improve")

    print()

    if summary['downtrend_count'] > 0:
        print("🔴 AVOID THESE ASSETS (in downtrend):")
        print("-"*80)
        for asset in summary['all_assets']:
            if not asset['should_trade'] and asset['trend'] == 'downtrend':
                print(f"  • {asset['symbol']}: {asset['reason']}")

    print()
    print("="*80)
