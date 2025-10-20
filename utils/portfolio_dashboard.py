"""
Portfolio Risk Dashboard and Logging
Displays correlation metrics, portfolio exposure, and risk analysis.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional


def print_portfolio_dashboard(correlation_manager, btc_prices: list, sol_prices: list, eth_prices: list):
    """
    Print a comprehensive portfolio risk dashboard to console.

    Args:
        correlation_manager: CorrelationManager instance
        btc_prices: BTC price history
        sol_prices: SOL price history
        eth_prices: ETH price history
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "PORTFOLIO RISK DASHBOARD")
    print("=" * 80)

    # Get portfolio state
    portfolio_state = correlation_manager.get_portfolio_state('transaction_log.json')

    # Section 1: Current Positions
    print("\n📊 CURRENT POSITIONS:")
    print("-" * 80)
    if portfolio_state['total_positions'] == 0:
        print("   No open positions")
    else:
        for position in portfolio_state['long_positions']:
            print(f"   LONG {position['symbol']}")
            print(f"      Entry: ${position['entry_price']:.2f} | Shares: {position['shares']:.4f}")
            print(f"      Value: ${position['usd_value']:.2f} | Opened: {position['entry_time']}")

    print(f"\n   Total Positions: {portfolio_state['total_positions']}")
    print(f"   Total Exposure: ${portfolio_state['total_exposure_usd']:,.2f}")
    print(f"   Correlation-Adjusted Risk: ${portfolio_state['correlation_adjusted_risk']:,.2f}")

    # Section 2: BTC Market Context
    print("\n🔷 BTC MARKET CONTEXT:")
    print("-" * 80)
    if correlation_manager.btc_sentiment:
        btc_s = correlation_manager.btc_sentiment
        print(f"   Trend: {btc_s.get('market_trend', 'N/A').upper()}")
        print(f"   Confidence: {btc_s.get('confidence_level', 'N/A').upper()}")
        print(f"   Recommendation: {btc_s.get('trade_recommendation', 'N/A').upper()}")
        print(f"   Support: ${btc_s.get('major_support', 'N/A')} | Resistance: ${btc_s.get('major_resistance', 'N/A')}")
        print(f"   Volume Trend: {btc_s.get('volume_trend', 'N/A')}")
    else:
        print("   No BTC sentiment data available")

    # Section 3: Correlation Matrix
    print("\n📈 CORRELATION ANALYSIS:")
    print("-" * 80)
    correlation_report = None  # Initialize to None
    if btc_prices and sol_prices and eth_prices:
        correlation_report = correlation_manager.generate_correlation_report(
            btc_prices, sol_prices, eth_prices
        )

        print("   Pairwise Correlations:")
        corr = correlation_report['correlations']
        print(f"      BTC ↔ SOL: {corr['BTC-SOL']:+.3f}")
        print(f"      BTC ↔ ETH: {corr['BTC-ETH']:+.3f}")
        print(f"      SOL ↔ ETH: {corr['SOL-ETH']:+.3f}")
        print(f"      Average:   {corr['average']:+.3f}")
        print(f"\n   Interpretation: {correlation_report['interpretation']}")

        # Relative Strength
        print("\n   Relative Strength vs BTC (7-day):")
        for asset_symbol, rs_data in correlation_report['relative_strength'].items():
            print(f"      {asset_symbol}:")
            print(f"         Performance: {rs_data['asset_change_pct']:+.2f}% (BTC: {rs_data['btc_change_pct']:+.2f}%)")
            print(f"         Outperformance: {rs_data['outperformance']:+.2f}%")
            print(f"         Category: {rs_data['strength_category']}")
    else:
        print("   Insufficient price data for correlation analysis")

    # Section 4: Risk Warnings
    print("\n⚠️  RISK ALERTS:")
    print("-" * 80)
    alerts = []

    # Check for over-concentration
    if portfolio_state['total_positions'] >= 2:
        if correlation_report and correlation_report['correlations']['average'] > 0.7:
            alerts.append("   🔴 HIGH CORRELATION: Multiple positions with 0.7+ correlation - limited diversification")

    # Check for position limits
    max_positions = correlation_manager.max_correlated_longs
    if portfolio_state['total_positions'] >= max_positions:
        alerts.append(f"   🔴 POSITION LIMIT: At max correlated positions ({portfolio_state['total_positions']}/{max_positions})")

    # Check for BTC trend vs altcoin positions
    if correlation_manager.btc_trend == 'bearish' and len(portfolio_state['long_symbols']) > 0:
        altcoins_long = [s for s in portfolio_state['long_symbols'] if s != 'BTC-USD']
        if altcoins_long:
            alerts.append(f"   ⚠️  BTC BEARISH: Holding altcoin longs ({', '.join(altcoins_long)}) in BTC downtrend")

    # Check for weak relative strength
    if correlation_report:
        for asset_symbol, rs_data in correlation_report['relative_strength'].items():
            asset_pair = f"{asset_symbol}-USD"
            if asset_pair in portfolio_state['long_symbols']:
                if rs_data['strength_category'] in ['strong_underperformer', 'mild_underperformer']:
                    alerts.append(f"   ⚠️  WEAK RELATIVE STRENGTH: {asset_symbol} underperforming BTC by {rs_data['outperformance']:+.2f}%")

    if not alerts:
        print("   ✅ No risk alerts at this time")
    else:
        for alert in alerts:
            print(alert)

    print("\n" + "=" * 80 + "\n")


def log_correlation_event(event_type: str, symbol: str, details: Dict, log_path: str = 'correlation_events.json'):
    """
    Log correlation-related trading events to a JSON file.

    Args:
        event_type: Type of event (e.g., 'trade_blocked', 'confidence_adjusted', 'position_sized')
        symbol: Trading pair symbol
        details: Event details dictionary
        log_path: Path to log file
    """
    event = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'symbol': symbol,
        'details': details
    }

    # Load existing log or create new
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
    else:
        log = []

    # Append event
    log.append(event)

    # Keep only last 1000 events
    if len(log) > 1000:
        log = log[-1000:]

    # Save log
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


def generate_correlation_summary_report(correlation_manager, btc_prices: list, sol_prices: list, eth_prices: list) -> str:
    """
    Generate a text summary report of current correlation state.

    Args:
        correlation_manager: CorrelationManager instance
        btc_prices: BTC price history
        sol_prices: SOL price history
        eth_prices: ETH price history

    Returns:
        Formatted text report string
    """
    report_lines = []
    report_lines.append("CORRELATION SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)

    # Portfolio state
    portfolio_state = correlation_manager.get_portfolio_state('transaction_log.json')
    report_lines.append(f"\nPositions: {portfolio_state['total_positions']}")
    report_lines.append(f"Total Exposure: ${portfolio_state['total_exposure_usd']:,.2f}")
    report_lines.append(f"Correlation-Adjusted Risk: ${portfolio_state['correlation_adjusted_risk']:,.2f}")

    # BTC context
    if correlation_manager.btc_sentiment:
        btc_trend = correlation_manager.btc_sentiment.get('market_trend', 'N/A')
        report_lines.append(f"\nBTC Trend: {btc_trend}")

    # Correlations
    if btc_prices and sol_prices and eth_prices:
        correlation_report = correlation_manager.generate_correlation_report(
            btc_prices, sol_prices, eth_prices
        )
        corr = correlation_report['correlations']
        report_lines.append(f"\nAverage Correlation: {corr['average']:.3f}")
        report_lines.append(correlation_report['interpretation'])

    return "\n".join(report_lines)
