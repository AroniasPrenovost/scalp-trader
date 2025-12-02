"""
Daily Summary Email Generator

Generates comprehensive daily trading summary emails with:
- Portfolio performance metrics
- Per-symbol performance breakdown
- Market context and insights
- Visual charts and screenshots
- Trading activity log
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.trade_context import load_transaction_history, calculate_wallet_metrics
from utils.email import send_email_notification


def get_daily_transactions(symbol: str, date: Optional[datetime] = None) -> List[Dict]:
    """
    Get all transactions for a symbol that occurred on a specific date.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC-USD')
        date: Date to filter transactions (defaults to today)

    Returns:
        List of transactions that occurred on the specified date
    """
    if date is None:
        date = datetime.now(timezone.utc)

    transactions = load_transaction_history(symbol)

    # Filter for transactions on the specified date
    daily_transactions = []
    for t in transactions:
        timestamp_str = t.get('timestamp', '')
        if timestamp_str:
            try:
                # Parse ISO format timestamp
                trans_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if trans_date.date() == date.date():
                    daily_transactions.append(t)
            except Exception as e:
                print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")

    return daily_transactions


def calculate_portfolio_metrics(wallets_config: List[Dict]) -> Dict:
    """
    Calculate overall portfolio metrics across all symbols.

    Args:
        wallets_config: List of wallet configurations from config.json

    Returns:
        Dictionary with portfolio-wide metrics
    """
    total_starting_capital = 0.0
    total_current_value = 0.0
    total_gross_profit = 0.0
    total_taxes = 0.0
    total_exchange_fees = 0.0
    total_net_profit = 0.0

    symbol_metrics = {}

    for wallet in wallets_config:
        if not wallet.get('enabled', True):
            continue

        symbol = wallet['symbol']
        starting_capital = wallet['starting_capital_usd']

        metrics = calculate_wallet_metrics(symbol, starting_capital)
        symbol_metrics[symbol] = metrics

        total_starting_capital += metrics['starting_capital_usd']
        total_current_value += metrics['current_usd']
        total_gross_profit += metrics['gross_profit']
        total_taxes += metrics['taxes']
        total_exchange_fees += metrics['exchange_fees']
        total_net_profit += metrics['total_profit']

    percentage_gain = ((total_current_value - total_starting_capital) / total_starting_capital * 100) if total_starting_capital > 0 else 0

    return {
        'total_starting_capital': total_starting_capital,
        'total_current_value': total_current_value,
        'percentage_gain': percentage_gain,
        'total_gross_profit': total_gross_profit,
        'total_taxes': total_taxes,
        'total_exchange_fees': total_exchange_fees,
        'total_net_profit': total_net_profit,
        'symbol_metrics': symbol_metrics
    }


def get_daily_trade_summary(wallets_config: List[Dict], date: Optional[datetime] = None) -> Dict:
    """
    Get summary of trades executed on a specific date.

    Args:
        wallets_config: List of wallet configurations
        date: Date to summarize (defaults to today)

    Returns:
        Dictionary with daily trade statistics
    """
    if date is None:
        date = datetime.now(timezone.utc)

    all_daily_trades = []
    trades_by_symbol = {}

    for wallet in wallets_config:
        if not wallet.get('enabled', True):
            continue

        symbol = wallet['symbol']
        daily_trades = get_daily_transactions(symbol, date)

        if daily_trades:
            trades_by_symbol[symbol] = daily_trades
            all_daily_trades.extend(daily_trades)

    # Calculate daily statistics
    total_trades = len(all_daily_trades)
    winning_trades = [t for t in all_daily_trades if t.get('total_profit', 0) > 0]
    losing_trades = [t for t in all_daily_trades if t.get('total_profit', 0) <= 0]

    daily_profit = sum(t.get('total_profit', 0) for t in all_daily_trades)
    daily_gross = sum(t.get('gross_profit', 0) for t in all_daily_trades)

    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

    best_trade = max(all_daily_trades, key=lambda x: x.get('total_profit', 0)) if all_daily_trades else None
    worst_trade = min(all_daily_trades, key=lambda x: x.get('total_profit', 0)) if all_daily_trades else None

    return {
        'date': date,
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'daily_net_profit': daily_profit,
        'daily_gross_profit': daily_gross,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'trades_by_symbol': trades_by_symbol
    }


def generate_portfolio_value_chart(wallets_config: List[Dict], days: int = 30) -> str:
    """
    Generate a chart showing portfolio value over time.

    Args:
        wallets_config: List of wallet configurations
        days: Number of days to show in chart

    Returns:
        Path to saved chart image
    """
    try:
        # Collect all transactions across symbols
        all_transactions = []
        for wallet in wallets_config:
            if wallet.get('enabled', True):
                symbol = wallet['symbol']
                transactions = load_transaction_history(symbol)
                for t in transactions:
                    t['symbol'] = symbol  # Ensure symbol is in transaction
                all_transactions.extend(transactions)

        if not all_transactions:
            return None

        # Sort by timestamp
        all_transactions.sort(key=lambda x: x.get('timestamp', ''))

        # Calculate cumulative portfolio value over time
        total_starting_capital = sum(w['starting_capital_usd'] for w in wallets_config if w.get('enabled', True))

        dates = []
        values = []
        current_value = total_starting_capital
        now = datetime.now(timezone.utc)

        dates.append(now - timedelta(days=days))
        values.append(total_starting_capital)

        for transaction in all_transactions:
            timestamp_str = transaction.get('timestamp', '')
            if not timestamp_str:
                continue

            try:
                trans_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                # Only include transactions within the date range
                if trans_date >= now - timedelta(days=days):
                    profit = transaction.get('total_profit', 0)
                    current_value += profit
                    dates.append(trans_date)
                    values.append(current_value)
            except Exception as e:
                print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")

        # Add current value as final point
        dates.append(now)
        values.append(current_value)

        # Create the chart
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(dates, values, linewidth=2, color='#2E86DE', marker='o', markersize=4)
        ax.fill_between(dates, values, alpha=0.3, color='#2E86DE')

        # Format
        ax.set_title(f'Portfolio Value - Last {days} Days', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)

        # Add starting and ending value annotations
        ax.annotate(f'${total_starting_capital:,.2f}',
                   xy=(dates[0], values[0]),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

        ax.annotate(f'${current_value:,.2f}',
                   xy=(dates[-1], values[-1]),
                   xytext=(10, -20),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

        plt.tight_layout()

        # Save chart
        os.makedirs('screenshots', exist_ok=True)
        chart_path = f'screenshots/portfolio_value_{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        return chart_path

    except Exception as e:
        print(f"Error generating portfolio value chart: {e}")
        return None


def generate_daily_pnl_chart(wallets_config: List[Dict], days: int = 30) -> str:
    """
    Generate a bar chart showing daily P&L over time.

    Args:
        wallets_config: List of wallet configurations
        days: Number of days to show

    Returns:
        Path to saved chart image
    """
    try:
        # Collect all transactions
        all_transactions = []
        for wallet in wallets_config:
            if wallet.get('enabled', True):
                symbol = wallet['symbol']
                transactions = load_transaction_history(symbol)
                all_transactions.extend(transactions)

        if not all_transactions:
            return None

        # Group transactions by day
        daily_pnl = {}
        now = datetime.now(timezone.utc)

        for transaction in all_transactions:
            timestamp_str = transaction.get('timestamp', '')
            if not timestamp_str:
                continue

            try:
                trans_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                date_key = trans_date.date()

                # Only include recent days
                if trans_date >= now - timedelta(days=days):
                    profit = transaction.get('total_profit', 0)
                    daily_pnl[date_key] = daily_pnl.get(date_key, 0) + profit
            except Exception as e:
                print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")

        if not daily_pnl:
            return None

        # Sort by date
        sorted_dates = sorted(daily_pnl.keys())
        dates = [datetime.combine(d, datetime.min.time()) for d in sorted_dates]
        pnl_values = [daily_pnl[d] for d in sorted_dates]

        # Create the chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Color bars based on profit/loss
        colors = ['#2ECC71' if v >= 0 else '#E74C3C' for v in pnl_values]
        ax.bar(dates, pnl_values, color=colors, alpha=0.7, width=0.8)

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Format
        ax.set_title(f'Daily P&L - Last {days} Days', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Profit/Loss (USD)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save chart
        os.makedirs('screenshots', exist_ok=True)
        chart_path = f'screenshots/daily_pnl_{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        return chart_path

    except Exception as e:
        print(f"Error generating daily P&L chart: {e}")
        return None


def generate_symbol_performance_chart(wallets_config: List[Dict]) -> str:
    """
    Generate a bar chart comparing performance across symbols.

    Args:
        wallets_config: List of wallet configurations

    Returns:
        Path to saved chart image
    """
    try:
        symbols = []
        returns = []

        for wallet in wallets_config:
            if not wallet.get('enabled', True):
                continue

            symbol = wallet['symbol']
            starting_capital = wallet['starting_capital_usd']

            metrics = calculate_wallet_metrics(symbol, starting_capital)

            # Extract symbol name (e.g., "BTC" from "BTC-USD")
            symbol_name = symbol.split('-')[0]
            symbols.append(symbol_name)
            returns.append(metrics['percentage_gain'])

        if not symbols:
            return None

        # Create the chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color bars based on profit/loss
        colors = ['#2ECC71' if v >= 0 else '#E74C3C' for v in returns]
        bars = ax.bar(symbols, returns, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}%',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Format
        ax.set_title('Performance by Symbol (All-Time)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Symbol', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save chart
        os.makedirs('screenshots', exist_ok=True)
        chart_path = f'screenshots/symbol_performance_{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        return chart_path

    except Exception as e:
        print(f"Error generating symbol performance chart: {e}")
        return None


def create_html_email_template(
    portfolio_metrics: Dict,
    daily_summary: Dict,
    wallets_config: List[Dict],
    portfolio_chart_path: Optional[str] = None,
    pnl_chart_path: Optional[str] = None,
    symbol_chart_path: Optional[str] = None
) -> tuple:
    """
    Create HTML email template for daily summary.

    Args:
        portfolio_metrics: Overall portfolio metrics
        daily_summary: Daily trade summary
        wallets_config: Wallet configurations
        portfolio_chart_path: Path to portfolio value chart
        pnl_chart_path: Path to daily P&L chart
        symbol_chart_path: Path to symbol performance chart

    Returns:
        Tuple of (subject, text_content, html_content)
    """
    date_str = daily_summary['date'].strftime('%B %d, %Y')

    # Subject line
    subject = f"Trading Summary - {date_str}"
    if daily_summary['total_trades'] > 0:
        profit_emoji = "📈" if daily_summary['daily_net_profit'] >= 0 else "📉"
        subject = f"{profit_emoji} Trading Summary - {date_str} ({daily_summary['total_trades']} trades)"

    # Text content (plain text fallback)
    text_content = f"""
Daily Trading Summary - {date_str}

PORTFOLIO OVERVIEW
Starting Capital: ${portfolio_metrics['total_starting_capital']:,.2f}
Current Value: ${portfolio_metrics['total_current_value']:,.2f}
Total Return: {portfolio_metrics['percentage_gain']:.2f}%
Net Profit: ${portfolio_metrics['total_net_profit']:,.2f}

DAILY ACTIVITY
Trades: {daily_summary['total_trades']} ({daily_summary['winning_trades']} wins, {daily_summary['losing_trades']} losses)
Win Rate: {daily_summary['win_rate']:.1f}%
Daily P&L: ${daily_summary['daily_net_profit']:,.2f}
"""

    # HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E86DE;
            border-bottom: 3px solid #2E86DE;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #2C3E50;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #2E86DE;
            padding-left: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #2E86DE;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2C3E50;
        }}
        .metric-value.positive {{
            color: #2ECC71;
        }}
        .metric-value.negative {{
            color: #E74C3C;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .trade-list {{
            margin: 20px 0;
        }}
        .trade-item {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            border-left: 4px solid #95A5A6;
        }}
        .trade-item.win {{
            border-left-color: #2ECC71;
        }}
        .trade-item.loss {{
            border-left-color: #E74C3C;
        }}
        .symbol-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .symbol-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border: 2px solid #e0e0e0;
        }}
        .symbol-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2C3E50;
            margin-bottom: 15px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #999;
            font-size: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Daily Trading Summary</h1>
        <p style="font-size: 14px; color: #666;">{date_str}</p>

        <h2>💼 Portfolio Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Starting Capital</div>
                <div class="metric-value">${portfolio_metrics['total_starting_capital']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Current Value</div>
                <div class="metric-value">${portfolio_metrics['total_current_value']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if portfolio_metrics['percentage_gain'] >= 0 else 'negative'}">
                    {portfolio_metrics['percentage_gain']:+.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Net Profit</div>
                <div class="metric-value {'positive' if portfolio_metrics['total_net_profit'] >= 0 else 'negative'}">
                    ${portfolio_metrics['total_net_profit']:+,.2f}
                </div>
            </div>
        </div>
"""

    # Add portfolio value chart if available
    if portfolio_chart_path and os.path.exists(portfolio_chart_path):
        html_content += f"""
        <div class="chart">
            <img src="cid:portfolio_chart" alt="Portfolio Value Chart" />
        </div>
"""

    # Daily activity section
    html_content += f"""
        <h2>📅 Today's Activity</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Trades Executed</div>
                <div class="metric-value">{daily_summary['total_trades']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{daily_summary['win_rate']:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Daily P&L</div>
                <div class="metric-value {'positive' if daily_summary['daily_net_profit'] >= 0 else 'negative'}">
                    ${daily_summary['daily_net_profit']:+,.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">W/L Ratio</div>
                <div class="metric-value">{daily_summary['winning_trades']}/{daily_summary['losing_trades']}</div>
            </div>
        </div>
"""

    # Add daily P&L chart if available
    if pnl_chart_path and os.path.exists(pnl_chart_path):
        html_content += f"""
        <div class="chart">
            <img src="cid:pnl_chart" alt="Daily P&L Chart" />
        </div>
"""

    # Per-symbol performance
    html_content += """
        <h2>🎯 Performance by Symbol</h2>
        <div class="symbol-grid">
"""

    for symbol, metrics in portfolio_metrics['symbol_metrics'].items():
        symbol_name = symbol.split('-')[0]
        html_content += f"""
            <div class="symbol-card">
                <div class="symbol-title">{symbol_name}</div>
                <table>
                    <tr>
                        <td>Starting</td>
                        <td style="text-align: right; font-weight: bold;">${metrics['starting_capital_usd']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Current</td>
                        <td style="text-align: right; font-weight: bold;">${metrics['current_usd']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Return</td>
                        <td style="text-align: right; font-weight: bold; color: {'#2ECC71' if metrics['percentage_gain'] >= 0 else '#E74C3C'};">
                            {metrics['percentage_gain']:+.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>Net Profit</td>
                        <td style="text-align: right; font-weight: bold; color: {'#2ECC71' if metrics['total_profit'] >= 0 else '#E74C3C'};">
                            ${metrics['total_profit']:+,.2f}
                        </td>
                    </tr>
                </table>
            </div>
"""

    html_content += """
        </div>
"""

    # Add symbol performance chart if available
    if symbol_chart_path and os.path.exists(symbol_chart_path):
        html_content += f"""
        <div class="chart">
            <img src="cid:symbol_chart" alt="Symbol Performance Chart" />
        </div>
"""

    # Today's trades detail
    if daily_summary['total_trades'] > 0:
        html_content += """
        <h2>📋 Today's Trades</h2>
        <div class="trade-list">
"""

        for symbol, trades in daily_summary['trades_by_symbol'].items():
            symbol_name = symbol.split('-')[0]
            for trade in trades:
                is_win = trade.get('total_profit', 0) >= 0
                win_class = 'win' if is_win else 'loss'
                win_emoji = '✅' if is_win else '❌'

                html_content += f"""
            <div class="trade-item {win_class}">
                <strong>{win_emoji} {symbol_name}</strong> -
                ${trade.get('buy_price', 0):,.2f} → ${trade.get('sell_price', 0):,.2f}
                ({trade.get('potential_profit_percentage', 0):+.2f}%)
                <br>
                <span style="font-size: 12px; color: #666;">
                    Profit: ${trade.get('total_profit', 0):+,.2f} |
                    Hold: {trade.get('time_held_position', 'N/A')} |
                    Exit: {trade.get('exit_analysis', {}).get('exit_trigger', 'N/A').replace('_', ' ').title()}
                </span>
            </div>
"""

        html_content += """
        </div>
"""

        # Best and worst trades
        if daily_summary['best_trade'] and daily_summary['worst_trade']:
            best = daily_summary['best_trade']
            worst = daily_summary['worst_trade']

            html_content += f"""
        <h2>🏆 Notable Trades</h2>
        <div class="metric-grid">
            <div class="metric-card" style="border-left-color: #2ECC71;">
                <div class="metric-label">Best Trade</div>
                <div class="metric-value positive">${best.get('total_profit', 0):,.2f}</div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                    {best.get('symbol', 'N/A').split('-')[0]} - {best.get('potential_profit_percentage', 0):.2f}%
                </div>
            </div>
            <div class="metric-card" style="border-left-color: #E74C3C;">
                <div class="metric-label">Worst Trade</div>
                <div class="metric-value negative">${worst.get('total_profit', 0):,.2f}</div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                    {worst.get('symbol', 'N/A').split('-')[0]} - {worst.get('potential_profit_percentage', 0):.2f}%
                </div>
            </div>
        </div>
"""
    else:
        html_content += """
        <h2>📋 Today's Activity</h2>
        <p style="color: #666; font-style: italic;">No trades executed today. The bot is monitoring markets for opportunities.</p>
"""

    # Footer
    html_content += f"""
        <div class="footer">
            <p>Generated by Scalp-Scripts Trading Bot</p>
            <p>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
    </div>
</body>
</html>
"""

    return subject, text_content, html_content


def send_daily_summary_email(wallets_config: List[Dict]) -> bool:
    """
    Generate and send the daily summary email.

    Args:
        wallets_config: List of wallet configurations from config.json

    Returns:
        Boolean indicating success
    """
    try:
        print("Generating daily summary email...")

        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(wallets_config)

        # Get daily trade summary
        daily_summary = get_daily_trade_summary(wallets_config)

        # Generate charts
        print("Generating charts...")
        portfolio_chart = generate_portfolio_value_chart(wallets_config, days=30)
        pnl_chart = generate_daily_pnl_chart(wallets_config, days=30)
        symbol_chart = generate_symbol_performance_chart(wallets_config)

        # Create email content
        subject, text_content, html_content = create_html_email_template(
            portfolio_metrics,
            daily_summary,
            wallets_config,
            portfolio_chart,
            pnl_chart,
            symbol_chart
        )

        # Send email with chart attachments
        # Note: Mailjet supports inline images via CID, but we'll attach the main portfolio chart
        print("Sending email...")
        send_email_notification(
            subject=subject,
            text_content=text_content,
            html_content=html_content,
            attachment_path=portfolio_chart
        )

        print("✅ Daily summary email sent successfully!")

        # Clean up chart files after email is sent
        charts_to_delete = [portfolio_chart, pnl_chart, symbol_chart]
        for chart_path in charts_to_delete:
            if chart_path and os.path.exists(chart_path):
                try:
                    os.remove(chart_path)
                    print(f"  ✓ Deleted chart: {os.path.basename(chart_path)}")
                except Exception as e:
                    print(f"  ⚠ Failed to delete {chart_path}: {e}")

        return True

    except Exception as e:
        print(f"❌ Error sending daily summary email: {e}")
        import traceback
        traceback.print_exc()
        return False


def should_send_daily_summary(target_hour: int = 8) -> bool:
    """
    Check if it's time to send the daily summary.

    Args:
        target_hour: Hour of day to send summary (0-23, default 8 AM)

    Returns:
        Boolean indicating if summary should be sent
    """
    now = datetime.now()  # Use local time for scheduling

    # Check if we're in the target hour
    if now.hour != target_hour:
        return False

    # Check if we've already sent today
    # This prevents sending multiple times if the bot runs multiple times in the target hour
    marker_file = f'screenshots/.daily_summary_sent_{now.strftime("%Y%m%d")}'

    if os.path.exists(marker_file):
        return False

    # Create marker file
    try:
        os.makedirs('screenshots', exist_ok=True)
        with open(marker_file, 'w') as f:
            f.write(now.isoformat())
        return True
    except Exception as e:
        print(f"Warning: Could not create marker file: {e}")
        return True  # Send anyway if we can't track


if __name__ == "__main__":
    # For testing: generate and send summary immediately
    import json

    with open('config.json', 'r') as f:
        config = json.load(f)

    send_daily_summary_email(config['wallets'])
