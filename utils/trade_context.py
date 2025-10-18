import os
import json
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Optional

def load_transaction_history(symbol: str) -> List[Dict]:
    """
    Load transaction history from transactions/data.json filtered by symbol.

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')

    Returns:
        List of transaction records for the specified symbol
    """
    try:
        file_path = os.path.join("transactions", "data.json")

        if not os.path.exists(file_path):
            print(f"No transaction history found at {file_path}")
            return []

        with open(file_path, 'r') as f:
            all_transactions = json.load(f)

        # Filter transactions for this symbol
        symbol_transactions = [t for t in all_transactions if t.get('symbol') == symbol]

        # Sort by timestamp (most recent first)
        symbol_transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return symbol_transactions

    except Exception as e:
        print(f"ERROR: Failed to load transaction history: {e}")
        return []


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string or None if failed
    """
    try:
        if not os.path.exists(image_path):
            return None

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        print(f"Warning: Could not encode image {image_path}: {e}")
        return None


def calculate_wallet_metrics(symbol: str, starting_capital_usd: float) -> Dict:
    """
    Calculate wallet performance metrics for a symbol.

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')
        starting_capital_usd: The initial capital allocated to this symbol

    Returns:
        Dictionary containing:
        {
            'starting_capital_usd': float,
            'current_usd': float,
            'percentage_gain': float,
            'gross_profit': float,
            'taxes': float,
            'exchange_fees': float,
            'total_profit': float
        }
    """
    transactions = load_transaction_history(symbol)

    # Calculate cumulative metrics from all completed trades
    total_profit = sum(t.get('total_profit', 0) for t in transactions)
    gross_profit = sum(t.get('gross_profit', 0) for t in transactions)
    taxes = sum(t.get('taxes', 0) for t in transactions)
    exchange_fees = sum(t.get('exchange_fees', 0) for t in transactions)

    # Current USD = starting capital + all profits/losses
    current_usd = starting_capital_usd + total_profit

    # Calculate percentage gain
    percentage_gain = ((current_usd - starting_capital_usd) / starting_capital_usd * 100) if starting_capital_usd > 0 else 0

    return {
        'starting_capital_usd': starting_capital_usd,
        'current_usd': current_usd,
        'percentage_gain': percentage_gain,
        'gross_profit': gross_profit,
        'taxes': taxes,
        'exchange_fees': exchange_fees,
        'total_profit': total_profit
    }


def _analyze_trade_patterns(transactions: List[Dict]) -> Dict:
    """
    Analyze trading patterns from transaction history to identify what works and what doesn't.

    Args:
        transactions: List of transaction records

    Returns:
        Dictionary containing pattern analysis insights
    """
    if not transactions:
        return {}

    insights = {}

    # Analyze by confidence level
    confidence_analysis = {}
    for conf_level in ['high', 'medium', 'low']:
        trades_at_level = [t for t in transactions if t.get('ai_model_data', {}).get('confidence_level') == conf_level]
        if trades_at_level:
            wins = [t for t in trades_at_level if t.get('total_profit', 0) > 0]
            win_rate = (len(wins) / len(trades_at_level) * 100) if trades_at_level else 0
            avg_profit = sum(t.get('potential_profit_percentage', 0) for t in trades_at_level) / len(trades_at_level)
            confidence_analysis[conf_level] = {
                'count': len(trades_at_level),
                'win_rate': win_rate,
                'avg_profit_pct': avg_profit
            }
    insights['by_confidence_level'] = confidence_analysis

    # Analyze by market trend
    trend_analysis = {}
    for trend in ['bullish', 'bearish', 'neutral', 'ranging']:
        trades_in_trend = [t for t in transactions if t.get('market_context_at_entry', {}).get('current_trend') == trend]
        if trades_in_trend:
            wins = [t for t in trades_in_trend if t.get('total_profit', 0) > 0]
            win_rate = (len(wins) / len(trades_in_trend) * 100) if trades_in_trend else 0
            avg_profit = sum(t.get('potential_profit_percentage', 0) for t in trades_in_trend) / len(trades_in_trend)
            trend_analysis[trend] = {
                'count': len(trades_in_trend),
                'win_rate': win_rate,
                'avg_profit_pct': avg_profit
            }
    insights['by_market_trend'] = trend_analysis

    # Analyze by exit trigger
    exit_analysis = {}
    for exit_type in ['profit_target', 'stop_loss', 'manual']:
        trades_by_exit = [t for t in transactions if t.get('exit_analysis', {}).get('exit_trigger') == exit_type]
        if trades_by_exit:
            avg_profit = sum(t.get('potential_profit_percentage', 0) for t in trades_by_exit) / len(trades_by_exit)
            exit_analysis[exit_type] = {
                'count': len(trades_by_exit),
                'avg_profit_pct': avg_profit
            }
    insights['by_exit_trigger'] = exit_analysis

    # Analyze volatility impact
    volatility_buckets = {'low': [], 'medium': [], 'high': [], 'extreme': []}
    for t in transactions:
        volatility = t.get('market_context_at_entry', {}).get('volatility_range_pct')
        if volatility is not None:
            if volatility < 15:
                volatility_buckets['low'].append(t)
            elif volatility < 30:
                volatility_buckets['medium'].append(t)
            elif volatility < 50:
                volatility_buckets['high'].append(t)
            else:
                volatility_buckets['extreme'].append(t)

    volatility_analysis = {}
    for vol_level, trades_list in volatility_buckets.items():
        if trades_list:
            wins = [t for t in trades_list if t.get('total_profit', 0) > 0]
            win_rate = (len(wins) / len(trades_list) * 100) if trades_list else 0
            avg_profit = sum(t.get('potential_profit_percentage', 0) for t in trades_list) / len(trades_list)
            volatility_analysis[vol_level] = {
                'count': len(trades_list),
                'win_rate': win_rate,
                'avg_profit_pct': avg_profit
            }
    insights['by_volatility'] = volatility_analysis

    # Analyze hold time patterns
    winning_trades = [t for t in transactions if t.get('total_profit', 0) > 0]
    losing_trades = [t for t in transactions if t.get('total_profit', 0) <= 0]

    winning_hold_times = [t.get('time_held_seconds') for t in winning_trades if t.get('time_held_seconds')]
    losing_hold_times = [t.get('time_held_seconds') for t in losing_trades if t.get('time_held_seconds')]

    insights['hold_time_analysis'] = {
        'avg_winning_hold_hours': (sum(winning_hold_times) / len(winning_hold_times) / 3600) if winning_hold_times else None,
        'avg_losing_hold_hours': (sum(losing_hold_times) / len(losing_hold_times) / 3600) if losing_hold_times else None,
    }

    # Analyze entry quality (distance from support)
    support_proximity_wins = []
    support_proximity_losses = []
    for t in transactions:
        tech_signals = t.get('technical_signals_at_entry', {})
        if 'price_vs_support' in tech_signals:
            if t.get('total_profit', 0) > 0:
                support_proximity_wins.append(tech_signals['price_vs_support'])
            else:
                support_proximity_losses.append(tech_signals['price_vs_support'])

    insights['entry_quality'] = {
        'sample_winning_support_distances': support_proximity_wins[:5],
        'sample_losing_support_distances': support_proximity_losses[:5],
    }

    return insights


def build_trading_context(symbol: str, max_trades: int = 10, include_screenshots: bool = True, starting_capital_usd: Optional[float] = None) -> Dict:
    """
    Build contextual information from past trades for LLM analysis.

    This function creates a comprehensive context object containing:
    - Historical trade performance for the symbol
    - Buy event screenshots (if available and enabled)
    - Performance metrics and patterns
    - Wallet metrics (if starting_capital_usd is provided)

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')
        max_trades: Maximum number of recent trades to include (default: 10)
        include_screenshots: Whether to include base64-encoded screenshots (default: True)
        starting_capital_usd: Initial capital allocated to this symbol (optional)

    Returns:
        Dictionary containing trading context with the following structure:
        {
            'symbol': str,
            'total_trades': int,
            'trades_included': int,
            'performance_summary': {
                'total_profit': float,
                'average_profit_percentage': float,
                'win_rate': float,
                'profitable_trades': int,
                'losing_trades': int
            },
            'wallet_metrics': {  # Only included if starting_capital_usd is provided
                'starting_capital_usd': float,
                'current_usd': float,
                'percentage_gain': float,
                'total_profit': float
            },
            'trades': [
                {
                    'buy_price': float,
                    'sell_price': float,
                    'profit_percentage': float,
                    'total_profit': float,
                    'time_held': str,
                    'buy_timestamp': str,
                    'sell_timestamp': str,
                    'screenshot_available': bool,
                    'screenshot_base64': str (optional)
                }
            ]
        }
    """
    transactions = load_transaction_history(symbol)

    if not transactions:
        result = {
            'symbol': symbol,
            'total_trades': 0,
            'trades_included': 0,
            'performance_summary': None,
            'trades': []
        }
        # Include wallet metrics even if no transactions yet
        if starting_capital_usd is not None:
            result['wallet_metrics'] = calculate_wallet_metrics(symbol, starting_capital_usd)
        return result

    # Limit to max_trades
    limited_transactions = transactions[:max_trades]

    # Calculate performance metrics
    total_profit = sum(t.get('total_profit', 0) for t in transactions)
    profit_percentages = [t.get('potential_profit_percentage', 0) for t in transactions]
    average_profit_percentage = sum(profit_percentages) / len(profit_percentages) if profit_percentages else 0

    profitable_trades = len([t for t in transactions if t.get('total_profit', 0) > 0])
    losing_trades = len([t for t in transactions if t.get('total_profit', 0) <= 0])
    win_rate = (profitable_trades / len(transactions) * 100) if transactions else 0

    # Enhanced analytics: Analyze patterns by market conditions
    pattern_insights = _analyze_trade_patterns(transactions)

    performance_summary = {
        'total_profit': total_profit,
        'average_profit_percentage': average_profit_percentage,
        'win_rate': win_rate,
        'profitable_trades': profitable_trades,
        'losing_trades': losing_trades,
        'pattern_insights': pattern_insights
    }

    # Build trade records with optional screenshots
    trades = []
    for transaction in limited_transactions:
        trade_record = {
            'buy_price': transaction.get('buy_price'),
            'sell_price': transaction.get('sell_price'),
            'profit_percentage': transaction.get('potential_profit_percentage'),
            'total_profit': transaction.get('total_profit'),
            'time_held': transaction.get('time_held_position'),
            'buy_timestamp': transaction.get('buy_timestamp'),
            'sell_timestamp': transaction.get('timestamp'),
            'screenshot_available': False
        }

        # Include screenshot if available and enabled
        if include_screenshots:
            screenshot_path = transaction.get('buy_screenshot_path')
            if screenshot_path and os.path.exists(screenshot_path):
                encoded_image = encode_image_to_base64(screenshot_path)
                if encoded_image:
                    trade_record['screenshot_available'] = True
                    trade_record['screenshot_base64'] = encoded_image

        trades.append(trade_record)

    result = {
        'symbol': symbol,
        'total_trades': len(transactions),
        'trades_included': len(limited_transactions),
        'performance_summary': performance_summary,
        'trades': trades
    }

    # Include wallet metrics if starting capital is provided
    if starting_capital_usd is not None:
        result['wallet_metrics'] = calculate_wallet_metrics(symbol, starting_capital_usd)

    return result


def format_context_for_llm(context: Dict) -> str:
    """
    Format the trading context into a human-readable string for LLM consumption.

    Args:
        context: The context dictionary from build_trading_context()

    Returns:
        Formatted string suitable for including in LLM prompts
    """
    if context['total_trades'] == 0:
        base_msg = f"No historical trading data available for {context['symbol']}."
        # Include wallet metrics even if no trades yet
        if 'wallet_metrics' in context:
            metrics = context['wallet_metrics']
            base_msg += f"""

Wallet Status:
- Starting Capital: ${metrics['starting_capital_usd']:.2f}
- Current Value: ${metrics['current_usd']:.2f}
- Gross Profit: ${metrics['gross_profit']:.2f}
- Taxes Paid: ${metrics['taxes']:.2f}
- Exchange Fees: ${metrics['exchange_fees']:.2f}
- Total Net Profit: ${metrics['total_profit']:.2f} ({metrics['percentage_gain']:.2f}%)
"""
        return base_msg

    summary = context['performance_summary']

    output = f"""
Historical Trading Performance for {context['symbol']}:
"""

    # Add wallet metrics section if available
    if 'wallet_metrics' in context:
        metrics = context['wallet_metrics']
        output += f"""
Wallet Status:
- Starting Capital: ${metrics['starting_capital_usd']:.2f}
- Current Value: ${metrics['current_usd']:.2f}
- Gross Profit: ${metrics['gross_profit']:.2f}
- Taxes Paid: ${metrics['taxes']:.2f}
- Exchange Fees: ${metrics['exchange_fees']:.2f}
- Total Net Profit: ${metrics['total_profit']:.2f} ({metrics['percentage_gain']:.2f}%)

"""

    output += f"""
Performance Summary (All-Time):
- Total Trades: {context['total_trades']}
- Win Rate: {summary['win_rate']:.1f}% ({summary['profitable_trades']} wins, {summary['losing_trades']} losses)
- Average Profit: {summary['average_profit_percentage']:.2f}% per trade
- Total Cumulative Profit: ${summary['total_profit']:.2f}
"""

    # Add pattern insights if available
    pattern_insights = summary.get('pattern_insights', {})
    if pattern_insights:
        output += "\n=== CRITICAL PATTERN INSIGHTS FROM PAST TRADES ===\n"

        # Confidence level analysis
        if 'by_confidence_level' in pattern_insights and pattern_insights['by_confidence_level']:
            output += "\n📊 Performance by AI Confidence Level:\n"
            for level, stats in pattern_insights['by_confidence_level'].items():
                output += f"  - {level.upper()}: {stats['count']} trades, {stats['win_rate']:.1f}% win rate, {stats['avg_profit_pct']:.2f}% avg profit\n"

        # Market trend analysis
        if 'by_market_trend' in pattern_insights and pattern_insights['by_market_trend']:
            output += "\n📈 Performance by Market Trend:\n"
            for trend, stats in pattern_insights['by_market_trend'].items():
                output += f"  - {trend.upper()}: {stats['count']} trades, {stats['win_rate']:.1f}% win rate, {stats['avg_profit_pct']:.2f}% avg profit\n"

        # Volatility analysis
        if 'by_volatility' in pattern_insights and pattern_insights['by_volatility']:
            output += "\n💹 Performance by Volatility Level:\n"
            for vol, stats in pattern_insights['by_volatility'].items():
                output += f"  - {vol.upper()}: {stats['count']} trades, {stats['win_rate']:.1f}% win rate, {stats['avg_profit_pct']:.2f}% avg profit\n"

        # Exit trigger analysis
        if 'by_exit_trigger' in pattern_insights and pattern_insights['by_exit_trigger']:
            output += "\n🎯 Performance by Exit Trigger:\n"
            for exit_type, stats in pattern_insights['by_exit_trigger'].items():
                output += f"  - {exit_type.replace('_', ' ').title()}: {stats['count']} trades, {stats['avg_profit_pct']:.2f}% avg profit\n"

        # Hold time analysis
        if 'hold_time_analysis' in pattern_insights:
            hold_time = pattern_insights['hold_time_analysis']
            if hold_time.get('avg_winning_hold_hours') or hold_time.get('avg_losing_hold_hours'):
                output += "\n⏱️ Hold Time Patterns:\n"
                if hold_time.get('avg_winning_hold_hours'):
                    output += f"  - Winning trades: avg {hold_time['avg_winning_hold_hours']:.1f} hours\n"
                if hold_time.get('avg_losing_hold_hours'):
                    output += f"  - Losing trades: avg {hold_time['avg_losing_hold_hours']:.1f} hours\n"

        # Entry quality
        if 'entry_quality' in pattern_insights:
            entry = pattern_insights['entry_quality']
            if entry.get('sample_winning_support_distances') or entry.get('sample_losing_support_distances'):
                output += "\n🎯 Entry Quality (Distance from Support):\n"
                if entry.get('sample_winning_support_distances'):
                    output += f"  - Winning entries: {', '.join(entry['sample_winning_support_distances'][:3])}\n"
                if entry.get('sample_losing_support_distances'):
                    output += f"  - Losing entries: {', '.join(entry['sample_losing_support_distances'][:3])}\n"

        output += "\n"

    output += f"""
Recent Trade History (Last {context['trades_included']} trades):
"""

    for i, trade in enumerate(context['trades'], 1):
        profit_indicator = "✓" if trade['total_profit'] > 0 else "✗"
        output += f"""
{i}. {profit_indicator} Trade from {trade['buy_timestamp'][:10]}:
   - Entry: ${trade['buy_price']:.6f} → Exit: ${trade['sell_price']:.6f}
   - Profit: {trade['profit_percentage']:.2f}% (${trade['total_profit']:.2f})
   - Time Held: {trade['time_held']}
   - Screenshot: {'Available' if trade['screenshot_available'] else 'Not available'}
"""

    # Build data-driven learning directives based on actual patterns
    learning_directives = f"""
HISTORICAL LEARNING DIRECTIVES - DATA-DRIVEN INSIGHTS:

1. CONFIDENCE CALIBRATION:
   - Current overall win rate: {summary['win_rate']:.1f}%
"""

    # Add confidence-specific guidance if data available
    if pattern_insights.get('by_confidence_level'):
        conf_data = pattern_insights['by_confidence_level']
        if 'high' in conf_data:
            learning_directives += f"   - HIGH confidence trades: {conf_data['high']['win_rate']:.1f}% win rate ({conf_data['high']['count']} trades) - "
            if conf_data['high']['win_rate'] < 70:
                learning_directives += "CALIBRATION NEEDED: High confidence should achieve >70% win rate\n"
            else:
                learning_directives += "Well calibrated, continue current criteria\n"

        if 'medium' in conf_data:
            learning_directives += f"   - MEDIUM confidence trades: {conf_data['medium']['win_rate']:.1f}% win rate - Consider avoiding these, focus on HIGH only\n"

    learning_directives += "\n2. MARKET CONDITION OPTIMIZATION:\n"

    # Add market trend insights
    if pattern_insights.get('by_market_trend'):
        trend_data = pattern_insights['by_market_trend']
        best_trend = max(trend_data.items(), key=lambda x: x[1]['win_rate']) if trend_data else None
        worst_trend = min(trend_data.items(), key=lambda x: x[1]['win_rate']) if trend_data else None

        if best_trend:
            learning_directives += f"   ✓ BEST: {best_trend[0].upper()} markets ({best_trend[1]['win_rate']:.1f}% win rate) - PRIORITIZE these conditions\n"
        if worst_trend and worst_trend[1]['count'] > 0:
            learning_directives += f"   ✗ WORST: {worst_trend[0].upper()} markets ({worst_trend[1]['win_rate']:.1f}% win rate) - "
            if worst_trend[1]['win_rate'] < 50:
                learning_directives += "AVOID trading in these conditions\n"
            else:
                learning_directives += "Exercise extra caution\n"

    # Add volatility insights
    if pattern_insights.get('by_volatility'):
        vol_data = pattern_insights['by_volatility']
        learning_directives += "\n3. VOLATILITY SWEET SPOT:\n"
        for vol_level in ['low', 'medium', 'high', 'extreme']:
            if vol_level in vol_data and vol_data[vol_level]['count'] > 0:
                learning_directives += f"   - {vol_level.upper()}: {vol_data[vol_level]['win_rate']:.1f}% win rate, {vol_data[vol_level]['avg_profit_pct']:.2f}% avg profit ({vol_data[vol_level]['count']} trades)\n"

    # Add hold time guidance
    if pattern_insights.get('hold_time_analysis'):
        hold_data = pattern_insights['hold_time_analysis']
        learning_directives += "\n4. OPTIMAL HOLD TIME EXPECTATIONS:\n"
        if hold_data.get('avg_winning_hold_hours'):
            learning_directives += f"   - Winners typically held: {hold_data['avg_winning_hold_hours']:.1f} hours\n"
            learning_directives += f"   - Set profit targets that align with this timeframe\n"
        if hold_data.get('avg_losing_hold_hours'):
            learning_directives += f"   - Losers typically held: {hold_data['avg_losing_hold_hours']:.1f} hours\n"
            if hold_data.get('avg_winning_hold_hours') and hold_data.get('avg_losing_hold_hours'):
                if hold_data['avg_losing_hold_hours'] > hold_data['avg_winning_hold_hours']:
                    learning_directives += f"   ⚠️  Losers held LONGER than winners - consider tighter stop losses or earlier exits on struggling positions\n"

    # Add exit trigger insights
    if pattern_insights.get('by_exit_trigger'):
        exit_data = pattern_insights['by_exit_trigger']
        learning_directives += "\n5. EXIT STRATEGY EFFECTIVENESS:\n"
        if 'profit_target' in exit_data:
            learning_directives += f"   - Profit targets hit: {exit_data['profit_target']['count']} times, {exit_data['profit_target']['avg_profit_pct']:.2f}% avg profit\n"
        if 'stop_loss' in exit_data:
            learning_directives += f"   - Stop losses hit: {exit_data['stop_loss']['count']} times, {exit_data['stop_loss']['avg_profit_pct']:.2f}% avg loss\n"
            if exit_data['stop_loss']['count'] > 2:
                learning_directives += f"   ⚠️  Multiple stop losses triggered - review entry quality and stop loss placement\n"

    learning_directives += f"""
6. POSITION SIZING RULES:
   - Win rate {summary['win_rate']:.1f}%: """

    if summary['win_rate'] < 60:
        learning_directives += "Below target - only trade EXCEPTIONAL setups with HIGH confidence\n"
    elif summary['win_rate'] < 75:
        learning_directives += "Good - maintain current strict criteria\n"
    else:
        learning_directives += "Excellent - maintain discipline, avoid overconfidence\n"

    learning_directives += f"""
7. CRITICAL RULES - APPLY TO CURRENT TRADE:
   - Use the pattern insights above to CALIBRATE your analysis
   - If current market conditions match your worst-performing scenarios, REDUCE confidence or SKIP trade
   - If current conditions match your best-performing scenarios, this validates the setup
   - Profit targets should be realistic based on historical achievement rates
   - Hold time expectations should align with historical winning patterns
   - Weight: 70% current technical setup + 30% historical pattern validation
"""

    output += learning_directives

    return output


def get_trade_screenshots_for_vision(context: Dict) -> List[Dict]:
    """
    Extract screenshot data formatted for OpenAI Vision API.

    Args:
        context: The context dictionary from build_trading_context()

    Returns:
        List of dictionaries formatted for OpenAI Vision API:
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,..."
                }
            }
        ]
    """
    screenshots = []

    for trade in context.get('trades', []):
        if trade.get('screenshot_available') and 'screenshot_base64' in trade:
            screenshots.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{trade['screenshot_base64']}",
                    "detail": "low"  # Use low detail to save tokens
                }
            })

    return screenshots


def prune_old_transactions(symbol: str, keep_count: int = 50) -> bool:
    """
    Remove older transactions for a symbol to manage context size.
    Keeps only the most recent 'keep_count' transactions.

    Args:
        symbol: The trading pair symbol (e.g., 'BTC-USD')
        keep_count: Number of recent transactions to keep (default: 50)

    Returns:
        Boolean indicating success
    """
    try:
        file_path = os.path.join("transactions", "data.json")

        if not os.path.exists(file_path):
            return False

        with open(file_path, 'r') as f:
            all_transactions = json.load(f)

        # Separate transactions for this symbol and others
        symbol_transactions = [t for t in all_transactions if t.get('symbol') == symbol]
        other_transactions = [t for t in all_transactions if t.get('symbol') != symbol]

        # Sort symbol transactions by timestamp (most recent first)
        symbol_transactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Keep only the most recent ones
        kept_transactions = symbol_transactions[:keep_count]
        removed_count = len(symbol_transactions) - len(kept_transactions)

        # Combine back together
        new_transactions = other_transactions + kept_transactions

        # Save back to file
        with open(file_path, 'w') as f:
            json.dump(new_transactions, f, indent=4)

        if removed_count > 0:
            print(f"✓ Pruned {removed_count} old transactions for {symbol}, kept {len(kept_transactions)}")

        return True

    except Exception as e:
        print(f"ERROR: Failed to prune transactions: {e}")
        return False
