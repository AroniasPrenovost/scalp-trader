"""
Strategy Performance Metrics

Tracks the performance of the Unified Adaptive AI Strategy:
- Win rate
- Average profit/loss
- Total P/L
- Strategy component breakdown (AMR vs AI enhancements)
- Confidence level accuracy

This enables data-driven optimization of the strategy.
"""

import json
import os
import time
from datetime import datetime


def log_trade_entry(symbol, opportunity_data, fill_price, fill_amount_usd):
    """
    Log when a trade is entered based on the Adaptive AI strategy.

    Args:
        symbol: Trading pair symbol (e.g. 'BTC-USD')
        opportunity_data: The opportunity dict from score_opportunity()
        fill_price: Actual fill price
        fill_amount_usd: Amount in USD

    Returns:
        Trade ID (timestamp-based) for tracking
    """
    trade_id = f"{symbol}_{int(time.time())}"

    trade_entry = {
        'trade_id': trade_id,
        'symbol': symbol,
        'strategy': 'adaptive_ai',
        'entry_time': time.time(),
        'entry_time_human': datetime.now().isoformat(),
        'entry_price': fill_price,
        'planned_entry': opportunity_data.get('entry_price'),
        'stop_loss': opportunity_data.get('stop_loss'),
        'profit_target': opportunity_data.get('profit_target'),
        'amount_usd': fill_amount_usd,
        'confidence': opportunity_data.get('confidence'),
        'trend': opportunity_data.get('trend'),
        'score': opportunity_data.get('score'),
        'amr_signal': {
            'deviation_from_ma': opportunity_data.get('amr_signal', {}).get('deviation_from_ma'),
            'reasoning': opportunity_data.get('amr_signal', {}).get('reasoning')
        },
        'ai_validation': {
            'confidence': opportunity_data.get('ai_validation', {}).get('confidence') if opportunity_data.get('ai_validation') else None,
            'reasoning': opportunity_data.get('ai_validation', {}).get('reasoning') if opportunity_data.get('ai_validation') else None
        },
        'exit_time': None,
        'exit_price': None,
        'exit_reason': None,
        'profit_loss_usd': None,
        'profit_loss_pct': None,
        'outcome': 'open'  # 'open', 'win', 'loss', 'breakeven'
    }

    # Save to metrics file
    _append_trade_to_metrics(trade_entry)

    print(f"📊 Strategy Metrics: Trade entry logged - {trade_id}")
    return trade_id


def log_trade_exit(symbol, entry_time, exit_price, exit_reason):
    """
    Log when a trade is exited.

    Args:
        symbol: Trading pair symbol
        entry_time: Timestamp when trade was entered
        exit_price: Actual exit price
        exit_reason: Reason for exit ('profit_target', 'stop_loss', 'manual', etc.)

    Returns:
        Updated trade record
    """
    # Find the open trade
    metrics_file = _get_metrics_file_path()

    if not os.path.exists(metrics_file):
        print(f"⚠️  No metrics file found to update")
        return None

    with open(metrics_file, 'r') as f:
        trades = [json.loads(line) for line in f if line.strip()]

    # Find matching trade
    trade_to_update = None
    for trade in trades:
        if trade['symbol'] == symbol and abs(trade['entry_time'] - entry_time) < 60:
            trade_to_update = trade
            break

    if not trade_to_update:
        print(f"⚠️  Could not find matching trade entry for {symbol}")
        return None

    # Calculate P/L
    entry_price = trade_to_update['entry_price']
    amount_usd = trade_to_update['amount_usd']

    profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
    profit_loss_usd = amount_usd * (profit_loss_pct / 100)

    # Determine outcome
    if profit_loss_pct > 0.5:
        outcome = 'win'
    elif profit_loss_pct < -0.5:
        outcome = 'loss'
    else:
        outcome = 'breakeven'

    # Update trade
    trade_to_update['exit_time'] = time.time()
    trade_to_update['exit_time_human'] = datetime.now().isoformat()
    trade_to_update['exit_price'] = exit_price
    trade_to_update['exit_reason'] = exit_reason
    trade_to_update['profit_loss_usd'] = profit_loss_usd
    trade_to_update['profit_loss_pct'] = profit_loss_pct
    trade_to_update['outcome'] = outcome

    # Rewrite metrics file
    with open(metrics_file, 'w') as f:
        for trade in trades:
            f.write(json.dumps(trade) + '\n')

    print(f"📊 Strategy Metrics: Trade exit logged - {outcome.upper()} ({profit_loss_pct:+.2f}%)")
    return trade_to_update


def get_strategy_performance_summary(days=30):
    """
    Get performance summary for the Adaptive AI strategy.

    Args:
        days: Number of days to include in summary (default: 30)

    Returns:
        Dictionary with performance metrics
    """
    metrics_file = _get_metrics_file_path()

    if not os.path.exists(metrics_file):
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'win_rate': 0,
            'total_profit_usd': 0,
            'avg_profit_pct': 0,
            'avg_loss_pct': 0,
            'by_confidence': {},
            'by_trend': {}
        }

    with open(metrics_file, 'r') as f:
        all_trades = [json.loads(line) for line in f if line.strip()]

    # Filter to completed trades within time window
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    completed_trades = [
        t for t in all_trades
        if t['outcome'] in ['win', 'loss', 'breakeven']
        and t['entry_time'] >= cutoff_time
    ]

    if not completed_trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'win_rate': 0,
            'total_profit_usd': 0,
            'avg_profit_pct': 0,
            'avg_loss_pct': 0,
            'by_confidence': {},
            'by_trend': {}
        }

    # Calculate metrics
    wins = [t for t in completed_trades if t['outcome'] == 'win']
    losses = [t for t in completed_trades if t['outcome'] == 'loss']
    breakeven = [t for t in completed_trades if t['outcome'] == 'breakeven']

    total_profit_usd = sum(t['profit_loss_usd'] for t in completed_trades)
    win_rate = len(wins) / len(completed_trades) * 100 if completed_trades else 0

    avg_profit_pct = sum(t['profit_loss_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss_pct = sum(t['profit_loss_pct'] for t in losses) / len(losses) if losses else 0

    # Breakdown by confidence level
    by_confidence = {}
    for conf_level in ['high', 'medium', 'low']:
        conf_trades = [t for t in completed_trades if t['confidence'] == conf_level]
        if conf_trades:
            conf_wins = len([t for t in conf_trades if t['outcome'] == 'win'])
            by_confidence[conf_level] = {
                'total': len(conf_trades),
                'wins': conf_wins,
                'win_rate': (conf_wins / len(conf_trades)) * 100,
                'avg_profit_usd': sum(t['profit_loss_usd'] for t in conf_trades) / len(conf_trades)
            }

    # Breakdown by trend
    by_trend = {}
    for trend in ['uptrend', 'sideways', 'downtrend']:
        trend_trades = [t for t in completed_trades if t['trend'] == trend]
        if trend_trades:
            trend_wins = len([t for t in trend_trades if t['outcome'] == 'win'])
            by_trend[trend] = {
                'total': len(trend_trades),
                'wins': trend_wins,
                'win_rate': (trend_wins / len(trend_trades)) * 100,
                'avg_profit_usd': sum(t['profit_loss_usd'] for t in trend_trades) / len(trend_trades)
            }

    return {
        'total_trades': len(completed_trades),
        'wins': len(wins),
        'losses': len(losses),
        'breakeven': len(breakeven),
        'win_rate': win_rate,
        'total_profit_usd': total_profit_usd,
        'avg_profit_pct': avg_profit_pct,
        'avg_loss_pct': avg_loss_pct,
        'by_confidence': by_confidence,
        'by_trend': by_trend
    }


def print_strategy_performance_report(days=30):
    """
    Print a formatted performance report for the Adaptive AI strategy.

    Args:
        days: Number of days to include in report
    """
    summary = get_strategy_performance_summary(days)

    print("\n" + "="*80)
    print(f"ADAPTIVE AI STRATEGY - PERFORMANCE REPORT (Last {days} days)")
    print("="*80)

    if summary['total_trades'] == 0:
        print("No completed trades in this time period.")
        print("="*80 + "\n")
        return

    print(f"Total Trades: {summary['total_trades']}")
    print(f"Wins: {summary['wins']} | Losses: {summary['losses']} | Breakeven: {summary['breakeven']}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Total P/L: ${summary['total_profit_usd']:+.2f}")
    print(f"Avg Win: {summary['avg_profit_pct']:+.2f}% | Avg Loss: {summary['avg_loss_pct']:+.2f}%")

    print("\n--- Performance by Confidence Level ---")
    for conf, data in summary['by_confidence'].items():
        print(f"{conf.upper()}: {data['total']} trades, {data['win_rate']:.1f}% win rate, ${data['avg_profit_usd']:+.2f} avg P/L")

    print("\n--- Performance by Market Trend ---")
    for trend, data in summary['by_trend'].items():
        print(f"{trend.upper()}: {data['total']} trades, {data['win_rate']:.1f}% win rate, ${data['avg_profit_usd']:+.2f} avg P/L")

    print("="*80 + "\n")


# Private helper functions

def _get_metrics_file_path():
    """Get the path to the strategy metrics file."""
    metrics_dir = 'strategy_metrics'
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    return os.path.join(metrics_dir, 'adaptive_ai_trades.jsonl')


def _append_trade_to_metrics(trade_entry):
    """Append a trade entry to the metrics file."""
    metrics_file = _get_metrics_file_path()
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(trade_entry) + '\n')
