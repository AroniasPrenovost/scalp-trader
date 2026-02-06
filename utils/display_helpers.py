import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from utils.wallet_helpers import load_transaction_history, calculate_wallet_metrics
from utils.coinbase import get_asset_price, get_last_order_from_local_json_ledger, detect_stored_coinbase_order_type


# Terminal colors (mirrors index.py Colors class)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'


def _color_value(value, positive_color=Colors.GREEN, negative_color=Colors.RED, zero_color=Colors.YELLOW):
    """Return the appropriate color for a numeric value."""
    if value > 0:
        return positive_color
    elif value < 0:
        return negative_color
    return zero_color


def _format_duration(seconds):
    """Format seconds into a human-readable duration string."""
    if seconds is None or seconds <= 0:
        return "N/A"
    hours = seconds / 3600
    if hours < 1:
        return f"{seconds/60:.0f}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        return f"{hours/24:.1f}d"


def print_periodic_summary(config, coinbase_client, coinbase_spot_fee, federal_tax_rate, bot_start_time=None):
    """
    Print a comprehensive 6-hour summary table showing:
    - Per-symbol: trades, wins/losses, win%, gross/net profit, fees, taxes, avg duration, status
    - Totals row
    - Last 3 trades
    - Capital deployment status
    """
    wallets = config.get('wallets', [])
    enabled_wallets = [w for w in wallets if w.get('enabled', False)]
    tradeable_wallets = [w for w in wallets if w.get('ready_to_trade', False)]

    # Use market_rotation config for actual capital numbers
    market_rotation = config.get('market_rotation', {})
    total_trading_capital = market_rotation.get('total_trading_capital_usd', 0)
    capital_per_position = market_rotation.get('capital_per_position', 0)

    # Collect per-symbol data
    rows = []
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_gross = 0.0
    total_net = 0.0
    total_fees = 0.0
    total_taxes = 0.0
    capital_deployed = 0.0
    all_recent_trades = []

    for wallet in enabled_wallets:
        symbol = wallet['symbol']
        starting_capital = wallet.get('starting_capital_usd', 0)
        ready_to_trade = wallet.get('ready_to_trade', False)

        # Get transaction history
        transactions = load_transaction_history(symbol)
        trades = len(transactions)
        wins = len([t for t in transactions if t.get('total_profit', 0) > 0])
        losses = trades - wins
        win_pct = (wins / trades * 100) if trades > 0 else 0

        # Get wallet metrics
        metrics = calculate_wallet_metrics(symbol, starting_capital)
        gross = metrics.get('gross_profit', 0)
        net = metrics.get('total_profit', 0)
        fees = metrics.get('exchange_fees', 0)
        taxes = metrics.get('taxes', 0)
        net_pct = metrics.get('percentage_gain', 0)

        # Average trade duration
        avg_duration_sec = None
        if transactions:
            durations = [t.get('time_held_seconds', 0) for t in transactions if t.get('time_held_seconds')]
            if durations:
                avg_duration_sec = sum(durations) / len(durations)

        # Check position status
        last_order = get_last_order_from_local_json_ledger(symbol)
        order_type = detect_stored_coinbase_order_type(last_order)
        has_position = order_type in ['placeholder', 'buy']

        status = "IDLE"
        unrealized_str = ""
        if has_position:
            if order_type == 'placeholder':
                status = "PENDING"
            else:
                order_data = last_order.get('order', last_order)
                if 'average_filled_price' in order_data:
                    try:
                        entry_p = float(order_data['average_filled_price'])
                        current_p = get_asset_price(coinbase_client, symbol)
                        if current_p:
                            unr_pct = ((current_p - entry_p) / entry_p) * 100
                            status = f"OPEN"
                            unrealized_str = f"{unr_pct:+.1f}%"
                            capital_deployed += capital_per_position
                    except (ValueError, TypeError):
                        status = "OPEN"
                        capital_deployed += capital_per_position
        elif not ready_to_trade:
            status = "DATA ONLY"

        # Accumulate totals (only for tradeable symbols)
        if ready_to_trade:
            total_trades += trades
            total_wins += wins
            total_losses += losses
            total_gross += gross
            total_net += net
            total_fees += fees
            total_taxes += taxes

        # Collect recent trades for "last 3" display
        for t in transactions[:3]:
            all_recent_trades.append({
                'symbol': symbol,
                'timestamp': t.get('timestamp', ''),
                'profit': t.get('total_profit', 0),
                'exit_trigger': t.get('exit_analysis', {}).get('exit_trigger', t.get('exit_trigger', 'unknown')),
                'duration': t.get('time_held_position', 'N/A'),
            })

        rows.append({
            'symbol': symbol,
            'ready': ready_to_trade,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_pct': win_pct,
            'gross': gross,
            'net': net,
            'net_pct': net_pct,
            'fees': fees,
            'taxes': taxes,
            'avg_duration': _format_duration(avg_duration_sec),
            'status': status,
            'unrealized': unrealized_str,
        })

    # Sort: tradeable first, then by net profit descending
    rows.sort(key=lambda r: (not r['ready'], -r['net']))

    # Print header
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*110}")
    print(f"  PERIODIC SUMMARY — {now_str}")
    print(f"{'='*110}{Colors.ENDC}")

    # Column headers
    hdr = (
        f"  {'Symbol':<10} {'Trades':>6} {'W/L':>7} {'Win%':>5} "
        f"{'Gross $':>10} {'Net $':>10} {'Net %':>7} "
        f"{'Fees $':>8} {'Tax $':>8} {'Avg Dur':>7} {'Status':<12}"
    )
    print(f"{Colors.DIM}{hdr}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*110}{Colors.ENDC}")

    # Per-symbol rows
    for r in rows:
        net_c = _color_value(r['net'])
        gross_c = _color_value(r['gross'])

        symbol_display = r['symbol'].replace('-USD', '')
        if not r['ready']:
            symbol_display = f"{Colors.DIM}{symbol_display}{Colors.ENDC}"

        status_display = r['status']
        if r['unrealized']:
            unr_c = _color_value(float(r['unrealized'].replace('%', '').replace('+', '')))
            status_display = f"{r['status']} {unr_c}{r['unrealized']}{Colors.ENDC}"

        wl_str = f"{r['wins']}/{r['losses']}" if r['trades'] > 0 else "-"
        win_pct_str = f"{r['win_pct']:.0f}%" if r['trades'] > 0 else "-"

        print(
            f"  {symbol_display:<10} {r['trades']:>6} {wl_str:>7} {win_pct_str:>5} "
            f"{gross_c}${r['gross']:>+9.2f}{Colors.ENDC} "
            f"{net_c}${r['net']:>+9.2f}{Colors.ENDC} "
            f"{net_c}{r['net_pct']:>+6.2f}%{Colors.ENDC} "
            f"${r['fees']:>7.2f} ${r['taxes']:>7.2f} "
            f"{r['avg_duration']:>7} {status_display:<12}"
        )

    # Totals row
    total_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0
    total_net_pct = (total_net / total_trading_capital * 100) if total_trading_capital > 0 else 0
    total_net_c = _color_value(total_net)
    total_gross_c = _color_value(total_gross)

    print(f"{Colors.CYAN}{'-'*110}{Colors.ENDC}")
    print(
        f"  {Colors.BOLD}{'TOTAL':<10} {total_trades:>6} {total_wins}/{total_losses:>5} {total_win_pct:>4.0f}% "
        f"{total_gross_c}${total_gross:>+9.2f}{Colors.ENDC}{Colors.BOLD} "
        f"{total_net_c}${total_net:>+9.2f}{Colors.ENDC}{Colors.BOLD} "
        f"{total_net_c}{total_net_pct:>+6.2f}%{Colors.ENDC}{Colors.BOLD} "
        f"${total_fees:>7.2f} ${total_taxes:>7.2f}{Colors.ENDC}"
    )

    # Capital deployment
    capital_idle = total_trading_capital - capital_deployed
    print(f"\n  {Colors.BOLD}Capital:{Colors.ENDC} ${total_trading_capital:,.0f} total | "
          f"{Colors.GREEN}${capital_deployed:,.0f} deployed{Colors.ENDC} | "
          f"{Colors.YELLOW}${capital_idle:,.0f} idle{Colors.ENDC}")

    # Bot uptime
    if bot_start_time:
        uptime_sec = time.time() - bot_start_time
        print(f"  {Colors.BOLD}Uptime:{Colors.ENDC} {_format_duration(uptime_sec)}")

    # Last 3 trades (across all symbols)
    all_recent_trades.sort(key=lambda t: t['timestamp'], reverse=True)
    last_3 = all_recent_trades[:3]
    if last_3:
        print(f"\n  {Colors.BOLD}Recent Trades:{Colors.ENDC}")
        for t in last_3:
            p_c = _color_value(t['profit'])
            ts = t['timestamp'][:19].replace('T', ' ') if t['timestamp'] else 'N/A'
            sym = t['symbol'].replace('-USD', '')
            print(f"    {sym:<6} {p_c}${t['profit']:>+8.2f}{Colors.ENDC}  "
                  f"{t['exit_trigger']:<20} {t['duration']:<10} {ts}")

    print(f"\n{Colors.CYAN}{'='*110}{Colors.ENDC}\n")


def print_position_breakdown(symbol, order_data, current_price, profit_calc, coinbase_spot_fee, federal_tax_rate, analysis=None):
    """
    Print a compact position P&L breakdown.

    Args:
        symbol: Trading pair (e.g. 'ATOM-USD')
        order_data: The order dict with entry info
        current_price: Current market price
        profit_calc: Dict from calculate_net_profit_from_price_move()
        coinbase_spot_fee: Fee percentage
        federal_tax_rate: Tax rate percentage
        analysis: Optional analysis dict (for RSI info)
    """
    entry_price = float(order_data.get('average_filled_price', 0))
    shares = float(order_data.get('filled_size', 0))

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*60}")
    print(f"  POSITION — {symbol}")
    print(f"{'─'*60}{Colors.ENDC}")

    # Position line
    price_chg_c = _color_value(profit_calc['price_change_pct'])
    print(f"  Entry: ${entry_price:.4f}  →  Current: ${current_price:.4f}  "
          f"({price_chg_c}{profit_calc['price_change_pct']:+.2f}%{Colors.ENDC})")
    print(f"  Shares: {shares:.8f}  |  Value: ${profit_calc['exit_value_usd']:.2f}")

    # Fees
    total_fees = profit_calc['entry_fee_usd'] + profit_calc['exit_fee_usd']
    print(f"  Fees: ${total_fees:.2f} ({coinbase_spot_fee}% × 2)  "
          f"[entry ${profit_calc['entry_fee_usd']:.2f} + exit ${profit_calc['exit_fee_usd']:.2f}]")

    # Tax
    if profit_calc['capital_gain_usd'] > 0:
        print(f"  Tax:  ${profit_calc['tax_usd']:.2f} ({federal_tax_rate}% on ${profit_calc['capital_gain_usd']:.2f} gain)")
    else:
        print(f"  Tax:  $0.00 (no gain)")

    # Net profit
    net_c = _color_value(profit_calc['net_profit_usd'])
    print(f"{Colors.BOLD}  Net:  {net_c}${profit_calc['net_profit_usd']:+.2f} ({profit_calc['net_profit_pct']:+.2f}%){Colors.ENDC}")

    # RSI status if RSI strategy
    if analysis and analysis.get('strategy') == 'rsi_mean_reversion':
        metrics = analysis.get('metrics', {})
        rsi_exit = metrics.get('rsi_exit_threshold')
        rsi_partial = metrics.get('rsi_partial_exit_threshold')
        if rsi_exit or rsi_partial:
            thresholds = []
            if rsi_partial:
                thresholds.append(f"partial={rsi_partial}")
            if rsi_exit:
                thresholds.append(f"full={rsi_exit}")
            print(f"  RSI exits: {', '.join(thresholds)}")

    print(f"{Colors.CYAN}{'─'*60}{Colors.ENDC}\n")
