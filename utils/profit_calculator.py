#!/usr/bin/env python3
"""
Atomic Profitability Calculator

Single source of truth for all profit/loss calculations.
Accounts for:
- Entry exchange fees (charged on buy order)
- Exit exchange fees (charged on sell order)
- Capital gains taxes (federal only, on profits)

All other modules should import and use these functions to ensure consistency.
"""

from typing import Dict, Optional


def calculate_required_shares_for_capital(
    capital_usd: float,
    price_per_share: float,
    entry_fee_pct: float
) -> Dict:
    """
    Calculate how many shares to buy given capital, accounting for entry fees.

    Formula:
        subtotal = capital_usd / (1 + entry_fee_pct/100)
        shares = subtotal / price_per_share
        entry_fee = subtotal * (entry_fee_pct/100)
        total_cost = subtotal + entry_fee = capital_usd

    Args:
        capital_usd: Total capital to deploy (including fees)
        price_per_share: Current price per share
        entry_fee_pct: Entry fee percentage (e.g., 0.6 for 0.6%)

    Returns:
        {
            'shares': float,  # Number of shares to buy
            'subtotal_usd': float,  # Pre-fee cost
            'entry_fee_usd': float,  # Entry fee amount
            'total_cost_usd': float,  # Total cost (should equal capital_usd)
            'cost_basis_per_share': float  # Average cost per share including fees
        }
    """
    # Calculate subtotal (pre-fee amount)
    fee_multiplier = 1 + (entry_fee_pct / 100)
    subtotal_usd = capital_usd / fee_multiplier

    # Calculate shares
    shares = subtotal_usd / price_per_share

    # Calculate entry fee
    entry_fee_usd = subtotal_usd * (entry_fee_pct / 100)

    # Total cost
    total_cost_usd = subtotal_usd + entry_fee_usd

    # Cost basis per share (for later profit calculations)
    cost_basis_per_share = total_cost_usd / shares

    return {
        'shares': shares,
        'subtotal_usd': subtotal_usd,
        'entry_fee_usd': entry_fee_usd,
        'total_cost_usd': total_cost_usd,
        'cost_basis_per_share': cost_basis_per_share
    }


def calculate_net_profit_from_price_move(
    entry_price: float,
    exit_price: float,
    shares: float,
    entry_fee_pct: float,
    exit_fee_pct: float,
    tax_rate_pct: float,
    cost_basis_usd: Optional[float] = None
) -> Dict:
    """
    Calculate net profit from a price move, accounting for all costs.

    This is the ATOMIC profitability function - single source of truth.

    Formula:
        1. entry_cost = (entry_price * shares) * (1 + entry_fee_pct/100)
           OR use cost_basis_usd if provided (already includes entry fee)
        2. exit_value_gross = exit_price * shares
        3. exit_fee = exit_value_gross * (exit_fee_pct/100)
        4. exit_value_net = exit_value_gross - exit_fee
        5. gross_profit = exit_value_net - entry_cost
        6. capital_gain = exit_value_net - entry_cost
        7. tax = (tax_rate_pct/100) * capital_gain if capital_gain > 0 else 0
        8. net_profit = gross_profit - tax

    Args:
        entry_price: Price per share when bought
        exit_price: Price per share when sold
        shares: Number of shares
        entry_fee_pct: Entry fee percentage (e.g., 0.6)
        exit_fee_pct: Exit fee percentage (e.g., 0.6)
        tax_rate_pct: Tax rate percentage (e.g., 37)
        cost_basis_usd: Optional pre-calculated cost basis (includes entry + entry fee)

    Returns:
        {
            'entry_value_usd': float,  # Entry order value (shares * entry_price)
            'entry_fee_usd': float,  # Entry fee charged
            'cost_basis_usd': float,  # Total cost (entry + entry fee)
            'exit_value_usd': float,  # Exit order value (shares * exit_price)
            'exit_fee_usd': float,  # Exit fee charged
            'exit_proceeds_usd': float,  # Exit value after exit fee
            'gross_profit_usd': float,  # Profit before taxes
            'capital_gain_usd': float,  # Taxable gain
            'tax_usd': float,  # Tax owed
            'net_profit_usd': float,  # Final profit after all costs
            'net_profit_pct': float,  # Net profit as % of cost basis
            'price_change_pct': float  # Raw price change %
        }
    """
    # Calculate entry cost
    entry_value_usd = entry_price * shares
    if cost_basis_usd is None:
        entry_fee_usd = entry_value_usd * (entry_fee_pct / 100)
        cost_basis_usd = entry_value_usd + entry_fee_usd
    else:
        # Cost basis provided - back-calculate entry fee
        entry_fee_usd = cost_basis_usd - entry_value_usd

    # Calculate exit proceeds
    exit_value_usd = exit_price * shares
    exit_fee_usd = exit_value_usd * (exit_fee_pct / 100)
    exit_proceeds_usd = exit_value_usd - exit_fee_usd

    # Calculate profit before taxes
    gross_profit_usd = exit_proceeds_usd - cost_basis_usd

    # Calculate capital gain (same as gross profit in this case)
    capital_gain_usd = gross_profit_usd

    # Calculate tax (only on gains)
    if capital_gain_usd > 0:
        tax_usd = capital_gain_usd * (tax_rate_pct / 100)
    else:
        tax_usd = 0.0

    # Calculate net profit
    net_profit_usd = gross_profit_usd - tax_usd

    # Calculate percentages
    net_profit_pct = (net_profit_usd / cost_basis_usd) * 100 if cost_basis_usd > 0 else 0
    price_change_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'entry_value_usd': entry_value_usd,
        'entry_fee_usd': entry_fee_usd,
        'cost_basis_usd': cost_basis_usd,
        'exit_value_usd': exit_value_usd,
        'exit_fee_usd': exit_fee_usd,
        'exit_proceeds_usd': exit_proceeds_usd,
        'gross_profit_usd': gross_profit_usd,
        'capital_gain_usd': capital_gain_usd,
        'tax_usd': tax_usd,
        'net_profit_usd': net_profit_usd,
        'net_profit_pct': net_profit_pct,
        'price_change_pct': price_change_pct
    }


def calculate_required_price_for_target_profit(
    entry_price: float,
    target_net_profit_pct: float,
    entry_fee_pct: float,
    exit_fee_pct: float,
    tax_rate_pct: float
) -> Dict:
    """
    Calculate the exit price required to achieve a target net profit percentage.

    This is useful for setting profit targets that account for all costs.

    Solving for exit_price where net_profit_pct = target:
        This requires iterative solving or approximation.

    Args:
        entry_price: Entry price per share
        target_net_profit_pct: Desired net profit % (e.g., 0.8 for 0.8%)
        entry_fee_pct: Entry fee %
        exit_fee_pct: Exit fee %
        tax_rate_pct: Tax rate %

    Returns:
        {
            'required_exit_price': float,  # Price needed to hit target
            'price_change_pct': float,  # % change from entry to exit
            'gross_price_move_pct': float  # Gross move needed
        }
    """
    # Iterative approach: test exit prices until we find target
    # Start with a rough estimate
    shares = 1.0  # Use 1 share for percentage calculations

    # Binary search for the right exit price
    low = entry_price
    high = entry_price * 2  # Assume max 100% move
    tolerance = 0.0001  # 0.01% tolerance

    required_exit_price = None

    for _ in range(100):  # Max 100 iterations
        mid = (low + high) / 2

        # Calculate profit at this price
        result = calculate_net_profit_from_price_move(
            entry_price=entry_price,
            exit_price=mid,
            shares=shares,
            entry_fee_pct=entry_fee_pct,
            exit_fee_pct=exit_fee_pct,
            tax_rate_pct=tax_rate_pct
        )

        net_profit_pct = result['net_profit_pct']

        # Check if we're close enough
        if abs(net_profit_pct - target_net_profit_pct) < tolerance:
            required_exit_price = mid
            break

        # Adjust search range
        if net_profit_pct < target_net_profit_pct:
            low = mid
        else:
            high = mid

    if required_exit_price is None:
        required_exit_price = high

    price_change_pct = ((required_exit_price - entry_price) / entry_price) * 100

    return {
        'required_exit_price': required_exit_price,
        'price_change_pct': price_change_pct,
        'gross_price_move_pct': price_change_pct
    }


def calculate_breakeven_price(
    entry_price: float,
    entry_fee_pct: float,
    exit_fee_pct: float,
    tax_rate_pct: float
) -> Dict:
    """
    Calculate the exit price needed to break even (net profit = $0).

    Args:
        entry_price: Entry price per share
        entry_fee_pct: Entry fee %
        exit_fee_pct: Exit fee %
        tax_rate_pct: Tax rate %

    Returns:
        {
            'breakeven_price': float,
            'breakeven_price_change_pct': float
        }
    """
    result = calculate_required_price_for_target_profit(
        entry_price=entry_price,
        target_net_profit_pct=0.0,
        entry_fee_pct=entry_fee_pct,
        exit_fee_pct=exit_fee_pct,
        tax_rate_pct=tax_rate_pct
    )

    return {
        'breakeven_price': result['required_exit_price'],
        'breakeven_price_change_pct': result['price_change_pct']
    }


# Example usage and validation
if __name__ == "__main__":
    # Test with realistic values
    print("="*80)
    print("PROFITABILITY CALCULATOR - VALIDATION")
    print("="*80)

    entry_price = 15.00
    capital = 4500
    entry_fee = 0.6
    exit_fee = 0.6
    tax_rate = 37.0

    print(f"\nScenario: ${capital} capital, entry ${entry_price}/share")
    print(f"Fees: {entry_fee}% entry, {exit_fee}% exit")
    print(f"Tax: {tax_rate}%")
    print()

    # Calculate shares
    share_calc = calculate_required_shares_for_capital(capital, entry_price, entry_fee)
    print(f"Shares to buy: {share_calc['shares']:.2f}")
    print(f"  Subtotal: ${share_calc['subtotal_usd']:.2f}")
    print(f"  Entry fee: ${share_calc['entry_fee_usd']:.2f}")
    print(f"  Total cost: ${share_calc['total_cost_usd']:.2f}")
    print(f"  Cost basis/share: ${share_calc['cost_basis_per_share']:.4f}")
    print()

    # Test different exit scenarios
    test_scenarios = [
        ("0.8% price move", entry_price * 1.008),
        ("1.5% price move", entry_price * 1.015),
        ("2.5% price move", entry_price * 1.025),
    ]

    for scenario_name, exit_price in test_scenarios:
        print(f"\n{scenario_name}: ${exit_price:.4f}")
        profit = calculate_net_profit_from_price_move(
            entry_price=entry_price,
            exit_price=exit_price,
            shares=share_calc['shares'],
            entry_fee_pct=entry_fee,
            exit_fee_pct=exit_fee,
            tax_rate_pct=tax_rate,
            cost_basis_usd=share_calc['total_cost_usd']
        )
        print(f"  Net profit: ${profit['net_profit_usd']:.2f} ({profit['net_profit_pct']:.2f}%)")
        print(f"  Breakdown: Exit ${profit['exit_value_usd']:.2f} - Fee ${profit['exit_fee_usd']:.2f} - Cost ${profit['cost_basis_usd']:.2f} - Tax ${profit['tax_usd']:.2f}")

    # Calculate breakeven
    print(f"\n{'='*80}")
    breakeven = calculate_breakeven_price(entry_price, entry_fee, exit_fee, tax_rate)
    print(f"Breakeven price: ${breakeven['breakeven_price']:.4f} ({breakeven['breakeven_price_change_pct']:+.2f}%)")

    # Calculate required price for 0.8% NET profit
    print(f"\n{'='*80}")
    target_08 = calculate_required_price_for_target_profit(entry_price, 0.8, entry_fee, exit_fee, tax_rate)
    print(f"For 0.8% NET profit, need: ${target_08['required_exit_price']:.4f} ({target_08['price_change_pct']:+.2f}% price move)")

    print(f"\n{'='*80}")
