#!/usr/bin/env python3
"""
Test the market rotation filter
"""

from utils.market_rotation_filter import print_rotation_summary, get_tradeable_assets, get_market_rotation_summary

if __name__ == "__main__":
    # Print summary
    print_rotation_summary()

    # Get tradeable assets
    tradeable = get_tradeable_assets()

    print("\n💰 CAPITAL ALLOCATION STRATEGY:")
    print("="*80)

    total_capital = 3500
    max_concurrent_trades = 2

    if tradeable:
        print(f"\nTotal Capital Available: ${total_capital:.2f}")
        print(f"Max Concurrent Trades: {max_concurrent_trades}")
        print(f"Capital per Trade: ${total_capital / max_concurrent_trades:.2f}")
        print()

        print(f"📊 Tradeable Assets ({len(tradeable)}):")
        for i, symbol in enumerate(tradeable, 1):
            if i <= max_concurrent_trades:
                print(f"  {i}. {symbol}: PRIORITY - Will trade if signal appears (${total_capital / max_concurrent_trades:.2f})")
            else:
                print(f"  {i}. {symbol}: BACKUP - Will trade if top {max_concurrent_trades} have no signals")

        print()
        if len(tradeable) > max_concurrent_trades:
            print(f"✅ With {len(tradeable)} tradeable assets, you have multiple opportunities!")
            print(f"   Bot will pick the best {max_concurrent_trades} setups and trade those.")
        elif len(tradeable) == max_concurrent_trades:
            print(f"✅ Perfect - {len(tradeable)} tradeable assets matches your concurrent trade limit")
        else:
            print(f"✅ {len(tradeable)} tradeable asset(s) - can use full ${total_capital:.2f} per trade")
    else:
        print(f"\nTotal Capital: ${total_capital:.2f}")
        print(f"⚠️  NO ASSETS TO TRADE - All assets in downtrend")
        print(f"   Recommendation: Wait for market conditions to improve")

    print("\n" + "="*80)
