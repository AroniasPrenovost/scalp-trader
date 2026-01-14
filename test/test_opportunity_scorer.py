"""
Test script for Market Rotation Opportunity Scorer

This script tests the opportunity scoring system to ensure it correctly:
1. Scans all enabled assets
2. Scores each one based on strategy signals
3. Returns the single best opportunity
4. Allows skipping if no good setups exist
"""

import json
from utils.opportunity_scorer import find_best_opportunity, print_opportunity_report, score_opportunity
from utils.coinbase import get_coinbase_client, get_asset_price
from utils.file_helpers import get_property_values_from_crypto_file
from utils.price_helpers import calculate_percentage_from_min

# Load config
def load_config(file_path='config.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    print("="*100)
    print("OPPORTUNITY SCORER TEST")
    print("="*100)
    print()

    # Load config
    config = load_config()

    # Get enabled wallets
    enabled_wallets = [wallet['symbol'] for wallet in config['wallets'] if wallet['enabled']]
    print(f"Enabled wallets: {enabled_wallets}")
    print()

    # Initialize Coinbase client
    coinbase_client = get_coinbase_client()

    # Data settings
    DATA_RETENTION_HOURS = config['data_retention']['max_hours']
    INTERVAL_SECONDS = config['data_retention']['interval_seconds']
    coinbase_data_directory = 'coinbase-data'

    # Test 1: Score individual opportunities
    print("TEST 1: Scoring individual opportunities")
    print("-"*100)

    all_opportunities = []

    for symbol in enabled_wallets:
        try:
            print(f"\nScoring {symbol}...")

            # Get current price
            current_price = get_asset_price(coinbase_client, symbol)
            print(f"  Current price: ${current_price:.4f}")

            # Get historical price data
            coin_prices_list = get_property_values_from_crypto_file(
                coinbase_data_directory,
                symbol,
                'price',
                max_age_hours=DATA_RETENTION_HOURS
            )

            if not coin_prices_list or len(coin_prices_list) == 0:
                print(f"  ⚠️  No price data available")
                continue

            print(f"  Price data points: {len(coin_prices_list)}")

            # Calculate 24h volatility
            volatility_window_hours = 24
            volatility_data_points = int(volatility_window_hours / (INTERVAL_SECONDS / 3600))
            recent_prices = coin_prices_list[-volatility_data_points:] if len(coin_prices_list) >= volatility_data_points else coin_prices_list

            min_price = min(recent_prices)
            max_price = max(recent_prices)
            range_percentage_from_min = calculate_percentage_from_min(min_price, max_price)

            print(f"  24h volatility: {range_percentage_from_min:.2f}%")

            # Score this opportunity
            opportunity = score_opportunity(
                symbol=symbol,
                config=config,
                coinbase_client=coinbase_client,
                coin_prices_list=coin_prices_list,
                current_price=current_price,
                range_percentage_from_min=range_percentage_from_min
            )

            all_opportunities.append(opportunity)

            print(f"  Score: {opportunity['score']:.1f}/100")
            print(f"  Signal: {opportunity['signal']}")
            print(f"  Strategy: {opportunity['strategy']}")
            print(f"  Trend: {opportunity['trend']}")
            print(f"  Can trade: {opportunity['can_trade']}")
            print(f"  Reasoning: {opportunity['reasoning']}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print("\n" + "="*100)
    print("TEST 2: Finding best opportunity")
    print("="*100)
    print()

    # Find the best opportunity
    best_opportunity = find_best_opportunity(
        config=config,
        coinbase_client=coinbase_client,
        enabled_symbols=enabled_wallets,
        interval_seconds=INTERVAL_SECONDS,
        data_retention_hours=DATA_RETENTION_HOURS
    )

    # Print detailed report
    print_opportunity_report(all_opportunities, best_opportunity)

    print("\n" + "="*100)
    print("TEST 3: Market Rotation Logic")
    print("="*100)
    print()

    if best_opportunity:
        min_score = config.get('market_rotation', {}).get('min_score_for_entry', 50)

        print(f"Best opportunity: {best_opportunity['symbol']}")
        print(f"Score: {best_opportunity['score']:.1f}/100")
        print(f"Minimum required score: {min_score}")
        print()

        if best_opportunity['score'] >= min_score:
            print(f"✅ TRADE: {best_opportunity['symbol']}")
            print(f"   Strategy: {best_opportunity['strategy']}")
            print(f"   Entry: ${best_opportunity['entry_price']:.4f}")
            print(f"   Stop Loss: ${best_opportunity['stop_loss']:.4f}")
            print(f"   Profit Target: ${best_opportunity['profit_target']:.4f}")
            if best_opportunity['risk_reward_ratio']:
                print(f"   Risk/Reward: 1:{best_opportunity['risk_reward_ratio']:.2f}")
            print()
            print(f"⏭  SKIP all other assets: {[s for s in enabled_wallets if s != best_opportunity['symbol']]}")
        else:
            print(f"⚠️  SKIP ALL TRADES - best score {best_opportunity['score']:.1f} below minimum {min_score}")
    else:
        print("⚠️  NO OPPORTUNITIES - all assets have open positions or no valid setups")

    print("\n" + "="*100)
    print("TEST COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main()
