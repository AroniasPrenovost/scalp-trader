#!/usr/bin/env python3
"""
Script to normalize volume_24h data in crypto files.

PROBLEM:
The volume_24h field has inconsistent formats in historical data:
- Backfilled data (from CoinGecko): Volume in USD (e.g., 11,983,928,457.88)
- Live data (from Coinbase): Volume in BTC (e.g., 5,200.00)

SOLUTION:
Convert all USD volumes to BTC by dividing by the price to match industry standard
(volume should be in base currency = BTC for BTC-USD pairs).

This aligns with:
1. Coinbase API standard (volume_24h in base currency)
2. Industry best practices (CryptoDataDownload, CoinMarketCap, etc.)
3. Fixed backfill_historical_data.py to prevent future inconsistencies
"""

import json
import shutil
import os
from datetime import datetime

def normalize_volume_data(file_path, threshold=100000, dry_run=False):
    """
    Normalize volume_24h data to use consistent BTC units.

    Args:
        file_path: Path to the crypto data JSON file
        threshold: Volume values above this are assumed to be in USD format
        dry_run: If True, only show what would be changed without modifying the file

    Returns:
        Dictionary with conversion statistics
    """
    print(f"\n{'='*60}")
    print(f"Normalizing volume data in: {file_path}")
    print(f"{'='*60}\n")

    # Read the data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Create backup
    if not dry_run:
        backup_path = f"{file_path}.backup_{int(datetime.now().timestamp())}"
        shutil.copy2(file_path, backup_path)
        print(f"✓ Created backup: {backup_path}\n")

    # Statistics
    stats = {
        'total_entries': len(data),
        'converted': 0,
        'already_btc': 0,
        'errors': 0,
        'skipped_zero_price': 0
    }

    # Process each entry
    print("Processing entries...")
    for i, entry in enumerate(data):
        try:
            volume = float(entry.get('volume_24h', 0))
            price = float(entry.get('price', 0))

            # Check if volume needs conversion (likely in USD format)
            if volume > threshold:
                if price == 0:
                    if stats['skipped_zero_price'] == 0:  # Only print first occurrence
                        print(f"⚠ Warning: Entries with volume > {threshold:,} but price = 0 detected, skipping these")
                    stats['skipped_zero_price'] += 1
                    continue

                # Convert USD volume to BTC volume
                old_volume = volume
                new_volume = volume / price

                if dry_run and stats['converted'] < 5:  # Show first 5 examples in dry run
                    print(f"  Entry {i}: Would convert {old_volume:,.2f} USD → {new_volume:.8f} BTC (price: ${price:,.2f})")
                elif not dry_run:
                    entry['volume_24h'] = str(new_volume)

                stats['converted'] += 1

                # Show progress every 1000 entries
                if not dry_run and stats['converted'] % 1000 == 0:
                    print(f"  Converted {stats['converted']:,} entries so far...")

            else:
                # Already in BTC format
                stats['already_btc'] += 1

        except (ValueError, TypeError, KeyError) as e:
            if stats['errors'] == 0:  # Only print first error
                print(f"✗ Error processing entry {i}: {e}")
            stats['errors'] += 1
            continue

    # Write the normalized data back
    if not dry_run:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n✓ Data normalized and saved to {file_path}")

    # Print statistics
    print(f"\n{'='*60}")
    print("Normalization Summary:")
    print(f"{'='*60}")
    print(f"  Total entries:              {stats['total_entries']:,}")
    print(f"  Converted (USD → BTC):      {stats['converted']:,}")
    print(f"  Already in BTC format:      {stats['already_btc']:,}")
    print(f"  Skipped (price = 0):        {stats['skipped_zero_price']:,}")
    print(f"  Errors:                     {stats['errors']:,}")
    print(f"{'='*60}\n")

    if dry_run:
        print("⚠ DRY RUN MODE - No changes were made to the file")
        print("  Run with --apply to apply changes\n")
    else:
        print("✓ Normalization complete!")
        print(f"  Backup saved to: {backup_path}\n")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Normalize volume_24h data from USD to BTC (industry standard)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on BTC-USD (preview changes):
  python3 normalize_volume_data.py --dry-run

  # Apply normalization to BTC-USD:
  python3 normalize_volume_data.py --apply

  # Normalize all crypto files:
  python3 normalize_volume_data.py --all --apply

  # Normalize specific file:
  python3 normalize_volume_data.py -f ./coinbase-data/ETH-USD.json --apply
        """
    )
    parser.add_argument('--file', '-f', default='./coinbase-data/BTC-USD.json',
                        help='Path to the crypto data file (default: ./coinbase-data/BTC-USD.json)')
    parser.add_argument('--threshold', '-t', type=float, default=100000,
                        help='Volume threshold - values above this are assumed to be USD (default: 100000)')
    parser.add_argument('--dry-run', '-d', action='store_true', default=True,
                        help='Dry run mode - show what would be changed without modifying (DEFAULT)')
    parser.add_argument('--apply', '-a', action='store_true',
                        help='Apply changes (required to actually modify files)')
    parser.add_argument('--all', action='store_true',
                        help='Normalize all crypto files in coinbase-data directory')

    args = parser.parse_args()

    # Determine dry_run mode
    dry_run = not args.apply

    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - No files will be modified")
        print("Use --apply flag to actually apply changes")
        print("="*60)

    if args.all:
        # Process all JSON files in coinbase-data directory
        data_dir = './coinbase-data'
        if not os.path.exists(data_dir):
            print(f"Error: Directory {data_dir} does not exist")
            exit(1)

        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and not f.startswith('.')]

        if not json_files:
            print(f"No JSON files found in {data_dir}")
            exit(0)

        print(f"\nFound {len(json_files)} crypto data file(s) to process")

        total_stats = {
            'total_entries': 0,
            'converted': 0,
            'already_btc': 0,
            'errors': 0,
            'skipped_zero_price': 0
        }

        for json_file in sorted(json_files):
            file_path = os.path.join(data_dir, json_file)
            stats = normalize_volume_data(file_path, args.threshold, dry_run)

            # Aggregate stats
            for key in total_stats:
                total_stats[key] += stats[key]

        # Print overall summary
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY (All Files):")
        print(f"{'='*60}")
        print(f"  Files processed:            {len(json_files)}")
        print(f"  Total entries:              {total_stats['total_entries']:,}")
        print(f"  Converted (USD → BTC):      {total_stats['converted']:,}")
        print(f"  Already in BTC format:      {total_stats['already_btc']:,}")
        print(f"  Skipped (price = 0):        {total_stats['skipped_zero_price']:,}")
        print(f"  Errors:                     {total_stats['errors']:,}")
        print(f"{'='*60}\n")

        if dry_run:
            print("To apply these changes, run with --apply flag")
    else:
        # Process single file
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} does not exist")
            exit(1)

        normalize_volume_data(args.file, args.threshold, dry_run)

        if dry_run:
            print("To apply these changes, run with --apply flag")
