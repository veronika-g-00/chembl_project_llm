#!/usr/bin/env python3
"""
Export ChEMBL SQLite data to Parquet format for Spark processing.

This script extracts IC50 activity data from the ChEMBL SQLite database
and exports it to Parquet format for distributed processing with Spark.

Usage:
    python export_to_parquet.py                    # Export all data
    python export_to_parquet.py --limit 10000     # Export 10k records (testing)
    python export_to_parquet.py --organism "Homo sapiens"  # Filter by organism
"""
from __future__ import annotations

import argparse
from pathlib import Path
from libs.data_processing import load_chembl_data, export_to_parquet, CHEMBL_ACTIVITY_QUERY_LIGHT


def main():
    parser = argparse.ArgumentParser(description="Export ChEMBL data to Parquet")
    parser.add_argument(
        "--db",
        default="chembl_36.db",
        help="Path to ChEMBL SQLite database"
    )
    parser.add_argument(
        "--output",
        default="libs/datasets/chembl_raw.parquet",
        help="Output Parquet file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records (for testing)"
    )
    parser.add_argument(
        "--organism",
        default=None,
        help="Filter by target organism (e.g., 'Homo sapiens')"
    )
    args = parser.parse_args()

    # Validate database exists
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1

    # Load data from SQLite
    print("=" * 60)
    print("ChEMBL Data Export")
    print("=" * 60)

    df = load_chembl_data(
        db_path=db_path,
        query=CHEMBL_ACTIVITY_QUERY_LIGHT,
        limit=args.limit,
        organism_filter=args.organism
    )

    # Export to Parquet
    export_to_parquet(df, args.output)

    print("=" * 60)
    print("Export complete!")
    print(f"  Records: {len(df):,}")
    print(f"  Output:  {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
