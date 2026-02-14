"""
ChEMBL Data Processing Module

Functions for loading and processing chemical activity data from ChEMBL SQLite database.
Handles unit normalization, pIC50 calculation, and data cleaning.
"""
from __future__ import annotations

import sqlite3
import polars as pl
from pathlib import Path


# SQL query to extract IC50 activity data with molecular structures and properties
CHEMBL_ACTIVITY_QUERY = """
SELECT
    act.activity_id,
    act.molregno,
    cs.canonical_smiles,
    act.standard_value,
    act.standard_units,
    act.standard_type,
    act.standard_relation,
    act.pchembl_value,
    cp.mw_freebase,
    cp.alogp,
    cp.hba,
    cp.hbd,
    cp.psa,
    cp.rtb,
    cp.aromatic_rings,
    cp.qed_weighted,
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name,
    td.organism AS target_organism,
    a.confidence_score
FROM activities act
JOIN compound_structures cs ON act.molregno = cs.molregno
JOIN compound_properties cp ON act.molregno = cp.molregno
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
WHERE
    act.standard_type = 'IC50'
    AND act.standard_value IS NOT NULL
    AND cs.canonical_smiles IS NOT NULL
"""

# Lighter query without compound properties (faster for large extracts)
CHEMBL_ACTIVITY_QUERY_LIGHT = """
SELECT
    act.activity_id,
    act.molregno,
    cs.canonical_smiles,
    act.standard_value,
    act.standard_units,
    act.standard_type,
    act.standard_relation,
    act.pchembl_value,
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name,
    td.organism AS target_organism,
    a.confidence_score
FROM activities act
JOIN compound_structures cs ON act.molregno = cs.molregno
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
WHERE
    act.standard_type = 'IC50'
    AND act.standard_value IS NOT NULL
    AND cs.canonical_smiles IS NOT NULL
"""


def load_chembl_data(
    db_path: str | Path,
    query: str = None,
    limit: int = None,
    organism_filter: str = None
) -> pl.DataFrame:
    """
    Load ChEMBL activity data from SQLite database.

    Args:
        db_path: Path to chembl_36.db SQLite database
        query: Custom SQL query (uses default IC50 query if None)
        limit: Maximum number of rows to return (for testing)
        organism_filter: Filter by target organism (e.g., 'Homo sapiens')

    Returns:
        Polars DataFrame with activity data
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if query is None:
        query = CHEMBL_ACTIVITY_QUERY_LIGHT

    # Add organism filter if specified
    if organism_filter:
        query = query.rstrip().rstrip(';')
        query += f"\n    AND td.organism = '{organism_filter}'"

    # Add limit if specified
    if limit:
        query = query.rstrip().rstrip(';')
        query += f"\n    LIMIT {limit}"

    print(f"Loading data from {db_path}...")

    conn = sqlite3.connect(db_path)
    try:
        df = pl.read_database(query, conn)
        print(f"Loaded {len(df):,} records")
        return df
    finally:
        conn.close()


def impute_units(df: pl.DataFrame) -> pl.DataFrame:
    """
    Impute missing standard_units based on value range heuristics.

    Heuristic:
    - Values 0.01 to 1e6 without units are assumed to be nM
    - Tracks imputation with 'units_imputed' flag column

    Args:
        df: DataFrame with standard_value and standard_units columns

    Returns:
        DataFrame with imputed units and units_imputed flag
    """
    df = df.with_columns([
        pl.lit(False).alias("units_imputed")
    ])

    # Conditions for imputation
    missing_units = pl.col("standard_units").is_null()
    has_value = pl.col("standard_value").is_not_null()
    in_nm_range = (pl.col("standard_value") >= 0.01) & (pl.col("standard_value") <= 1e6)

    df = df.with_columns([
        pl.when(missing_units & has_value & in_nm_range)
        .then(pl.lit("nM"))
        .otherwise(pl.col("standard_units"))
        .alias("standard_units"),

        pl.when(missing_units & has_value & in_nm_range)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("units_imputed")
    ])

    return df


def normalize_to_molar(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert activity values to Molar concentration.

    Conversion factors:
    - nM: value * 1e-9
    - uM: value * 1e-6
    - mM: value * 1e-3
    - M: value (no conversion)

    Args:
        df: DataFrame with standard_value and standard_units columns

    Returns:
        DataFrame with new 'value_molar' column
    """
    df = df.with_columns([
        pl.when(pl.col("standard_units") == "nM")
        .then(pl.col("standard_value") * 1e-9)
        .when(pl.col("standard_units") == "uM")
        .then(pl.col("standard_value") * 1e-6)
        .when(pl.col("standard_units") == "mM")
        .then(pl.col("standard_value") * 1e-3)
        .when(pl.col("standard_units") == "M")
        .then(pl.col("standard_value"))
        .otherwise(None)
        .alias("value_molar")
    ])

    return df


def compute_pIC50(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate pIC50 from Molar concentration.

    Formula: pIC50 = -log10(IC50_molar)

    If pchembl_value exists, uses that instead (pre-calculated by ChEMBL).

    Args:
        df: DataFrame with value_molar column (or pchembl_value)

    Returns:
        DataFrame with new 'pIC50' column
    """
    # Ensure value_molar exists
    if "value_molar" not in df.columns:
        df = normalize_to_molar(df)

    df = df.with_columns([
        pl.when(pl.col("pchembl_value").is_not_null())
        .then(pl.col("pchembl_value"))
        .otherwise(-pl.col("value_molar").log10())
        .alias("pIC50")
    ])

    return df


def clean_data(
    df: pl.DataFrame,
    pIC50_min: float = 3.0,
    pIC50_max: float = 12.0,
    deduplicate_by_smiles: bool = True
) -> pl.DataFrame:
    """
    Clean and filter activity data.

    Steps:
    1. Remove records with null SMILES or pIC50
    2. Remove infinite pIC50 values
    3. Filter pIC50 to valid range (default: 3-12)
    4. Optionally deduplicate by SMILES (keeps first occurrence)

    Args:
        df: DataFrame with canonical_smiles and pIC50 columns
        pIC50_min: Minimum valid pIC50 value
        pIC50_max: Maximum valid pIC50 value
        deduplicate_by_smiles: Whether to keep only one record per SMILES

    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)

    # Filter nulls and invalid values
    df_clean = df.filter(
        pl.col("canonical_smiles").is_not_null() &
        pl.col("pIC50").is_not_null() &
        pl.col("pIC50").is_finite() &
        (pl.col("pIC50") >= pIC50_min) &
        (pl.col("pIC50") <= pIC50_max)
    )

    # Deduplicate by SMILES if requested
    if deduplicate_by_smiles:
        df_clean = df_clean.unique(subset=["canonical_smiles"], keep="first")

    final_count = len(df_clean)
    print(f"Cleaned: {initial_count:,} -> {final_count:,} records ({final_count/initial_count*100:.1f}% retained)")

    return df_clean


def load_and_process_chembl(
    db_path: str | Path,
    limit: int = None,
    organism_filter: str = None,
    include_properties: bool = False
) -> pl.DataFrame:
    """
    Complete pipeline: load, process, and clean ChEMBL data.

    Args:
        db_path: Path to chembl_36.db
        limit: Maximum records to load (for testing)
        organism_filter: Filter by target organism (e.g., 'Homo sapiens')
        include_properties: Include molecular properties (slower query)

    Returns:
        Cleaned DataFrame ready for analysis/ML
    """
    query = CHEMBL_ACTIVITY_QUERY if include_properties else CHEMBL_ACTIVITY_QUERY_LIGHT

    # Load raw data
    df = load_chembl_data(db_path, query=query, limit=limit, organism_filter=organism_filter)

    # Process
    print("Processing data...")
    df = impute_units(df)
    df = normalize_to_molar(df)
    df = compute_pIC50(df)
    df = clean_data(df)

    return df


def export_to_parquet(
    df: pl.DataFrame,
    output_path: str | Path,
    compression: str = "zstd"
) -> None:
    """
    Export DataFrame to Parquet format.

    Args:
        df: DataFrame to export
        output_path: Output file path
        compression: Compression codec (zstd, snappy, gzip, lz4)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(output_path, compression=compression)
    print(f"Exported to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    # Example usage
    import sys

    db_path = Path(__file__).parent.parent / "chembl_36.db"

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    # Load sample data (1000 records for testing)
    df = load_and_process_chembl(db_path, limit=1000, organism_filter="Homo sapiens")

    print("\nSample data:")
    print(df.head())

    print("\nSchema:")
    print(df.schema)

    print("\npIC50 statistics:")
    print(df.select("pIC50").describe())
