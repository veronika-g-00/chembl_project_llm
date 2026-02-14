"""
ChEMBL Spark ETL Job

Distributed processing of ChEMBL activity data using PySpark.
Reads from Parquet, performs transformations, and outputs processed dataset.

Usage:
    # Run locally
    spark-submit spark_etl_job.py --input data.parquet --output processed.parquet

    # Run on cluster (from Docker)
    spark-submit --master spark://spark-master:7077 spark_etl_job.py \
        --input /opt/workspace/data/chembl_raw.parquet \
        --output /opt/workspace/data/chembl_processed.parquet
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log10, lit, round as spark_round
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType
import argparse


def create_spark_session(app_name: str = "ChEMBL_ETL", master: str = None) -> SparkSession:
    """
    Create and configure Spark session.

    Args:
        app_name: Application name shown in Spark UI
        master: Spark master URL (None for local mode, spark://host:7077 for cluster)

    Returns:
        Configured SparkSession
    """
    builder = SparkSession.builder.appName(app_name)

    if master:
        builder = builder.master(master)

    builder = builder \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "8")

    return builder.getOrCreate()


def normalize_units(df):
    """
    Convert activity values to Molar concentration.

    Conversion factors:
    - nM: value * 1e-9
    - uM: value * 1e-6
    - mM: value * 1e-3
    - M: value (no conversion)
    """
    return df.withColumn(
        "value_molar",
        when(col("standard_units") == "nM", col("standard_value") * 1e-9)
        .when(col("standard_units") == "uM", col("standard_value") * 1e-6)
        .when(col("standard_units") == "mM", col("standard_value") * 1e-3)
        .when(col("standard_units") == "M", col("standard_value"))
        .otherwise(None)
    )


def compute_pIC50(df):
    """
    Calculate pIC50 = -log10(IC50_molar).

    Uses pchembl_value if available, otherwise calculates from value_molar.
    """
    # Ensure value_molar exists
    if "value_molar" not in df.columns:
        df = normalize_units(df)

    return df.withColumn(
        "pIC50",
        when(col("pchembl_value").isNotNull(), col("pchembl_value"))
        .otherwise(-log10(col("value_molar")))
    ).withColumn(
        "pIC50",
        spark_round(col("pIC50"), 2)
    )


def impute_units(df):
    """
    Impute missing standard_units based on value range heuristics.

    Values 0.01 to 1e6 without units are assumed to be nM.
    """
    return df.withColumn(
        "standard_units",
        when(
            col("standard_units").isNull() &
            col("standard_value").isNotNull() &
            (col("standard_value") >= 0.01) &
            (col("standard_value") <= 1e6),
            lit("nM")
        ).otherwise(col("standard_units"))
    ).withColumn(
        "units_imputed",
        when(
            col("standard_units").isNull() &
            col("standard_value").isNotNull() &
            (col("standard_value") >= 0.01) &
            (col("standard_value") <= 1e6),
            lit(True)
        ).otherwise(lit(False))
    )


def clean_data(df, pIC50_min: float = 3.0, pIC50_max: float = 12.0):
    """
    Clean and filter activity data.

    - Remove null SMILES and pIC50
    - Filter pIC50 to valid range (3-12)
    - Remove duplicate SMILES (keep first)
    """
    return df.filter(
        col("canonical_smiles").isNotNull() &
        col("pIC50").isNotNull() &
        (col("pIC50") >= pIC50_min) &
        (col("pIC50") <= pIC50_max)
    ).dropDuplicates(["canonical_smiles"])


def process_chembl_data(spark, input_path: str, output_path: str, organism_filter: str = None):
    """
    Complete ETL pipeline for ChEMBL data.

    Args:
        spark: SparkSession
        input_path: Input Parquet file path
        output_path: Output Parquet file path
        organism_filter: Optional filter by target organism (e.g., 'Homo sapiens')
    """
    print(f"Reading data from {input_path}...")
    df = spark.read.parquet(input_path)

    initial_count = df.count()
    print(f"Loaded {initial_count:,} records")

    # Apply organism filter if specified
    if organism_filter:
        df = df.filter(col("target_organism") == organism_filter)
        print(f"After organism filter: {df.count():,} records")

    # Processing pipeline
    print("Processing data...")
    df = impute_units(df)
    df = normalize_units(df)
    df = compute_pIC50(df)
    df = clean_data(df)

    final_count = df.count()
    print(f"After cleaning: {final_count:,} records ({final_count/initial_count*100:.1f}% retained)")

    # Select output columns
    output_cols = [
        "activity_id",
        "molregno",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_relation",
        "pIC50",
        "target_chembl_id",
        "target_name",
        "target_organism",
        "confidence_score"
    ]

    # Filter to existing columns only
    existing_cols = [c for c in output_cols if c in df.columns]
    df_output = df.select(existing_cols)

    # Write output
    print(f"Saving processed data to {output_path}...")
    df_output.write.mode("overwrite").parquet(output_path)

    print("ETL job completed successfully!")

    return df_output


def main():
    parser = argparse.ArgumentParser(description="ChEMBL Spark ETL Job")
    parser.add_argument("--input", required=True, help="Input Parquet file path")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument("--master", default=None, help="Spark master URL (e.g., spark://spark-master:7077)")
    parser.add_argument("--organism", default=None, help="Filter by target organism (e.g., 'Homo sapiens')")
    args = parser.parse_args()

    # Create Spark session
    spark = create_spark_session(master=args.master)

    try:
        # Run ETL pipeline
        process_chembl_data(
            spark=spark,
            input_path=args.input,
            output_path=args.output,
            organism_filter=args.organism
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
