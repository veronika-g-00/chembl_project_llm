"""
ChEMBL ETL Pipeline DAG

Apache Airflow DAG for orchestrating the ChEMBL data processing pipeline.
Schedules Spark ETL jobs to process chemical activity data.

The pipeline:
1. Reads raw ChEMBL data from Parquet
2. Processes: unit normalization, pIC50 calculation, cleaning
3. Outputs processed dataset ready for analysis
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


# Default arguments for the DAG
default_args = {
    'owner': 'chembl-pipeline',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id='chembl_etl_pipeline',
    default_args=default_args,
    description='Process ChEMBL IC50 activity data using Spark',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['chembl', 'etl', 'spark'],
) as dag:

    # Task: Run Spark ETL job
    # Uses BashOperator to submit Spark job to the cluster
    spark_etl_task = BashOperator(
        task_id='spark_chembl_etl',
        bash_command="""
            /opt/spark/bin/spark-submit \
                --master spark://spark-master:7077 \
                --deploy-mode client \
                --driver-memory 1g \
                --executor-memory 1g \
                --total-executor-cores 2 \
                /opt/workspace/spark_etl_job.py \
                --input /opt/workspace/libs/datasets/chembl_raw.parquet \
                --output /opt/workspace/libs/datasets/chembl_processed.parquet
        """,
    )

    # Task: Verify output
    verify_output_task = BashOperator(
        task_id='verify_output',
        bash_command="""
            if [ -d /opt/workspace/libs/datasets/chembl_processed.parquet ]; then
                echo "Output directory exists"
                ls -la /opt/workspace/libs/datasets/chembl_processed.parquet/
                echo "ETL job completed successfully!"
            else
                echo "ERROR: Output not found!"
                exit 1
            fi
        """,
    )

    # Define task dependencies
    spark_etl_task >> verify_output_task
