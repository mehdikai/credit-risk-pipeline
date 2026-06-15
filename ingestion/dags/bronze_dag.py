"""Airflow DAG — bronze_ingestion.

One task per raw source file: validate (GE) then write untouched Parquet
to the Bronze layer. See ingestion/bronze_ingest.py for the actual logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pendulum

from airflow.sdk import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dag(
    dag_id="bronze_ingestion",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["bronze", "lakehouse"],
)
def bronze_ingestion():
    @task
    def ingest(table_name: str):
        from ingestion.bronze_ingest import ingest_table

        return ingest_table(table_name)

    for table_name in ["application_record", "credit_record", "credit_risk_dataset"]:
        ingest.override(task_id=f"ingest_{table_name}")(table_name)


bronze_ingestion()
