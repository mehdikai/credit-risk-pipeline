"""Airflow DAG — lakehouse_pipeline.

Bronze -> Silver medallion pipeline:
  - bronze: one task per raw source file, validates (GE) and writes
    untouched Parquet to the Bronze layer (ingestion/bronze_ingest.py)
  - silver_clean: joins, dedups, derives the TARGET label via vintage
    analysis, and writes silver_clean as a Delta table
    (ingestion/silver_clean.py)
  - silver_woe: WoE-bins + IV-selects features from silver_clean, writes
    silver_woe and the woe_bins artifact (ingestion/silver_woe.py)

Gold feature-selection tasks (Week 5) will be appended to this same DAG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pendulum

from airflow.sdk import dag, task
from airflow.sdk import TaskGroup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dag(
    dag_id="lakehouse_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["lakehouse"],
)
def lakehouse_pipeline():
    @task
    def ingest_bronze(table_name: str):
        from ingestion.bronze_ingest import ingest_table

        return ingest_table(table_name)

    @task
    def build_silver_clean(_bronze_results):
        from ingestion.silver_clean import build_silver_clean as _build

        return _build()

    @task
    def build_silver_woe(_silver_clean_result):
        from ingestion.silver_woe import build_silver_woe as _build

        return _build()

    with TaskGroup(group_id="bronze"):
        bronze_results = [
            ingest_bronze.override(task_id=f"ingest_{table_name}")(table_name)
            for table_name in ["application_record", "credit_record", "credit_risk_dataset"]
        ]

    silver_clean_result = build_silver_clean(bronze_results)
    build_silver_woe(silver_clean_result)


lakehouse_pipeline()
