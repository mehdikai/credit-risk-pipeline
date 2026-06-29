"""STEP 1 — BRONZE.

Raw CSVs from data/raw/ are validated against critical schema/identity
checks, then written as-is (untouched) to the Bronze layer as Parquet.
Soft checks are logged as warnings and never block the write.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.config import BRONZE_PATH
from dq_checks.bronze_suites import SUITES
from dq_checks.ge_utils import validate_dataframe

RAW_DATA_DIR = Path("data/raw")

TABLES = {
    "application_record": "application_record.csv",
    "credit_record": "credit_record.csv",
    "credit_risk_dataset": "credit_risk_dataset.csv",
}


def _file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_manifest(entry):
    manifest_path = Path(BRONZE_PATH) / "_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def ingest_table(table_name):
    csv_path = RAW_DATA_DIR / TABLES[table_name]
    df = pd.read_csv(csv_path)

    critical, warn = SUITES[table_name]()

    critical_result = validate_dataframe(df, f"{table_name}_critical", critical)
    if not critical_result["success"]:
        raise ValueError(f"Bronze critical GE checks failed for {table_name}")

    warn_result = validate_dataframe(df, f"{table_name}_warn", warn)
    if not warn_result["success"]:
        print(f"WARNING: soft GE checks failed for {table_name} (see Data Docs)")

    out_dir = Path(BRONZE_PATH) / table_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{table_name}.parquet"
    df.to_parquet(out_path, index=False)

    entry = {
        "table": table_name,
        "source_file": str(csv_path),
        "source_hash": _file_hash(csv_path),
        "row_count": len(df),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(out_path),
        "ge_critical_success": critical_result["success"],
        "ge_warn_success": warn_result["success"],
    }
    _append_manifest(entry)
    return entry


def main():
    for table_name in TABLES:
        entry = ingest_table(table_name)
        print(f"Ingested {table_name}: {entry['row_count']} rows -> {entry['output_path']}")


if __name__ == "__main__":
    main()
