"""STEP 3 — GOLD: final ML-ready feature table, versioned in Delta Lake.

Reads silver_woe, keeps only the WoE-encoded feature columns + TARGET
(drops ID and any helper/raw columns), validates the strict output
contract, and writes gold_features as a Delta table. Delta's transaction
log gives this table versioning for free — each run's version number is
also logged to _version_log.json for a quick human-readable history.
"""
import json
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable, write_deltalake

from config.config import GOLD_PATH, SILVER_PATH
from dq_checks.ge_utils import validate_dataframe
from dq_checks.gold_suites import gold_expectations

GOLD_VERSION_LOG = Path(GOLD_PATH) / "_version_log.json"


def build_gold_features():
    silver_woe = DeltaTable(f"{SILVER_PATH}silver_woe").to_pandas()

    feature_columns = [c for c in silver_woe.columns if c.endswith("_woe")]
    gold = silver_woe[feature_columns + ["TARGET"]].copy()

    # --- final type checks: everything must be numeric for sklearn ---
    non_numeric = [c for c in gold.columns if not pd.api.types.is_numeric_dtype(gold[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns found in gold_features: {non_numeric}")

    # --- output contract check ---
    result = validate_dataframe(gold, "gold_gold_features", gold_expectations(feature_columns))
    if not result["success"]:
        raise ValueError("Gold output-contract checks failed for gold_features")

    # --- write to Delta ---
    out_path = f"{GOLD_PATH}gold_features"
    write_deltalake(out_path, gold, mode="overwrite", schema_mode="overwrite")

    delta_version = DeltaTable(out_path).version()
    _append_version_log({
        "version": delta_version,
        "feature_columns": feature_columns,
        "row_count": len(gold),
        "target_rate_pct": round(gold["TARGET"].mean() * 100, 2),
    })

    return {
        "feature_columns": feature_columns,
        "rows_gold_features": len(gold),
        "delta_version": delta_version,
        "output_path": out_path,
    }


def _append_version_log(entry):
    GOLD_VERSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(GOLD_VERSION_LOG.read_text()) if GOLD_VERSION_LOG.exists() else []
    log.append(entry)
    GOLD_VERSION_LOG.write_text(json.dumps(log, indent=2, default=str))


def main():
    stats = build_gold_features()
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
