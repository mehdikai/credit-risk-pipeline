"""STEP 2 — SILVER: clean, join, dedup, default label (vintage analysis).

Reads Bronze Parquet via DuckDB, dedups application_record on ID, derives
the TARGET (default) label from credit_record via vintage analysis,
engineers a handful of base features, fixes known nulls, and writes the
result to the Silver layer as a Delta table.
"""
import duckdb

from config.config import BRONZE_PATH, SILVER_PATH
from dq_checks.ge_utils import validate_dataframe
from dq_checks.silver_suites import silver_input_expectations, silver_output_expectations

try:
    from deltalake import write_deltalake
except ImportError:  # pragma: no cover
    write_deltalake = None


def build_silver_clean():
    con = duckdb.connect()

    app_path = f"{BRONZE_PATH}application_record/application_record.parquet"
    credit_path = f"{BRONZE_PATH}credit_record/credit_record.parquet"

    app_raw = con.execute(f"SELECT * FROM read_parquet('{app_path}')").df()

    # --- input contract check (pre-clean, on raw application_record) ---
    input_result = validate_dataframe(
        app_raw, "silver_input_application_record", silver_input_expectations()
    )
    if not input_result["success"]:
        print("WARNING: Silver input-contract checks did not fully pass (see Data Docs)")

    # --- dedup application_record on ID (keep first occurrence) ---
    app_dedup = con.execute(f"""
        SELECT * EXCLUDE (_rn, _seq) FROM (
            SELECT *, row_number() OVER (PARTITION BY ID ORDER BY _seq) AS _rn
            FROM (
                SELECT *, row_number() OVER () AS _seq
                FROM read_parquet('{app_path}')
            )
        ) WHERE _rn = 1
    """).df()

    # --- default label via vintage analysis on credit_record ---
    # A client is "bad" (TARGET=1) if STATUS ever reaches 60+ days overdue
    # (status codes 2-5) in their credit history.
    bad_clients = con.execute(f"""
        SELECT ID, MAX(CASE WHEN STATUS IN ('2','3','4','5') THEN 1 ELSE 0 END) AS TARGET
        FROM read_parquet('{credit_path}')
        GROUP BY ID
    """).df()

    con.register("app_dedup", app_dedup)
    con.register("bad_clients", bad_clients)
    joined = con.execute("""
        SELECT a.*, b.TARGET
        FROM app_dedup a
        INNER JOIN bad_clients b ON a.ID = b.ID
    """).df()

    # --- feature engineering / null fixes ---
    joined["AGE_YEARS"] = (-joined["DAYS_BIRTH"] / 365).round(0).astype(int)
    joined["IS_UNEMPLOYED"] = (joined["DAYS_EMPLOYED"] > 0).astype(int)
    # DAYS_EMPLOYED uses a 365243 sentinel for unemployed/pensioners (positive
    # value). Map that to 0 employed years rather than leaving it null, so
    # downstream WoE binning (Week 4) doesn't have to special-case nulls.
    joined["EMPLOYED_YEARS"] = joined["DAYS_EMPLOYED"].apply(
        lambda x: round(-x / 365, 1) if x < 0 else 0.0
    )
    joined["OCCUPATION_TYPE"] = joined["OCCUPATION_TYPE"].fillna("Unknown")

    # Domain ratio features — both outperform every raw shortlisted feature
    # on IV (Week 4 experiment) and stay interpretable for the BAM scorecard
    # narrative ("share of life spent employed", "income per dependent").
    # Denominators are safe: AGE_YEARS >= 18 and CNT_FAM_MEMBERS >= 1 always.
    joined["EMPLOYED_RATIO"] = joined["EMPLOYED_YEARS"] / joined["AGE_YEARS"]
    joined["INCOME_PER_FAM_MEMBER"] = joined["AMT_INCOME_TOTAL"] / joined["CNT_FAM_MEMBERS"]

    # --- output contract check (post-clean) ---
    output_result = validate_dataframe(
        joined, "silver_output_silver_clean", silver_output_expectations()
    )
    if not output_result["success"]:
        raise ValueError("Silver output-contract checks failed for silver_clean")

    # --- write to Delta ---
    out_path = f"{SILVER_PATH}silver_clean"
    write_deltalake(out_path, joined, mode="overwrite", schema_mode="overwrite")

    return {
        "rows_bronze_application_record": len(app_raw),
        "rows_deduped_application_record": len(app_dedup),
        "rows_credit_record_clients": len(bad_clients),
        "rows_silver_clean": len(joined),
        "default_rate_pct": round(joined["TARGET"].mean() * 100, 2),
        "output_path": out_path,
    }


def main():
    stats = build_silver_clean()
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
