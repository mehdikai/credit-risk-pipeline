"""STEP 2 (cont.) — SILVER: WoE binning, IV-based feature selection.

Loads silver_clean, bins + WoE-encodes the 14 shortlisted features with
scorecardpy, drops features outside the IV usefulness range, and writes
the encoded table (silver_woe) plus the woe_bins artifact. The woe_bins
artifact is the single source of truth Week 9 (live scoring) reuses to
transform new applicants identically to how training data was encoded.
"""
import json
import pickle
from pathlib import Path

import scorecardpy as sc
from deltalake import DeltaTable, write_deltalake

from config.config import SILVER_PATH
from dq_checks.ge_utils import validate_dataframe
from dq_checks.silver_woe_suites import silver_woe_expectations

SHORTLISTED_FEATURES = [
    "AGE_YEARS", "AMT_INCOME_TOTAL", "EMPLOYED_YEARS", "IS_UNEMPLOYED", "CODE_GENDER",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    # engineered ratios (added after IV experiment: both outperform every
    # raw feature above while staying interpretable for the scorecard)
    "EMPLOYED_RATIO", "INCOME_PER_FAM_MEMBER",
]

# Features with IV < IV_MIN carry too little predictive signal to be worth
# the model complexity; IV > IV_MAX is treated as suspiciously strong
# (likely overfit / a near-leak of the target) per standard scorecard practice.
IV_MIN = 0.02
IV_MAX = 0.5

WOE_BINS_DIR = Path(SILVER_PATH) / "woe_bins"
IV_REPORT_PATH = Path("docs/iv_report.json")
WOE_PLOTS_DIR = Path("docs/woe_plots")


def build_silver_woe():
    silver_clean = DeltaTable(f"{SILVER_PATH}silver_clean").to_pandas()
    sub = silver_clean[SHORTLISTED_FEATURES + ["TARGET"]]

    bins = sc.woebin(sub, y="TARGET", positive="1", print_step=0)

    iv_table = {feature: round(float(b["total_iv"].iloc[0]), 4) for feature, b in bins.items()}
    selected_features = [f for f, iv in iv_table.items() if IV_MIN <= iv <= IV_MAX]
    dropped_features = {f: iv for f, iv in iv_table.items() if f not in selected_features}

    IV_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    IV_REPORT_PATH.write_text(json.dumps({
        "iv_by_feature": iv_table,
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "iv_min": IV_MIN,
        "iv_max": IV_MAX,
    }, indent=2))

    WOE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plots = sc.woebin_plot(bins)
    for feature, fig in plots.items():
        fig.savefig(WOE_PLOTS_DIR / f"{feature}_woebin.png", dpi=150, bbox_inches="tight")

    if not selected_features:
        raise ValueError(
            f"No feature met the IV selection range [{IV_MIN}, {IV_MAX}] — "
            f"observed IVs: {iv_table}"
        )

    selected_bins = {f: bins[f] for f in selected_features}

    WOE_BINS_DIR.mkdir(parents=True, exist_ok=True)
    with open(WOE_BINS_DIR / "woe_bins.pkl", "wb") as f:
        pickle.dump(selected_bins, f)

    encoded = sc.woebin_ply(
        silver_clean[["ID"] + selected_features + ["TARGET"]], selected_bins
    )
    woe_columns = [c for c in encoded.columns if c.endswith("_woe")]

    output_result = validate_dataframe(
        encoded, "silver_woe_silver_woe", silver_woe_expectations(woe_columns)
    )
    if not output_result["success"]:
        raise ValueError("Silver WoE output-contract checks failed for silver_woe")

    out_path = f"{SILVER_PATH}silver_woe"
    write_deltalake(out_path, encoded, mode="overwrite", schema_mode="overwrite")

    return {
        "features_evaluated": len(iv_table),
        "features_selected": selected_features,
        "features_dropped": dropped_features,
        "rows_silver_woe": len(encoded),
        "output_path": out_path,
    }


def main():
    stats = build_silver_woe()
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
