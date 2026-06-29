"""STEP 5 — MODEL EVALUATION: KS, Gini, AUC, confusion matrix, threshold selection.

Reloads the exact train/test split from Week 6 (same random_state/test_size
applied to the same gold_features table -> deterministic, reproducible
split) and the trained model bundle, scores the held-out test set, and
reports the standard scorecard evaluation metrics plus quantile-based
Approve/Refer/Decline decision bands.
"""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from deltalake import DeltaTable
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from config.config import GOLD_PATH, MLFLOW_URI
from ml.train import MODEL_PATH, RANDOM_STATE, TEST_SIZE

PLOTS_DIR = Path("docs/eval_plots")
EVAL_REPORT_PATH = Path("docs/evaluation_report.json")

# Quantile-based decision bands on predicted PD (probability of default).
# Below the 50th percentile -> Approve, above the 90th -> Decline, between
# the two -> Refer (manual review). Chosen because the ~1.69% base default
# rate makes a single hard cutoff uninformative for business framing.
APPROVE_QUANTILE = 0.50
DECLINE_QUANTILE = 0.90


def _load_test_set():
    gold = DeltaTable(f"{GOLD_PATH}gold_features").to_pandas()
    feature_columns = [c for c in gold.columns if c != "TARGET"]
    X = gold[feature_columns]
    y = gold["TARGET"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test


def _ks_statistic(y_true, y_proba):
    good_scores = y_proba[y_true == 0]
    bad_scores = y_proba[y_true == 1]
    thresholds = np.unique(y_proba)
    cum_good = np.array([(good_scores <= t).mean() for t in thresholds])
    cum_bad = np.array([(bad_scores <= t).mean() for t in thresholds])
    diffs = np.abs(cum_bad - cum_good)
    ks_idx = int(np.argmax(diffs))
    return float(diffs[ks_idx]), float(thresholds[ks_idx]), thresholds, cum_good, cum_bad


def evaluate_model():
    X_test, y_test = _load_test_set()

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    y_proba = model.predict_proba(X_test[bundle["feature_columns"]])[:, 1]
    y_true = y_test.values

    auc = roc_auc_score(y_true, y_proba)
    gini = 2 * auc - 1
    ks, ks_threshold, thresholds, cum_good, cum_bad = _ks_statistic(y_true, y_proba)

    y_pred_ks = (y_proba >= ks_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_ks)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", color="steelblue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, cum_good, label="Cumulative % Good", color="seagreen")
    plt.plot(thresholds, cum_bad, label="Cumulative % Bad", color="firebrick")
    plt.axvline(ks_threshold, linestyle="--", color="grey",
                label=f"KS={ks:.3f} @ PD={ks_threshold:.3f}")
    plt.xlabel("Predicted PD threshold")
    plt.ylabel("Cumulative proportion")
    plt.title("KS Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ks_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.hist(y_proba[y_true == 0], bins=40, alpha=0.6, label="Good (TARGET=0)",
              color="seagreen", density=True)
    plt.hist(y_proba[y_true == 1], bins=40, alpha=0.6, label="Bad (TARGET=1)",
              color="firebrick", density=True)
    plt.xlabel("Predicted PD")
    plt.ylabel("Density")
    plt.title("Score Distribution by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "score_distribution.png", dpi=150)
    plt.close()

    # --- quantile-based decision bands ---
    approve_cutoff = float(np.quantile(y_proba, APPROVE_QUANTILE))
    decline_cutoff = float(np.quantile(y_proba, DECLINE_QUANTILE))

    bands = pd.cut(
        y_proba,
        bins=[-np.inf, approve_cutoff, decline_cutoff, np.inf],
        labels=["Approve", "Refer", "Decline"],
    )
    band_stats = (
        pd.DataFrame({"band": bands, "TARGET": y_true})
        .groupby("band", observed=True)["TARGET"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "default_rate"})
    )

    report = {
        "auc": round(float(auc), 4),
        "gini": round(float(gini), 4),
        "ks": round(float(ks), 4),
        "ks_threshold_pd": round(float(ks_threshold), 4),
        "confusion_matrix_at_ks_threshold": {
            "labels": ["good(0)", "bad(1)"],
            "matrix": cm.tolist(),
        },
        "overall_test_default_rate": round(float(y_true.mean()), 4),
        "decision_bands": {
            "approve_cutoff_pd": round(approve_cutoff, 4),
            "decline_cutoff_pd": round(decline_cutoff, 4),
            "band_stats": {
                str(band): {
                    "count": int(row["count"]),
                    "default_rate": round(float(row["default_rate"]), 4),
                }
                for band, row in band_stats.iterrows()
            },
        },
    }
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("credit_risk_scorecard")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({"auc": auc, "gini": gini, "ks": ks})
        mlflow.log_artifact(str(PLOTS_DIR / "roc_curve.png"))
        mlflow.log_artifact(str(PLOTS_DIR / "ks_curve.png"))
        mlflow.log_artifact(str(PLOTS_DIR / "score_distribution.png"))
        mlflow.log_artifact(str(EVAL_REPORT_PATH))

    return report


def main():
    report = evaluate_model()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
