"""STEP 4 — MODEL TRAINING: Logistic Regression + credit scorecard.

Loads gold_features, trains a class-balanced Logistic Regression, converts
it (together with the woe_bins artifact from Week 4) into a point-based
credit scorecard, and tracks everything in MLflow. Full performance
evaluation (KS/Gini/AUC, threshold selection) is Week 7 — this step only
logs basic train/test accuracy as a sanity check.
"""
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import scorecardpy as sc
from deltalake import DeltaTable
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config.config import GOLD_PATH, MLFLOW_URI, SILVER_PATH

WOE_BINS_PATH = Path(SILVER_PATH) / "woe_bins" / "woe_bins.pkl"
MODEL_PATH = Path("ml/model.pkl")
SCORECARD_PATH = Path("ml/scorecard.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Scorecard calibration per the architecture doc: 600 points at 50:1
# good:bad odds, doubling every 20 points (BAM-friendly, interpretable scale).
SCORECARD_POINTS0 = 600
SCORECARD_ODDS0 = 1 / 50
SCORECARD_PDO = 20


def train_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("credit_risk_scorecard")

    gold_table = DeltaTable(f"{GOLD_PATH}gold_features")
    gold = gold_table.to_pandas()
    gold_version = gold_table.version()

    feature_columns = [c for c in gold.columns if c != "TARGET"]
    X = gold[feature_columns]
    y = gold["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    with open(WOE_BINS_PATH, "rb") as f:
        woe_bins = pickle.load(f)

    with mlflow.start_run(run_name="logistic_regression_scorecard") as run:
        model = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)

        mlflow.log_params({
            "solver": model.solver,
            "C": model.C,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_features": len(feature_columns),
            "gold_features_version": gold_version,
        })
        mlflow.log_metrics({
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        })
        mlflow.set_tag("woe_bins_path", str(WOE_BINS_PATH))
        mlflow.sklearn.log_model(model, name="model")
        mlflow.log_artifact(str(WOE_BINS_PATH), artifact_path="woe_bins")

        # --- credit scorecard: LR coefficients + woe_bins -> point scale ---
        card = sc.scorecard(
            woe_bins, model, X.columns,
            points0=SCORECARD_POINTS0, odds0=SCORECARD_ODDS0, pdo=SCORECARD_PDO,
        )
        scorecard_df = pd.concat(card.values(), ignore_index=True)
        SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        scorecard_df.to_csv(SCORECARD_PATH, index=False)
        mlflow.log_artifact(str(SCORECARD_PATH))

        # --- model bundle for Week 9 live scoring: model + woe_bins together,
        # so transforming a new applicant never drifts from how training data
        # was encoded ---
        bundle = {
            "model": model,
            "woe_bins": woe_bins,
            "feature_columns": feature_columns,
            "scorecard_params": {
                "points0": SCORECARD_POINTS0,
                "odds0": SCORECARD_ODDS0,
                "pdo": SCORECARD_PDO,
            },
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        mlflow.log_artifact(str(MODEL_PATH))

        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": str(MODEL_PATH),
        "scorecard_path": str(SCORECARD_PATH),
    }


def main():
    stats = train_model()
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
