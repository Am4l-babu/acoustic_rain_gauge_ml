"""
Model Training & Evaluation (Stage 4)
=======================================
Trains and evaluates models on the Stage 3 train/test splits
(data/processed/train.csv, test.csv, built by feature_extraction.py):

  1. Rain / no-rain classification
       - baseline : Logistic Regression (class_weight="balanced")
       - primary  : XGBoost classifier (scale_pos_weight from the
                     training split's actual imbalance)
  2. Rainfall amount regression (mm)
       - XGBoost regressor, trained directly on rainfall_mm
         (0 for dry clips)

Metrics are saved to docs/model_evaluation_report.json; a confusion
matrix and feature-importance chart for the XGBoost classifier are
saved as PNGs in docs/ (a first pass at what Stage 5 formalizes
further, e.g. per-month generalization and error analysis).

Run:
    python src/train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR   = REPO_ROOT / "docs"

FEATURE_COLS = [
    "rms", "peak", "par",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "zero_crossing_rate", "energy_variance",
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
]

XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
)


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/feature_extraction.py first.")
    return pd.read_csv(path)


def evaluate_classifier(name: str, y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    print(f"\n  {name}")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")
    return metrics


def evaluate_regressor(name: str, y_true, y_pred) -> dict:
    metrics = {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print(f"\n  {name}")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")
    return metrics


def main():
    print("=" * 68)
    print("  STAGE 4 - MODEL TRAINING & EVALUATION")
    print("=" * 68)

    print("\n[1] Loading train/test splits...")
    train_df = load_split("train")
    test_df = load_split("test")
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
    X_train = scaler.transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train_cls = train_df["is_rainy"].to_numpy()
    y_test_cls = test_df["is_rainy"].to_numpy()
    y_train_reg = train_df["rainfall_mm"].to_numpy()
    y_test_reg = test_df["rainfall_mm"].to_numpy()

    all_metrics = {}

    # ---------------- Classification: baseline ----------------
    print("\n[2] Training baseline Logistic Regression classifier...")
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
    logreg.fit(X_train, y_train_cls)
    logreg_pred = logreg.predict(X_test)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]
    all_metrics["logistic_regression"] = evaluate_classifier(
        "Logistic Regression (baseline)", y_test_cls, logreg_pred, logreg_proba)

    # ---------------- Classification: XGBoost ----------------
    print("\n[3] Training XGBoost classifier...")
    scale_pos_weight = (y_train_cls == 0).sum() / (y_train_cls == 1).sum()
    xgb_clf = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc",
                             **XGB_PARAMS)
    xgb_clf.fit(X_train, y_train_cls)
    xgb_pred = xgb_clf.predict(X_test)
    xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
    all_metrics["xgboost_classifier"] = evaluate_classifier(
        "XGBoost (primary)", y_test_cls, xgb_pred, xgb_proba)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[4] Saving confusion matrix (XGBoost)...")
    cm = confusion_matrix(y_test_cls, xgb_pred)
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Dry", "Rainy"], yticklabels=["Dry", "Rainy"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("XGBoost Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()
    print(f"  Saved: {DOCS_DIR / 'confusion_matrix.png'}")

    print("\n[5] Saving feature importance chart (XGBoost)...")
    importances = pd.Series(xgb_clf.feature_importances_, index=FEATURE_COLS).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="#4C72B0")
    plt.title("XGBoost Feature Importance (rain/no-rain)")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "feature_importance.png", dpi=120)
    plt.close()
    print(f"  Saved: {DOCS_DIR / 'feature_importance.png'}")

    # ---------------- Regression: XGBoost ----------------
    print("\n[6] Training XGBoost regressor (rainfall_mm)...")
    xgb_reg = XGBRegressor(**XGB_PARAMS)
    xgb_reg.fit(X_train, y_train_reg)
    reg_pred = np.clip(xgb_reg.predict(X_test), 0, None)  # rainfall can't be negative
    all_metrics["xgboost_regressor"] = evaluate_regressor(
        "XGBoost Regressor (rainfall_mm)", y_test_reg, reg_pred)

    print("\n[7] Saving models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, MODELS_DIR / "logreg_baseline.pkl")
    xgb_clf.save_model(MODELS_DIR / "xgb_classifier.json")
    xgb_reg.save_model(MODELS_DIR / "xgb_regressor.json")
    print(f"  Saved: {MODELS_DIR / 'logreg_baseline.pkl'}")
    print(f"  Saved: {MODELS_DIR / 'xgb_classifier.json'}")
    print(f"  Saved: {MODELS_DIR / 'xgb_regressor.json'}")

    print("\n[8] Saving metrics report...")
    with open(DOCS_DIR / "model_evaluation_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved: {DOCS_DIR / 'model_evaluation_report.json'}")

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68)


if __name__ == "__main__":
    main()
