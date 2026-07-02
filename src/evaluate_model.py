"""
Deeper Evaluation & Tuning (Stage 5)
======================================
Goes past the single aggregate AUC/F1 numbers from Stage 4 to answer
three concrete questions about the XGBoost rain/no-rain classifier:

  1. Does it generalize evenly across recording campaigns, or do a
     few months carry all the error?
  2. Precision was weak (0.46) at the default 0.5 threshold. Is that
     a modeling problem, or just the wrong operating point? -- this
     is checked BEFORE reaching for hyperparameter tuning, since a
     threshold adjustment is free (no retraining) and is the standard
     first lever for a class-weighted classifier.
  3. Does light hyperparameter tuning (searched on a held-out
     validation slice of *training* data only, never on test) beat
     the Stage 4 baseline on the real test set?

Run:
    python src/evaluate_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, mean_squared_error,
)
from xgboost import XGBClassifier

from feature_extraction import chronological_split, FEATURE_COLS

# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR   = REPO_ROOT / "docs"

VAL_FRACTION = 0.15  # carved out of TRAIN only, per campaign; test.csv is never touched

# A small, deliberately modest search -- not an exhaustive grid.
PARAM_CANDIDATES = [
    dict(n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=1),  # Stage 4 baseline
    dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=1),
    dict(n_estimators=300, max_depth=8, learning_rate=0.05, min_child_weight=1),
    dict(n_estimators=300, max_depth=6, learning_rate=0.03, min_child_weight=1),
    dict(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=5),
    dict(n_estimators=500, max_depth=8, learning_rate=0.03, min_child_weight=5),
]


def load_models_and_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
    clf = XGBClassifier()
    clf.load_model(MODELS_DIR / "xgb_classifier.json")
    return train_df, test_df, scaler, clf


# ============================================================
# 1. Per-campaign generalization check
# ============================================================
def per_campaign_breakdown(test_df, X_test, y_test, y_pred, y_proba):
    rows = []
    for month, idx in test_df.groupby("month_folder").groups.items():
        pos = test_df.index.get_indexer(idx)
        yt, yp, ypr = y_test[pos], y_pred[pos], y_proba[pos]
        row = {
            "month_folder": month,
            "n": len(pos),
            "rainy_frac": float(yt.mean()),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "roc_auc": roc_auc_score(yt, ypr) if len(set(yt)) > 1 else None,
        }
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("f1")
    return result


# ============================================================
# 2. Threshold analysis (precision/recall tradeoff)
# ============================================================
def threshold_analysis(y_test, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)

    best_f1_idx = np.argmax(f1s[:-1])  # last point has no matching threshold
    best_f1_threshold = float(thresholds[best_f1_idx])

    high_precision_idx = np.argmax(precisions[:-1] >= 0.6)
    high_precision_threshold = (
        float(thresholds[high_precision_idx]) if precisions[high_precision_idx] >= 0.6 else None
    )

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, color="#4C72B0")
    plt.scatter([recalls[best_f1_idx]], [precisions[best_f1_idx]], color="#DD8452",
                zorder=5, label=f"best-F1 threshold={best_f1_threshold:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Tradeoff (XGBoost Classifier)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "precision_recall_curve.png", dpi=120)
    plt.close()

    return {
        "default_threshold": 0.5,
        "best_f1_threshold": best_f1_threshold,
        "best_f1_precision": float(precisions[best_f1_idx]),
        "best_f1_recall": float(recalls[best_f1_idx]),
        "high_precision_threshold": high_precision_threshold,
    }


# ============================================================
# 3. Error analysis: what's different about the mistakes?
# ============================================================
def error_analysis(test_df, y_test, y_pred):
    tp = (y_test == 1) & (y_pred == 1)
    fp = (y_test == 0) & (y_pred == 1)
    tn = (y_test == 0) & (y_pred == 0)
    fn = (y_test == 1) & (y_pred == 0)

    summary = test_df[FEATURE_COLS].groupby(
        np.select([tp, fp, tn, fn], ["TP", "FP", "TN", "FN"], default="?")
    ).mean().round(4)

    print("\n  Mean feature values by outcome:")
    print(summary.to_string())

    fp_months = test_df.loc[fp, "month_folder"].value_counts().head(5)
    fn_months = test_df.loc[fn, "month_folder"].value_counts().head(5)
    print(f"\n  Top campaigns for false positives (predicted rainy, actually dry):\n{fp_months.to_string()}")
    print(f"\n  Top campaigns for false negatives (predicted dry, actually rainy):\n{fn_months.to_string()}")

    return {
        "feature_means_by_outcome": summary,
        "top_false_positive_campaigns": {k: int(v) for k, v in fp_months.items()},
        "top_false_negative_campaigns": {k: int(v) for k, v in fn_months.items()},
    }


# ============================================================
# 4. Light hyperparameter search (validation carved from TRAIN only)
# ============================================================
def tune_classifier(train_df):
    tune_train, val = chronological_split(train_df, VAL_FRACTION)
    scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
    X_tune = scaler.transform(tune_train[FEATURE_COLS])
    X_val = scaler.transform(val[FEATURE_COLS])
    y_tune = tune_train["is_rainy"].to_numpy()
    y_val = val["is_rainy"].to_numpy()
    spw = (y_tune == 0).sum() / (y_tune == 1).sum()

    results = []
    for params in PARAM_CANDIDATES:
        model = XGBClassifier(scale_pos_weight=spw, eval_metric="auc",
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1, **params)
        model.fit(X_tune, y_tune)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_proba))
        results.append({**params, "val_auc": val_auc})
        print(f"    {params} -> val AUC={val_auc:.4f}")

    results_sorted = sorted(results, key=lambda r: r["val_auc"], reverse=True)
    best_params = {k: v for k, v in results_sorted[0].items() if k != "val_auc"}
    return best_params, results_sorted


def main():
    print("=" * 68)
    print("  STAGE 5 - DEEPER EVALUATION & TUNING")
    print("=" * 68)

    print("\n[1] Loading test data and Stage 4 model...")
    train_df, test_df, scaler, clf = load_models_and_data()
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_test = test_df["is_rainy"].to_numpy()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print(f"  Test rows: {len(test_df):,}")

    print("\n[2] Per-campaign generalization check...")
    per_campaign = per_campaign_breakdown(test_df, X_test, y_test, y_pred, y_proba)
    print(per_campaign.to_string(index=False))
    per_campaign.to_csv(DOCS_DIR / "per_campaign_evaluation.csv", index=False)
    print(f"  Saved: {DOCS_DIR / 'per_campaign_evaluation.csv'}")

    print("\n[3] Threshold analysis (precision/recall tradeoff)...")
    threshold_results = threshold_analysis(y_test, y_proba)
    for k, v in threshold_results.items():
        print(f"    {k:24s}: {v}")
    print(f"  Saved: {DOCS_DIR / 'precision_recall_curve.png'}")

    print("\n[4] Error analysis (default threshold=0.5)...")
    error_summary = error_analysis(test_df, y_test, y_pred)
    error_summary["feature_means_by_outcome"].to_csv(DOCS_DIR / "error_analysis_feature_means.csv")
    print(f"  Saved: {DOCS_DIR / 'error_analysis_feature_means.csv'}")

    print("\n[5] Light hyperparameter search (tuned on a TRAIN-only validation slice)...")
    best_params, search_results = tune_classifier(train_df)
    print(f"\n  Best params by validation AUC: {best_params}")

    print("\n[6] Refitting best params on full training set, evaluating on test...")
    X_train_full = scaler.transform(train_df[FEATURE_COLS])
    y_train_full = train_df["is_rainy"].to_numpy()
    spw_full = (y_train_full == 0).sum() / (y_train_full == 1).sum()
    tuned_clf = XGBClassifier(scale_pos_weight=spw_full, eval_metric="auc",
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1, **best_params)
    tuned_clf.fit(X_train_full, y_train_full)
    tuned_pred = tuned_clf.predict(X_test)
    tuned_proba = tuned_clf.predict_proba(X_test)[:, 1]

    baseline_auc = float(roc_auc_score(y_test, y_proba))
    tuned_auc = float(roc_auc_score(y_test, tuned_proba))
    print(f"  Baseline (Stage 4) test AUC: {baseline_auc:.4f}")
    print(f"  Tuned    (Stage 5) test AUC: {tuned_auc:.4f}")

    improved = bool(tuned_auc > baseline_auc)
    if improved:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tuned_clf.save_model(MODELS_DIR / "xgb_classifier_tuned.json")
        print(f"  Tuning improved AUC -> saved {MODELS_DIR / 'xgb_classifier_tuned.json'}")
    else:
        print("  Tuning did not beat the Stage 4 baseline -- keeping xgb_classifier.json as-is.")

    print("\n[7] Saving Stage 5 report...")
    report = {
        "baseline_test_auc": baseline_auc,
        "tuned_test_auc": tuned_auc,
        "tuning_improved_model": improved,
        "best_params_by_val_auc": best_params,
        "threshold_analysis": threshold_results,
        "top_false_positive_campaigns": error_summary["top_false_positive_campaigns"],
        "top_false_negative_campaigns": error_summary["top_false_negative_campaigns"],
        "hyperparameter_search_results": search_results,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCS_DIR / "stage5_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {DOCS_DIR / 'stage5_evaluation_report.json'}")

    print("\n" + "=" * 68)
    print("  DONE")
    print("=" * 68)


if __name__ == "__main__":
    main()
