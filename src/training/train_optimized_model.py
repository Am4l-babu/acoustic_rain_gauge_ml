"""
Stage 8 (cont.) — Train XGBoost classifier + regressor on the SHAP-selected
top-N features from the master feature store, evaluated on the same
per-campaign chronological train/test split used by the Stage 4 baseline
(train_model.py), so results are directly comparable.

master_feature_extraction.py extracted features from every raw clip under
arg_cleaned_dataset, without Stage 3's duration_category / rainfall_mm_cap
filters and without split labels. This script re-derives the same filtered
population and split by joining the master store's features onto
data/processed/train.csv|test.csv (Stage 3's already-filtered, already-split
output) via (month_folder, audio_filename), rather than training on
optimized_rain_dataset.parquet as-is (which has the unfiltered population,
no split, and no is_rainy column).

Run:
    python src/training/train_optimized_model.py --master-store-dir F:\\master_feature_store \\
        --shap-csv-classifier F:\\feature_importance_shap_is_rainy.csv \\
        --shap-csv-regressor F:\\feature_importance_shap.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs" / "reports"

XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
)


def load_master_store(store_dir, feature_cols):
    files = sorted(Path(store_dir).glob("master_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No chunks found in {store_dir}")
    keep = feature_cols + ["source_folder", "audio_filename"]
    print(f"  Loading {len(files)} chunk(s) from {store_dir}...")
    df = pd.concat((pd.read_parquet(f, columns=keep) for f in files), ignore_index=True)
    print(f"  Loaded {len(df):,} rows from master store")
    before = len(df)
    df = df.drop_duplicates(subset=["source_folder", "audio_filename"])
    dupes = before - len(df)
    if dupes:
        print(f"  Dropped {dupes:,} duplicate (source_folder, audio_filename) rows "
              f"(known upstream CSV duplication, e.g. December_2024_rain_data)")
    return df


def attach_features(split_df, master_df, name):
    merged = split_df.merge(
        master_df,
        left_on=["month_folder", "audio_filename"],
        right_on=["source_folder", "audio_filename"],
        how="inner",
    )
    dropped = len(split_df) - len(merged)
    print(f"  {name}: {len(split_df):,} rows -> {len(merged):,} matched "
          f"({dropped:,} unmatched, {dropped / len(split_df):.2%})")
    return merged


def evaluate_classifier(name, y_true, y_pred, y_proba):
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


def evaluate_regressor(name, y_true, y_pred):
    metrics = {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print(f"\n  {name}")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")
    return metrics


def top_features_from_csv(path, top_n):
    importance_df = pd.read_csv(path)
    return (importance_df.sort_values("shap_importance", ascending=False)
            .head(top_n)["feature"].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-store-dir", type=str, required=True)
    parser.add_argument("--shap-csv-classifier", type=str, required=True,
                         help="SHAP ranking computed against is_rainy")
    parser.add_argument("--shap-csv-regressor", type=str, required=True,
                         help="SHAP ranking computed against rainfall_mm")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    print("=" * 68)
    print("  STAGE 8 - TRAIN ON SHAP-SELECTED FEATURES (per-target feature sets)")
    print("  (evaluated on the Stage 4 baseline's exact filtered population + split)")
    print("=" * 68)

    print("\n[1] Loading SHAP feature rankings...")
    clf_features = top_features_from_csv(args.shap_csv_classifier, args.top_n)
    reg_features = top_features_from_csv(args.shap_csv_regressor, args.top_n)
    all_feature_cols = sorted(set(clf_features) | set(reg_features))
    print(f"  Classifier features: {len(clf_features)} | Regressor features: {len(reg_features)} "
          f"| Union to load: {len(all_feature_cols)}")

    print("\n[2] Loading Stage 3 train/test splits (filtered + chronologically split population)...")
    usecols = ["audio_filename", "month_folder", "rainfall_mm", "is_rainy"]
    train_split = pd.read_csv(DATA_DIR / "train.csv", usecols=usecols)
    test_split = pd.read_csv(DATA_DIR / "test.csv", usecols=usecols)
    print(f"  Train: {len(train_split):,} rows | Test: {len(test_split):,} rows")

    print("\n[3] Loading master feature store...")
    master_df = load_master_store(args.master_store_dir, all_feature_cols)

    print("\n[4] Joining master-store features onto the Stage 3 split (month_folder + audio_filename)...")
    train_df = attach_features(train_split, master_df, "Train")
    test_df = attach_features(test_split, master_df, "Test")

    y_train_cls, y_test_cls = train_df["is_rainy"].to_numpy(), test_df["is_rainy"].to_numpy()
    y_train_reg, y_test_reg = train_df["rainfall_mm"].to_numpy(), test_df["rainfall_mm"].to_numpy()

    all_metrics = {"classifier_features": clf_features, "regressor_features": reg_features}

    print("\n[5] Training XGBoost classifier (is_rainy) on is_rainy-ranked features...")
    X_train_cls, X_test_cls = train_df[clf_features].fillna(0), test_df[clf_features].fillna(0)
    scale_pos_weight = (y_train_cls == 0).sum() / (y_train_cls == 1).sum()
    xgb_clf = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", **XGB_PARAMS)
    xgb_clf.fit(X_train_cls, y_train_cls)
    xgb_pred = xgb_clf.predict(X_test_cls)
    xgb_proba = xgb_clf.predict_proba(X_test_cls)[:, 1]
    all_metrics["xgboost_classifier"] = evaluate_classifier(
        "XGBoost classifier (is_rainy-ranked SHAP features)", y_test_cls, xgb_pred, xgb_proba)

    print("\n[6] Training XGBoost regressor (rainfall_mm) on rainfall_mm-ranked features...")
    X_train_reg, X_test_reg = train_df[reg_features].fillna(0), test_df[reg_features].fillna(0)
    xgb_reg = XGBRegressor(**XGB_PARAMS)
    xgb_reg.fit(X_train_reg, y_train_reg)
    reg_pred = np.clip(xgb_reg.predict(X_test_reg), 0, None)
    all_metrics["xgboost_regressor"] = evaluate_regressor(
        "XGBoost regressor (rainfall_mm-ranked SHAP features)", y_test_reg, reg_pred)

    print("\n[7] Hurdle model: regressor trained ONLY on rainy training rows...")
    train_rainy_mask = y_train_cls == 1
    test_rainy_mask = y_test_cls == 1
    print(f"  Rainy training rows: {train_rainy_mask.sum():,} / {len(train_df):,} "
          f"({train_rainy_mask.mean():.1%})")
    X_train_rainy = train_df.loc[train_rainy_mask, reg_features].fillna(0)
    y_train_rainy = y_train_reg[train_rainy_mask]
    hurdle_reg = XGBRegressor(**XGB_PARAMS)
    hurdle_reg.fit(X_train_rainy, y_train_rainy)

    # Diagnostic: regressor's own accuracy, evaluated only on truly-rainy test rows
    # (isolates "how good is the amount model" from "how good is the rain/no-rain gate")
    X_test_rainy = test_df.loc[test_rainy_mask, reg_features].fillna(0)
    y_test_rainy = y_test_reg[test_rainy_mask]
    rainy_only_pred = np.clip(hurdle_reg.predict(X_test_rainy), 0, None)
    all_metrics["hurdle_regressor_on_rainy_only"] = evaluate_regressor(
        "Hurdle regressor, evaluated on true-rainy test rows only (diagnostic)",
        y_test_rainy, rainy_only_pred)

    # Full pipeline: classifier gates whether to predict 0 or run the amount regressor,
    # evaluated on the whole test set -- this is what's directly comparable to
    # xgboost_regressor above (same population, same metric).
    hurdle_full_pred = np.zeros(len(test_df))
    predicted_rainy_mask = xgb_pred == 1
    if predicted_rainy_mask.any():
        X_test_predicted_rainy = test_df.loc[predicted_rainy_mask, reg_features].fillna(0)
        hurdle_full_pred[predicted_rainy_mask] = np.clip(hurdle_reg.predict(X_test_predicted_rainy), 0, None)
    all_metrics["hurdle_full_pipeline"] = evaluate_regressor(
        "Hurdle full pipeline (classifier gate + rain-only regressor), whole test set",
        y_test_reg, hurdle_full_pred)

    print("\n[8] Soft-gated hurdle: classifier probability * rain-only regressor (all test rows)...")
    X_test_all_rainy_model = test_df[reg_features].fillna(0)
    rain_only_pred_all = np.clip(hurdle_reg.predict(X_test_all_rainy_model), 0, None)
    soft_gated_pred = xgb_proba * rain_only_pred_all
    all_metrics["hurdle_soft_gated"] = evaluate_regressor(
        "Soft-gated hurdle (P(rainy) * rain-only regressor), whole test set",
        y_test_reg, soft_gated_pred)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "model_evaluation_report_optimized.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 68)
    print("  COMPARISON TO STAGE 4 BASELINE (13 hand-picked scalar features)")
    print(f"  Baseline           : AUC 0.883 | R2 0.155")
    print(f"  Optimized (single) : AUC {all_metrics['xgboost_classifier']['roc_auc']:.3f} "
          f"| R2 {all_metrics['xgboost_regressor']['r2']:.3f}")
    print(f"  Hurdle (hard gate) : R2 {all_metrics['hurdle_full_pipeline']['r2']:.3f}")
    print(f"  Hurdle (soft gate) : R2 {all_metrics['hurdle_soft_gated']['r2']:.3f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
