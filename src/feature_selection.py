"""
Stage 8 — Feature selection over the 176-feature master store using SHAP.

Loads every chunk written by master_feature_extraction.py, trains a quick
XGBoost regressor, ranks features by mean |SHAP value|, and writes the top-N
down to a slim Parquet file for training.

Usage:
    python src/feature_selection.py                  # top 30 (default)
    python src/feature_selection.py --top-n 20
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

STORE_DIR = Path(r"D:\master_feature_store")
OUTPUT_PATH = Path(r"D:\optimized_rain_dataset.parquet")
META_COLS = ["timestamp", "rainfall_mm", "is_rainy", "source_folder", "audio_filename"]


def load_store():
    files = sorted(STORE_DIR.glob("master_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No chunks found in {STORE_DIR} — run master_feature_extraction.py first")
    print(f"Loading {len(files)} chunk(s) from {STORE_DIR}...")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def select_features(df, top_n=30, shap_sample_size=5000, seed=42):
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].fillna(0)
    y = df["rainfall_mm"]

    print(f"Training XGBoost on {len(feature_cols)} features, {len(X)} rows...")
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=seed)
    model.fit(X, y)

    sample_n = min(shap_sample_size, len(X))
    print(f"Computing SHAP values on a {sample_n}-row sample...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.sample(sample_n, random_state=seed))

    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": shap_importance,
    }).sort_values("shap_importance", ascending=False)

    top_features = importance_df.head(top_n)["feature"].tolist()
    return top_features, importance_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--shap-sample-size", type=int, default=5000)
    args = parser.parse_args()

    df = load_store()
    top_features, importance_df = select_features(df, top_n=args.top_n, shap_sample_size=args.shap_sample_size)

    print(f"\nTop {args.top_n} features by SHAP importance:")
    for f in top_features:
        score = importance_df.loc[importance_df["feature"] == f, "shap_importance"].values[0]
        print(f"  {f:35s} {score:.4f}")

    optimized_df = df[top_features + ["rainfall_mm"]]
    optimized_df.to_parquet(OUTPUT_PATH)
    print(f"\nSaved optimized dataset ({len(top_features)} features + target) -> {OUTPUT_PATH}")

    importance_csv = OUTPUT_PATH.parent / "feature_importance_shap.csv"
    importance_df.to_csv(importance_csv, index=False)
    print(f"Saved full SHAP importance ranking -> {importance_csv}")


if __name__ == "__main__":
    main()
