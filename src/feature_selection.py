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

# OS/mount agnostic: derived from this script's own location (two levels up
# is the HDD's root, whether that's a Windows drive letter or a Linux mount
# point) rather than a hardcoded drive, matching master_feature_extraction.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = _PROJECT_ROOT / "master_feature_store"
OUTPUT_PATH = _PROJECT_ROOT / "optimized_rain_dataset.parquet"
META_COLS = ["timestamp", "rainfall_mm", "is_rainy", "source_folder", "audio_filename"]


def load_store(store_dir=None):
    store_dir = Path(store_dir) if store_dir else STORE_DIR
    files = sorted(store_dir.glob("master_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No chunks found in {store_dir} — run master_feature_extraction.py first")
    print(f"Loading {len(files)} chunk(s) from {store_dir}...")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def select_features(df, top_n=30, shap_sample_size=5000, seed=42, target="rainfall_mm"):
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].fillna(0)
    y = df[target]

    print(f"Training XGBoost on {len(feature_cols)} features, {len(X)} rows (target={target})...")
    if target == "is_rainy":
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=seed,
                                   scale_pos_weight=scale_pos_weight, eval_metric="auc")
    else:
        model = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=seed)
    model.fit(X, y)

    sample_n = min(shap_sample_size, len(X))
    print(f"Computing SHAP values on a {sample_n}-row sample...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.sample(sample_n, random_state=seed))
    if isinstance(shap_values, list):  # binary XGBClassifier -> list of 2 class arrays
        shap_values = shap_values[1]

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
    parser.add_argument("--store-dir", type=str, default=None, help=f"Override master store directory (default: {STORE_DIR})")
    parser.add_argument("--output-path", type=str, default=None, help=f"Override output parquet path (default: {OUTPUT_PATH})")
    parser.add_argument("--target", type=str, default="rainfall_mm", choices=["rainfall_mm", "is_rainy"],
                         help="Target to rank features against (default: rainfall_mm)")
    args = parser.parse_args()

    target_suffix = "" if args.target == "rainfall_mm" else f"_{args.target}"
    default_output = OUTPUT_PATH.parent / f"{OUTPUT_PATH.stem}{target_suffix}{OUTPUT_PATH.suffix}"
    output_path = Path(args.output_path) if args.output_path else default_output

    df = load_store(store_dir=args.store_dir)
    top_features, importance_df = select_features(df, top_n=args.top_n, shap_sample_size=args.shap_sample_size,
                                                    target=args.target)

    print(f"\nTop {args.top_n} features by SHAP importance:")
    for f in top_features:
        score = importance_df.loc[importance_df["feature"] == f, "shap_importance"].values[0]
        print(f"  {f:35s} {score:.4f}")

    optimized_df = df[top_features + [args.target]]
    optimized_df.to_parquet(output_path)
    print(f"\nSaved optimized dataset ({len(top_features)} features + target) -> {output_path}")

    importance_csv = output_path.parent / f"feature_importance_shap{target_suffix}.csv"
    importance_df.to_csv(importance_csv, index=False)
    print(f"Saved full SHAP importance ranking -> {importance_csv}")


if __name__ == "__main__":
    main()
