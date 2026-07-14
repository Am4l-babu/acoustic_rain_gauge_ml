"""
Ensemble: full-scale CNN (raw-audio MFCC) x XGBoost-optimized (SHAP-selected
scalar features) regressors
============================================================================
Both individual models plateau around R^2 0.22-0.28 on the full test set
(CNN 0.277, XGBoost-optimized 0.226 -- see docs/dl_ablation_full_report.json
and docs/model_evaluation_report_optimized.json). They see different
representations of the same audio (CNN: full 2D MFCC time series; XGBoost:
hand-engineered spectral/wavelet/loudness scalar means), so their errors are
more likely to be uncorrelated than ensembling CNN/LSTM/Transformer against
each other (same MFCC input -- correlated errors, limited diversity gain).

This script:
  1. Recomputes the exact train-set MFCC mean/std used at training time
     (needed to normalize the test MFCC identically -- these stats were never
     saved to disk during training; shared across all three architectures
     since they were trained on the same normalized tensors), in chunks so
     the 15GB train cache is never fully materialized in RAM.
  2. Runs the CNN, LSTM, and Transformer checkpoints over the cached test
     MFCC to get per-clip predictions from all three (streamed in batches,
     same reason) -- not just CNN, so ensemble_stack.py can weigh all three
     "experts" instead of only the best individual one.
  3. Re-trains the XGBoost regressor on SHAP-selected features exactly as
     train_optimized_model.py does (that script never persisted the model,
     only its metrics), to get matching test predictions.
  4. Joins both prediction sets on (month_folder, audio_filename) -- the
     XGBoost path's master-store inner join drops a few unmatched rows, so
     the two prediction arrays aren't guaranteed to be the same length.
  5. Also retrains the XGBoost classifier (SHAP-selected is_rainy features)
     to get P(rainy) per clip -- a candidate third input for a stacked
     meta-learner (src/ensemble_stack.py), since it's a differently-shaped
     signal again (binary gate) from either regressor's raw amount estimate.
  6. Evaluates: CNN alone, XGBoost alone, simple 50/50 average, and a
     weight swept 0.0-1.0 to find the best linear blend.
  7. Saves per-row predictions (cnn_pred, xgb_pred, xgb_proba, rainfall_mm)
     to models/ensemble_predictions.parquet so ensemble_stack.py can try
     learned meta-models without repeating the expensive CNN inference /
     XGBoost retraining steps above.

Run:
    python src/ensemble_predict.py --master-store-dir F:\\master_feature_store \\
        --shap-csv-regressor F:\\feature_importance_shap.csv \\
        --shap-csv-classifier F:\\feature_importance_shap_is_rainy.csv \\
        --dl-models-dir F:\\acoustic_rain_gauge_ml\\models \\
        --top-n 30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from dl_models import MODEL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs"
MODELS_DIR = REPO_ROOT / "models"

XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
)


def compute_train_mfcc_stats(train_cache_path, chunk_size=50_000):
    """Recompute the per-coefficient mean/std that train_dl_model.py derived
    from the train MFCC cache. mmap=True + chunked accumulation keeps this
    from requiring the full 15GB tensor resident in RAM at once."""
    print(f"  Computing train-set MFCC mean/std from {train_cache_path} (mmap, chunked)...")
    cache = torch.load(train_cache_path, mmap=True)
    mfcc = cache["mfcc"]
    n, n_mfcc, n_frames = mfcc.shape
    running_sum = torch.zeros(n_mfcc, dtype=torch.float64)
    running_sumsq = torch.zeros(n_mfcc, dtype=torch.float64)
    count = 0
    for start in range(0, n, chunk_size):
        chunk = mfcc[start:start + chunk_size].to(torch.float64)
        running_sum += chunk.sum(dim=(0, 2))
        running_sumsq += (chunk ** 2).sum(dim=(0, 2))
        count += chunk.shape[0] * chunk.shape[2]
        print(f"    {min(start + chunk_size, n):,} / {n:,} rows", flush=True)
    mean = (running_sum / count).to(torch.float32).view(1, n_mfcc, 1)
    var = (running_sumsq / count - (running_sum / count) ** 2).clamp_min(0)
    std = var.sqrt().to(torch.float32).view(1, n_mfcc, 1).clamp_min(1e-6)
    return mean, std


def predict_dl_model(arch_name, test_cache_path, mean, std, checkpoint_path, device, batch_size=512):
    """Runs any of the three MFCC architectures (cnn/lstm/transformer) over the
    cached test MFCC. All three were trained on the same normalized MFCC
    tensors in one train_dl_model.py invocation, so the same mean/std applies
    to every architecture -- not just the one it happened to be computed
    alongside."""
    print(f"  Running {arch_name.upper()} inference over {test_cache_path} (mmap, batched)...")
    cache = torch.load(test_cache_path, mmap=True)
    mfcc, y = cache["mfcc"], cache["y"]
    n = mfcc.shape[0]

    model = MODEL_REGISTRY[arch_name]().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    mean_dev, std_dev = mean.to(device), std.to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = mfcc[start:start + batch_size].to(device)
            batch = (batch - mean_dev) / std_dev
            pred = model(batch).cpu().numpy()
            preds.append(pred)
    preds = np.clip(np.concatenate(preds), 0, None)
    targets = y.numpy()
    return preds, targets


def load_master_store(store_dir, feature_cols):
    files = sorted(Path(store_dir).glob("master_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No chunks found in {store_dir}")
    keep = feature_cols + ["source_folder", "audio_filename"]
    print(f"  Loading {len(files)} chunk(s) from {store_dir}...")
    df = pd.concat((pd.read_parquet(f, columns=keep) for f in files), ignore_index=True)
    df = df.drop_duplicates(subset=["source_folder", "audio_filename"])
    return df


def attach_features(split_df, master_df, name):
    merged = split_df.merge(
        master_df, left_on=["month_folder", "audio_filename"],
        right_on=["source_folder", "audio_filename"], how="inner",
    )
    print(f"  {name}: {len(split_df):,} rows -> {len(merged):,} matched")
    return merged


def top_features_from_csv(path, top_n):
    importance_df = pd.read_csv(path)
    return (importance_df.sort_values("shap_importance", ascending=False)
            .head(top_n)["feature"].tolist())


def evaluate(name, y_true, y_pred):
    metrics = {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print(f"  {name:45s} RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-store-dir", type=str, required=True)
    parser.add_argument("--shap-csv-regressor", type=str, required=True)
    parser.add_argument("--shap-csv-classifier", type=str, required=True)
    parser.add_argument("--dl-models-dir", type=str, required=True,
                         help="Directory with dl_cnn_mfcc.pt, mfcc_cache_full_raw_train.pt, mfcc_cache_full_raw_test.pt")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    dl_dir = Path(args.dl_models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1] Loading Stage 3 test split...")
    usecols = ["audio_filename", "month_folder", "rainfall_mm", "is_rainy"]
    test_split = pd.read_csv(DATA_DIR / "test.csv", usecols=usecols)
    print(f"  Test: {len(test_split):,} rows")

    print("\n[2] Neural-network predictions (raw-audio MFCC path: CNN, LSTM, Transformer)...")
    mean, std = compute_train_mfcc_stats(dl_dir / "mfcc_cache_full_raw_train.pt")
    cnn_df = test_split.copy()
    for arch in ["cnn", "lstm", "transformer"]:
        pred, target = predict_dl_model(
            arch, dl_dir / "mfcc_cache_full_raw_test.pt", mean, std,
            dl_dir / f"dl_{arch}_mfcc.pt", device,
        )
        assert len(pred) == len(test_split), f"{arch} cache/test.csv row count mismatch"
        cnn_df[f"{arch}_pred"] = pred
        # Sanity check: cache target should match test.csv's rainfall_mm at the same row
        mismatch = np.abs(target - cnn_df["rainfall_mm"].to_numpy()).max()
        print(f"    Row-order sanity check ({arch}, max |cached_y - test.csv rainfall_mm|): {mismatch:.6f}")

    print("\n[3] XGBoost-optimized predictions (scalar hand-engineered feature path)...")
    reg_features = top_features_from_csv(args.shap_csv_regressor, args.top_n)
    clf_features = top_features_from_csv(args.shap_csv_classifier, args.top_n)
    all_features = sorted(set(reg_features) | set(clf_features))
    print(f"  Regressor features: {len(reg_features)} | Classifier features: {len(clf_features)} "
          f"| Union to load: {len(all_features)}")
    master_df = load_master_store(args.master_store_dir, all_features)
    usecols_train = usecols
    train_split = pd.read_csv(DATA_DIR / "train.csv", usecols=usecols_train)
    train_df = attach_features(train_split, master_df, "Train")
    test_df = attach_features(test_split, master_df, "Test")

    X_train, y_train = train_df[reg_features].fillna(0), train_df["rainfall_mm"].to_numpy()
    X_test = test_df[reg_features].fillna(0)
    xgb_reg = XGBRegressor(**XGB_PARAMS)
    xgb_reg.fit(X_train, y_train)
    xgb_pred = np.clip(xgb_reg.predict(X_test), 0, None)

    print("\n  Retraining XGBoost classifier (is_rainy-ranked features) for P(rainy)...")
    X_train_cls, y_train_cls = train_df[clf_features].fillna(0), train_df["is_rainy"].to_numpy()
    X_test_cls = test_df[clf_features].fillna(0)
    scale_pos_weight = (y_train_cls == 0).sum() / (y_train_cls == 1).sum()
    xgb_clf = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", **XGB_PARAMS)
    xgb_clf.fit(X_train_cls, y_train_cls)
    xgb_proba = xgb_clf.predict_proba(X_test_cls)[:, 1]

    xgb_df = test_df[["audio_filename", "month_folder", "rainfall_mm", "is_rainy"]].copy()
    xgb_df["xgb_pred"] = xgb_pred
    xgb_df["xgb_proba"] = xgb_proba

    print("\n[4] Aligning both prediction sets on (month_folder, audio_filename)...")
    combined = cnn_df.merge(
        xgb_df, on=["audio_filename", "month_folder"], suffixes=("", "_xgb"),
    )
    dropped = len(cnn_df) - len(combined)
    print(f"  Combined: {len(combined):,} rows ({dropped:,} dropped, unmatched by master-store join)")
    y_true = combined["rainfall_mm"].to_numpy()
    cnn_p = combined["cnn_pred"].to_numpy()
    xgb_p = combined["xgb_pred"].to_numpy()

    print("\n[5] Evaluating individual models and ensembles on the common test subset...")
    results = {}
    results["cnn_alone"] = evaluate("CNN alone", y_true, cnn_p)
    results["lstm_alone"] = evaluate("LSTM alone", y_true, combined["lstm_pred"].to_numpy())
    results["transformer_alone"] = evaluate("Transformer alone", y_true, combined["transformer_pred"].to_numpy())
    results["xgb_alone"] = evaluate("XGBoost-optimized alone", y_true, xgb_p)
    results["average_50_50"] = evaluate("Simple average (50/50)", y_true, 0.5 * cnn_p + 0.5 * xgb_p)

    print("\n  Sweeping blend weight w in [cnn*w + xgb*(1-w)]...")
    best_w, best_r2 = None, -float("inf")
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * cnn_p + (1 - w) * xgb_p
        r2 = r2_score(y_true, blend)
        if r2 > best_r2:
            best_r2, best_w = r2, w
    print(f"  Best weight: w_cnn={best_w:.2f}, w_xgb={1 - best_w:.2f} -> R2={best_r2:.4f}")
    results["best_weighted_blend"] = evaluate(
        f"Best weighted blend (w_cnn={best_w:.2f})", y_true, best_w * cnn_p + (1 - best_w) * xgb_p)
    results["best_weighted_blend"]["w_cnn"] = float(best_w)

    print("\n" + "=" * 68)
    print("  ENSEMBLE SUMMARY")
    print("=" * 68)
    print(f"  {'Model':<35}{'R2':<10}")
    for name, m in results.items():
        print(f"  {name:<35}{m['r2']:<10.4f}")
    print(f"  Baseline (Stage 4 XGBoost, 13 features): R2=0.155")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "ensemble_cnn_xgb_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = MODELS_DIR / "ensemble_predictions.parquet"
    combined[["audio_filename", "month_folder", "rainfall_mm", "is_rainy",
              "cnn_pred", "lstm_pred", "transformer_pred", "xgb_pred", "xgb_proba"]].to_parquet(pred_path, index=False)
    print(f"Saved per-row predictions -> {pred_path} (for ensemble_stack.py)")

    # Persist everything needed to run this same pipeline on ONE new clip at
    # inference time (predict.py --ensemble) -- none of this was saved before,
    # since this script previously only ever reported metrics.
    torch.save({"mean": mean, "std": std}, MODELS_DIR / "cnn_mfcc_norm_stats.pt")
    xgb_reg.save_model(str(MODELS_DIR / "xgb_regressor_optimized.json"))
    xgb_clf.save_model(str(MODELS_DIR / "xgb_classifier_optimized.json"))
    with open(MODELS_DIR / "xgb_optimized_features.json", "w") as f:
        json.dump({"regressor_features": reg_features, "classifier_features": clf_features}, f, indent=2)
    print(f"Saved production artifacts -> cnn_mfcc_norm_stats.pt, xgb_regressor_optimized.json, "
          f"xgb_classifier_optimized.json, xgb_optimized_features.json")


if __name__ == "__main__":
    main()
