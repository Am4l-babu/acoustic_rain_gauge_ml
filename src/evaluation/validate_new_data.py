"""
Held-out validation on genuinely new data (Stage 9)
=====================================================
Validates the production ensemble (CNN + LSTM + Transformer + XGBoost-optimized,
stacked -- see src/inference/ensemble_predict.py, R2=0.5429 on the original test
split, docs/reports/ensemble_stack_report.json) against audio recorded 2026-06-18
through 2026-07-15, none of which existed when the model was trained. Ground
truth is data/mechanical_data.csv (tipping-bucket gauge), aligned to each clip
by nearest timestamp within a 2-minute tolerance -- the same rule
features/data_cleaning.py uses for the training set.

Reads each clip exactly once per worker (single librosa.load), computing the
176 master features AND keeping the raw waveform for the DL models in the same
pass, instead of the 3 separate reads predict.py's predict_ensemble() does per
clip (fine for one-off CLI use, wasteful at ~180k clips). DL inference is
batched per chunk (GPU) rather than one clip at a time.

Run:
    python src/evaluation/validate_new_data.py
    python src/evaluation/validate_new_data.py --limit 2000   # smoke test
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ on sys.path for sibling packages

from features.master_feature_extraction import (
    extract_time_and_teager, extract_spectral, extract_mfcc,
    extract_mel_bands, extract_wavelets, extract_packets, extract_rhythm,
)
from dl.dl_dataset import CLIP_SAMPLES, MFCCExtractor
from dl.dl_models import MODEL_REGISTRY

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
# OS/mount agnostic: derived from this script's own location rather than a
# hardcoded drive letter, since the HDD holding both the repo and the dataset
# can show up as any letter on Windows (F:\, E:\, ...) or any mount point on
# Linux (/media/user/E-HDD, /mnt/..., ...). Project layout is always
# <hdd_root>/acoustic_rain_gauge_ml/src/evaluation/validate_new_data.py, so
# the HDD root is 3 levels up -- same convention as
# features/master_feature_extraction.py's _PROJECT_ROOT.
_HDD_ROOT   = Path(__file__).resolve().parents[3]
MODELS_DIR  = REPO_ROOT / "models"
DOCS_DIR    = REPO_ROOT / "docs" / "reports"
DATA_DIR    = REPO_ROOT / "data"
RESULTS_DIR = DOCS_DIR / "new_data_validation_chunks"

TARGET_SR = 8000

# The June 18 - July 15 2026 folders on the dataset drive. Two are named after
# the day the recorder was restarted rather than the dates they contain
# ("july 07" holds 2026-06-29..07-06, "july15" holds 07-08..07-15) -- verified
# by listing filenames, not by folder name.
AUDIO_ROOT = _HDD_ROOT / "arg_dataset_unzip"
NEW_DATA_FOLDERS = [
    "2026_06_18", "2026_06_19", "2026_06_20", "2026_06_21",
    "2026_06_22", "2026_06_23", "2026_06_24", "2026_06_25",
    "july 07", "july15",
]
# 2026-06-26..06-28 have no audio (recorder gap, confirmed against mechanical
# data which has no corresponding gap -- a recorder outage, not a dry spell).

MECH_CSV = DATA_DIR / "mechanical_data.csv"

CHUNK_SIZE = 5000
NUM_WORKERS = min(6, max(1, mp.cpu_count() - 2))

_TS_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")


def _parse_timestamp(filename: str):
    m = _TS_PATTERN.search(Path(filename).stem)
    if not m:
        return None
    g = m.groups()
    try:
        return datetime(int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]))
    except ValueError:
        return None


def load_mechanical_data(path=None):
    df = pd.read_csv(path or MECH_CSV)
    time_col, rain_col = df.columns[0], df.columns[1]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    df[rain_col] = (df[rain_col].astype(str).str.replace("mm", "", regex=False).str.strip())
    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0)
    return df, time_col, rain_col


def scan_new_clips(limit=None):
    tasks = []
    for folder in NEW_DATA_FOLDERS:
        folder_path = AUDIO_ROOT / folder
        if not folder_path.exists():
            print(f"  WARNING: folder not found, skipping: {folder_path}")
            continue
        for wav_path in folder_path.glob("*.wav"):
            ts = _parse_timestamp(wav_path.name)
            if ts is None:
                continue
            tasks.append({"audio_path": str(wav_path), "audio_filename": wav_path.name, "timestamp": ts})
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def align_to_mechanical(tasks):
    audio_df = pd.DataFrame(tasks).sort_values("timestamp").reset_index(drop=True)
    mech_df, time_col, rain_col = load_mechanical_data()
    aligned = pd.merge_asof(
        audio_df, mech_df, left_on="timestamp", right_on=time_col,
        direction="nearest", tolerance=pd.Timedelta(minutes=2),
    )
    aligned["rainfall_mm"] = aligned[rain_col].fillna(0.0)
    aligned["is_aligned"] = aligned[time_col].notna()
    aligned["is_rainy"] = (aligned["rainfall_mm"] > 0).astype(int)
    return aligned


# ============================================================
# WORKER: one audio read -> 176 master features + raw waveform
# ============================================================
def _extract_features_and_waveform(audio_path: str):
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, duration=None)
        if len(y) < TARGET_SR:
            return None

        features = {}
        features.update(extract_time_and_teager(y))
        features.update(extract_spectral(y, sr))
        features.update(extract_mfcc(y, sr))
        features.update(extract_mel_bands(y, sr))
        features.update(extract_wavelets(y))
        features.update(extract_packets(y, sr))
        features.update(extract_rhythm(y, sr))

        if len(y) != CLIP_SAMPLES:
            y = np.pad(y, (0, CLIP_SAMPLES - len(y))) if len(y) < CLIP_SAMPLES else y[:CLIP_SAMPLES]

        return audio_path, features, y.astype(np.float32)
    except Exception:
        return None


def load_ensemble_artifacts(device):
    norm = torch.load(MODELS_DIR / "cnn_mfcc_norm_stats.pt", map_location=device)
    mean, std = norm["mean"].to(device), norm["std"].to(device)
    extractor = MFCCExtractor().to(device)

    dl_models = {}
    for arch in ["cnn", "lstm", "transformer"]:
        m = MODEL_REGISTRY[arch]().to(device)
        m.load_state_dict(torch.load(MODELS_DIR / f"dl_{arch}_mfcc.pt", map_location=device))
        m.eval()
        dl_models[arch] = m

    with open(MODELS_DIR / "xgb_optimized_features.json") as f:
        feat_lists = json.load(f)
    xgb_reg = XGBRegressor()
    xgb_reg.load_model(str(MODELS_DIR / "xgb_regressor_optimized.json"))
    xgb_clf = XGBClassifier()
    xgb_clf.load_model(str(MODELS_DIR / "xgb_classifier_optimized.json"))

    with open(MODELS_DIR / "ensemble_stacker_config.json") as f:
        stacker_config = json.load(f)
    stacker = XGBRegressor()
    stacker.load_model(str(MODELS_DIR / "ensemble_stacker.json"))

    return {
        "mean": mean, "std": std, "extractor": extractor, "dl_models": dl_models,
        "reg_features": feat_lists["regressor_features"], "clf_features": feat_lists["classifier_features"],
        "xgb_reg": xgb_reg, "xgb_clf": xgb_clf,
        "stacker": stacker, "stacker_config": stacker_config,
    }


def run_ensemble_batch(waveforms: np.ndarray, master_feat_df: pd.DataFrame, artifacts, device, gpu_batch_size=512):
    dl_preds = {arch: [] for arch in artifacts["dl_models"]}
    with torch.no_grad():
        for start in range(0, len(waveforms), gpu_batch_size):
            batch = torch.from_numpy(waveforms[start:start + gpu_batch_size]).to(device)
            mfcc = (artifacts["extractor"](batch) - artifacts["mean"]) / artifacts["std"]
            for arch, model in artifacts["dl_models"].items():
                dl_preds[arch].append(np.clip(model(mfcc).cpu().numpy(), 0, None))
    dl_preds = {arch: np.concatenate(chunks) for arch, chunks in dl_preds.items()}

    reg_row = master_feat_df.reindex(columns=artifacts["reg_features"], fill_value=0)
    clf_row = master_feat_df.reindex(columns=artifacts["clf_features"], fill_value=0)
    xgb_pred = np.clip(artifacts["xgb_reg"].predict(reg_row), 0, None)
    xgb_proba = artifacts["xgb_clf"].predict_proba(clf_row)[:, 1]

    config = artifacts["stacker_config"]
    signals = {
        "cnn_pred": dl_preds["cnn"], "lstm_pred": dl_preds["lstm"],
        "transformer_pred": dl_preds["transformer"],
        "xgb_pred": xgb_pred, "xgb_proba": xgb_proba,
    }
    stack_input = np.column_stack([signals[f] for f in config["features"]])
    stacker_pred = artifacts["stacker"].predict(stack_input)

    if config["type"] == "hurdle_hard_gate":
        final_pred = np.where(xgb_proba >= config["gate_threshold"], np.clip(stacker_pred, 0, None), 0.0)
    else:
        final_pred = np.clip(stacker_pred, 0, None)

    return pd.DataFrame({
        "cnn_pred_mm": dl_preds["cnn"], "lstm_pred_mm": dl_preds["lstm"],
        "transformer_pred_mm": dl_preds["transformer"], "xgb_pred_mm": xgb_pred,
        "rain_probability": xgb_proba, "estimated_rainfall_mm": final_pred,
    })


def compute_metrics_and_plots(results_df):
    y_true = results_df["rainfall_mm"].to_numpy()
    y_pred = results_df["estimated_rainfall_mm"].to_numpy()
    is_rainy_true = results_df["is_rainy"].to_numpy()
    rain_proba = results_df["rain_probability"].to_numpy()
    is_rainy_pred = (rain_proba >= 0.5).astype(int)

    regression = {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_clips": int(len(results_df)),
    }
    classification = {
        "precision": float(precision_score(is_rainy_true, is_rainy_pred, zero_division=0)),
        "recall": float(recall_score(is_rainy_true, is_rainy_pred, zero_division=0)),
        "f1": float(f1_score(is_rainy_true, is_rainy_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(is_rainy_true, rain_proba)) if len(set(is_rainy_true)) > 1 else None,
        "rainy_clips": int(is_rainy_true.sum()),
        "dry_clips": int((is_rainy_true == 0).sum()),
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs actual scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=6, alpha=0.3, color="#4C72B0")
    lim = max(y_true.max(), y_pred.max(), 1.0)
    plt.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    plt.xlabel("True rainfall (mm)")
    plt.ylabel("Predicted rainfall (mm)")
    plt.title(f"New-data validation (n={len(results_df):,}) — R2={regression['r2']:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "new_data_validation_scatter.png", dpi=120)
    plt.close()

    # 2. Residuals
    residuals = y_pred - y_true
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=80, color="#DD8452")
    plt.xlabel("Predicted - True (mm)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "new_data_validation_residuals.png", dpi=120)
    plt.close()

    # 3. Confusion matrix
    cm = confusion_matrix(is_rainy_true, is_rainy_pred, labels=[0, 1])
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["Dry", "Rainy"])
    plt.yticks([0, 1], ["Dry", "Rainy"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (threshold=0.5)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "new_data_validation_confusion_matrix.png", dpi=120)
    plt.close()

    # 4. ROC curve
    if classification["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(is_rainy_true, rain_proba)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="#55A868", label=f"AUC={classification['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(DOCS_DIR / "new_data_validation_roc.png", dpi=120)
        plt.close()

    # 5. Time series overlay
    ts_df = results_df.sort_values("timestamp")
    plt.figure(figsize=(14, 5))
    plt.plot(ts_df["timestamp"], ts_df["rainfall_mm"], label="True (mechanical)", color="#4C72B0", lw=0.8)
    plt.plot(ts_df["timestamp"], ts_df["estimated_rainfall_mm"], label="Predicted (ensemble)",
              color="#DD8452", lw=0.8, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Rainfall (mm)")
    plt.title("Predicted vs true rainfall over the validation window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "new_data_validation_timeseries.png", dpi=120)
    plt.close()

    return regression, classification


def main():
    global AUDIO_ROOT, MODELS_DIR, DOCS_DIR, DATA_DIR, RESULTS_DIR, MECH_CSV

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N clips (smoke test)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--audio-root", type=str, default=None,
                         help=f"Override audio root, i.e. wherever arg_dataset_unzip lives (default: {AUDIO_ROOT})")
    parser.add_argument("--output-dir", type=str, default=None,
                         help=f"Override report/plot output directory (default: {DOCS_DIR})")
    parser.add_argument("--mech-csv", type=str, default=None,
                         help=f"Override mechanical ground-truth CSV path (default: {MECH_CSV})")
    args = parser.parse_args()
    num_workers = args.workers or NUM_WORKERS

    if args.audio_root:
        AUDIO_ROOT = Path(args.audio_root)
    if args.output_dir:
        DOCS_DIR = Path(args.output_dir)
        RESULTS_DIR = DOCS_DIR / "new_data_validation_chunks"
    if args.mech_csv:
        MECH_CSV = Path(args.mech_csv)

    print("=" * 70)
    print("STAGE 9 — HELD-OUT VALIDATION ON NEW DATA (2026-06-18 .. 2026-07-15)")
    print("=" * 70)

    print("\n[1] Scanning new audio clips...")
    tasks = scan_new_clips(limit=args.limit)
    print(f"  Found {len(tasks):,} clips")
    if not tasks:
        print("ABORTING: no clips found — check AUDIO_ROOT / drive letter.")
        return

    print("\n[2] Aligning to mechanical_data.csv (nearest, 2-min tolerance)...")
    aligned = align_to_mechanical(tasks)
    print(f"  Aligned: {aligned['is_aligned'].sum():,}/{len(aligned):,}  "
          f"Rainy: {(aligned['rainfall_mm'] > 0).sum():,}  Dry: {(aligned['rainfall_mm'] == 0).sum():,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[3] Loading ensemble artifacts (device={device})...")
    artifacts = load_ensemble_artifacts(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    t_start = time.time()
    n_chunks = (len(aligned) - 1) // CHUNK_SIZE + 1

    for chunk_num, start in enumerate(range(0, len(aligned), CHUNK_SIZE), start=1):
        chunk_out = RESULTS_DIR / f"chunk_{chunk_num:04d}.parquet"
        if chunk_out.exists():
            all_results.append(pd.read_parquet(chunk_out))
            print(f"  Chunk {chunk_num}/{n_chunks}: already done, skipping")
            continue

        chunk_meta = aligned.iloc[start:start + CHUNK_SIZE].reset_index(drop=True)
        t0 = time.time()

        with mp.Pool(num_workers) as pool:
            worker_results = pool.map(_extract_features_and_waveform, chunk_meta["audio_path"].tolist(), chunksize=32)

        valid_idx, feat_rows, waveforms = [], [], []
        for i, r in enumerate(worker_results):
            if r is None:
                continue
            _path, feats, wav = r
            valid_idx.append(i)
            feat_rows.append(feats)
            waveforms.append(wav)

        if not feat_rows:
            print(f"  Chunk {chunk_num}/{n_chunks}: no valid clips, skipping")
            continue

        feat_df = pd.DataFrame(feat_rows)
        wav_batch = np.stack(waveforms)
        meta_valid = chunk_meta.iloc[valid_idx].reset_index(drop=True)

        preds = run_ensemble_batch(wav_batch, feat_df, artifacts, device)
        chunk_result = pd.concat([
            meta_valid[["audio_filename", "timestamp", "rainfall_mm", "is_rainy"]].reset_index(drop=True),
            preds.reset_index(drop=True),
        ], axis=1)
        chunk_result.to_parquet(chunk_out, index=False)
        all_results.append(chunk_result)

        elapsed_chunk = time.time() - t0
        ms_per_clip = elapsed_chunk / len(chunk_meta) * 1000
        elapsed_total = time.time() - t_start
        clips_done = start + len(chunk_meta)
        eta = (len(aligned) - clips_done) * (elapsed_total / clips_done) if clips_done else 0
        print(f"  Chunk {chunk_num}/{n_chunks}: {len(feat_rows)}/{len(chunk_meta)} ok, "
              f"{elapsed_chunk:.1f}s ({ms_per_clip:.1f} ms/clip) | "
              f"elapsed={elapsed_total/60:.1f}min ETA={eta/60:.1f}min")

    total_elapsed = time.time() - t_start
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\n[4] Computing metrics on {len(results_df):,} validated clips...")
    regression, classification = compute_metrics_and_plots(results_df)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Regression : RMSE={regression['rmse']:.4f}  MAE={regression['mae']:.4f}  R2={regression['r2']:.4f}")
    print(f"  Classification (threshold=0.5): Precision={classification['precision']:.4f}  "
          f"Recall={classification['recall']:.4f}  F1={classification['f1']:.4f}  "
          f"ROC-AUC={classification['roc_auc']}")
    print(f"  Reference  : original test-split ensemble R2=0.5429 (docs/reports/ensemble_stack_report.json)")
    print(f"\n  Total wall time: {total_elapsed/60:.1f} min ({total_elapsed/len(results_df)*1000:.1f} ms/clip)")

    report = {
        "run_date": datetime.now().isoformat(),
        "n_clips": len(results_df),
        "validation_window": "2026-06-18 to 2026-07-15",
        "regression": regression,
        "classification": classification,
        "reference_test_split_r2": 0.5429394145661929,
        "elapsed_seconds": total_elapsed,
        "ms_per_clip": total_elapsed / len(results_df) * 1000,
    }
    with open(DOCS_DIR / "new_data_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {DOCS_DIR / 'new_data_validation_report.json'}")
    print(f"  Plots: new_data_validation_{{scatter,residuals,confusion_matrix,roc,timeseries}}.png")


if __name__ == "__main__":
    main()
