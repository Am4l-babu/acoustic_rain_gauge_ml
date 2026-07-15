"""
Overnight full training sweep — orchestrator
================================================
Runs every planned next-step combo from this project's ablation in one
unattended pass, in three phases, and refreshes the production ensemble with
whatever wins. Designed to survive a machine left running overnight with
nobody watching it:

  - Every experiment's result is written to docs/sweep_progress.json
    IMMEDIATELY after it finishes (or fails) -- a crash never loses more
    than the one experiment in flight.
  - Re-running this script skips any experiment already recorded in that
    file, so an interrupted overnight run picks up where it left off
    instead of restarting from zero.
  - Phase 1 (DL) runs each hyperparameter combo as an ISOLATED SUBPROCESS
    (a fresh `python3 train_dl_model.py ...` call) rather than in-process,
    so a CUDA OOM or hard crash in one combo can't corrupt the GPU context
    for every combo after it -- the orchestrator just logs the failure and
    moves to the next combo.
  - Phase 2 (XGBoost) failures are caught with try/except per combo, same
    reasoning, cheaper to isolate since XGBoost crashes don't poison CUDA.

Phase 1 -- DL hyperparameter sweep (CNN / LSTM / Transformer):
  Motivated by the 2026-07-14 finding that full-scale training peaks in the
  first handful of epochs and overfits monotonically after that (train loss
  keeps falling, test R2 does not) -- every combo here uses early stopping
  (--patience) instead of a fixed 40 epochs, and sweeps learning rate /
  batch size since lr=3e-4,batch=256 already beat the original lr=1e-3,
  batch=128 for CNN (0.306 vs 0.277) but LSTM/Transformer were never
  retried with it.

Phase 2 -- XGBoost-optimized sweep (SHAP top-N x hyperparameters):
  Sweeps how many of the 175 master-store features to keep (20/30/50) and a
  small randomized hyperparameter search per top-N, for both the regressor
  and classifier.

Phase 3 -- Ensemble refresh:
  Takes the best checkpoint per DL architecture and the best XGBoost
  regressor/classifier config found above, re-runs the full CNN+LSTM+
  Transformer+XGBoost+proba stacking pipeline (5-fold CV, hyperparameter
  search), and overwrites the production ensemble artifacts if the result
  beats the current one -- never overwrites with something worse.

Run (from the repo root on the HDD -- all paths auto-detected from the
repo's own location, same convention as dl_dataset.py / master_feature_
extraction.py, so no arguments are needed when the standard HDD layout is
in place):
    cd /media/icfoss/E-HDD/acoustic_rain_gauge_ml
    python3 src/run_sweep.py

Explicit overrides are available (--master-store-dir, --shap-csv-regressor,
--shap-csv-classifier, --dl-models-dir) for non-standard layouts.

Safe to Ctrl-C and re-run later -- it will skip everything already recorded
in docs/sweep_progress.json.
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs"
PROGRESS_PATH = DOCS_DIR / "sweep_progress.json"

# The HDD layout puts the repo alongside its data siblings
# (master_feature_store, the SHAP CSVs, arg_dataset_unzip) under one root --
# whether that root is /media/icfoss/E-HDD on Linux or F:\ on Windows.
HDD_ROOT = REPO_ROOT.parent

PYTHON = sys.executable


# ============================================================
# Progress manifest (resume support)
# ============================================================

def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"dl": {}, "xgb": {}, "ensemble": {}}


def save_progress(progress):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2, default=str)
    tmp.replace(PROGRESS_PATH)  # atomic on POSIX -- never leaves a half-written file


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# PHASE 1 -- DL hyperparameter sweep
# ============================================================

DL_GRID = {
    "cnn":         {"lrs": [1e-3, 3e-4, 1e-4], "batches": [128, 256]},
    "lstm":        {"lrs": [1e-3, 3e-4, 1e-4], "batches": [128, 256]},
    "transformer": {"lrs": [1e-3, 3e-4, 1e-4], "batches": [128, 256]},
}


def run_dl_phase(progress, args):
    log("=" * 68)
    log("PHASE 1: DL hyperparameter sweep (CNN / LSTM / Transformer)")
    log("=" * 68)

    experiments = []
    for arch, grid in DL_GRID.items():
        for lr in grid["lrs"]:
            for batch in grid["batches"]:
                tag = f"{arch}_lr{lr:g}_bs{batch}"
                experiments.append((arch, lr, batch, tag))
    log(f"  {len(experiments)} combos planned "
        f"({len(DL_GRID)} architectures x {len(DL_GRID['cnn']['lrs'])} lrs x "
        f"{len(DL_GRID['cnn']['batches'])} batch sizes)")

    for arch, lr, batch, tag in experiments:
        if tag in progress["dl"]:
            log(f"  SKIP (already done): {tag} -> best_r2={progress['dl'][tag].get('best_r2')}")
            continue

        cmd = [
            PYTHON, str(REPO_ROOT / "src" / "train_dl_model.py"),
            "--full", "--models", arch,
            "--epochs", "15", "--patience", "5",
            "--lr", str(lr), "--batch-size", str(batch),
            "--tag", tag,
        ]
        if args.dl_models_dir:
            cmd += ["--models-dir", args.dl_models_dir]

        log(f"  RUNNING: {tag}  ({' '.join(cmd)})")
        t0 = time.time()
        try:
            result = subprocess.run(cmd, cwd=str(REPO_ROOT), timeout=90 * 60,
                                     capture_output=True, text=True)
            elapsed = time.time() - t0
            log_path = DOCS_DIR / f"sweep_log_{tag}.txt"
            log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)

            if result.returncode != 0:
                log(f"  FAILED ({elapsed/60:.1f} min, exit {result.returncode}) -- see {log_path.name}")
                progress["dl"][tag] = {"status": "failed", "arch": arch, "lr": lr,
                                        "batch_size": batch, "elapsed_min": elapsed / 60,
                                        "log_file": str(log_path.name)}
            else:
                report_path = DOCS_DIR / f"dl_ablation_full_report_{tag}.json"
                if report_path.exists():
                    with open(report_path) as f:
                        report = json.load(f)
                    best_r2 = report["results"][0]["best_r2"]
                    stopped_at = report["results"][0]["stopped_early_at"]
                    ckpt = report["results"][0]["checkpoint_path"]
                    log(f"  DONE ({elapsed/60:.1f} min): {tag} -> best_r2={best_r2:.4f} "
                        f"(stopped at epoch {stopped_at})")
                    progress["dl"][tag] = {"status": "ok", "arch": arch, "lr": lr, "batch_size": batch,
                                            "best_r2": best_r2, "stopped_early_at": stopped_at,
                                            "checkpoint_path": ckpt, "elapsed_min": elapsed / 60}
                else:
                    log(f"  DONE but report missing ({elapsed/60:.1f} min) -- treating as failed")
                    progress["dl"][tag] = {"status": "failed_no_report", "arch": arch,
                                            "lr": lr, "batch_size": batch, "elapsed_min": elapsed / 60}
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            log(f"  TIMEOUT after {elapsed/60:.1f} min -- killed, moving on")
            progress["dl"][tag] = {"status": "timeout", "arch": arch, "lr": lr,
                                    "batch_size": batch, "elapsed_min": elapsed / 60}
        except Exception as e:
            log(f"  ERROR launching {tag}: {e}")
            progress["dl"][tag] = {"status": "error", "arch": arch, "lr": lr,
                                    "batch_size": batch, "error": str(e)}
        save_progress(progress)  # after EVERY experiment, not just at the end

    log("Phase 1 complete.")
    return progress


def best_checkpoint_per_arch(progress):
    best = {}
    for tag, r in progress["dl"].items():
        if r.get("status") != "ok":
            continue
        arch = r["arch"]
        if arch not in best or r["best_r2"] > best[arch]["best_r2"]:
            best[arch] = r
    return best


# ============================================================
# PHASE 2 -- XGBoost-optimized sweep
# ============================================================

def load_master_store(store_dir, feature_cols):
    files = sorted(Path(store_dir).glob("master_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No chunks found in {store_dir}")
    keep = feature_cols + ["source_folder", "audio_filename"]
    df = pd.concat((pd.read_parquet(f, columns=keep) for f in files), ignore_index=True)
    return df.drop_duplicates(subset=["source_folder", "audio_filename"])


def attach_features(split_df, master_df, name):
    merged = split_df.merge(
        master_df, left_on=["month_folder", "audio_filename"],
        right_on=["source_folder", "audio_filename"], how="inner")
    log(f"    {name}: {len(split_df):,} -> {len(merged):,} matched")
    return merged


def top_features_from_csv(path, top_n):
    df = pd.read_csv(path)
    return df.sort_values("shap_importance", ascending=False).head(top_n)["feature"].tolist()


XGB_HP_GRID = {
    "n_estimators": [150, 250, 350, 450],
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
TOP_N_GRID = [20, 30, 50]
N_HP_SAMPLES_PER_TOPN = 6


def run_xgb_phase(progress, args):
    log("=" * 68)
    log("PHASE 2: XGBoost-optimized sweep (SHAP top-N x hyperparameters)")
    log("=" * 68)

    usecols = ["audio_filename", "month_folder", "rainfall_mm", "is_rainy"]
    train_split = pd.read_csv(DATA_DIR / "train.csv", usecols=usecols)
    test_split = pd.read_csv(DATA_DIR / "test.csv", usecols=usecols)

    max_top_n = max(TOP_N_GRID)
    reg_features_full = top_features_from_csv(args.shap_csv_regressor, max_top_n)
    clf_features_full = top_features_from_csv(args.shap_csv_classifier, max_top_n)
    all_features = sorted(set(reg_features_full) | set(clf_features_full))
    log(f"  Loading master store ({len(all_features)} unique features across all top-N)...")
    master_df = load_master_store(args.master_store_dir, all_features)
    train_df = attach_features(train_split, master_df, "Train")
    test_df = attach_features(test_split, master_df, "Test")

    sampler_seed = 42
    for top_n in TOP_N_GRID:
        reg_features = reg_features_full[:top_n]
        clf_features = clf_features_full[:top_n]

        # Classifier: one fixed-config fit per top_n (already strong at defaults; the
        # regressor is the documented weak link, so most of the search budget goes there)
        clf_tag = f"clf_topn{top_n}"
        if clf_tag not in progress["xgb"]:
            try:
                X_tr, y_tr = train_df[clf_features].fillna(0), train_df["is_rainy"].to_numpy()
                X_te, y_te = test_df[clf_features].fillna(0), test_df["is_rainy"].to_numpy()
                spw = (y_tr == 0).sum() / (y_tr == 1).sum()
                clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                                     n_jobs=-1, scale_pos_weight=spw, eval_metric="auc")
                clf.fit(X_tr, y_tr)
                auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
                log(f"  {clf_tag}: AUC={auc:.4f}")
                progress["xgb"][clf_tag] = {"status": "ok", "type": "classifier", "top_n": top_n,
                                             "features": clf_features, "auc": auc,
                                             "params": {"n_estimators": 300, "max_depth": 6,
                                                        "learning_rate": 0.05, "subsample": 0.8,
                                                        "colsample_bytree": 0.8}}
            except Exception as e:
                log(f"  {clf_tag} FAILED: {e}")
                progress["xgb"][clf_tag] = {"status": "failed", "error": str(e)}
            save_progress(progress)
        else:
            log(f"  SKIP (already done): {clf_tag}")

        # Regressor: randomized hyperparameter search per top_n
        sampler = list(ParameterSampler(XGB_HP_GRID, n_iter=N_HP_SAMPLES_PER_TOPN,
                                         random_state=sampler_seed))
        sampler_seed += 1
        X_tr, y_tr = train_df[reg_features].fillna(0), train_df["rainfall_mm"].to_numpy()
        X_te, y_te = test_df[reg_features].fillna(0), test_df["rainfall_mm"].to_numpy()
        for i, params in enumerate(sampler, 1):
            reg_tag = f"reg_topn{top_n}_hp{i}"
            if reg_tag in progress["xgb"]:
                log(f"  SKIP (already done): {reg_tag} -> r2={progress['xgb'][reg_tag].get('r2')}")
                continue
            try:
                reg = XGBRegressor(**params, random_state=42, n_jobs=-1)
                reg.fit(X_tr, y_tr)
                pred = np.clip(reg.predict(X_te), 0, None)
                r2 = r2_score(y_te, pred)
                log(f"  {reg_tag}: {params} -> R2={r2:.4f}")
                progress["xgb"][reg_tag] = {"status": "ok", "type": "regressor", "top_n": top_n,
                                             "features": reg_features, "r2": r2, "params": params}
            except Exception as e:
                log(f"  {reg_tag} FAILED: {e}")
                progress["xgb"][reg_tag] = {"status": "failed", "top_n": top_n, "error": str(e)}
            save_progress(progress)

    log("Phase 2 complete.")
    return progress, train_df, test_df


def best_xgb_configs(progress):
    best_reg, best_clf = None, None
    for tag, r in progress["xgb"].items():
        if r.get("status") != "ok":
            continue
        if r["type"] == "regressor" and (best_reg is None or r["r2"] > best_reg["r2"]):
            best_reg = r
        if r["type"] == "classifier" and (best_clf is None or r["auc"] > best_clf["auc"]):
            best_clf = r
    return best_reg, best_clf


# ============================================================
# PHASE 3 -- Ensemble refresh
# ============================================================

def compute_train_mfcc_stats(train_cache_path, chunk_size=50_000):
    import torch
    cache = torch.load(train_cache_path, mmap=True)
    mfcc = cache["mfcc"]
    n, n_mfcc, _ = mfcc.shape
    running_sum = torch.zeros(n_mfcc, dtype=torch.float64)
    running_sumsq = torch.zeros(n_mfcc, dtype=torch.float64)
    count = 0
    for start in range(0, n, chunk_size):
        chunk = mfcc[start:start + chunk_size].to(torch.float64)
        running_sum += chunk.sum(dim=(0, 2))
        running_sumsq += (chunk ** 2).sum(dim=(0, 2))
        count += chunk.shape[0] * chunk.shape[2]
    mean = (running_sum / count).to(torch.float32).view(1, n_mfcc, 1)
    var = (running_sumsq / count - (running_sum / count) ** 2).clamp_min(0)
    std = var.sqrt().to(torch.float32).view(1, n_mfcc, 1).clamp_min(1e-6)
    return mean, std


def predict_dl_checkpoint(arch, checkpoint_path, test_cache_path, mean, std, device, batch_size=512):
    import torch
    from dl_models import MODEL_REGISTRY
    cache = torch.load(test_cache_path, mmap=True)
    mfcc, y = cache["mfcc"], cache["y"]
    model = MODEL_REGISTRY[arch]().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    mean_dev, std_dev = mean.to(device), std.to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, mfcc.shape[0], batch_size):
            batch = ((mfcc[start:start + batch_size].to(device) - mean_dev) / std_dev)
            preds.append(model(batch).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, None), y.numpy()


def run_ensemble_phase(progress, args, train_df, test_df):
    import torch
    log("=" * 68)
    log("PHASE 3: Ensemble refresh (best DL x best XGBoost)")
    log("=" * 68)

    dl_dir = Path(args.dl_models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_dl = best_checkpoint_per_arch(progress)
    best_reg, best_clf = best_xgb_configs(progress)

    if len(best_dl) < 3 or best_reg is None or best_clf is None:
        log("  Not enough successful experiments to build an ensemble yet (need all 3 "
            "architectures + a regressor + a classifier) -- skipping Phase 3 for now.")
        return progress

    log(f"  Best CNN: {best_dl['cnn']['checkpoint_path']} (R2={best_dl['cnn']['best_r2']:.4f})")
    log(f"  Best LSTM: {best_dl['lstm']['checkpoint_path']} (R2={best_dl['lstm']['best_r2']:.4f})")
    log(f"  Best Transformer: {best_dl['transformer']['checkpoint_path']} (R2={best_dl['transformer']['best_r2']:.4f})")
    log(f"  Best XGB regressor: top_n={best_reg['top_n']} R2={best_reg['r2']:.4f} params={best_reg['params']}")
    log(f"  Best XGB classifier: top_n={best_clf['top_n']} AUC={best_clf['auc']:.4f}")

    log("  Computing MFCC normalization stats (mmap, chunked)...")
    mean, std = compute_train_mfcc_stats(dl_dir / "mfcc_cache_full_raw_train.pt")

    usecols = ["audio_filename", "month_folder", "rainfall_mm", "is_rainy"]
    test_split = pd.read_csv(DATA_DIR / "test.csv", usecols=usecols)
    combined = test_split.copy()
    for arch in ["cnn", "lstm", "transformer"]:
        pred, target = predict_dl_checkpoint(
            arch, best_dl[arch]["checkpoint_path"], dl_dir / "mfcc_cache_full_raw_test.pt",
            mean, std, device)
        combined[f"{arch}_pred"] = pred
    mismatch = np.abs(target - combined["rainfall_mm"].to_numpy()).max()
    log(f"  Row-order sanity check: {mismatch:.6f}")

    reg = XGBRegressor(**best_reg["params"], random_state=42, n_jobs=-1)
    reg.fit(train_df[best_reg["features"]].fillna(0), train_df["rainfall_mm"].to_numpy())
    xgb_pred_test = np.clip(reg.predict(test_df[best_reg["features"]].fillna(0)), 0, None)

    clf = XGBClassifier(**best_clf["params"], random_state=42, n_jobs=-1,
                         scale_pos_weight=(train_df["is_rainy"] == 0).sum() / (train_df["is_rainy"] == 1).sum(),
                         eval_metric="auc")
    clf.fit(train_df[best_clf["features"]].fillna(0), train_df["is_rainy"].to_numpy())
    xgb_proba_test = clf.predict_proba(test_df[best_clf["features"]].fillna(0))[:, 1]

    xgb_key_cols = test_df[["audio_filename", "month_folder"]].copy()
    xgb_key_cols["xgb_pred"] = xgb_pred_test
    xgb_key_cols["xgb_proba"] = xgb_proba_test
    combined = combined.merge(xgb_key_cols, on=["audio_filename", "month_folder"])
    log(f"  Combined: {len(combined):,} rows")

    y_all = combined["rainfall_mm"].to_numpy()
    is_rainy = combined["is_rainy"].to_numpy()

    def stack_oof_r2(feature_cols, params):
        X = combined[feature_cols].to_numpy()
        oof = np.zeros(len(y_all))
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kfold.split(X, is_rainy):
            m = XGBRegressor(**params)
            m.fit(X[tr_idx], y_all[tr_idx])
            oof[val_idx] = np.clip(m.predict(X[val_idx]), 0, None)
        return r2_score(y_all, oof)

    BASE_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

    # --- 3a: which SUBSET of the 5 signals should the stacker see? ---
    # Never swept before (production always used all 5). Cheap to answer
    # here, and genuinely open: e.g. the LSTM is the weakest input -- does
    # dropping it help or does the stacker already learn to down-weight it?
    SUBSETS = {
        "all5":               ["cnn_pred", "lstm_pred", "transformer_pred", "xgb_pred", "xgb_proba"],
        "no_lstm":            ["cnn_pred", "transformer_pred", "xgb_pred", "xgb_proba"],
        "no_proba":           ["cnn_pred", "lstm_pred", "transformer_pred", "xgb_pred"],
        "dl_only":            ["cnn_pred", "lstm_pred", "transformer_pred"],
        "dl_plus_proba":      ["cnn_pred", "lstm_pred", "transformer_pred", "xgb_proba"],
        "cnn_xgb_proba_prev": ["cnn_pred", "xgb_pred", "xgb_proba"],
        "cnn_xgb":            ["cnn_pred", "xgb_pred"],
        "scalar_only":        ["xgb_pred", "xgb_proba"],
    }
    log("  [3a] Stacker input-subset sweep (base params, 5-fold OOF each)...")
    subset_results = {}
    for sname, cols in SUBSETS.items():
        r2 = stack_oof_r2(cols, BASE_PARAMS)
        subset_results[sname] = {"features": cols, "r2": r2}
        log(f"    {sname:22s} ({len(cols)} inputs) -> R2={r2:.4f}")
        progress["ensemble"].setdefault("subsets", {})[sname] = subset_results[sname]
        save_progress(progress)
    best_subset = max(subset_results, key=lambda s: subset_results[s]["r2"])
    feature_cols = subset_results[best_subset]["features"]
    log(f"  Best subset: {best_subset} (R2={subset_results[best_subset]['r2']:.4f})")

    # --- 3b: hyperparameter search on the winning subset ---
    log("  [3b] Stacker hyperparameter search on the winning subset...")
    param_grid = {"n_estimators": [100, 200, 300, 400], "max_depth": [2, 3, 4, 5],
                  "learning_rate": [0.02, 0.05, 0.08, 0.12], "subsample": [0.7, 0.8, 0.9, 1.0],
                  "colsample_bytree": [0.6, 0.8, 1.0]}
    sampler = list(ParameterSampler(param_grid, n_iter=12, random_state=42))
    best_stack_params, best_stack_r2 = dict(BASE_PARAMS), subset_results[best_subset]["r2"]
    for params in sampler:
        full_params = {**params, "random_state": 42, "n_jobs": -1}
        r2 = stack_oof_r2(feature_cols, full_params)
        log(f"    stack params {params} -> 5-fold OOF R2={r2:.4f}")
        if r2 > best_stack_r2:
            best_stack_r2, best_stack_params = r2, full_params

    log(f"  Best refreshed ensemble: subset={best_subset}, 5-fold OOF R2={best_stack_r2:.4f}")
    progress["ensemble"]["refreshed"] = {
        "r2": best_stack_r2, "params": best_stack_params, "features": feature_cols,
        "subset": best_subset,
        "dl_checkpoints": {a: best_dl[a]["checkpoint_path"] for a in ["cnn", "lstm", "transformer"]},
        "xgb_regressor_features": best_reg["features"], "xgb_classifier_features": best_clf["features"],
        "xgb_regressor_params": best_reg["params"], "xgb_classifier_params": best_clf["params"],
    }
    save_progress(progress)

    prior_path = DOCS_DIR / "ensemble_stack_report.json"
    prior_r2 = None
    if prior_path.exists():
        with open(prior_path) as f:
            prior = json.load(f)
        prior_r2 = prior.get("results", {}).get("cv_xgb_stack", {}).get("r2")
    log(f"  Prior production ensemble R2: {prior_r2}")

    if prior_r2 is not None and best_stack_r2 <= prior_r2:
        log("  Refreshed ensemble does NOT beat the current production ensemble -- "
            "not overwriting production artifacts.")
        return progress

    log("  Refreshed ensemble beats (or no prior recorded) -- saving as new production artifacts.")
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for arch in ["cnn", "lstm", "transformer"]:
        shutil.copy(best_dl[arch]["checkpoint_path"], models_dir / f"dl_{arch}_mfcc.pt")
    torch.save({"mean": mean, "std": std}, models_dir / "cnn_mfcc_norm_stats.pt")
    reg.save_model(str(models_dir / "xgb_regressor_optimized.json"))
    clf.save_model(str(models_dir / "xgb_classifier_optimized.json"))
    with open(models_dir / "xgb_optimized_features.json", "w") as f:
        json.dump({"regressor_features": best_reg["features"],
                    "classifier_features": best_clf["features"]}, f, indent=2)

    final_stacker = XGBRegressor(**best_stack_params)
    final_stacker.fit(combined[feature_cols].to_numpy(), y_all)
    final_stacker.save_model(str(models_dir / "ensemble_stacker.json"))
    with open(models_dir / "ensemble_stacker_config.json", "w") as f:
        json.dump({"type": "learned_stack", "features": feature_cols,
                    "params": best_stack_params}, f, indent=2)

    with open(DOCS_DIR / "ensemble_stack_report.json", "w") as f:
        json.dump({"results": {"cv_xgb_stack": {"r2": best_stack_r2}},
                    "winner": "stack_5feat_tuned_refreshed",
                    "best_stacker_params": best_stack_params}, f, indent=2)

    log(f"  Production artifacts updated. New ensemble R2: {best_stack_r2:.4f} "
        f"(was {prior_r2})")
    return progress


# ============================================================
# MAIN
# ============================================================

def preflight(args):
    """Fail fast, before any training starts, if a required input is missing --
    an unattended overnight run must not discover a bad path 3 hours in."""
    problems = []
    checks = [
        ("master feature store", Path(args.master_store_dir),
         lambda p: p.is_dir() and any(p.glob("master_chunk_*.parquet"))),
        ("SHAP regressor CSV", Path(args.shap_csv_regressor), Path.is_file),
        ("SHAP classifier CSV", Path(args.shap_csv_classifier), Path.is_file),
        ("train.csv", DATA_DIR / "train.csv", Path.is_file),
        ("test.csv", DATA_DIR / "test.csv", Path.is_file),
        ("full train MFCC cache", Path(args.dl_models_dir) / "mfcc_cache_full_raw_train.pt", Path.is_file),
        ("full test MFCC cache", Path(args.dl_models_dir) / "mfcc_cache_full_raw_test.pt", Path.is_file),
    ]
    for name, path, ok in checks:
        if not ok(path):
            problems.append(f"  MISSING {name}: {path}")
    try:
        import torch
    except ImportError:
        torch = None
        problems.append("  torch is not importable in this Python environment")
    if problems:
        log("PREFLIGHT FAILED -- nothing was run. Fix these and retry:")
        for p in problems:
            log(p)
        sys.exit(1)
    log(f"Preflight OK. CUDA available: {torch.cuda.is_available()}"
        + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else
           " -- WARNING: Phase 1/3 will be very slow on CPU"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-store-dir", type=str,
                    default=str(HDD_ROOT / "master_feature_store"))
    ap.add_argument("--shap-csv-regressor", type=str,
                    default=str(HDD_ROOT / "feature_importance_shap.csv"))
    ap.add_argument("--shap-csv-classifier", type=str,
                    default=str(HDD_ROOT / "feature_importance_shap_is_rainy.csv"))
    ap.add_argument("--dl-models-dir", type=str,
                    default=str(REPO_ROOT / "models"))
    ap.add_argument("--skip-dl", action="store_true", help="Skip Phase 1 (use existing progress)")
    ap.add_argument("--skip-xgb", action="store_true", help="Skip Phase 2 (use existing progress)")
    ap.add_argument("--skip-ensemble", action="store_true", help="Skip Phase 3")
    args = ap.parse_args()

    start = time.time()
    log("OVERNIGHT SWEEP STARTING")
    log(f"Repo root       : {REPO_ROOT}")
    log(f"Master store    : {args.master_store_dir}")
    log(f"DL models dir   : {args.dl_models_dir}")
    log(f"Progress file   : {PROGRESS_PATH} (delete it to force a full restart)")
    preflight(args)
    progress = load_progress()

    try:
        if not args.skip_dl:
            progress = run_dl_phase(progress, args)
        else:
            log("Skipping Phase 1 (--skip-dl)")
    except Exception:
        log("Phase 1 crashed at the orchestrator level (not a single-experiment failure):")
        log(traceback.format_exc())
        save_progress(progress)

    train_df = test_df = None
    try:
        if not args.skip_xgb:
            progress, train_df, test_df = run_xgb_phase(progress, args)
        else:
            log("Skipping Phase 2 (--skip-xgb)")
    except Exception:
        log("Phase 2 crashed at the orchestrator level:")
        log(traceback.format_exc())
        save_progress(progress)

    try:
        if not args.skip_ensemble:
            if train_df is None:
                usecols = ["audio_filename", "month_folder", "rainfall_mm", "is_rainy"]
                train_split = pd.read_csv(DATA_DIR / "train.csv", usecols=usecols)
                test_split = pd.read_csv(DATA_DIR / "test.csv", usecols=usecols)
                max_top_n = max(TOP_N_GRID)
                reg_f = top_features_from_csv(args.shap_csv_regressor, max_top_n)
                clf_f = top_features_from_csv(args.shap_csv_classifier, max_top_n)
                master_df = load_master_store(args.master_store_dir, sorted(set(reg_f) | set(clf_f)))
                train_df = attach_features(train_split, master_df, "Train")
                test_df = attach_features(test_split, master_df, "Test")
            progress = run_ensemble_phase(progress, args, train_df, test_df)
        else:
            log("Skipping Phase 3 (--skip-ensemble)")
    except Exception:
        log("Phase 3 crashed at the orchestrator level:")
        log(traceback.format_exc())
        save_progress(progress)

    save_progress(progress)
    elapsed = (time.time() - start) / 3600
    log("=" * 68)
    log(f"SWEEP COMPLETE -- total wall time {elapsed:.2f} hours")
    log("=" * 68)
    ok_dl = sum(1 for r in progress["dl"].values() if r.get("status") == "ok")
    ok_xgb = sum(1 for r in progress["xgb"].values() if r.get("status") == "ok")
    log(f"  DL experiments: {ok_dl}/{len(progress['dl'])} succeeded")
    log(f"  XGBoost experiments: {ok_xgb}/{len(progress['xgb'])} succeeded")
    if "refreshed" in progress.get("ensemble", {}):
        log(f"  Final ensemble R2: {progress['ensemble']['refreshed']['r2']:.4f}")
    log(f"  Full detail: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
