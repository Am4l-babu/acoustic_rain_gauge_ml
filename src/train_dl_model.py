"""
SARID-style Deep Learning Ablation (Stage 7)
================================================
Replicates SARID's feature x architecture ablation study on our data,
scoped to what their own paper's results and our disk/compute budget
justify: MFCC only (their best-performing feature by a clear margin) across
CNN / LSTM / Transformer, regression-only (predicting rainfall_mm directly
-- our classifier is already solid at AUC 0.893, the actual gap is the
weak Stage 4 regressor at R^2=0.155).

Pilot mode (default) trains on a random subset to get a fast, cheap
comparison across architectures before committing to a multi-hour full run.
Reuses the existing per-campaign train/test split from
data/processed/train.csv and test.csv (Stage 3 output) -- no new sampling
logic beyond a fixed-seed random subsample for the pilot.

MFCC is precomputed ONCE per split (shared across all architectures) via
dl_dataset.precompute_mfcc(), not recomputed per epoch. Source audio lives
on a mechanical HDD (F:\\arg_dataset_unzip); a first attempt at on-the-fly
per-epoch random-shuffle loading measured ~55ms/file random access vs
~7.6ms/file sequential -- a shuffled 40,000-row epoch would cost ~37 minutes
of I/O alone, repeated every epoch, which is infeasible. Precomputing once
in sequential order and caching in memory avoids this entirely.

Run:
    python src/train_dl_model.py --pilot --models cnn,lstm,transformer --epochs 20
    python src/train_dl_model.py --full --models transformer --epochs 40
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dl_dataset import MFCCExtractor, precompute_mfcc
from dl_models import MODEL_REGISTRY

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR   = REPO_ROOT / "docs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PILOT_TRAIN_N = 40_000
PILOT_TEST_N  = 10_000
RANDOM_SEED   = 42


def load_split(name: str, n: int | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def evaluate(model, loader) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for mfcc, y in loader:
            mfcc = mfcc.to(DEVICE)
            pred = model(mfcc).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    preds = np.clip(np.concatenate(preds), 0, None)
    targets = np.concatenate(targets)
    return {
        "rmse": float(mean_squared_error(targets, preds) ** 0.5),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds)),
    }


def train_one_model(name: str, train_loader, test_loader, epochs: int, lr: float) -> dict:
    print(f"\n{'='*68}\n  Training: MFCC + {name.upper()}\n{'='*68}")

    model = MODEL_REGISTRY[name]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Full-scale training does ~30x more optimizer steps per epoch than the
    # pilot (607K vs 40K rows at the same batch size) with the same fixed lr
    # -- measured directly: the full Transformer run's test R2 oscillated
    # between -0.59 and +0.11 across 40 epochs despite train_loss decreasing
    # smoothly throughout, the signature of a step size too large to stay
    # stable over that many updates. Cosine decay brings lr toward 0 by the
    # final epoch regardless of how many steps/epoch, so pilot and full runs
    # both end up stable rather than only the (fewer-steps) pilot being lucky.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.SmoothL1Loss()

    history = []
    best_r2 = -float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for mfcc, y in train_loader:
            mfcc, y = mfcc.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(mfcc)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * mfcc.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start
        metrics = evaluate(model, test_loader)
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "epoch_seconds": epoch_time, "lr": scheduler.get_last_lr()[0], **metrics})

        print(f"  Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | "
              f"R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f} | "
              f"MAE={metrics['mae']:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | {epoch_time:.1f}s/epoch")

        if epoch == 1:
            eta_min = epoch_time * epochs / 60
            print(f"  [ETA] first epoch took {epoch_time:.1f}s -> "
                  f"~{eta_min:.1f} min for all {epochs} epochs of this model")

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"dl_{name}_mfcc.pt")
    print(f"  Saved best checkpoint (R2={best_r2:.4f}): "
          f"{MODELS_DIR / f'dl_{name}_mfcc.pt'}")

    return {"model": name, "feature": "mfcc", "best_r2": best_r2, "history": history}


def main():
    parser = argparse.ArgumentParser(description="SARID-style MFCC ablation")
    parser.add_argument("--models", default="cnn,lstm,transformer",
                         help="Comma-separated subset of: cnn,lstm,transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=None,
                         help="Workers for MFCC precompute (0 on Windows, 8 on Linux by default)")
    parser.add_argument("--audio-root", type=str, default=None,
                         help="Override arg_dataset_unzip location (auto-detected from this script's own drive/mount by default)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pilot", action="store_true", default=True,
                        help=f"Train on a random {PILOT_TRAIN_N:,}/{PILOT_TEST_N:,} "
                             "train/test subset (default)")
    group.add_argument("--full", action="store_true",
                        help="Train on the full train/test sets (multi-hour)")
    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = 0 if platform.system() == "Windows" else 8

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pilot = not args.full
    train_n = PILOT_TRAIN_N if pilot else None
    test_n = PILOT_TEST_N if pilot else None

    print(f"\nLoading {'pilot subset' if pilot else 'FULL'} train/test splits...")
    train_df = load_split("train", train_n)
    test_df = load_split("test", test_n)
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    print(f"  Train rainy: {(train_df['rainfall_mm'] > 0).mean():.1%} | "
          f"Test rainy: {(test_df['rainfall_mm'] > 0).mean():.1%}")

    print("\nPrecomputing MFCC once per split (shared across all architectures)...")
    extractor = MFCCExtractor().to(DEVICE)
    # "_raw" distinguishes these from older cache files computed by the
    # previous MFCCExtractor, which divided each clip by its own peak value
    # before caching -- that erased absolute loudness irreversibly, so old
    # cache files can't be reused/patched, only recomputed from raw audio.
    mode_tag = ("full" if not pilot else "pilot") + "_raw"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_cache_path = MODELS_DIR / f"mfcc_cache_{mode_tag}_train.pt"
    test_cache_path = MODELS_DIR / f"mfcc_cache_{mode_tag}_test.pt"

    if train_cache_path.exists():
        print(f"  Found cached train MFCC -> {train_cache_path}, loading instead of recomputing...")
        cache = torch.load(train_cache_path)
        train_mfcc, train_y = cache["mfcc"], cache["y"]
        print(f"  Train MFCC cache: {tuple(train_mfcc.shape)} (loaded from disk)")
    else:
        t0 = time.time()
        train_mfcc, train_y = precompute_mfcc(train_df, extractor, DEVICE,
                                               batch_size=256, num_workers=args.num_workers,
                                               audio_root=args.audio_root,
                                               checkpoint_dir=MODELS_DIR / f"mfcc_checkpoints_{mode_tag}_train")
        print(f"  Train MFCC cache: {tuple(train_mfcc.shape)} in {time.time()-t0:.1f}s")
        torch.save({"mfcc": train_mfcc, "y": train_y}, train_cache_path)
        print(f"  Saved train MFCC cache -> {train_cache_path} (resume point if training crashes later)")

    if test_cache_path.exists():
        print(f"  Found cached test MFCC -> {test_cache_path}, loading instead of recomputing...")
        cache = torch.load(test_cache_path)
        test_mfcc, test_y = cache["mfcc"], cache["y"]
        print(f"  Test MFCC cache : {tuple(test_mfcc.shape)} (loaded from disk)")
    else:
        t0 = time.time()
        test_mfcc, test_y = precompute_mfcc(test_df, extractor, DEVICE,
                                             batch_size=256, num_workers=args.num_workers,
                                             audio_root=args.audio_root,
                                             checkpoint_dir=MODELS_DIR / f"mfcc_checkpoints_{mode_tag}_test")
        print(f"  Test MFCC cache : {tuple(test_mfcc.shape)} in {time.time()-t0:.1f}s")
        torch.save({"mfcc": test_mfcc, "y": test_y}, test_cache_path)
        print(f"  Saved test MFCC cache -> {test_cache_path} (resume point if training crashes later)")

    print("\nApplying global per-coefficient normalization (train-set-derived, no test leakage)...")
    print("  This replaces the old per-clip peak normalization, which erased the")
    print("  absolute loudness differences between clips that SHAP found were the")
    print("  strongest rainfall_mm predictors in the scalar-feature pipeline.")
    mfcc_mean = train_mfcc.mean(dim=(0, 2), keepdim=True)
    mfcc_std = train_mfcc.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    train_mfcc = (train_mfcc - mfcc_mean) / mfcc_std
    test_mfcc = (test_mfcc - mfcc_mean) / mfcc_std
    print(f"  Per-coefficient mean/std shape: {tuple(mfcc_mean.shape)} (computed once from train, applied to both splits)")

    train_loader = DataLoader(TensorDataset(train_mfcc, train_y), batch_size=args.batch_size,
                               shuffle=True, num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(TensorDataset(test_mfcc, test_y), batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    models_to_run = [m.strip() for m in args.models.split(",")]
    results = []
    run_start = time.time()
    for name in models_to_run:
        result = train_one_model(name, train_loader, test_loader, args.epochs, args.lr)
        results.append(result)

    print(f"\n{'='*68}\n  ABLATION SUMMARY (mode={'pilot' if pilot else 'full'})\n{'='*68}")
    print(f"  {'Model':<15}{'Feature':<10}{'Best R2':<10}")
    for r in results:
        print(f"  {r['model']:<15}{r['feature']:<10}{r['best_r2']:<10.4f}")
    print(f"  Baseline (Stage 4 XGBoost regressor): R2=0.155")
    print(f"  Total wall time: {(time.time() - run_start)/60:.1f} min")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = "dl_ablation_pilot_report.json" if pilot else "dl_ablation_full_report.json"
    with open(DOCS_DIR / out_name, "w") as f:
        json.dump({"mode": "pilot" if pilot else "full", "results": results}, f, indent=2)
    print(f"  Saved: {DOCS_DIR / out_name}")


if __name__ == "__main__":
    main()
