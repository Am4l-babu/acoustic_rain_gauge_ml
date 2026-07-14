"""
Diagnose Stage 7's full-scale training instability (mid-scale, RAM-safe)
============================================================================
Every full-scale DL architecture scored *worse* than its own 40K-row pilot,
and epoch-by-epoch R2 bounced around instead of climbing smoothly even with
cosine LR decay (see docs/dl_ablation_full_report.json). The natural next
question is whether a lower learning rate / larger batch size stabilizes
training -- but re-running the actual full-scale training (607,673 rows)
requires holding ~19GB of cached MFCC tensors in RAM at once (train_dl_model.py
does a plain torch.load, not mmap), and this machine only has 16GB total --
confirmed directly: attempting it dropped free RAM to ~500MB and the process
was paging heavily rather than training.

Instead, this script slices a random 150,000-row / 30,000-row mid-scale
subset directly out of the existing full-scale MFCC cache via mmap (so the
full 15GB/3.8GB tensors are never materialized in RAM, only the slice is),
then trains CNN under the ORIGINAL config (lr=1e-3, batch=128) and a
CANDIDATE config (lr=3e-4, batch=256) on the *identical* mid-scale data, and
compares epoch-to-epoch R2 stability between the two -- a cheap, RAM-safe
proxy for whether the candidate config would help at full scale, without
needing the other (more powerful, more RAM) machine the original full-scale
runs used.

Run:
    python src/diagnose_dl_stability.py --dl-models-dir F:\\acoustic_rain_gauge_ml\\models
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_models import MODEL_REGISTRY
from train_dl_model import evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MID_TRAIN_N = 40_000
MID_TEST_N = 10_000
SEED = 42


def load_mid_scale_slice(cache_path, n, seed, n_blocks=12):
    """mmap the big cache and copy out `n` rows as a handful of large
    CONTIGUOUS blocks spread across the file, rather than n individually
    scattered random rows. Scattered single-row gather on this drive measured
    as extremely slow (a first attempt at ~180,000 scattered rows didn't
    finish in over an hour); a contiguous block is one sequential disk read,
    and spreading a few blocks across the file still samples different
    campaigns/time-periods for reasonable diversity, at a fraction of the
    seek cost."""
    print(f"    Opening {cache_path} (mmap)...", flush=True)
    cache = torch.load(cache_path, mmap=True)
    mfcc, y = cache["mfcc"], cache["y"]
    total = mfcc.shape[0]
    n = min(n, total)
    block_size = n // n_blocks
    rng = np.random.default_rng(seed)
    max_start = total - block_size
    starts = sorted(rng.choice(max_start, size=n_blocks, replace=False))
    chunks_mfcc, chunks_y = [], []
    for i, s in enumerate(starts):
        print(f"    Block {i+1}/{n_blocks}: rows {s:,}-{s+block_size:,}...", flush=True)
        chunks_mfcc.append(mfcc[s:s + block_size].clone())
        chunks_y.append(y[s:s + block_size].clone())
    return torch.cat(chunks_mfcc), torch.cat(chunks_y)


def train_config(name, train_mfcc, train_y, test_mfcc, test_y, lr, batch_size, epochs=40):
    print(f"\n{'='*68}\n  Config: {name} (lr={lr}, batch_size={batch_size})\n{'='*68}", flush=True)
    train_loader = DataLoader(TensorDataset(train_mfcc, train_y), batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(TensorDataset(test_mfcc, test_y), batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    model = MODEL_REGISTRY["cnn"]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.SmoothL1Loss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
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
        metrics = evaluate(model, test_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "epoch_seconds": time.time() - t0, **metrics})
        print(f"  Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | R2={metrics['r2']:.4f} | "
              f"{time.time()-t0:.1f}s/epoch", flush=True)

    r2_values = [h["r2"] for h in history]
    # Stability window: last 10 epochs, once the LR has decayed substantially
    tail = r2_values[-10:]
    summary = {
        "config": name, "lr": lr, "batch_size": batch_size,
        "best_r2": max(r2_values),
        "final_r2": r2_values[-1],
        "tail_r2_mean": float(np.mean(tail)),
        "tail_r2_std": float(np.std(tail)),
        "full_r2_std": float(np.std(r2_values)),
        "history": history,
    }
    print(f"\n  {name}: best R2={summary['best_r2']:.4f} | final R2={summary['final_r2']:.4f} | "
          f"last-10-epoch std={summary['tail_r2_std']:.4f} (lower = more stable)")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl-models-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    dl_dir = Path(args.dl_models_dir)

    print(f"Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"\nSlicing a mid-scale subset ({MID_TRAIN_N:,} train / {MID_TEST_N:,} test) "
          f"from the full-scale cache as contiguous blocks (RAM-safe, seek-friendly)...", flush=True)
    t0 = time.time()
    train_mfcc, train_y = load_mid_scale_slice(dl_dir / "mfcc_cache_full_raw_train.pt", MID_TRAIN_N, SEED)
    test_mfcc, test_y = load_mid_scale_slice(dl_dir / "mfcc_cache_full_raw_test.pt", MID_TEST_N, SEED + 1)
    print(f"  Train slice: {tuple(train_mfcc.shape)} | Test slice: {tuple(test_mfcc.shape)} "
          f"in {time.time()-t0:.1f}s", flush=True)

    print("\nApplying per-coefficient normalization (mid-scale-train-derived)...", flush=True)
    mean = train_mfcc.mean(dim=(0, 2), keepdim=True)
    std = train_mfcc.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    train_mfcc = (train_mfcc - mean) / std
    test_mfcc = (test_mfcc - mean) / std

    results = []
    results.append(train_config("original (lr=1e-3, batch=128)", train_mfcc, train_y, test_mfcc, test_y,
                                 lr=1e-3, batch_size=128, epochs=args.epochs))
    results.append(train_config("candidate (lr=3e-4, batch=256)", train_mfcc, train_y, test_mfcc, test_y,
                                 lr=3e-4, batch_size=256, epochs=args.epochs))

    print(f"\n{'='*68}\n  STABILITY COMPARISON (mid-scale, {MID_TRAIN_N:,} rows, CNN only)\n{'='*68}")
    print(f"  {'Config':<32}{'Best R2':<10}{'Final R2':<10}{'Last-10 std':<12}")
    for r in results:
        print(f"  {r['config']:<32}{r['best_r2']:<10.4f}{r['final_r2']:<10.4f}{r['tail_r2_std']:<12.4f}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "dl_stability_diagnostic_report.json"
    with open(out_path, "w") as f:
        json.dump({"mid_train_n": MID_TRAIN_N, "mid_test_n": MID_TEST_N, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
