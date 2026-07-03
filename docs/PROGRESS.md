# Project Progress — Acoustic Rain Gauge ML

_Last updated: 2026-07-03_

This file tracks where the project stands: what's done, what's in progress, and what's next. For full technical detail on Stages 1-6, see [README.md](../README.md#-roadmap). This file focuses on **Stage 7 (Deep Learning Ablation)**, which is the active work.

---

## ✅ Past Work (Stages 1-6) — Summary

| Stage | Deliverable | Result |
|---|---|---|
| **1. Data Cleaning** | `src/data_cleaning.py` — discover/timestamp/align/extract, resume-safe | 780,725 clips processed, ~4h22m CPU pipeline |
| **2. EDA** | `notebooks/eda.ipynb` | Surfaced duration confound (short clips in early campaigns) |
| **3. Feature Engineering** | `src/feature_extraction.py` | Dropped confounded clips + 204-row sensor-artifact outliers (rainfall_mm≈655mm); per-campaign chronological split (fixed 8.9%/39.4% rainy-rate mismatch → 15.1%/14.6%) |
| **4. Model Training** | `src/train_model.py` | XGBoost classifier: **AUC-ROC 0.883**; XGBoost regressor: **R²=0.155** (weak — scalar MFCC means lose temporal structure) |
| **5. Evaluation** | `src/evaluate_model.py` | Threshold tuning (0.5→0.78) lifted precision 0.46→0.70; light HP search improved AUC to **0.893**; false positives are loud non-rain sounds |
| **6. Real-Time Inference** | `src/predict.py` | End-to-end single-clip scoring at tuned threshold, OOD flagging for non-10-15s clips |

**Core problem motivating Stage 7:** the Stage 4 regressor (R²=0.155) is weak because XGBoost was fed *time-averaged* MFCC means — a single scalar per coefficient — discarding all temporal structure within a 10s clip. The SARID paper (reference implementation, `external_repos/SARID/`, gitignored) solves the same audio→rainfall problem with **R²=0.765** by feeding the full 2D MFCC time-series into a CNN/LSTM/Transformer.

---

## 🔄 Current Stage — Stage 7: Deep Learning Ablation (MFCC × CNN/LSTM/Transformer)

**Goal:** Replicate SARID's feature×architecture ablation on our data to see if sequence models recover the temporal structure XGBoost throws away, before committing to a multi-hour full-scale run.

### What's built (all done)
- **`src/dl_dataset.py`** — `WaveformDataset` (raw waveform + target via `soundfile`) + `MFCCExtractor` (GPU-batched `torchaudio.transforms.MFCC`, per-clip peak-normalized) + `precompute_mfcc()` (reads every clip once in DataFrame order, caches MFCC tensor in memory — avoids the measured 55ms/file random-read penalty on the source HDD vs 7.6ms/file sequential).
- **`src/dl_models.py`** — `CNNRegressor`, `LSTMRegressor`, `TransformerRegressor`, reimplemented cleanly from the SARID paper's Section 3.2 description (their actual code has unfilled template placeholders and shape bugs, so it wasn't reusable as-is). All take MFCC input `(B, 40, 157)` → scalar `rainfall_mm`.
- **`src/train_dl_model.py`** — training loop (`SmoothL1Loss`, Adam), pilot mode (40k/10k random subset) vs full mode (607k/152k), per-epoch timing + R²/RMSE/MAE, best-checkpoint saving, JSON report output.
- **Windows fix:** `DataLoader(num_workers=0)` auto-detected on Windows (multiprocessing spawn requires a `__main__` guard that doesn't fit this pipeline's structure) — confirmed via `src/diagnose_dl.py` step-by-step diagnostic.

### Pilot ablation — COMPLETE ✅ (2026-07-03)

Trained on a fixed-seed 40,000-row train / 10,000-row test subsample, 20 epochs each, MFCC precomputed once and cached (43.3 min train + 9.6 min test, one-time cost per run).

| Model | Best R² | RMSE | MAE | Epoch time | vs. Stage 4 baseline (R²=0.155) |
|---|---|---|---|---|---|
| **Transformer** | **0.3469** | 0.3257 | 0.1184 | ~15s/epoch | **2.24× better** |
| CNN | 0.3173 | 0.3330 | 0.1233 | ~8.5s/epoch | 2.05× better |
| LSTM | 0.2705 | 0.3443 | 0.1296 | ~7-12s/epoch | 1.74× better |

**Total pilot wall time: 12.0 min** (training only; precomputation ~53 min happened once, not repeated per model).

**Conclusion:** All three architectures beat the XGBoost baseline substantially, confirming the hypothesis — temporal structure matters. Transformer is the current best performer. Artifacts saved:
- `models/dl_cnn_mfcc.pt`, `models/dl_lstm_mfcc.pt`, `models/dl_transformer_mfcc.pt`
- `docs/dl_ablation_pilot_report.json` (full per-epoch history)

### Decision point (open, awaiting go-ahead)

Full-scale training on the full 607,673-row train set was estimated at **~9-10 hours** (6.5h train MFCC precompute + 2.5h test precompute + 0.5-1h training, one-time I/O cost since source audio is on a mechanical HDD). Not yet started — a mid-scale run (~150k rows, ~2-3h) was proposed as a cheaper way to check whether R² keeps improving with more data before committing to the full 10-hour run.

---

## ⏭️ Next Steps

1. **Decide pilot → scale-up path**: run mid-scale (150k, ~2-3h) to check R² trend, or commit directly to full-scale (607k, ~9-10h) on the Transformer architecture.
2. **Full-scale training** (once triggered): 30-50 epochs with early stopping on validation R² plateau, Transformer only (pilot winner).
3. **Evaluate & document**: compare full-scale R²/RMSE/MAE against Stage 4 baseline and SARID's reported 0.765; save comparison report to `docs/`.
4. **Update README**: add Stage 7 to the roadmap table with final numbers; fix the "sample rate unconfirmed" note (confirmed: 8kHz, mono, 10.0s exactly, verified in this stage's investigation).
5. **Optional**: wire the winning DL model into `src/predict.py` as an alternative regressor to the XGBoost one.

---

## 📌 Key Facts for Future Reference

- Audio: 8000 Hz, mono, exactly 10.0s (80,000 samples) for the `10-15s`-duration training subset — 607,673 train / 151,927 test rows.
- MFCC shape per clip: `(40 coefficients, 157 time frames)`.
- Hardware: RTX 4060 Laptop GPU (8GB VRAM); source audio (`F:\arg_dataset_unzip`) is a mechanical HDD — sequential reads ~7.6ms/file, random reads ~55ms/file (7.3× penalty), hence the precompute-once-cache-in-memory design.
- `precompute_mfcc()` must re-run per training invocation (cache isn't persisted across script runs) — this is the ~53min (pilot) / ~9h (full) fixed cost before any training epoch begins.
