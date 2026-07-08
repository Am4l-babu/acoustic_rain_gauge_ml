# Project Progress — Acoustic Rain Gauge ML

_Last updated: 2026-07-07_

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

## ✅ Stage 8 — Master Feature Store & SHAP Selection — COMPLETE (2026-07-07)

**Goal:** Give the XGBoost regressor (R²=0.155 baseline) a much richer, hand-crafted feature set as a cheaper alternative/complement to the Stage 7 DL path — 175 features across 7 families (time-domain/Teager, spectral incl. flux, MFCC, dense mel-band stats, wavelet, histogram-packet rhythm, onset/tempo), computed in parallel and written to chunked Parquet, then narrowed down via SHAP.

### What's built
- **`src/master_feature_extraction.py`** — supersedes `src/advanced_feature_extraction.py` (63 features); this is a superset, so run one or the other, not both.
- **`src/feature_selection.py`** — XGBoost + `shap.TreeExplainer`, ranks all 175 features by mean |SHAP value|, keeps top-N, saves `optimized_rain_dataset.parquet` + a full importance CSV.

### Bugs caught while integrating the originally-proposed script
1. **Wasteful resampling**: the draft config loaded every clip at 16kHz, forcing librosa to upsample from the true native 8kHz (confirmed in Stage 7). Upsampling fabricates no information above the real 4kHz Nyquist limit and measured ~22ms/clip of pure overhead. Fixed: `TARGET_SR = 8000`.
2. **`spectral_contrast` crashes at 8kHz**: its default band edges (`fmin=200, n_bands=6`) reach 25,600 Hz — past Nyquist at both 8kHz *and* 16kHz, but only raises `librosa.util.exceptions.ParameterError` at the lower rate in practice. Fixed: `n_bands=3` when `sr <= 8000`.
3. **Scan loop ignored `--limit` until after the fact** (caught 2026-07-04, first real run against real audio): the dataset scan called `path.exists()` for all ~780,744 rows across every CSV *before* slicing to `--limit`, so a "2000-clip smoke test" actually did a full-dataset filesystem scan first — on the mechanical HDD this made the smoke test take as long as the full run. Fixed: the scan loop now stops as soon as it's collected `limit` valid tasks.
4. **`NUM_WORKERS = cpu_count()-2` (18 on this machine) crashed under real memory pressure**: with only ~1.9GB free RAM (other apps using the rest of 16GB), spawning 18 worker processes — each importing numpy/scipy/librosa (~200-400MB per worker) — caused several workers to fail with `ImportError: DLL load failed... The paging file is too small`. 2/5 clips silently failed in the first real test. Fixed: default lowered to `min(6, cpu_count()-2)`; added `--workers` override for machines with more free RAM.
5. **Audio path portability**: `audio_full_path` in the CSVs is an absolute path baked in at Stage 1 time (e.g. `F:\arg_dataset_unzip\...`). If the HDD mounts under a different drive letter on another PC, every lookup would silently fail (the script just skips missing files). Fixed: `resolve_audio_path()` anchors on the `arg_dataset_unzip` folder name and remaps to wherever `--audio-root` (or the auto-detected script's own drive) actually is, plus a loud `WARNING` if >0% of checked files are missing.

### ETA — measured against real audio (2026-07-04)
First real end-to-end run (previous numbers below were from a synthetic clip benchmark, since the source HDD wasn't connected at integration time):

| | Result |
|---|---|
| **2000-clip smoke test, 6 workers** | 2000/2000 succeeded, 0 nulls, **31.6ms/clip** (62s total) |
| **Full 780,725 clips, extrapolated at 6 workers** | **~6.9 hours** |

This supersedes the earlier "~1-2h with 18 workers" estimate — that assumed 18 parallel workers, which we now know crashes on this machine under real memory pressure (see bug #4 above). 6 workers is the safe default; raise via `--workers` only on a machine with confirmed free RAM (~0.5GB/worker).

Original synthetic-clip benchmark, kept for reference:

| | Feature compute | + I/O (seq. HDD) | + I/O (random HDD, worst case) |
|---|---|---|---|
| At 8kHz (fixed) | ~30ms/clip | ~38ms/clip | ~85ms/clip |
| At 16kHz (original draft) | ~59ms/clip | ~67ms/clip | ~114ms/clip |

- **Output size**: ~1.5-3GB of Parquet (measured via a representative synthetic write), comfortably inside the ~30GB free on the dataset drive at time of writing.

### New CLI overrides (added for the multi-PC / HDD-migration workflow)
`master_feature_extraction.py` and `feature_selection.py` now accept `--data-dir`, `--output-dir`, `--audio-root`, `--workers` (extraction) and `--store-dir`, `--output-path` (selection). All default to auto-detecting the drive the script itself is running from, so the same code works whether the project lives on `D:`, `F:`, or any other letter a plugged-in HDD gets assigned.

An automated wrapper — `docs/NEW_PC_SETUP/RUN_EXTRACTION.ps1` — chains `pip install` → smoke test → validate output is non-empty → full run, aborting with a clear message at any failure point instead of silently proceeding.

### Full-scale extraction — COMPLETE (2026-07-07, run on Ubuntu i9-13900K, 8 workers)

| | Result |
|---|---|
| **Clips processed** | 780,725 (780,713 after 12 unreadable/short clips dropped) |
| **Wall time** | **4.34 hours** (20.0ms/clip — faster than the 31.6ms/clip Windows-based estimate) |
| **Output** | 157 chunked Parquet files, ~0.77GB total, `master_feature_store/` |
| **Known data quirk** | 16,556 rows in `December_2024_rain_data` are byte-identical duplicates (upstream CSV duplication, harmless — dropped via `drop_duplicates` before any join) |

### SHAP feature selection — COMPLETE, per-target (2026-07-07)

Initial run ranked all 175 features against `rainfall_mm` only and reused that ranking for both models — this **hurt classifier AUC** (0.842 vs 0.883 baseline) because the top regression-ranked features aren't the best rain/no-rain discriminators. Fixed by adding `--target {rainfall_mm, is_rainy}` to `feature_selection.py`, so the classifier and regressor now each get their own top-30 SHAP-ranked feature set:

- Regression-ranked top features: dominated by dense mel-band means (`mel_band_8_mean`, `mel_band_7_mean`, ...), wavelet variance, spectral flux.
- Classification-ranked top features: also mel-band-dominated but a different subset (`mel_band_27_mean`, `mel_band_36_mean`, ...) plus `mfcc_3_mean`, `fd_spectral_contrast_mean`.

Evaluated via new **`src/train_optimized_model.py`**, which joins the master store's features onto `data/processed/train.csv`/`test.csv` (Stage 3's already-filtered, already-split population) rather than trusting `optimized_rain_dataset.parquet` as a standalone train/test source — this keeps the comparison to the Stage 4 baseline apples-to-apples (same 607,673/151,927 row split, same filters).

| Model | Metric | Stage 4 baseline (13 features) | Stage 8 optimized (30 SHAP features, per-target) |
|---|---|---|---|
| Classifier | AUC-ROC | 0.883 | **0.887** |
| Regressor | R² | 0.155 | **0.226** (+46% relative) |

Regressor improvement is substantial; classifier improvement is modest but real (already-strong baseline, less headroom). Artifacts:
- `F:\optimized_rain_dataset.parquet` / `..._is_rainy.parquet` — top-30 feature sets + target
- `F:\feature_importance_shap.csv` / `..._is_rainy.csv` — full 175-feature SHAP rankings per target
- `docs/model_evaluation_report_optimized.json` — final metrics + exact feature lists used

### Not yet done
- Not yet merged into `train_model.py`/`predict.py` as the default feature source (still a standalone comparison script).
- Hurdle-model variant (regressor trained only on rainy samples, not all 780k rows incl. ~85% zeros) not yet tried — likely the next lever on the regression side.
- No fusion yet with Stage 7's DL embeddings (pilot R²=0.3469, still the single biggest lever available).

---

## 📌 Key Facts for Future Reference

- Audio: 8000 Hz, mono, exactly 10.0s (80,000 samples) for the `10-15s`-duration training subset — 607,673 train / 151,927 test rows.
- MFCC shape per clip: `(40 coefficients, 157 time frames)`.
- Hardware: RTX 4060 Laptop GPU (8GB VRAM); source audio (`F:\arg_dataset_unzip`) is a mechanical HDD — sequential reads ~7.6ms/file, random reads ~55ms/file (7.3× penalty), hence the precompute-once-cache-in-memory design.
- `precompute_mfcc()` must re-run per training invocation (cache isn't persisted across script runs) — this is the ~53min (pilot) / ~9h (full) fixed cost before any training epoch begins.
