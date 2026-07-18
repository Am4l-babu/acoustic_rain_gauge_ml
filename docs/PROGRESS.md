# Project Progress — Acoustic Rain Gauge ML

_Last updated: 2026-07-16_

This file tracks where the project stands: what's done, what's in progress, and what's next. For full technical detail on the stage-by-stage history, see [README.md](../README.md#-roadmap).

**Current state:** classifier AUC-ROC **0.887**; regressor **R² = 0.5429** (learned stacking ensemble, Stage 9). Edge firmware (ESP32-S3 + INMP441) built and streaming. Stages 1-11 complete.

**Active work:** Phase A of the [Roadmap](ROADMAP.md) — cheap, high-information experiments that need no new data. See [Next Steps](#-next-steps) below.

**Research grounding:** [DEEP_RESEARCH_ANALYSIS.md](DEEP_RESEARCH_ANALYSIS.md) (hidden inferences, physics, ranked directions) · [ROADMAP.md](ROADMAP.md) · [RESEARCH_PAPER_ANALYSIS.md](RESEARCH_PAPER_ANALYSIS.md) (172-paper library).

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

### Full-scale training — COMPLETE, with an unresolved finding ⚠️

Full-scale runs (607,673 rows) were executed across the Stage 10 sweep. **Every architecture performed worse at full scale than at its own 40k pilot** — the opposite of the expected trend:

| Model | Pilot R² (40k) | Full-scale best R² (607k) |
|---|---|---|
| CNN | 0.317 | 0.306 → 0.095 by epoch 40 (declined) |
| Transformer | 0.347 | 0.111 (epoch 7) → −0.108 by epoch 40 (declined) |

A learning-rate schedule was added as a partial fix; `diagnose_dl_stability.py` measured that lower-LR/larger-batch is more *stable* (lower epoch-to-epoch variance) but did **not** resolve the underlying problem. **This remains open** — see the new hypothesis in Next Steps below.

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

### Follow-ups from Stage 8 — resolved since

- ~~Hurdle-model variant not yet tried~~ → **tried, and it lost** (see Stage 9).
- ~~No fusion yet with Stage 7's DL embeddings~~ → **done, and it's the project's best result** (see Stage 9).
- Still open: not merged into `train_model.py`/`predict.py` as the default feature source (still a standalone comparison script). `train_optimized_model.py` also never persists its trained models to disk, which forces `ensemble_predict.py` to retrain them from scratch — noted in its own code comments.

---

## ✅ Stage 9 — Ensembling & Stacking — COMPLETE (2026-07-14) — **best result in the project**

**Goal:** the four model families each plateau around R² 0.22-0.28 individually. If they make *partially independent* errors, combining them should beat any one of them.

They do. Results on the identical 607,673/151,927 split ([`reports/ensemble_cnn_xgb_report.json`](reports/ensemble_cnn_xgb_report.json), [`reports/ensemble_stack_report.json`](reports/ensemble_stack_report.json)):

| Configuration | R² | RMSE | MAE |
|---|---|---|---|
| CNN alone | 0.277 | 0.319 | 0.107 |
| LSTM alone | 0.219 | 0.331 | 0.092 |
| Transformer alone | 0.268 | 0.321 | 0.075 |
| XGBoost alone | 0.226 | 0.330 | 0.104 |
| Simple 50/50 average | 0.314 | 0.311 | 0.103 |
| Best weighted blend (w_cnn=0.6) | 0.316 | 0.310 | 0.103 |
| **Learned XGBoost stacking meta-model** | **0.5429** | — | — |

Winner: `stack_5feat_tuned_refreshed` — a 5-feature XGBoost meta-model over the base models' predictions (`max_depth=5`, `n_estimators=300`, `lr=0.02`, `subsample=0.7`), 5-fold CV, then refreshed via the Stage 10 input-subset sweep.

**Hurdle model — tried twice, lost twice:** hard-gate R² **−0.097**, soft-gate **0.076**, vs 0.226 for a single always-on regressor. A hard-gate variant on the stacked features reached 0.412 — still below the plain soft stack.

> ⚠️ **Important caveat on how to read that result.** It refutes gating on **rain/no-rain** specifically. It does *not* refute mixture-of-experts in general. Per Lee & Zawadzki (2005), the gate that matters physically is rain **regime** (convective/stratiform/DSD-family) — a rain/no-rain gate adds classifier errors without reducing the DSD variance that actually causes the error floor, so it multiplies error without buying anything. See [DEEP_RESEARCH_ANALYSIS.md §2.3](DEEP_RESEARCH_ANALYSIS.md#23-the-dsd-ceiling--why-a-single-regression-function-cannot-win). The stack's unusually large gain over its base models is arguably evidence that it's *already* doing implicit regime-conditioning — which suggests making it explicit should help further.

---

## ✅ Stage 10 — Automated Sweep Infrastructure — COMPLETE (2026-07-14)

**`src/training/run_sweep.py`** — a resume-safe, failure-tolerant orchestrator for unattended overnight experiment batches: 59 experiments across three phases (DL hyperparameter grid × XGBoost configs × ensemble input-subset search). Results stream incrementally to [`reports/sweep_progress.json`](reports/sweep_progress.json), so a crash mid-run costs one experiment rather than the night. A new production ensemble is promoted **only** if it measurably beats the deployed one — which is how the current R² = 0.5429 stacker was found.

Per-configuration DL logs live in `docs/sweep_log_{model}_lr{lr}_bs{bs}.txt` and `docs/reports/dl_ablation_full_report_*.json` (18 configs: 3 architectures × 3 learning rates × 2 batch sizes).

---

## ✅ Stage 11 — Edge Hardware Bring-Up — COMPLETE (2026-07-15)

**`firmware/xiao_esp32s3_inmp441_test/`** — PlatformIO project migrating acoustic sensing from the original analog electret + LM393 module to a **Seeed XIAO ESP32-S3 + INMP441 I2S MEMS mic**: wiring/bring-up, WAV capture over serial, live Wi-Fi audio streaming to the trained ensemble, and `MODE_DUAL_COMPARE` / `MODE_DUAL_LEVEL_METER` for validating the sensor swap side-by-side before retiring the analog module.

**Why the digital mic matters beyond convenience:** the INMP441 has a stable, reproducible absolute sensitivity (−26 dBFS ±1 dB) across units. Since rainfall amount is encoded largely in **absolute acoustic level** (the same reason per-clip peak normalization was removed in Stage 7), unit-to-unit gain consistency is a *correctness* requirement — an analog module's gain variation would masquerade as rainfall variation across deployments.

GPIO choice (7/8/9) deliberately avoids every boot-strapping pin and the native-USB pins, so serial monitoring and reflashing always stay usable, leaving D0–D7 free for a tipping-bucket reed switch, SD card, or status LED.

---

## ⏭️ Next Steps

Full plan: [ROADMAP.md](ROADMAP.md). Rationale and ranked scoring: [DEEP_RESEARCH_ANALYSIS.md](DEEP_RESEARCH_ANALYSIS.md).

**Phase A — cheap, high-information, no new data required.** These come first because two of them may change what metric the project should be optimizing at all.

1. **Integration-time scaling analysis** (~1 day, no retraining). Our R² is measured per **10-second clip** against a **0.2 mm-per-tip** bucket. At 2 mm/h that's one tip every ~6 minutes, so roughly **97% of light-rain clips carry a label that reflects where the tip boundary fell** — not the rain during those 10 seconds. That's label quantization noise, and no model can fit it because it isn't a function of the audio. Xavier et al. (2024) got R² 0.62 → **0.85+** from the same model purely by aggregating to hourly; Joss & Waldvogel (1969) and Lee & Zawadzki (2005) both derive why. We have `timestamp` on every row: group the *existing* test-set predictions by campaign, resample to 1/5/15 min and 1/3 h, plot R² vs integration time. Either outcome is informative and publishable.
2. **Leave-one-campaign-out CV** (compute only). Train on 18 campaigns, test on the 19th, rotate. Every published result in this field is single-site; **nobody else has the data to measure cross-deployment generalization.** Note the concern this tests: our top *regression* features (mel bands 7-8 ≈ 400-572 Hz) sit in exactly the band Xavier et al. **rejected as overfitting-prone**, choosing a wider, lower-correlating band specifically to buy site-transferability. Per-campaign chronological splitting protects against *temporal* leakage but not against *surface/site* non-transferability.
3. **Physics-band features** (~1 day). Add integrated Welch-PSD scalars over 0–797 Hz and 1641–2719 Hz (`nperseg=1024`) — the two windows Xavier's frequency sweep found, which independently match our own SHAP-top bands.
4. **Spectral kurtosis + modulation spectrum** (~2 days). Separates impulsive rain from stationary wind *within the same band* — no energy feature can do this. Absent from the entire 172-paper library. Targets our documented dominant false-positive mode (loud non-rain sounds).
5. **Per-campaign dry-ambient calibration** (~1 day). We have 666,310 dry clips as a free per-campaign ambient reference. Following Ma & Nystuen's self-calibration, normalize per campaign against dry-clip ambient rather than per clip against its own peak — removes cross-campaign mic/gain/mounting differences without touching within-campaign amplitude, which is the signal.

**Phase B — new structure and comparisons.**

6. **Run our pipeline on the public SARID dataset.** Calibrates our 0.5429 against their 0.765/0.787 on *their* data. If we score near theirs, our field-data number becomes quantified evidence that field conditions are harder — turning an apparent shortfall into the actual contribution. If we don't, we've found a real deficiency cheaply.
7. **Regime-conditioned mixture-of-experts** — see the Stage 9 caveat above.
8. **Resolve the DL degradation with a new hypothesis.** Current explanation is a learning-rate/step-count mismatch, and the LR schedule only partially fixed it. **Alternative worth testing:** the pilot is a random 40k subsample **across all campaigns** (IID-shuffled), while the full run is 607k rows with **per-campaign chronological structure** and campaigns ranging from ~9% to ~93% rainy. Those are not the same distribution — the pilot is a *balanced, decorrelated* view and the full run is a *clustered, heteroscedastic* one. If so it isn't an optimizer bug at all; more data made the problem harder by bringing campaign distribution shift with it. Two cheap diagnostics settle it: (a) shuffled vs chronological 607k pass; (b) 40k from a single campaign vs 40k across all.

**Phase C — blocked on new data. Start the unblock now.**

9. **Put an anemometer on the next campaign.** We record **zero** wind data. Six papers in the library say that's a problem (Kochendorfer 2017: unshielded gauges catch <50% above 5 m/s, and wind+temperature alone corrects bias −27%→−4%; Pensieri 2015: 4 m/s erases drizzle's acoustic signature; Habib 1999: the correction must be applied at ~1-min resolution or it *overestimates* the bias). This is data collection, not modeling, and it gates the two highest-ceiling ideas in the analysis (wind-corrected labels; wind estimated from the same audio).

> **On why wind features haven't obviously helped anyone:** Monti & Ntalampiras found that *naively concatenating* meteorological parameters did **not** improve performance. That isn't a contradiction of the six papers above — it means wind enters **multiplicatively, not additively**. Wind doesn't add to the rainfall signal; it modulates the transfer function between rainfall and sound *and* corrupts the label via gauge undercatch. Concatenating wind as feature #176 asks the model to learn an interaction from a main effect. Use it as a **label correction** or a **multiplicative gate** instead.

---

## 📌 Key Facts for Future Reference

- Audio: 8000 Hz, mono, exactly 10.0s (80,000 samples) for the `10-15s`-duration training subset — 607,673 train / 151,927 test rows.
- MFCC shape per clip: `(40 coefficients, 157 time frames)`.
- Hardware: RTX 4060 Laptop GPU (8GB VRAM); source audio (`F:\arg_dataset_unzip`) is a mechanical HDD — sequential reads ~7.6ms/file, random reads ~55ms/file (7.3× penalty), hence the precompute-once-cache-in-memory design.
- `precompute_mfcc()` must re-run per training invocation (cache isn't persisted across script runs) — this is the ~53min (pilot) / ~9h (full) fixed cost before any training epoch begins.
