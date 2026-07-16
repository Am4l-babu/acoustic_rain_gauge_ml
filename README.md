# 🌧️ Acoustic Rain Gauge ML

**Can a microphone replace a rain gauge?**

This project trains a machine learning model to estimate rainfall — whether it's raining, and how hard — purely from the *sound* of rain hitting a surface, using a traditional mechanical tipping-bucket gauge as ground truth.

<p align="left">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/status-Stage%208%20complete%20(AUC%200.887%2C%20R²%200.226)-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/dataset-780%2C725%20audio%20clips-orange" alt="Dataset size">
  <img src="https://img.shields.io/badge/span-Dec%202023%20→%20Jun%202026-lightgrey" alt="Time span">
</p>

---

## 💡 The Idea

Rain gauges have measured rainfall the same way for centuries: a tipping bucket that clicks every time a fixed volume of water falls into it. It's reliable, but coarse, mechanical, and needs physical upkeep.

Rain also makes a very distinctive *sound* — the intensity, texture, and frequency content of rain hitting a surface changes with drop size and rainfall rate. This project asks: **if we record that sound and pair it with real tipping-bucket readings at the same moment in time, can a model learn to read rainfall straight off the audio?**

If it works, a microphone becomes a low-maintenance, potentially higher-resolution sensor — either standing in for a mechanical gauge, or cross-validating one.

```
   🎙️  Microphone            🪣  Tipping-bucket gauge
        │                            │
        │  raw audio (.wav)          │  timestamped mm readings (.csv)
        ▼                            ▼
   ┌─────────────────────────────────────────┐
   │        time-aligned, labeled dataset      │
   └─────────────────────────────────────────┘
                        │
                        ▼
              🧠  ML model (XGBoost)
                        │
                        ▼
        "it's raining, ~X mm in this interval"
```

---

## 🔬 How It Works — The Pipeline

Every monthly batch of recordings goes through the same five-step pipeline:

| Step | What happens | Why |
|---|---|---|
| **1. Discover** | Recursively scan the folder for every `.wav` file | Recordings are nested under daily sub-folders, not flat |
| **2. Timestamp** | Regex-parse the recording time out of each filename (e.g. `20240501_143022.wav` → `2024-05-01 14:30:22`) | The audio itself carries no timestamp metadata — the filename is the only clock |
| **3. Align** | `pandas.merge_asof` nearest-neighbor join against the mechanical CSV, **2-minute tolerance** | The microphone and the tipping bucket are two independent clocks that never tick in perfect sync |
| **4. Extract features** | Compute acoustic features per clip — in parallel on CPU, or batched on GPU | Turns raw waveforms into a fixed-size numeric feature vector any ML model can consume |
| **5. Save** | Write `cleaned_aligned_data.csv` + `metadata.json` per folder, with resume-on-restart | Full runs take hours across 780K+ files — restart-safety matters |

Two interchangeable implementations exist for step 4:

- **[`data_cleaning.py`](src/features/data_cleaning.py)** — CPU pipeline. A `multiprocessing.Pool` of 16 workers extracts features with `librosa`/`soundfile`. This is the version the full dataset was actually processed with.
- **[`data_cleaning_gpu.py`](src/features/data_cleaning_gpu.py)** — experimental GPU pipeline. Batches waveforms onto CUDA and computes every feature as a single vectorized `torchaudio` pass (spectrogram → centroid/bandwidth/rolloff/MFCC), aiming for a 4-8x speedup over CPU. Not yet used for a production run.

---

## 🎛️ Features Extracted Per Clip

| Feature | What it captures |
|---|---|
| `rms` | Overall loudness — louder generally means more rain |
| `peak` | Maximum instantaneous amplitude — peak drop intensity |
| `par` | Peak-to-average ratio — separates heavy bursts from steady drizzle |
| `spectral_centroid` | "Center of mass" of the frequency spectrum |
| `spectral_bandwidth` | How spread out the frequency content is |
| `spectral_rolloff` | Frequency below which 85% of the signal's energy sits |
| `zero_crossing_rate` | How noise-like vs. tonal the signal is |
| `energy_variance` | Variance of energy across 50ms frames — steady rain vs. intermittent bursts |
| `mfcc_0` … `mfcc_4` | Mel-frequency cepstral coefficients — overall spectral "texture" |
| `duration_sec` | Clip length (mostly 10–15s; a couple of months are shorter — see [Notes](#-honest-notes)) |

Each row is labeled with `timestamp`, `rainfall_mm` (the target), and `is_aligned` (a data-quality flag).

---

## 📊 Dataset, So Far

The full pipeline has run end-to-end once already:

| | |
|---|---|
| **Monthly folders processed** | 19 (Dec 2023 → Jun 2026) |
| **Total audio clips** | 780,725 |
| **Aligned to a mechanical reading (±2 min)** | 699,145 |
| **Rainy samples** | 114,415 (14.7%) |
| **Dry samples** | 666,310 (85.3%) |
| **Feature extraction errors** | 1 (0.0001%) |
| **Total processing time** | ~4h 22m (CPU pipeline) |

Class imbalance (≈15% rainy) is expected — it doesn't rain most of the time — but it's real and will need explicit handling (class weights / resampling) at training time.

---

## 🛠️ Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/numpy-array%20math-013243?logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-time--series%20alignment-150458?logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/librosa-audio%20features-orange">
  <img src="https://img.shields.io/badge/soundfile-fast%20WAV%20I%2FO-lightgrey">
  <img src="https://img.shields.io/badge/PyTorch%20%2B%20torchaudio-GPU%20pipeline-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-model-4B8BBE">
  <img src="https://img.shields.io/badge/scikit--learn-metrics%20%2F%20preprocessing-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/matplotlib%20%2B%20seaborn-visualization-11557C">
  <img src="https://img.shields.io/badge/PyWavelets-multi--scale%20DSP-lightgrey">
  <img src="https://img.shields.io/badge/SHAP-feature%20selection-8A2BE2">
</p>

| Layer | Tools |
|---|---|
| Audio I/O & DSP | `soundfile`, `librosa`, `torchaudio`, `PyWavelets` |
| Data wrangling | `pandas`, `numpy`, `pyarrow` (Parquet) |
| Parallelism | `multiprocessing.Pool` (CPU) / CUDA batching via `torch` (GPU) |
| Modeling | `xgboost`, `scikit-learn` |
| Feature selection | `shap` (TreeExplainer over XGBoost) |
| Visualization | `matplotlib`, `seaborn` |
| UX | `tqdm` progress bars, JSON metadata, resume-on-restart |

> **Note:** `torch`/`torchaudio` are required to run `data_cleaning_gpu.py` but aren't yet pinned in [requirements.txt](requirements.txt) — add them if you pick up the GPU path.

---

## 📁 Project Structure

```
acoustic_rain_gauge_ml/
│
├── src/
│   ├── data_cleaning.py       # Main pipeline: alignment + CPU feature extraction
│   ├── data_cleaning_gpu.py   # Experimental: batched GPU feature extraction
│   ├── analyze_dataset_v2.py  # Recursive dataset structure/health report
│   ├── dry_run_test.py        # Small-sample sanity check before a full run
│   ├── feature_extraction.py  # Stage 3: duration-filter, per-campaign split, scaling
│   ├── train_model.py         # Stage 4: XGBoost classifier + regressor training
│   ├── evaluate_model.py      # Stage 5: threshold tuning, error analysis, light HP search
│   ├── predict.py             # Stage 6: single-clip real-time inference
│   ├── dl_dataset.py          # Stage 7: waveform Dataset + GPU-batched MFCC extractor
│   ├── dl_models.py           # Stage 7: CNN/LSTM/Transformer regression heads
│   ├── train_dl_model.py      # Stage 7: DL ablation training loop (pilot/full modes)
│   ├── advanced_feature_extraction.py  # 63-feature pilot (superseded by master_feature_extraction.py)
│   ├── master_feature_extraction.py    # Stage 8: 175-feature master store, parallel + resume-safe
│   ├── feature_selection.py            # Stage 8: SHAP-ranked top-N feature selection
│   └── utils.py               # (stub) shared helpers — TODO
│
├── data/
│   ├── raw/                   # Original audio + tipping-bucket CSVs (kept off-repo)
│   ├── processed/             # Cleaned, aligned, feature-extracted CSVs
│   └── external/              # Any external datasets or metadata
│
├── docs/
│   ├── dataset_analysis_report_v2.txt   # Full recursive dataset scan
│   └── PROGRESS.md            # Stage 7 status: pilot results, next steps, key facts
│
├── models/                    # Trained model artifacts + feature_scaler.pkl
├── notebooks/
│   └── eda.ipynb              # Stage 2: combined-dataset exploration
└── requirements.txt
```

---

## 🗺️ Roadmap

- [x] **Stage 1 — Data Cleaning**: discover, timestamp, align, extract features, resume-safe — *done, 780,725 clips processed*
- [x] **Stage 2 — Exploratory Data Analysis** ([`notebooks/eda.ipynb`](notebooks/eda.ipynb)): combined all monthly CSVs, plotted rainfall & feature distributions, correlation heatmap, rainy-vs-dry comparisons — *surfaced the duration confound noted below*
- [x] **Stage 3 — Feature Engineering** ([`src/features/feature_extraction.py`](src/features/feature_extraction.py)): dropped the confounded short clips, filtered 204 rows of a repeating ~655mm sensor artifact (see below), split chronologically **per recording campaign** (a single global time cutoff was tried first and produced a severe train/test rainy-rate mismatch — 8.9% vs 39.4% — since campaigns like Feb-Mar 2026 are ~93% rainy; per-campaign splitting fixed it to 15.1% / 14.6%), computed class weights, fit a scaler on train only
- [x] **Stage 4 — Model Training** ([`src/training/train_model.py`](src/training/train_model.py)): Logistic Regression baseline, XGBoost rain/no-rain classifier (**AUC-ROC 0.883**, recall 0.76, precision 0.46), XGBoost regressor for rainfall amount (**R² 0.155**) — confusion matrix and feature-importance chart in [`docs/`](docs/); `mfcc_2` is the single most important feature for rain detection
- [x] **Stage 5 — Evaluation** ([`src/evaluation/evaluate_model.py`](src/evaluation/evaluate_model.py)): per-campaign breakdown showed the weak Stage 4 precision was mostly a **threshold problem, not a modeling one** — raising the decision threshold from 0.5 to ~0.78 lifts precision 0.46 → 0.70 at a recall cost of 0.76 → 0.62, no retraining needed; error analysis found false positives are acoustically loud non-rain sounds (elevated `spectral_rolloff`/`peak`) that resemble rain; a light train-only-validated hyperparameter search improved test AUC 0.883 → **0.893**
- [x] **Stage 6 — Real-Time Inference** ([`src/inference/predict.py`](src/inference/predict.py)): scores a single WAV clip end-to-end — extracts the same features used in training, classifies at the Stage 5 tuned threshold (read live from `docs/reports/stage5_evaluation_report.json`, not hardcoded), estimates rainfall in mm, and flags predictions as out-of-distribution if the clip isn't `10-15s`

  ```bash
  python src/inference/predict.py path/to/clip.wav                # tuned threshold + mm estimate
  python src/inference/predict.py path/to/clip.wav --threshold 0.5 # override the operating threshold
  python src/inference/predict.py path/to/clip.wav --no-mm         # classification only
  ```

- [x] **Stage 7 — Deep Learning Ablation** ([`src/training/train_dl_model.py`](src/training/train_dl_model.py), pilot complete; full-scale training pending): the Stage 4 regressor (R²=0.155) was weak because XGBoost was fed *time-averaged* MFCC means — a single scalar per coefficient — discarding all temporal structure within a clip. Reimplemented the SARID paper's feature×architecture ablation (their code has unfilled template placeholders and shape bugs, so it wasn't reusable as-is): a GPU-batched MFCC extractor ([`src/dl/dl_dataset.py`](src/dl/dl_dataset.py)) feeds the full `(40, 157)` time-series into a CNN, LSTM, and Transformer regressor ([`src/dl/dl_models.py`](src/dl/dl_models.py)), one clip's MFCC precomputed once and cached in memory (source audio lives on a mechanical HDD — random-order reads measured 7.3× slower than sequential, making per-epoch on-the-fly loading infeasible). Pilot run (40,000 train / 10,000 test rows, 20 epochs each) confirmed the hypothesis — every architecture beat the baseline by 1.7-2.2×:

  | Model | Best R² | vs. baseline (0.155) |
  |---|---|---|
  | **Transformer** | **0.3469** | **2.24×** |
  | CNN | 0.3173 | 2.05× |
  | LSTM | 0.2705 | 1.74× |

  See [`docs/PROGRESS.md`](docs/PROGRESS.md) for full pilot metrics, timing, and the next decision point (mid-scale vs. full 607K-row training run, ETA ~2-3h vs. ~9-10h on a mechanical HDD).

  ```bash
  python src/training/train_dl_model.py --pilot --models cnn,lstm,transformer --epochs 20  # pilot (fast)
  python src/training/train_dl_model.py --full --models transformer --epochs 40           # full-scale (multi-hour)
  ```

- [x] **Stage 8 — Master Feature Store & Selection** ([`src/features/master_feature_extraction.py`](src/features/master_feature_extraction.py), [`src/features/feature_selection.py`](src/features/feature_selection.py), [`src/training/train_optimized_model.py`](src/training/train_optimized_model.py)): expands the hand-crafted scalar feature set to **175 features per clip** across 7 families — time-domain/Teager energy, spectral (incl. flux), MFCC, dense mel-band statistics (40 bands × mean/std), wavelet decomposition, histogram-packet rhythm, and onset/tempo — written to chunked, resume-safe Parquet. Supersedes [`src/features/advanced_feature_extraction.py`](src/features/advanced_feature_extraction.py) (that script's 5 feature families are all reimplemented here as a superset — run one or the other, not both, since several column names overlap). Fixed while integrating: (1) the extractor loaded audio at 16kHz, forcing every clip to resample up from its true native 8kHz, costing ~22ms/clip for nothing — `TARGET_SR` is now 8000; (2) `spectral_contrast`'s default band edges reach past Nyquist at 8kHz — fixed by reducing `n_bands` to 3 at low sample rates; (3) the dataset scan checked `path.exists()` for all ~780k rows *before* applying `--limit`, so a "smoke test" scanned the whole dataset first — fixed to stop scanning once enough valid clips are found; (4) the default worker count (`cpu_count()-2`, 18 on a 20-core machine) crashed with `DLL load failed... paging file too small` under real memory pressure — lowered to `min(6, cpu_count()-2)`, overridable via `--workers`; (5) `audio_full_path` is an absolute path baked in from whichever drive letter existed at Stage 1 time — now remapped via `--audio-root`/auto-detected drive so a plugged-in HDD getting a different letter on another PC doesn't silently break every file lookup.

  `feature_selection.py` trains a quick XGBoost model over the full store and uses SHAP (`TreeExplainer`) to rank every feature by mean |SHAP value|, keeping only the top-N — with a `--target {rainfall_mm, is_rainy}` flag added after an early finding: ranking features against `rainfall_mm` alone and reusing that list for the classifier actually **hurt** classification (AUC 0.842 vs 0.883 baseline), since the best rainfall-*amount* predictors aren't necessarily the best rain/no-rain discriminators. Each model now gets its own top-30 SHAP-ranked feature set. `train_optimized_model.py` evaluates both on the exact same population and per-campaign chronological split as the Stage 4 baseline (joining onto `data/processed/train.csv`/`test.csv` rather than trusting the standalone optimized Parquet, which lacks split labels and Stage 3's duration/sensor-artifact filters).

  ```bash
  python src/features/master_feature_extraction.py                          # full run, all clips
  python src/features/feature_selection.py --top-n 30 --target rainfall_mm  # regression-ranked top-30
  python src/features/feature_selection.py --top-n 30 --target is_rainy     # classification-ranked top-30
  python src/training/train_optimized_model.py --master-store-dir <dir> \
      --shap-csv-classifier <is_rainy_csv> --shap-csv-regressor <rainfall_mm_csv>
  ```

  **Full-scale run completed 2026-07-07** on an i9-13900K (8 workers, Ubuntu): all 780,725 clips in **4.34 hours** (20.0ms/clip — faster than the 31.6ms/clip Windows-based estimate). Output: 157 chunked Parquet files, ~0.77GB total.

  **Result — both models beat the Stage 4 baseline**, evaluated on the identical 607,673/151,927-row split:

  | Model | Metric | Stage 4 baseline | Stage 8 optimized |
  |---|---|---|---|
  | Classifier | AUC-ROC | 0.883 | **0.887** |
  | Regressor | R² | 0.155 | **0.226** (+46% relative) |

  See [`docs/PROGRESS.md`](docs/PROGRESS.md) for the full feature lists, the duplicate-row data quirk found and fixed in `December_2024_rain_data`, and next steps (hurdle-model regressor, Stage 7 DL fusion).

---

## 🚀 Getting Started

```bash
# 1. Set up the environment
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Sanity-check the pipeline on a small sample
python tests/dry_run_test.py

# 3. Run the full cleaning pipeline (CPU)
python src/features/data_cleaning.py

# (Optional) GPU-accelerated version — requires torch + torchaudio
python src/features/data_cleaning_gpu.py

# 4. Inspect dataset structure/health
python src/evaluation/analyze_dataset_v2.py
```

> Source and destination drive paths are currently hardcoded at the top of each script (`SOURCE_DRIVE`, `DESTINATION_DRIVE`) — update them for your own machine before running.

---

## 📝 Honest Notes

A few details worth knowing before building on top of this:

- **Alignment tolerance is 2 minutes**, not 5 — every clip's nearest mechanical reading must fall within that window or it's marked `is_aligned = False` and labeled dry by default.
- **Clip duration isn't perfectly uniform.** ~97% of clips are 10–15s, but **January 2024 (100%) and December 2023 (96%)** are short, sub-5-second clips with a measurably different acoustic profile and rainy-rate — `feature_extraction.py` (Stage 3) drops these before training rather than mixing durations.
- **Audio sample rate**: confirmed empirically (Stage 7 investigation) at **8000 Hz, mono, exactly 10.0s (80,000 samples)** across all 19 recording campaigns in the `10-15s`-duration training subset.
- The GPU pipeline is a working, independent implementation, not a drop-in replacement yet — it hasn't been cross-validated feature-by-feature against the CPU pipeline's output.
- **A handful of mechanical readings are sensor artifacts, not real rain.** 204 rows carry `rainfall_mm` ≈ 655.2–655.33 identically across unrelated recording campaigns (real readings top out at 21.6mm) — almost certainly a corrupted or overflowed counter. `feature_extraction.py` drops anything above 50mm before it reaches training; this originally produced a broken regressor (R² of **-0.045**, worse than predicting the mean) until caught by actually running the pipeline.

---

*Built by [Am4l-babu](https://github.com/Am4l-babu) — an ongoing project turning the sound of rain into data.*
