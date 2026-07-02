# 🌧️ Acoustic Rain Gauge ML

**Can a microphone replace a rain gauge?**

This project trains a machine learning model to estimate rainfall — whether it's raining, and how hard — purely from the *sound* of rain hitting a surface, using a traditional mechanical tipping-bucket gauge as ground truth.

<p align="left">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/status-feature%20engineering%20complete-brightgreen" alt="Status">
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

- **[`data_cleaning.py`](src/data_cleaning.py)** — CPU pipeline. A `multiprocessing.Pool` of 16 workers extracts features with `librosa`/`soundfile`. This is the version the full dataset was actually processed with.
- **[`data_cleaning_gpu.py`](src/data_cleaning_gpu.py)** — experimental GPU pipeline. Batches waveforms onto CUDA and computes every feature as a single vectorized `torchaudio` pass (spectrogram → centroid/bandwidth/rolloff/MFCC), aiming for a 4-8x speedup over CPU. Not yet used for a production run.

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
</p>

| Layer | Tools |
|---|---|
| Audio I/O & DSP | `soundfile`, `librosa`, `torchaudio` |
| Data wrangling | `pandas`, `numpy` |
| Parallelism | `multiprocessing.Pool` (CPU) / CUDA batching via `torch` (GPU) |
| Modeling (planned) | `xgboost`, `scikit-learn` |
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
│   ├── train_model.py         # (stub) XGBoost training & evaluation — TODO
│   └── utils.py               # (stub) shared helpers — TODO
│
├── data/
│   ├── raw/                   # Original audio + tipping-bucket CSVs (kept off-repo)
│   ├── processed/             # Cleaned, aligned, feature-extracted CSVs
│   └── external/              # Any external datasets or metadata
│
├── docs/
│   └── dataset_analysis_report_v2.txt   # Full recursive dataset scan
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
- [x] **Stage 3 — Feature Engineering** ([`src/feature_extraction.py`](src/feature_extraction.py)): dropped the confounded short clips, split chronologically **per recording campaign** (a single global time cutoff was tried first and produced a severe train/test rainy-rate mismatch — 8.9% vs 39.4% — since campaigns like Feb-Mar 2026 are ~93% rainy; per-campaign splitting fixed it to 15.1% / 14.6%), computed class weights, fit a scaler on train only
- [ ] **Stage 4 — Model Training**: baseline classifier → XGBoost rain/no-rain classifier → regression for rainfall amount, cross-validated on time folds
- [ ] **Stage 5 — Evaluation**: precision/recall/F1/AUC-ROC, RMSE/MAE, feature importance, per-month generalization check
- [ ] **Stage 6 — Real-Time Inference** *(stretch goal)*: a lightweight script that scores a live WAV clip in real time

---

## 🚀 Getting Started

```bash
# 1. Set up the environment
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Sanity-check the pipeline on a small sample
python src/dry_run_test.py

# 3. Run the full cleaning pipeline (CPU)
python src/data_cleaning.py

# (Optional) GPU-accelerated version — requires torch + torchaudio
python src/data_cleaning_gpu.py

# 4. Inspect dataset structure/health
python src/analyze_dataset_v2.py
```

> Source and destination drive paths are currently hardcoded at the top of each script (`SOURCE_DRIVE`, `DESTINATION_DRIVE`) — update them for your own machine before running.

---

## 📝 Honest Notes

A few details worth knowing before building on top of this:

- **Alignment tolerance is 2 minutes**, not 5 — every clip's nearest mechanical reading must fall within that window or it's marked `is_aligned = False` and labeled dry by default.
- **Clip duration isn't perfectly uniform.** ~97% of clips are 10–15s, but **January 2024 (100%) and December 2023 (96%)** are short, sub-5-second clips with a measurably different acoustic profile and rainy-rate — `feature_extraction.py` (Stage 3) drops these before training rather than mixing durations.
- **Audio sample rate isn't recorded anywhere in the pipeline output** — don't assume a specific rate without checking the source files directly.
- The GPU pipeline is a working, independent implementation, not a drop-in replacement yet — it hasn't been cross-validated feature-by-feature against the CPU pipeline's output.

---

*Built by [Am4l-babu](https://github.com/Am4l-babu) — an ongoing project turning the sound of rain into data.*
