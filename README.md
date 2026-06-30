# Acoustic Rain Gauge ML

Machine learning project for estimating rainfall characteristics (e.g., intensity, droplet size distribution) using acoustic data from rain gauges, validated against tipping bucket data.

## Project Structure

```
acoustic_rain_gauge_ml/
│
├── data/
│   ├── raw/                  # Original, untouched audio and tipping bucket CSVs
│   ├── processed/            # Cleaned, aligned, and feature-extracted CSVs
│   └── external/             # Any external datasets or metadata
│
├── src/                      # Source code (Python scripts)
│   ├── data_cleaning.py      # Handles missing data and anomalies
│   ├── feature_extraction.py # Calculates PAR, Variance, FFT features
│   ├── train_model.py        # XGBoost training and evaluation
│   └── utils.py              # Helper functions
│
├── models/                   # Saved .pkl or .json model files
├── notebooks/                # Jupyter notebooks for EDA (Exploratory Data Analysis)
└── docs/                     # Documentation, architecture diagrams
```
