"""
Real-Time Inference (Stage 6)
================================
Scores a single WAV clip: rain / no-rain classification at the
Stage 5 tuned decision threshold, plus an optional rainfall (mm)
estimate from the Stage 4 regressor.

Reuses data_cleaning._extract_features so inference-time features are
computed identically to how the training data was built -- no
reimplementation of the feature math here.

Run:
    python src/predict.py path/to/clip.wav
    python src/predict.py path/to/clip.wav --threshold 0.5
    python src/predict.py path/to/clip.wav --no-mm
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from data_cleaning import _extract_features, _dur_category

REPO_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR   = REPO_ROOT / "docs"

FEATURE_COLS = [
    "rms", "peak", "par",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "zero_crossing_rate", "energy_variance",
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
]

# The only duration Stage 3 kept for training (see README's duration-
# confound note) -- predictions on other durations are out-of-distribution.
TRAINED_DURATION_CATEGORY = "10-15s"


def _default_threshold() -> float:
    """
    Stage 5's best-F1 operating threshold, read from its report so this
    script can't silently drift out of sync with whatever tuning found.
    Falls back to the neutral 0.5 if Stage 5 hasn't been run yet.
    """
    report_path = DOCS_DIR / "stage5_evaluation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        return report["threshold_analysis"]["best_f1_threshold"]
    return 0.5


def _load_classifier():
    """Prefer the Stage 5 tuned classifier; fall back to the Stage 4 baseline."""
    tuned_path = MODELS_DIR / "xgb_classifier_tuned.json"
    baseline_path = MODELS_DIR / "xgb_classifier.json"
    model = XGBClassifier()
    if tuned_path.exists():
        model.load_model(str(tuned_path))
        return model, tuned_path.name
    model.load_model(str(baseline_path))
    return model, baseline_path.name


def predict(wav_path: str, threshold: float | None = None, estimate_mm: bool = True) -> dict:
    if threshold is None:
        threshold = _default_threshold()

    raw = _extract_features(str(wav_path))
    if not raw.get("success"):
        raise RuntimeError(f"Feature extraction failed for {wav_path}: {raw.get('error')}")

    duration_category = _dur_category(raw["duration_sec"])
    out_of_distribution = duration_category != TRAINED_DURATION_CATEGORY

    scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
    features_row = pd.DataFrame([{col: raw[col] for col in FEATURE_COLS}])
    X = scaler.transform(features_row)

    clf, clf_file = _load_classifier()
    rain_proba = float(clf.predict_proba(X)[0, 1])

    result = {
        "audio_path": str(wav_path),
        "duration_sec": raw["duration_sec"],
        "duration_category": duration_category,
        "out_of_distribution": out_of_distribution,
        "rain_probability": rain_proba,
        "threshold_used": threshold,
        "is_rainy": bool(rain_proba >= threshold),
        "classifier_model": clf_file,
    }

    if estimate_mm:
        reg = XGBRegressor()
        reg.load_model(str(MODELS_DIR / "xgb_regressor.json"))
        result["estimated_rainfall_mm"] = max(0.0, float(reg.predict(X)[0]))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Score a WAV clip: rain/no-rain classification + optional mm estimate.")
    parser.add_argument("wav_path", help="Path to a WAV audio clip")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Rain-probability decision threshold "
                              "(default: Stage 5's tuned best-F1 threshold)")
    parser.add_argument("--no-mm", action="store_true", help="Skip the rainfall amount estimate")
    args = parser.parse_args()

    result = predict(args.wav_path, threshold=args.threshold, estimate_mm=not args.no_mm)

    print("=" * 60)
    print(f"  {Path(args.wav_path).name}")
    print("=" * 60)
    print(f"  Duration          : {result['duration_sec']:.1f}s ({result['duration_category']})")
    if result["out_of_distribution"]:
        print(f"  WARNING           : model was only trained on "
              f"{TRAINED_DURATION_CATEGORY} clips -- this prediction is out-of-distribution")
    print(f"  Rain probability  : {result['rain_probability']:.3f}")
    print(f"  Threshold used    : {result['threshold_used']:.3f}")
    print(f"  Prediction        : {'RAINY' if result['is_rainy'] else 'DRY'}")
    if "estimated_rainfall_mm" in result:
        print(f"  Estimated rainfall: {result['estimated_rainfall_mm']:.3f} mm")
    print(f"  Classifier used   : {result['classifier_model']}")


if __name__ == "__main__":
    main()
