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
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from xgboost import XGBClassifier, XGBRegressor

from data_cleaning import _extract_features, _dur_category
from dl_dataset import CLIP_SAMPLES, MFCCExtractor
from dl_models import MODEL_REGISTRY
from master_feature_extraction import extract_all_features

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


def _predict_dl_models(wav_path: str, device: torch.device) -> dict[str, float]:
    """Runs all three full-scale MFCC architectures (CNN, LSTM, Transformer)
    on one clip. Requires cnn_mfcc_norm_stats.pt -- the exact train-set
    mean/std these models were normalized against at training time (see
    ensemble_predict.py); shared across all three since they were trained on
    the same normalized tensors in one invocation."""
    norm = torch.load(MODELS_DIR / "cnn_mfcc_norm_stats.pt", map_location=device)
    mean, std = norm["mean"].to(device), norm["std"].to(device)

    y, _sr = sf.read(str(wav_path), dtype="float32")
    if len(y) != CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y))) if len(y) < CLIP_SAMPLES else y[:CLIP_SAMPLES]
    waveform = torch.from_numpy(y).unsqueeze(0).to(device)

    extractor = MFCCExtractor().to(device)
    with torch.no_grad():
        mfcc = (extractor(waveform) - mean) / std

    preds = {}
    for arch in ["cnn", "lstm", "transformer"]:
        model = MODEL_REGISTRY[arch]().to(device)
        model.load_state_dict(torch.load(MODELS_DIR / f"dl_{arch}_mfcc.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            preds[arch] = max(0.0, model(mfcc).cpu().item())
    return preds


def _predict_optimized(wav_path: str) -> tuple[float, float]:
    """Runs the Stage 8 XGBoost regressor + classifier (SHAP-selected
    hand-engineered feature path) on one clip."""
    with open(MODELS_DIR / "xgb_optimized_features.json") as f:
        feat_lists = json.load(f)
    reg_features, clf_features = feat_lists["regressor_features"], feat_lists["classifier_features"]

    raw = extract_all_features((str(wav_path), {}))
    if raw is None:
        raise RuntimeError(f"Master feature extraction failed for {wav_path}")

    reg_row = pd.DataFrame([{c: raw.get(c, 0) for c in reg_features}])
    clf_row = pd.DataFrame([{c: raw.get(c, 0) for c in clf_features}])

    xgb_reg = XGBRegressor()
    xgb_reg.load_model(str(MODELS_DIR / "xgb_regressor_optimized.json"))
    xgb_pred = max(0.0, float(xgb_reg.predict(reg_row)[0]))

    xgb_clf = XGBClassifier()
    xgb_clf.load_model(str(MODELS_DIR / "xgb_classifier_optimized.json"))
    xgb_proba = float(xgb_clf.predict_proba(clf_row)[0, 1])

    return xgb_pred, xgb_proba


def predict_ensemble(wav_path: str) -> dict:
    """Full ensemble pipeline (the best-performing configuration found in this
    project's ablation): CNN + LSTM + Transformer (raw-audio MFCC) +
    XGBoost-optimized (SHAP-selected scalar features) + its P(rainy), combined
    by whichever stacker configuration ensemble_stack.py's 5-fold CV search
    found best -- see models/ensemble_stacker_config.json for which features
    and combination logic that turned out to be, and docs/ensemble_stack_report.json
    for the accuracy this achieves vs. the simpler baseline pipeline above."""
    raw = _extract_features(str(wav_path))
    if not raw.get("success"):
        raise RuntimeError(f"Feature extraction failed for {wav_path}: {raw.get('error')}")
    duration_category = _dur_category(raw["duration_sec"])
    out_of_distribution = duration_category != TRAINED_DURATION_CATEGORY

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_preds = _predict_dl_models(wav_path, device)
    xgb_pred, xgb_proba = _predict_optimized(wav_path)

    with open(MODELS_DIR / "ensemble_stacker_config.json") as f:
        config = json.load(f)

    signals = {
        "cnn_pred": dl_preds["cnn"], "lstm_pred": dl_preds["lstm"],
        "transformer_pred": dl_preds["transformer"],
        "xgb_pred": xgb_pred, "xgb_proba": xgb_proba,
    }
    stacker = XGBRegressor()
    stacker.load_model(str(MODELS_DIR / "ensemble_stacker.json"))

    if config["type"] == "hurdle_hard_gate":
        amount = float(stacker.predict(np.array([[signals[f] for f in config["features"]]]))[0])
        final_pred = max(0.0, amount) if xgb_proba >= config["gate_threshold"] else 0.0
    else:
        stack_input = np.array([[signals[f] for f in config["features"]]])
        final_pred = max(0.0, float(stacker.predict(stack_input)[0]))

    return {
        "audio_path": str(wav_path),
        "duration_sec": raw["duration_sec"],
        "duration_category": duration_category,
        "out_of_distribution": out_of_distribution,
        "cnn_pred_mm": dl_preds["cnn"],
        "lstm_pred_mm": dl_preds["lstm"],
        "transformer_pred_mm": dl_preds["transformer"],
        "xgb_pred_mm": xgb_pred,
        "rain_probability": xgb_proba,
        "is_rainy": bool(xgb_proba >= 0.5),
        "estimated_rainfall_mm": final_pred,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Score a WAV clip: rain/no-rain classification + optional mm estimate.")
    parser.add_argument("wav_path", help="Path to a WAV audio clip")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Rain-probability decision threshold "
                              "(default: Stage 5's tuned best-F1 threshold)")
    parser.add_argument("--no-mm", action="store_true", help="Skip the rainfall amount estimate")
    parser.add_argument("--ensemble", action="store_true",
                         help="Use the stacked CNN + XGBoost-optimized ensemble instead of the "
                              "Stage 4/5 baseline pipeline (higher accuracy, see docs/ensemble_stack_report.json)")
    args = parser.parse_args()

    if args.ensemble:
        result = predict_ensemble(args.wav_path)
        print("=" * 60)
        print(f"  {Path(args.wav_path).name}  (ensemble pipeline)")
        print("=" * 60)
        print(f"  Duration          : {result['duration_sec']:.1f}s ({result['duration_category']})")
        if result["out_of_distribution"]:
            print(f"  WARNING           : models were only trained on "
                  f"{TRAINED_DURATION_CATEGORY} clips -- this prediction is out-of-distribution")
        print(f"  CNN estimate      : {result['cnn_pred_mm']:.3f} mm")
        print(f"  LSTM estimate     : {result['lstm_pred_mm']:.3f} mm")
        print(f"  Transformer est.  : {result['transformer_pred_mm']:.3f} mm")
        print(f"  XGBoost estimate  : {result['xgb_pred_mm']:.3f} mm")
        print(f"  Rain probability  : {result['rain_probability']:.3f}")
        print(f"  Prediction        : {'RAINY' if result['is_rainy'] else 'DRY'}")
        print(f"  Estimated rainfall: {result['estimated_rainfall_mm']:.3f} mm  (stacked ensemble)")
        return

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
