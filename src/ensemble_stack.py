"""
Stacked meta-learner over CNN + LSTM + Transformer + XGBoost-optimized predictions
=====================================================================================
ensemble_predict.py showed a fixed-weight blend of CNN (raw-audio MFCC) and
XGBoost-optimized (SHAP-selected scalar features) beats either model alone,
because the two see different representations of the same audio and make
largely uncorrelated errors. This script:

  1. Extends the stacker to all FIVE available signals: cnn_pred, lstm_pred,
     transformer_pred, xgb_pred, xgb_proba -- not just CNN + XGBoost. Since
     stacking already proved combining different model *families* helps,
     adding the two previously-unused neural nets as more "experts" for the
     meta-model to weigh is a natural extension.
  2. Small randomized hyperparameter search over the stacker's own XGBoost
     settings (it previously just reused generic params borrowed from
     elsewhere in the project), scored via 5-fold CV.
  3. Compares the winning soft-gated stack (proba is just one input among
     several, the model learns its own gating) against an explicit hard-gate
     hurdle (classify rain/no-rain first; only regress an amount for clips
     called rainy, force 0 otherwise) -- the Stage 8 hurdle idea, combined
     with the neural-net regressors this time instead of only XGBoost.
  4. Reports 5-fold out-of-fold R2 over all 151,927 rows for every variant,
     then refits the single best-performing configuration on all rows and
     saves it as the production stacker.

Run:
    python src/ensemble_stack.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR = REPO_ROOT / "docs"

BASE_STACK_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
ALL_FEATURES = ["cnn_pred", "lstm_pred", "transformer_pred", "xgb_pred", "xgb_proba"]
DL_ONLY = ["cnn_pred", "lstm_pred", "transformer_pred"]


def evaluate(name, y_true, y_pred):
    metrics = {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print(f"  {name:60s} RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")
    return metrics


def cv_oof_predict(X, y, is_rainy, params, n_splits=5, seed=42):
    """5-fold stratified (by is_rainy) out-of-fold predictions for one feature
    set / hyperparameter combo -- every row scored by a model that never saw
    it during fitting."""
    oof = np.zeros(len(y))
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kfold.split(X, is_rainy):
        model = XGBRegressor(**params)
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = np.clip(model.predict(X[val_idx]), 0, None)
    return oof


def main():
    pred_path = MODELS_DIR / "ensemble_predictions.parquet"
    print(f"Loading {pred_path}...")
    df = pd.read_parquet(pred_path)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    y_all = df["rainfall_mm"].to_numpy()
    is_rainy = df["is_rainy"].to_numpy()
    results = {}

    # ---- Part 1: individual models + simple combinations, all 5-fold OOF ----
    print("\n[1] Individual models (5-fold OOF, all 151,927 rows)...")
    for col in ALL_FEATURES:
        if col == "xgb_proba":
            continue
        results[f"{col}_alone"] = evaluate(f"{col} alone", y_all, df[col].to_numpy())

    print("\n[2] Stacking with CNN + XGBoost only (previous best, for comparison)...")
    X3 = df[["cnn_pred", "xgb_pred", "xgb_proba"]].to_numpy()
    oof3 = cv_oof_predict(X3, y_all, is_rainy, BASE_STACK_PARAMS)
    results["stack_3feat_cnn_xgb"] = evaluate("Stack: cnn + xgb + proba (previous)", y_all, oof3)

    print("\n[3] Stacking with all 5 signals (cnn, lstm, transformer, xgb, proba)...")
    X5 = df[ALL_FEATURES].to_numpy()
    oof5 = cv_oof_predict(X5, y_all, is_rainy, BASE_STACK_PARAMS)
    results["stack_5feat_all"] = evaluate("Stack: cnn + lstm + transformer + xgb + proba", y_all, oof5)

    # ---- Part 4: small randomized hyperparameter search on the 5-feature stack ----
    print("\n[4] Hyperparameter search for the stacker (5-feature input, 5-fold CV each)...")
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.05, 0.08, 0.12],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    sampler = list(ParameterSampler(param_grid, n_iter=12, random_state=42))
    best_params, best_r2, best_oof = None, -float("inf"), None
    for i, params in enumerate(sampler, start=1):
        full_params = {**params, "random_state": 42, "n_jobs": -1}
        oof = cv_oof_predict(X5, y_all, is_rainy, full_params)
        r2 = r2_score(y_all, oof)
        print(f"    [{i:2d}/{len(sampler)}] {params} -> R2={r2:.4f}")
        if r2 > best_r2:
            best_r2, best_params, best_oof = r2, full_params, oof
    print(f"\n  Best params found: {best_params}")
    results["stack_5feat_tuned"] = evaluate("Stack: 5-feature, tuned hyperparameters", y_all, best_oof)

    # ---- Part 5: hard-gate hurdle vs. soft-gate (proba as raw input) ----
    print("\n[5] Hard-gate hurdle (classify first, zero out predicted-dry clips)...")
    # Regressor trained on all 3 DL signals only (no proba leakage into the amount model),
    # then explicitly zeroed wherever the classifier calls the clip dry.
    Xdl = df[DL_ONLY].to_numpy()
    oof_dl_amount = cv_oof_predict(Xdl, y_all, is_rainy, BASE_STACK_PARAMS)
    hard_gated = np.where(df["xgb_proba"].to_numpy() >= 0.5, oof_dl_amount, 0.0)
    results["hurdle_hard_gate"] = evaluate("Hurdle (hard gate on proba>=0.5, DL-only amount)", y_all, hard_gated)

    print("\n" + "=" * 72)
    print("  FINAL SUMMARY (all 5-fold OOF over the same 151,927 rows)")
    print("=" * 72)
    print(f"  {'Model':<60}{'R2':<10}")
    for name, m in results.items():
        print(f"  {name:<60}{m['r2']:<10.4f}")
    print(f"  Baseline (Stage 4 XGBoost, 13 features): R2=0.155")

    # ---- Refit the single winning configuration on ALL rows, save it ----
    winner_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\n  Winner: {winner_name} (R2={results[winner_name]['r2']:.4f})")

    if winner_name == "hurdle_hard_gate":
        print("  Refitting production hurdle amount-model (DL-only) on all rows...")
        production = XGBRegressor(**BASE_STACK_PARAMS)
        production.fit(Xdl, y_all)
        stack_path = MODELS_DIR / "ensemble_stacker.json"
        production.save_model(str(stack_path))
        with open(MODELS_DIR / "ensemble_stacker_config.json", "w") as f:
            import json
            json.dump({"type": "hurdle_hard_gate", "features": DL_ONLY, "gate_threshold": 0.5}, f, indent=2)
    else:
        feats = ALL_FEATURES if "5feat" in winner_name else ["cnn_pred", "xgb_pred", "xgb_proba"]
        params = best_params if winner_name == "stack_5feat_tuned" else BASE_STACK_PARAMS
        print(f"  Refitting production stacker on all rows (features={feats})...")
        production = XGBRegressor(**params)
        production.fit(df[feats].to_numpy(), y_all)
        stack_path = MODELS_DIR / "ensemble_stacker.json"
        production.save_model(str(stack_path))
        with open(MODELS_DIR / "ensemble_stacker_config.json", "w") as f:
            import json
            json.dump({"type": "learned_stack", "features": feats, "params": params}, f, indent=2)
    print(f"  Saved: {stack_path} + ensemble_stacker_config.json")
    print(f"  NOTE: the 5-fold OOF R2 above (not this model's own training-set fit) is the honest")
    print(f"  estimate of how this production model performs on new, unseen clips.")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "ensemble_stack_report.json"
    import json
    with open(out_path, "w") as f:
        json.dump({"results": results, "winner": winner_name, "best_stacker_params": best_params}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
