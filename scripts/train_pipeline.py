"""scripts/train_pipeline.py
============================
Standalone training script — can be run from CLI or CI/CD.

Usage:
    python scripts/train_pipeline.py [--model RandomForest]
"""

import argparse
import json
import logging
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import numpy as np
import config
from src.data import load_raw
from src.data.preprocess import cap_outliers
from src.features import build_all_features
from src.models.train import (
    build_regression_models, cross_validate_models,
    tune_random_forest, train_final_model, save_model,
)
from src.models.evaluate import regression_report, compare_models


def main(model_name: str = "RandomForest", tune: bool = True) -> None:
    logger.info("=" * 60)
    logger.info("Pakistan Economic Analytics — Training Pipeline")
    logger.info("=" * 60)

    # ── 1. Load & preprocess ─────────────────────────────────────────────
    logger.info("Step 1/5: Loading data…")
    df = load_raw()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df = cap_outliers(df, numeric_cols)

    # ── 2. Feature engineering ───────────────────────────────────────────
    logger.info("Step 2/5: Engineering features…")
    df_feat = build_all_features(df)
    config.DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(config.PROCESSED_FILE)
    logger.info("Processed data: %s → %s", df_feat.shape, config.PROCESSED_FILE)

    # ── 3. Build X, y ────────────────────────────────────────────────────
    drop_cols = ["gdp_growth_category", "inflation_category", "decade",
                 "key_events", "gdp_usd_bn", "gdp_per_capita_usd"]
    feat_matrix = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])
    feat_matrix = feat_matrix.select_dtypes(include=np.number).fillna(0)

    target = config.TARGET_REG
    X = feat_matrix.drop(columns=[target])
    y = feat_matrix[target]
    feature_names = X.columns.tolist()

    X_train, X_test = X.iloc[:-5], X.iloc[-5:]
    y_train, y_test = y.iloc[:-5], y.iloc[-5:]
    logger.info("Train: %d rows | Test: %d rows | Features: %d",
                len(X_train), len(X_test), len(feature_names))

    # ── 4. Cross-validate baseline models ────────────────────────────────
    logger.info("Step 3/5: Cross-validating models…")
    models = build_regression_models()
    cv_results = cross_validate_models(models, X_train, y_train)
    logger.info("\n%s", cv_results.sort_values("mean_score").round(4).to_string())

    # ── 5. Tune & train best model ────────────────────────────────────────
    logger.info("Step 4/5: Tuning %s…", model_name)
    if tune and model_name == "RandomForest":
        best_model = tune_random_forest(X_train, y_train)
    else:
        best_model = train_final_model(X_train, y_train, model_name)

    # ── 6. Evaluate on test set ───────────────────────────────────────────
    logger.info("Step 5/5: Evaluating on test set…")
    y_pred  = best_model.predict(X_test)
    metrics = regression_report(y_test, y_pred, model_name=model_name)

    # ── 7. Persist artifacts ──────────────────────────────────────────────
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "best_model.pkl"
    save_model(best_model, model_path)

    feat_path = config.MODELS_DIR / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(model_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    manifest = {
        "model_name":   model_name,
        "trained_on":   f"{X_train.index[0]}–{X_train.index[-1]}",
        "test_years":   X_test.index.tolist(),
        "n_features":   len(feature_names),
        "metrics":      metrics,
        "model_md5":    md5,
    }
    with open(config.MODELS_DIR / "experiment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("✅ Training complete!")
    logger.info("   Model   → %s  (MD5: %s)", model_path, md5[:8])
    logger.info("   RMSE    = %.4f%%", metrics["RMSE"])
    logger.info("   R²      = %.4f",   metrics["R2"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GDP growth forecast model")
    parser.add_argument("--model",  default="RandomForest",
                        choices=["Ridge", "Lasso", "ElasticNet",
                                 "RandomForest", "GradientBoosting"])
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip GridSearchCV hyperparameter tuning")
    args = parser.parse_args()
    main(model_name=args.model, tune=not args.no_tune)
