"""src/models/train.py
======================
Model training pipelines for regression (GDP growth forecasting) and
classification (GDP growth category prediction).

Design principles
-----------------
* All models are scikit-learn compatible (Pipeline objects).
* Hyper-parameter tuning uses TimeSeriesSplit to respect temporal order.
* Artifacts (models + scaler) are persisted with joblib.
"""

import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_tscv(n_splits: int = 5) -> TimeSeriesSplit:
    """Return a TimeSeriesSplit respecting temporal ordering."""
    return TimeSeriesSplit(n_splits=n_splits)


def build_regression_models() -> dict[str, Pipeline]:
    """
    Return a catalogue of regression pipelines.

    Each pipeline: RobustScaler → estimator.
    """
    models = {
        "Ridge": Pipeline([
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Lasso": Pipeline([
            ("scaler", RobustScaler()),
            ("model", Lasso(alpha=0.1, max_iter=5000)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", RobustScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", RobustScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", RobustScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1,
                max_depth=4, random_state=42
            )),
        ]),
    }
    return models


def cross_validate_models(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> pd.DataFrame:
    """
    Cross-validate all models with TimeSeriesSplit.

    Returns a DataFrame with mean and std of the scoring metric.
    """
    tscv = get_tscv(n_splits)
    results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=tscv, scoring=scoring, n_jobs=-1)
        results[name] = {
            "mean_score": np.mean(scores),
            "std_score":  np.std(scores),
        }
        logger.info("%s  mean=%.4f  std=%.4f", name, results[name]["mean_score"],
                    results[name]["std_score"])
    return pd.DataFrame(results).T


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series,
    param_grid: dict | None = None, n_splits: int = 5,
) -> Pipeline:
    """
    Grid-search the RandomForest hyper-parameters with TimeSeriesSplit CV.
    """
    from config import RF_PARAM_GRID
    if param_grid is None:
        param_grid = {f"model__{k}": v for k, v in RF_PARAM_GRID.items()}

    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    gs = GridSearchCV(
        pipe, param_grid,
        cv=get_tscv(n_splits),
        scoring="neg_root_mean_squared_error",
        refit=True, n_jobs=-1, verbose=1,
    )
    gs.fit(X_train, y_train)
    logger.info("Best RF params: %s", gs.best_params_)
    logger.info("Best RF CV RMSE: %.4f", -gs.best_score_)
    return gs.best_estimator_


def train_final_model(
    X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "RandomForest"
) -> Pipeline:
    """Fit the chosen model on all training data and return the pipeline."""
    models = build_regression_models()
    pipe = models[model_name]
    pipe.fit(X_train, y_train)
    logger.info("Final %s model trained on %d samples.", model_name, len(X_train))
    return pipe


def save_model(model, path: str | Path) -> None:
    """Persist a trained model (or pipeline) to disk via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)


def load_model(path: str | Path):
    """Load a persisted model from disk."""
    return joblib.load(path)
