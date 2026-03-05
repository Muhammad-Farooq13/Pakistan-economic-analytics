"""src/models/predict.py
========================
Inference utilities: load a persisted pipeline and generate point
forecasts or scenario-based predictions.
"""

from pathlib import Path

import pandas as pd

from .train import load_model
from config import MODELS_DIR


def predict_gdp_growth(
    features: pd.DataFrame,
    model_path: str | Path | None = None,
) -> pd.Series:
    """
    Generate GDP growth forecasts for given feature rows.

    Parameters
    ----------
    features : DataFrame whose columns match the training feature set.
    model_path : path to a persisted joblib model; defaults to best model.

    Returns
    -------
    pd.Series of predicted GDP growth percentages.
    """
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pkl"
    model = load_model(model_path)
    preds = model.predict(features)
    return pd.Series(preds, index=features.index, name="predicted_gdp_growth")


def scenario_forecast(
    base_row: dict,
    scenarios: dict[str, dict],
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run named economic scenarios against the model.

    Parameters
    ----------
    base_row   : dict of feature values representing the 'baseline' year.
    scenarios  : dict of {scenario_name: {feature: override_value, …}}

    Returns
    -------
    DataFrame with one row per scenario and the predicted growth.
    """
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pkl"
    model = load_model(model_path)

    records = []
    for name, overrides in scenarios.items():
        row = {**base_row, **overrides}
        df_row = pd.DataFrame([row])
        pred = model.predict(df_row)[0]
        records.append({"scenario": name, "predicted_gdp_growth": round(pred, 2),
                         **overrides})
    return pd.DataFrame(records).set_index("scenario")
