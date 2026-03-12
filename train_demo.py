"""
train_demo.py — lightweight demo model for Streamlit Cloud.

Trains a Ridge regression pipeline directly from the raw CSV
without the full training pipeline overhead. Saves a self-contained
pickle at models/saved/pakistan_demo.pkl which streamlit_app.py loads.

Usage:
    python train_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import config
from src.data import load_raw
from src.features import build_all_features

DEMO_PKL = config.MODELS_DIR / "pakistan_demo.pkl"
RANDOM_STATE = 42


CORE_FEATURES = [
    # Macro
    "inflation_cpi_pct", "policy_rate_pct", "pkr_per_usd",
    "public_debt_gdp_pct", "tax_revenue_gdp_pct",
    # External
    "remittances_usd_bn", "exports_usd_bn", "imports_usd_bn",
    "forex_reserves_usd_bn", "fdi_inflows_usd_bn",
    # Structural
    "population_mn", "imf_program_active",
    # Engineered (stable)
    "gdp_growth_lag1", "inflation_lag1",
    "forex_months_import", "trade_openness", "pkr_yoy_change",
]


def build_Xy(df_feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [f for f in CORE_FEATURES if f in df_feat.columns]
    X = df_feat[available].fillna(0)
    y = df_feat[config.TARGET_REG]
    return X, y


def main() -> None:
    print("Loading data and engineering features...")
    df_raw  = load_raw()
    df_feat = build_all_features(df_raw)

    X, y = build_Xy(df_feat)
    feature_names = X.columns.tolist()
    print(f"  {len(y)} samples x {len(feature_names)} features")

    # Temporal split: last 5 years = test
    X_train, X_test = X.iloc[:-5], X.iloc[-5:]
    y_train, y_test = y.iloc[:-5], y.iloc[-5:]

    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  Ridge(alpha=10.0)),  # stronger regularisation for tiny dataset
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    test_r2  = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_r2 = cross_val_score(pipe, X, y, cv=tscv, scoring="r2")

    metrics = {
        "test_r2":    round(float(test_r2), 4),
        "test_mae":   round(float(test_mae), 4),
        "cv_r2_mean": round(float(cv_r2.mean()), 4),
        "cv_r2_std":  round(float(cv_r2.std()), 4),
    }
    print(f"  Test R²={metrics['test_r2']:.4f} | MAE={metrics['test_mae']:.3f}%")
    print(f"  CV R²={metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

    # Build predictions for full period (for back-test chart in app)
    all_preds = pipe.predict(X)
    pred_series = pd.Series(all_preds, index=df_feat.index, name="Predicted")

    import joblib
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":         pipe,
        "features":      feature_names,
        "metrics":       metrics,
        "df_feat_index": df_feat.index.tolist(),
        "pred_series":   pred_series.tolist(),
        "test_start":    int(df_feat.index[-5]),
    }
    joblib.dump(payload, DEMO_PKL)
    print(f"\nDemo model saved → {DEMO_PKL}")

    # Save feature names alongside (so app can load either way)
    with open(config.MODELS_DIR / "pakistan_demo_features.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature list saved → models/saved/pakistan_demo_features.json")


if __name__ == "__main__":
    main()
