"""src/data/preprocess.py
========================
Cleaning and preprocessing pipeline.  Each step is a pure function so
the pipeline is fully composable and unit-testable.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ── Step 1: Outlier capping (Winsorization) ────────────────────────────────
def cap_outliers(df: pd.DataFrame, cols: list[str], lower: float = 0.02,
                 upper: float = 0.98) -> pd.DataFrame:
    """Winsorize numeric columns at the given quantile bounds."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        lo, hi = df[col].quantile([lower, upper])
        df[col] = df[col].clip(lo, hi)
    return df


# ── Step 2: Missing value imputation ───────────────────────────────────────
def impute_missing(df: pd.DataFrame, strategy: str = "linear") -> pd.DataFrame:
    """
    Interpolate missing values using the specified pandas strategy.

    Parameters
    ----------
    strategy : {'linear', 'time', 'ffill'}
    """
    df = df.copy()
    n_missing_before = df.isnull().sum().sum()
    if strategy == "ffill":
        df = df.ffill().bfill()
    else:
        df = df.interpolate(method=strategy, limit_direction="both")
    n_missing_after = df.isnull().sum().sum()
    logger.info("Imputed %d → %d missing values", n_missing_before, n_missing_after)
    return df


# ── Step 3: Scale numeric features ────────────────────────────────────────
def scale_features(
    df: pd.DataFrame, feature_cols: list[str], scaler=None
) -> tuple[pd.DataFrame, RobustScaler]:
    """
    Apply RobustScaler to ``feature_cols`` (robust to outliers).

    Returns the transformed DataFrame and the fitted scaler so it can
    be persisted for inference.
    """
    df = df.copy()
    if scaler is None:
        scaler = RobustScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


# ── Step 4: Time-aware train / test split ─────────────────────────────────
def temporal_split(
    df: pd.DataFrame, test_size: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the time-series dataset.  The **last** ``test_size`` years
    become the test set; everything before is training.

    Using a purely temporal split avoids look-ahead bias that would
    occur with random shuffling.
    """
    train = df.iloc[:-test_size]
    test  = df.iloc[-test_size:]
    logger.info("Train: %d rows | Test: %d rows", len(train), len(test))
    return train, test
