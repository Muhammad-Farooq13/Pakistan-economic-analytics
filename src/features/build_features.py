"""src/features/build_features.py
==================================
Domain-driven feature engineering for Pakistan macroeconomic data.

Each feature is documented with its economic rationale so the code
doubles as technical commentary for interviews and write-ups.
"""

import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    """
    Autoregressive lag features.

    Rationale: economic momentum – last year's GDP growth is a strong
    predictor of this year's growth.
    """
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, col: str, windows: list[int]
) -> pd.DataFrame:
    """
    Rolling mean (trend smoother).

    Rationale: captures medium-term trend rather than single-year noise.
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    return df


def add_pkr_depreciation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year-on-year percentage change in PKR/USD exchange rate.

    Rationale: rapid depreciation signals external stress, raises import
    costs, and curtails real purchasing power.
    """
    df = df.copy()
    df["pkr_yoy_change"] = df["pkr_per_usd"].pct_change() * 100
    return df


def add_forex_cover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Foreign exchange reserves expressed in months of import cover.

    Formula: (forex_reserves / imports) * 12
    Rationale: IMF benchmark is ≥3 months; falling below signals crisis.
    """
    df = df.copy()
    df["forex_months_import"] = (
        df["forex_reserves_usd_bn"] / df["imports_usd_bn"]
    ) * 12
    return df


def add_trade_openness(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Exports + Imports) / GDP – classic measure of openness.

    Rationale: more open economies are more exposed to external shocks
    but also grow faster when global demand is strong.
    """
    df = df.copy()
    df["trade_openness"] = (
        (df["exports_usd_bn"] + df["imports_usd_bn"]) / df["gdp_usd_bn"]
    ) * 100
    return df


def add_external_pressure_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite external-pressure index (EPI).

    EPI = (trade_balance / gdp) - (forex_reserves / gdp)
    More negative → greater external stress.

    Rationale: captures the simultaneous strain from current-account
    deficit and dwindling reserves.
    """
    df = df.copy()
    df["ext_pressure_index"] = (
        df["trade_balance_usd_bn"] / df["gdp_usd_bn"]
        - df["forex_reserves_usd_bn"] / df["gdp_usd_bn"]
    )
    return df


def add_remittances_growth(df: pd.DataFrame) -> pd.DataFrame:
    """YoY growth rate of remittances – Pakistan's economic lifeline."""
    df = df.copy()
    df["remittances_growth"] = df["remittances_usd_bn"].pct_change() * 100
    return df


def add_fdi_growth(df: pd.DataFrame) -> pd.DataFrame:
    """YoY growth rate of FDI inflows."""
    df = df.copy()
    df["fdi_growth"] = df["fdi_inflows_usd_bn"].pct_change() * 100
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline: apply every feature transformation in order.
    Drops rows with NaN introduced by lag/rolling (first 1-2 years).
    """
    df = add_lag_features(df, "gdp_growth_pct", lags=[1, 2])
    df = add_lag_features(df, "inflation_cpi_pct", lags=[1])
    df = add_rolling_features(df, "gdp_growth_pct", windows=[3])
    df = add_rolling_features(df, "inflation_cpi_pct", windows=[3])
    df = add_pkr_depreciation(df)
    df = add_forex_cover(df)
    df = add_trade_openness(df)
    df = add_external_pressure_index(df)
    df = add_remittances_growth(df)
    df = add_fdi_growth(df)
    df = df.dropna()
    return df
