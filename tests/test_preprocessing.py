"""tests/test_preprocessing.py
================================
Unit tests for data loading and preprocessing pipeline.

Run:
    pytest tests/ -v --cov=src
"""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_raw, basic_info
from src.data.preprocess import (
    cap_outliers, impute_missing, temporal_split
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def sample_df():
    """Small synthetic DataFrame for fast unit tests."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "year":            range(2000, 2000 + n),
        "gdp_growth_pct":  np.random.normal(4, 2, n),
        "inflation_cpi_pct": np.random.uniform(3, 20, n),
        "pkr_per_usd":     np.linspace(55, 280, n),
    }).set_index("year")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_raw
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadRaw:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_year_index(self, raw_df):
        assert raw_df.index.name == "year"
        assert raw_df.index.dtype == int

    def test_expected_columns(self, raw_df):
        required = ["gdp_growth_pct", "inflation_cpi_pct", "pkr_per_usd",
                    "forex_reserves_usd_bn", "remittances_usd_bn"]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_year_range(self, raw_df):
        assert raw_df.index.min() == 2000
        assert raw_df.index.max() == 2025

    def test_no_nulls_in_key_cols(self, raw_df):
        key_cols = ["gdp_growth_pct", "inflation_cpi_pct"]
        assert raw_df[key_cols].isnull().sum().sum() == 0

    def test_row_count(self, raw_df):
        assert len(raw_df) == 26          # 2000–2025 inclusive

    def test_gdp_positive(self, raw_df):
        assert (raw_df["gdp_usd_bn"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: basic_info
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicInfo:
    def test_keys(self, raw_df):
        info = basic_info(raw_df)
        assert "shape" in info
        assert "missing_counts" in info
        assert "year_range" in info

    def test_year_range_values(self, raw_df):
        info = basic_info(raw_df)
        assert info["year_range"] == (2000, 2025)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: cap_outliers
# ─────────────────────────────────────────────────────────────────────────────

class TestCapOutliers:
    def test_values_clipped(self, sample_df):
        cols = ["gdp_growth_pct"]
        result = cap_outliers(sample_df, cols, lower=0.05, upper=0.95)
        q_lo = sample_df["gdp_growth_pct"].quantile(0.05)
        q_hi = sample_df["gdp_growth_pct"].quantile(0.95)
        assert result["gdp_growth_pct"].min() >= q_lo - 1e-9
        assert result["gdp_growth_pct"].max() <= q_hi + 1e-9

    def test_does_not_mutate_input(self, sample_df):
        orig_min = sample_df["gdp_growth_pct"].min()
        _ = cap_outliers(sample_df, ["gdp_growth_pct"])
        assert sample_df["gdp_growth_pct"].min() == orig_min

    def test_ignores_missing_columns(self, sample_df):
        """Should silently skip non-existent columns."""
        result = cap_outliers(sample_df, ["nonexistent_col"])
        assert "nonexistent_col" not in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests: impute_missing
# ─────────────────────────────────────────────────────────────────────────────

class TestImputeMissing:
    def test_fills_nans(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = impute_missing(df)
        assert result["a"].isnull().sum() == 0

    def test_linear_interpolation(self):
        df = pd.DataFrame({"a": [0.0, np.nan, 4.0]})
        result = impute_missing(df, strategy="linear")
        assert abs(result["a"].iloc[1] - 2.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Tests: temporal_split
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalSplit:
    def test_split_sizes(self, raw_df):
        train, test = temporal_split(raw_df, test_size=5)
        assert len(test) == 5
        assert len(train) == len(raw_df) - 5

    def test_no_overlap(self, raw_df):
        train, test = temporal_split(raw_df, test_size=5)
        assert len(set(train.index) & set(test.index)) == 0

    def test_train_precedes_test(self, raw_df):
        train, test = temporal_split(raw_df, test_size=5)
        assert train.index.max() < test.index.min()
