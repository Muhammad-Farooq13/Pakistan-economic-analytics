"""tests/test_features.py
=========================
Unit tests for the feature engineering pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_lag_features, add_rolling_features, add_pkr_depreciation,
    add_forex_cover, add_trade_openness, add_external_pressure_index,
    build_all_features,
)


@pytest.fixture(scope="module")
def base_df():
    """Synthetic 10-year dataset for feature tests."""
    np.random.seed(0)
    n = 15
    return pd.DataFrame({
        "gdp_growth_pct":       np.random.uniform(1, 8, n),
        "gdp_usd_bn":           np.linspace(100, 400, n),
        "inflation_cpi_pct":    np.random.uniform(3, 25, n),
        "pkr_per_usd":          np.linspace(60, 280, n),
        "forex_reserves_usd_bn": np.random.uniform(5, 25, n),
        "exports_usd_bn":       np.random.uniform(10, 35, n),
        "imports_usd_bn":       np.random.uniform(15, 55, n),
        "remittances_usd_bn":   np.linspace(1, 35, n),
        "fdi_inflows_usd_bn":   np.random.uniform(0.3, 5, n),
        "trade_balance_usd_bn": np.random.uniform(-30, -5, n),
    }, index=range(2000, 2000 + n))


class TestLagFeatures:
    def test_lag1_created(self, base_df):
        result = add_lag_features(base_df, "gdp_growth_pct", [1])
        assert "gdp_growth_lag1" in result.columns

    def test_lag1_values_shifted(self, base_df):
        result = add_lag_features(base_df, "gdp_growth_pct", [1])
        # First value of lag1 should be NaN
        assert pd.isna(result["gdp_growth_lag1"].iloc[0])
        # Second value should equal first original
        assert result["gdp_growth_lag1"].iloc[1] == base_df["gdp_growth_pct"].iloc[0]

    def test_multiple_lags(self, base_df):
        result = add_lag_features(base_df, "gdp_growth_pct", [1, 2, 3])
        for lag in [1, 2, 3]:
            assert f"gdp_growth_lag{lag}" in result.columns

    def test_no_mutation(self, base_df):
        original_cols = set(base_df.columns)
        _ = add_lag_features(base_df, "gdp_growth_pct", [1])
        assert set(base_df.columns) == original_cols


class TestRollingFeatures:
    def test_rolling_mean_created(self, base_df):
        result = add_rolling_features(base_df, "gdp_growth_pct", [3])
        assert "gdp_growth_ma3" in result.columns

    def test_rolling_mean_close_to_average(self, base_df):
        result = add_rolling_features(base_df, "gdp_growth_pct", [3])
        # At index 4: mean of rows 2,3,4
        expected = base_df["gdp_growth_pct"].iloc[2:5].mean()
        actual   = result["gdp_growth_ma3"].iloc[4]
        assert abs(actual - expected) < 1e-9


class TestPKRDepreciation:
    def test_column_created(self, base_df):
        result = add_pkr_depreciation(base_df)
        assert "pkr_yoy_change" in result.columns

    def test_first_row_nan(self, base_df):
        result = add_pkr_depreciation(base_df)
        assert pd.isna(result["pkr_yoy_change"].iloc[0])

    def test_positive_depreciation_for_rising_pkr(self, base_df):
        result = add_pkr_depreciation(base_df)
        # PKR is rising (more PKR per USD) → positive depreciation
        assert result["pkr_yoy_change"].dropna().mean() > 0


class TestForexCover:
    def test_column_created(self, base_df):
        result = add_forex_cover(base_df)
        assert "forex_months_import" in result.columns

    def test_values_positive(self, base_df):
        result = add_forex_cover(base_df)
        assert (result["forex_months_import"] > 0).all()

    def test_formula_correct(self, base_df):
        result = add_forex_cover(base_df)
        expected = (base_df["forex_reserves_usd_bn"] /
                    base_df["imports_usd_bn"]) * 12
        pd.testing.assert_series_equal(
            result["forex_months_import"], expected,
            check_names=False, check_exact=False, rtol=1e-9
        )


class TestTradeOpenness:
    def test_column_created(self, base_df):
        result = add_trade_openness(base_df)
        assert "trade_openness" in result.columns

    def test_values_range(self, base_df):
        result = add_trade_openness(base_df)
        assert (result["trade_openness"] > 0).all()
        assert (result["trade_openness"] < 200).all()   # sanity bound


class TestBuildAllFeatures:
    def test_returns_dataframe(self, base_df):
        result = build_all_features(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_engineered_cols(self, base_df):
        result = build_all_features(base_df)
        expected = [
            "gdp_growth_lag1", "gdp_growth_lag2", "inflation_lag1",
            "gdp_growth_ma3", "pkr_yoy_change",
            "forex_months_import", "trade_openness",
        ]
        for col in expected:
            assert col in result.columns

    def test_no_nans_after_build(self, base_df):
        result = build_all_features(base_df)
        # After dropna() the result should be clean
        assert result.isnull().sum().sum() == 0

    def test_fewer_rows_due_to_lag_dropna(self, base_df):
        result = build_all_features(base_df)
        assert len(result) < len(base_df)
