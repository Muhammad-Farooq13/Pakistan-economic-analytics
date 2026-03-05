"""
config.py
=========
Central configuration for the Pakistan Economic Indicators project.
All paths, constants, and hyper-parameter grids live here so every
module imports from one source of truth.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent
DATA_RAW_DIR    = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR   = ROOT_DIR / "data" / "processed"
MODELS_DIR      = ROOT_DIR / "models" / "saved"
REPORTS_DIR     = ROOT_DIR / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"

RAW_DATA_FILE   = DATA_RAW_DIR / "pakistan_economic_indicators_2000_2025.csv"
PROCESSED_FILE  = DATA_PROC_DIR / "features.csv"
SCALER_PATH     = MODELS_DIR / "scaler.pkl"
ENCODER_PATH    = MODELS_DIR / "label_encoder.pkl"

# ── Target & ID columns ────────────────────────────────────────────────────
TARGET_REG      = "gdp_growth_pct"           # regression target
TARGET_CLF      = "gdp_growth_category"      # classification target
TIME_COL        = "year"

# ── Feature groups ─────────────────────────────────────────────────────────
MACRO_FEATURES  = [
    "inflation_cpi_pct", "unemployment_pct", "policy_rate_pct",
    "pkr_per_usd", "public_debt_gdp_pct", "tax_revenue_gdp_pct",
]
EXTERNAL_FEATURES = [
    "remittances_usd_bn", "exports_usd_bn", "imports_usd_bn",
    "trade_balance_usd_bn", "forex_reserves_usd_bn",
    "current_account_usd_bn", "fdi_inflows_usd_bn",
]
STRUCTURAL_FEATURES = [
    "population_mn", "literacy_rate_pct",
    "agriculture_gdp_pct", "services_gdp_pct", "industry_gdp_pct",
    "mobile_per_100", "imf_program_active",
]
RATIO_FEATURES  = [
    "remittances_gdp_pct", "exports_gdp_pct",
    "imports_gdp_pct", "fdi_gdp_pct",
]
ENGINEERED_FEATURES = [
    "gdp_growth_lag1", "gdp_growth_lag2", "inflation_lag1",
    "gdp_growth_ma3", "inflation_ma3",
    "pkr_yoy_change", "forex_months_import",
    "trade_openness", "ext_pressure_index",
    "remittances_growth", "fdi_growth",
]

ALL_FEATURES = (
    MACRO_FEATURES + EXTERNAL_FEATURES + STRUCTURAL_FEATURES
    + RATIO_FEATURES + ENGINEERED_FEATURES
)

# ── Model training ─────────────────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.2          # time-aware split (last N rows)
CV_FOLDS        = 5

# Random-Forest hyper-parameter grid (used in GridSearchCV)
RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 4, 6],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}

# XGBoost hyper-parameter grid
XGB_PARAM_GRID = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample":     [0.8, 1.0],
}

# ── Plotting defaults ──────────────────────────────────────────────────────
PLOT_STYLE   = "seaborn-v0_8-whitegrid"
PALETTE      = "tab10"
FIG_DPI      = 150
FIG_SIZE     = (12, 6)
