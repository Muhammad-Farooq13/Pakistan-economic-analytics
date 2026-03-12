# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-03-12

### Added
- **Full ML pipeline** for Pakistan GDP growth forecasting (2000–2025):
  - `src/data/load_data.py` — CSV loader with ordinal encoding for categorical targets
  - `src/data/preprocess.py` — outlier capping, linear interpolation, temporal split
  - `src/features/build_features.py` — 35-feature matrix: 11 engineered features
    (GDP/inflation lags, rolling MAs, PKR depreciation, forex import cover, trade openness,
    external pressure index, remittances/FDI growth rates)
  - `src/models/train.py` — 5 regression pipelines (Ridge, Lasso, ElasticNet,
    RandomForest, GradientBoosting) with `TimeSeriesSplit` CV and GridSearchCV tuning
  - `src/models/evaluate.py` — regression metrics report (MAE, RMSE, MAPE, R²)
  - `src/models/predict.py` — artefact loader and inference helper
  - `scripts/train_pipeline.py` — full training script with model selection and
    experiment manifest (JSON)
- **35 pytest unit tests** covering data loading, preprocessing, and feature engineering
- `streamlit_app.py` — root-level 6-page interactive dashboard:
  - 📊 Overview (KPI cards, dual-axis GDP chart, sparklines, key milestones)
  - 📈 Trends (multi-indicator z-score comparison, IMF programme vlines)
  - 🔍 Correlations (scatter + Pearson/Spearman stats, full heatmap)
  - 🤖 GDP Forecast (ML back-test actual vs predicted, test period shading)
  - 🎯 Scenario Analysis (2026 GDP simulator, gauge chart, policy interpretation)
  - ℹ️ About (stack, structure, quick-start)
- `train_demo.py` — lightweight Ridge demo model trainer (no GridSearch)
  → `models/saved/pakistan_demo.pkl`
- `models/saved/best_model.pkl` — pre-trained RandomForest (88 KB, versioned) for
  Streamlit Cloud (no training required at deploy time)
- `.streamlit/config.toml` — light theme with Pakistan-green accent (`#01411C`)
- `runtime.txt` — `python-3.11` for Streamlit Cloud
- `packages.txt` — empty (no apt packages needed)
- `requirements.txt` — Plotly, Streamlit, statsmodels, shap, xgboost, lightgbm, etc.
- `requirements-ci.txt` — minimal CI subset (already present)

### Changed
- CI upgraded: `codecov/codecov-action@v4` → `@v5`
- `.gitignore` updated: `best_model.pkl`, `feature_names.json`, `experiment_manifest.json`,
  `pakistan_demo.pkl`, `pakistan_demo_features.json` excluded from the blanket ignore
  so pre-trained artifacts travel with the repo
- `trendline="ols"` removed from correlation scatter in `streamlit_app.py` — avoids
  a hard dependency on `statsmodels` being available at Streamlit Cloud startup

### Fixed
- N/A — all 35 tests passed from initial commit

---

## [Unreleased]

- SHAP feature importance visualisation tab
- Classification model for GDP growth category (Recession / Low / Moderate / High)
- Multi-step (3-year) GDP growth forecast with confidence bands