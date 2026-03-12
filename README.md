# Pakistan Economic Analytics

[![CI](https://github.com/Muhammad-Farooq13/Pakistan-economic-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/Pakistan-economic-analytics/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end GDP growth forecasting system** — 26 years of Pakistan macroeconomic data (2000–2025), 35 engineered features, temporal cross-validation, and an interactive Streamlit dashboard.

---

## Live Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pakistan-economic-analytics.streamlit.app)

**6 pages:**
| Page | Content |
|------|---------|
| 📊 Overview | KPI cards (GDP, inflation, FX, PKR/USD), GDP timeline, external sector sparklines |
| 📈 Trends | Multi-indicator z-score comparison, IMF programme markers, external sector chart |
| 🔍 Correlations | Scatter + Pearson/Spearman statistics, full correlation heatmap |
| 🤖 GDP Forecast | ML back-test: actual vs predicted (2002–2025), test period highlighted |
| 🎯 Scenario Analysis | 2026 GDP growth simulator — adjust 6 macro levers, see gauge + interpretation |
| ℹ️ About | Technical stack, project structure, quick-start |

---

## Dataset

**`data/raw/pakistan_economic_indicators_2000_2025.csv`** — 26 annual observations × 30+ columns

| Category | Indicators |
|----------|-----------|
| Output | GDP (USD bn), GDP growth (%), GDP per capita |
| Monetary | Inflation CPI, policy rate, PKR/USD |
| Fiscal | Public debt (% GDP), tax revenue (% GDP) |
| External | Exports, imports, remittances, forex reserves, FDI, current account |
| Structural | Population, literacy rate, sector shares (agri/services/industry), mobile penetration |
| Categorical | IMF programme active, GDP growth category, inflation category, key events |

**Feature engineering adds 11 derived features:** GDP/inflation lags, 3-year moving averages, PKR YoY depreciation, forex import cover (months), trade openness, external pressure index, remittances/FDI growth rates.

---

## Model Results

Training uses **TimeSeriesSplit** to prevent look-ahead bias (2002–2020 train, 2021–2025 test).

| Model | CV RMSE | Test RMSE | Test R² |
|-------|---------|-----------|---------|
| **RandomForest** ✅ | 1.697% | 2.178% | 0.073 |
| ElasticNet | 2.431% | — | — |
| GradientBoosting | 2.516% | — | — |
| Ridge | 2.522% | — | — |

*Note: The 2021–2025 test period spans post-COVID recovery (→ +6.1%), then a sharp reversal (−0.04% in 2023), making test R² highly sensitive to the 5 test years. CV RMSE is the more meaningful comparison metric.*

---

## Project Structure

```
pakeco/
├── data/
│   ├── raw/                       # Original CSV (26 rows × 30+ cols)
│   └── processed/                 # Engineered features (git-ignored)
├── src/
│   ├── data/
│   │   ├── load_data.py           # CSV loader + type coercions
│   │   └── preprocess.py          # Outlier capping, imputation, temporal split
│   ├── features/build_features.py # 35-feature engineering pipeline
│   ├── models/
│   │   ├── train.py               # 5-model catalogue, TimeSeriesSplit CV, GridSearch
│   │   ├── predict.py             # Artefact loader + inference
│   │   └── evaluate.py            # Regression metrics report
│   └── visualization/plots.py
├── models/saved/
│   ├── best_model.pkl             # Versioned RandomForest pipeline (~88 KB)
│   ├── feature_names.json         # 35 feature names
│   └── experiment_manifest.json   # Training run metadata
├── scripts/train_pipeline.py      # Full training script (with GridSearch tuning)
├── train_demo.py                  # Lightweight Ridge demo model trainer
├── streamlit_app.py               # Interactive 6-page dashboard
├── app/app.py                     # Original Streamlit app
├── notebooks/01_EDA.ipynb         # End-to-end analysis notebook
├── tests/                         # 35 unit tests (pytest)
├── config.py                      # Central path & constant registry
├── requirements.txt               # Full deps (includes streamlit, plotly)
├── requirements-ci.txt            # Minimal CI deps
├── runtime.txt                    # python-3.11 (Streamlit Cloud)
└── pyproject.toml
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Muhammad-Farooq13/Pakistan-economic-analytics
cd Pakistan-economic-analytics

# 2. Install
pip install -r requirements.txt

# 3. Launch dashboard (demo model already bundled)
streamlit run streamlit_app.py

# 4. Retrain the full model (optional — GridSearch, ~15s)
python scripts/train_pipeline.py

# 5. Run tests
pytest tests/ -v --cov=src
```

---

## License

MIT © Muhammad Farooq