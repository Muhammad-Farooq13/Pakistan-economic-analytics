# 🇵🇰 Pakistan Economic Analytics
### GDP Growth Forecasting & Macroeconomic Health Dashboard (2000–2025)

[![CI](https://github.com/Muhammad-Farooq13/Pakistan-economic-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/Pakistan-economic-analytics/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Problem Statement

Pakistan's macroeconomic trajectory is shaped by a complex interplay of fiscal policy, monetary conditions, external shocks, and structural factors. This project delivers:

1. **A reproducible ML pipeline** that forecasts annual GDP growth rate from 30+ macro indicators
2. **An interactive dashboard** for real-time economic scenario simulation
3. **A fully documented codebase** that serves as a reference for end-to-end data science projects

**Success Criteria:** Test RMSE < 1.5% on held-out years for GDP growth prediction.

---

## 🏗️ Project Architecture

```
pakistan-economic-analytics/
│
├── 📂 data/
│   ├── raw/                    ← Original CSV (2000–2025, 26×32)
│   └── processed/              ← Feature-engineered dataset
│
├── 📂 src/                     ← Importable library (pip install -e .)
│   ├── data/
│   │   ├── load_data.py        ← Loading + type coercions
│   │   └── preprocess.py       ← Outlier capping, imputation, scaling, splitting
│   ├── features/
│   │   └── build_features.py   ← 11 domain-driven engineered features
│   ├── models/
│   │   ├── train.py            ← CV, GridSearchCV, TimeSeriesSplit
│   │   ├── evaluate.py         ← MAE, RMSE, MAPE, R², feature importance
│   │   └── predict.py          ← Inference + scenario forecasting
│   └── visualization/
│       └── plots.py            ← Publication-quality Matplotlib/Seaborn charts
│
├── 📂 notebooks/
│   └── 01_EDA.ipynb            ← Full end-to-end analysis (9 sections)
│
├── 📂 app/
│   └── app.py                  ← Streamlit dashboard (5 pages)
│
├── 📂 models/saved/            ← Persisted model + feature manifest
├── 📂 reports/figures/         ← Auto-generated charts
├── 📂 tests/                   ← Unit tests (pytest + coverage)
├── 📂 .github/workflows/       ← CI/CD pipeline (GitHub Actions)
│
├── config.py                   ← Single source of truth for paths & constants
├── requirements.txt
├── setup.py
└── Makefile
```

---

## 📊 Dataset Overview

| Property | Detail |
|---|---|
| Source | Compiled from World Bank, IMF, SBP, PBS |
| Period | 2000–2025 (26 annual observations) |
| Features | 32 raw + 11 engineered = **43 total** |
| Target | `gdp_growth_pct` (regression) |

**Feature Groups:**
- **Macro:** Inflation, unemployment, policy rate, PKR/USD, public debt
- **External:** Remittances, exports, imports, trade balance, FX reserves, FDI
- **Structural:** Population, literacy, sectoral shares (agri/services/industry)
- **Engineered:** Lag features, rolling means, PKR depreciation, FX import cover, trade openness, external pressure index

---

## 🤖 Modeling Approach

| Step | Method |
|---|---|
| **Validation** | `TimeSeriesSplit` (k=5) — no look-ahead leakage |
| **Baseline models** | Ridge, Lasso, ElasticNet |
| **Advanced models** | RandomForest, GradientBoosting |
| **Tuning** | `GridSearchCV` on temporal CV folds |
| **Metrics** | MAE, RMSE, MAPE, R² |
| **Explainability** | Feature importance, permutation importance, SHAP |
| **Serialization** | `joblib` + JSON manifest with MD5 checksum |

---

## 🚀 Quickstart

### 1. Clone and install
```bash
git clone https://github.com/Muhammad-Farooq13/Pakistan-economic-analytics.git
cd pakistan-economic-analytics
pip install -r requirements.txt
pip install -e .
```

### 2. Run the analysis notebook
```bash
jupyter lab notebooks/01_EDA.ipynb
# Run all cells — this trains and saves the model
```

### 3. Launch the dashboard
```bash
streamlit run app/app.py
```

### 4. Run tests
```bash
pytest tests/ -v --cov=src
```

---

## 📈 Key Results

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Ridge (baseline) | ~1.8% | ~1.4% | ~0.65 |
| GradientBoosting | ~1.3% | ~1.0% | ~0.78 |
| **RandomForest (tuned)** | **~1.2%** | **~0.9%** | **~0.82** |

*Test set = last 5 years (2021–2025). Trained on 2002–2020.*

---

## 🎯 Scenario Analysis

The dashboard includes a **live policy scenario simulator** covering:
- Monetary shocks (inflation, policy rate)
- Currency depreciation (PKR/USD)
- External sector (remittances, FX reserves)
- Fiscal stress (public debt)

Each scenario produces a GDP growth point-estimate with a gauge visualization.

---

## 🛠️ Development

```bash
make lint      # flake8 + isort check
make format    # black + isort formatting
make test      # pytest with coverage report
make clean     # remove __pycache__, .coverage
```

---

## 📁 Reproducibility

Each training run saves:
- `models/saved/best_model.pkl` — trained pipeline
- `models/saved/feature_names.json` — exact feature list
- `models/saved/experiment_manifest.json` — metrics, params, MD5 hash

---

## 🎓 Skills Demonstrated

> This project is designed to signal the following competencies to employers:

| Skill Domain | Evidence |
|---|---|
| **Data Engineering** | Modular pipeline, versioned ingestion, outlier handling |
| **Feature Engineering** | 11 domain-driven features with economic rationale |
| **ML Modeling** | 5 models, GridSearchCV, TimeSeriesSplit |
| **Statistical Analysis** | ADF tests, correlation analysis, mutual information |
| **Software Engineering** | Clean code, type hints, docstrings, unit tests, CI/CD |
| **Visualization** | matplotlib, seaborn, Plotly (interactive) |
| **Deployment** | Production-quality Streamlit app, persisted artifacts |
| **Version Control** | Git workflow, GitHub Actions CI, structured commits |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Data Sources

- [State Bank of Pakistan](https://www.sbp.org.pk) — monetary & FX data
- [World Bank Open Data](https://data.worldbank.org) — GDP, population, trade
- [IMF World Economic Outlook](https://www.imf.org/en/Publications/WEO) — macro indicators
- [Pakistan Bureau of Statistics](https://www.pbs.gov.pk) — domestic statistics
