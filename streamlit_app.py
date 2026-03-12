"""
streamlit_app.py — Pakistan Economic Analytics Dashboard
=========================================================
Root-level entry point for Streamlit Cloud.

Pages:
  📊 Overview         — KPI cards, GDP timeline, sparklines
  📈 Trends           — Multi-indicator trend comparison
  🔍 Correlations     — Feature correlation explorer + heatmap
  🤖 GDP Forecast     — ML back-test: actual vs predicted
  🎯 Scenario Analysis — 2026 policy simulator
  ℹ️ About            — Technical stack and project info

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Ensure repo root is on path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.data import load_raw
from src.features import build_all_features

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Economic Analytics",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  (Pakistan green palette)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800; color: #01411C;
        border-bottom: 3px solid #01411C; padding-bottom: 8px;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f5e9, #ffffff);
        border-radius: 12px; padding: 18px 22px;
        border-left: 5px solid #01411C; margin-bottom: 12px;
    }
    .metric-label { font-size: 0.85rem; color: #555; font-weight: 600; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #01411C; }
    .metric-delta { font-size: 0.9rem; color: #888; }
    .insight-box {
        background: #fff8e1; border-radius: 10px; padding: 14px 18px;
        border-left: 4px solid #FFA000; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    df_raw  = load_raw()
    df_feat = build_all_features(df_raw)
    return df_raw, df_feat


@st.cache_resource(show_spinner="Loading model…")
def load_trained_model():
    import joblib

    # Prefer the full trained model; fall back to lightweight demo
    for pkl_name, feat_name in [
        ("best_model.pkl",     "feature_names.json"),
        ("pakistan_demo.pkl",  "pakistan_demo_features.json"),
    ]:
        model_path = config.MODELS_DIR / pkl_name
        feat_path  = config.MODELS_DIR / feat_name
        if model_path.exists() and feat_path.exists():
            artifact = joblib.load(model_path)
            # Handle both plain Pipeline and demo dict payload
            if isinstance(artifact, dict):
                model    = artifact["model"]
                features = artifact["features"]
            else:
                model = artifact
                with open(feat_path) as f:
                    features = json.load(f)
            return model, features

    return None, None


df_raw, df_feat = load_data()
model, feature_names = load_trained_model()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/3/32/Flag_of_Pakistan.svg",
        width=140,
    )
    st.markdown("### 🇵🇰 Pakistan Economic Analytics")
    st.markdown("*GDP Forecasting & Economic Health Dashboard*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "📈 Trends", "🔍 Correlations",
         "🤖 GDP Forecast", "🎯 Scenario Analysis", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    year_range = st.slider(
        "Year Range", int(df_raw.index.min()), int(df_raw.index.max()),
        (2000, 2025)
    )

df_filtered = df_raw.loc[year_range[0]:year_range[1]]


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Overview ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown('<div class="main-header">🇵🇰 Pakistan Economic Dashboard</div>',
                unsafe_allow_html=True)
    st.caption(f"Data: 2000–2025 | {len(df_raw)} annual observations | 30+ indicators")

    # ── KPI cards ─────────────────────────────────────────────────────────
    latest = df_raw.iloc[-1]
    prev   = df_raw.iloc[-2]
    col1, col2, col3, col4, col5 = st.columns(5)

    def kpi(col, label, value, delta=None, fmt="{:.1f}"):
        col.metric(label, fmt.format(value),
                   f"{delta:+.1f} YoY" if delta is not None else None)

    kpi(col1, "GDP (USD bn)",      latest["gdp_usd_bn"],
        latest["gdp_usd_bn"] - prev["gdp_usd_bn"])
    kpi(col2, "GDP Growth (%)",    latest["gdp_growth_pct"],
        latest["gdp_growth_pct"] - prev["gdp_growth_pct"])
    kpi(col3, "Inflation (%)",     latest["inflation_cpi_pct"],
        latest["inflation_cpi_pct"] - prev["inflation_cpi_pct"])
    kpi(col4, "FX Reserves ($bn)", latest["forex_reserves_usd_bn"],
        latest["forex_reserves_usd_bn"] - prev["forex_reserves_usd_bn"])
    kpi(col5, "PKR/USD",           latest["pkr_per_usd"],
        latest["pkr_per_usd"] - prev["pkr_per_usd"])

    st.divider()

    col_l, col_r = st.columns([2, 1])

    # ── GDP timeline ─────────────────────────────────────────────────────
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["gdp_usd_bn"],
            fill="tozeroy", name="GDP (USD bn)",
            line=dict(color="#1565C0", width=2.5),
            fillcolor="rgba(21,101,192,0.15)",
        ))
        fig.add_trace(go.Bar(
            x=df_filtered.index, y=df_filtered["gdp_growth_pct"],
            name="Growth (%)", yaxis="y2",
            marker_color=[
                "#C62828" if v < 0 else "#01411C"
                for v in df_filtered["gdp_growth_pct"]
            ],
            opacity=0.7,
        ))
        fig.update_layout(
            title="GDP Size & Annual Growth Rate",
            yaxis=dict(title="GDP (USD Billion)"),
            yaxis2=dict(title="Growth (%)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Key Milestones")
        milestone_years = [2005, 2009, 2020, 2023, 2025]
        for y in milestone_years:
            if y in df_raw.index:
                ev = df_raw.loc[y, "key_events"]
                gr = df_raw.loc[y, "gdp_growth_pct"]
                colour = "green" if gr > 3 else "red" if gr < 1 else "orange"
                st.markdown(f"**{y}** — :{colour}[{gr:+.1f}%]  \n_{ev}_")

    # ── Sparklines ─────────────────────────────────────────────────────────
    cB1, cB2, cB3, cB4 = st.columns(4)

    def sparkline(col_widget, series, title, color):
        fig = px.area(x=df_filtered.index, y=df_filtered[series],
                      title=title, color_discrete_sequence=[color])
        fig.update_layout(height=200, showlegend=False,
                          margin=dict(l=10, r=10, t=30, b=10),
                          template="plotly_white",
                          xaxis_title="", yaxis_title="")
        col_widget.plotly_chart(fig, use_container_width=True)

    sparkline(cB1, "inflation_cpi_pct",     "Inflation (%)",       "#E53935")
    sparkline(cB2, "remittances_usd_bn",    "Remittances ($bn)",   "#1E88E5")
    sparkline(cB3, "forex_reserves_usd_bn", "FX Reserves ($bn)",   "#00897B")
    sparkline(cB4, "pkr_per_usd",           "PKR / USD",           "#6D4C41")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Trends ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Trends":
    st.subheader("📈 Multi-Indicator Trend Comparison")
    st.caption("Select up to 6 indicators to compare on a normalised (z-score) axis.")

    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    exclude = ["gdp_growth_category", "inflation_category", "decade", "imf_program_active"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    selected = st.multiselect(
        "Choose indicators:",
        numeric_cols,
        default=["gdp_growth_pct", "inflation_cpi_pct",
                 "remittances_usd_bn", "forex_reserves_usd_bn"],
        max_selections=6,
    )

    if selected:
        normalise = st.checkbox("Normalise (z-score) for comparison", value=True)
        df_plot = df_filtered[selected].copy()
        if normalise:
            df_plot = (df_plot - df_plot.mean()) / df_plot.std()

        fig = px.line(df_plot, x=df_plot.index, y=selected,
                      title="Selected Indicators Over Time",
                      height=480, template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.D3)

        imf_years = df_filtered.index[df_filtered["imf_program_active"] == 1].tolist()
        for yr in imf_years:
            fig.add_vline(x=yr, line_dash="dot",
                          line_color="rgba(200,0,0,0.25)", line_width=1)
        if imf_years:
            fig.add_annotation(
                x=imf_years[0], y=df_plot.max().max() * 0.9,
                text="● IMF Programs", showarrow=False,
                font=dict(size=10, color="red"),
            )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("External Sector Deep Dive")
    df_ext = df_filtered[["exports_usd_bn", "imports_usd_bn",
                           "remittances_usd_bn"]].copy()
    fig2 = go.Figure()
    for col, color in [("exports_usd_bn",    "#43A047"),
                       ("imports_usd_bn",    "#E53935"),
                       ("remittances_usd_bn","#1E88E5")]:
        fig2.add_trace(go.Scatter(
            x=df_ext.index, y=df_ext[col],
            name=col.replace("_usd_bn", "").title(),
            line=dict(width=2.5), fill="tozeroy",
        ))
    fig2.update_layout(height=400, template="plotly_white",
                       title="Exports, Imports & Remittances (USD bn)")
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Correlations ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Correlations":
    st.subheader("🔍 Feature Correlation Explorer")
    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    exclude = ["gdp_growth_category", "inflation_category", "decade", "imf_program_active"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    col_a = st.selectbox("X-axis feature", numeric_cols,
                         index=numeric_cols.index("remittances_usd_bn"))
    col_b = st.selectbox("Y-axis feature", numeric_cols,
                         index=numeric_cols.index("gdp_growth_pct"))

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter(df_filtered, x=col_a, y=col_b,
                         color=[str(y) for y in df_filtered.index],
                         height=450, template="plotly_white",
                         title=f"{col_b} vs {col_a}",
                         labels={"color": "Year"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        from scipy.stats import pearsonr, spearmanr
        pair = df_filtered[[col_a, col_b]].dropna()
        if len(pair) > 2:
            x_, y_ = pair.values.T
            pr, pp = pearsonr(x_, y_)
            sr, sp = spearmanr(x_, y_)
            st.metric("Pearson r",  f"{pr:.3f}", f"p={pp:.3f}")
            st.metric("Spearman ρ", f"{sr:.3f}", f"p={sp:.3f}")
        st.caption("Hover over scatter points to see the year.")

    st.divider()
    st.subheader("Full Correlation Heatmap")
    sel_hmap = st.multiselect(
        "Select columns for heatmap:",
        numeric_cols,
        default=["gdp_growth_pct", "inflation_cpi_pct", "policy_rate_pct",
                 "pkr_per_usd", "forex_reserves_usd_bn", "remittances_usd_bn",
                 "current_account_usd_bn", "public_debt_gdp_pct"],
    )
    if sel_hmap:
        corr = df_filtered[sel_hmap].corr()
        fig_h = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, height=500,
                          title="Pearson Correlation Matrix")
        st.plotly_chart(fig_h, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: GDP Forecast ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 GDP Forecast":
    st.subheader("🤖 ML-Powered GDP Growth Forecast")

    if model is None:
        st.warning(
            "⚠️ No trained model found. "
            "Run `python scripts/train_pipeline.py` then refresh."
        )
    else:
        X_all = df_feat.reindex(columns=feature_names).fillna(0)
        preds = model.predict(X_all)
        pred_series  = pd.Series(preds, index=df_feat.index, name="Predicted")
        actual_series = df_feat[config.TARGET_REG]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_series.index, y=actual_series,
                                 name="Actual", mode="lines+markers",
                                 line=dict(color="#1565C0", width=2.5)))
        fig.add_trace(go.Scatter(x=pred_series.index, y=pred_series,
                                 name="Predicted", mode="lines+markers",
                                 line=dict(color="#FF5722", width=2, dash="dash")))
        test_start = df_feat.index[-5]
        fig.add_vrect(x0=test_start, x1=df_feat.index[-1],
                      fillcolor="rgba(255,193,7,0.12)", line_width=0,
                      annotation_text="Test Period",
                      annotation_position="top left")
        fig.update_layout(
            title="GDP Growth: Actual vs Model Predictions (2002–2025)",
            xaxis_title="Year", yaxis_title="GDP Growth (%)",
            height=450, template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        y_test = actual_series.iloc[-5:]
        y_pred = pred_series.iloc[-5:]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}%")
        c2.metric("Test MAE",  f"{mean_absolute_error(y_test, y_pred):.3f}%")
        c3.metric("Test R²",   f"{r2_score(y_test, y_pred):.3f}")
        c4.metric("Test Years", f"{len(y_test)}")

        st.info(
            "📌 A temporal (time-based) train/test split is used to prevent "
            "look-ahead bias — the model is never trained on future data. "
            "With only 26 annual observations, test R² varies significantly "
            "due to regime changes (COVID, post-IMF stabilisation)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Scenario Analysis ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Scenario Analysis":
    st.subheader("🎯 2026 GDP Growth — Policy Scenario Simulator")
    st.caption(
        "Adjust macro levers below and the model will instantly estimate "
        "the implied GDP growth rate for 2026."
    )

    if model is None:
        st.warning("Train the model first by running `python scripts/train_pipeline.py`.")
    else:
        base_row = df_feat.reindex(columns=feature_names).fillna(0).iloc[-1].to_dict()

        col_sliders, col_result = st.columns([1, 1])

        with col_sliders:
            st.markdown("**Monetary & Fiscal**")
            inflation   = st.slider("Inflation CPI (%)",       1.0, 40.0,
                                    float(round(base_row.get("inflation_cpi_pct", 8), 1)), 0.5)
            policy_rate = st.slider("Policy Rate (%)",         5.0, 25.0,
                                    float(round(base_row.get("policy_rate_pct", 11), 1)), 0.25)
            pub_debt    = st.slider("Public Debt (% GDP)",     40.0, 100.0,
                                    float(round(base_row.get("public_debt_gdp_pct", 65), 1)), 1.0)
            st.markdown("**External Sector**")
            rem_growth  = st.slider("Remittances YoY Growth (%)", -20.0, 40.0,
                                    float(round(base_row.get("remittances_growth", 10), 1)), 1.0)
            forex_cover = st.slider("FX Import Cover (months)",    0.5, 8.0,
                                    float(round(base_row.get("forex_months_import", 3), 1)), 0.1)
            pkr_change  = st.slider("PKR Depreciation YoY (%)",   -5.0, 40.0,
                                    float(round(base_row.get("pkr_yoy_change", 5), 1)), 0.5)

        scenario_row = {
            **base_row,
            "inflation_cpi_pct":   inflation,
            "policy_rate_pct":     policy_rate,
            "public_debt_gdp_pct": pub_debt,
            "remittances_growth":  rem_growth,
            "forex_months_import": forex_cover,
            "pkr_yoy_change":      pkr_change,
        }
        X_scenario = pd.DataFrame([scenario_row])[feature_names].fillna(0)
        predicted_growth = float(model.predict(X_scenario)[0])

        with col_result:
            colour = "green" if predicted_growth > 3 else "red" if predicted_growth < 1 else "orange"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted GDP Growth 2026</div>
                <div class="metric-value" style="color: {colour}">
                    {predicted_growth:+.2f}%
                </div>
                <div class="metric-delta">Based on your scenario inputs</div>
            </div>
            """, unsafe_allow_html=True)

            base_growth = float(df_raw.iloc[-1]["gdp_growth_pct"])
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_growth,
                delta={"reference": base_growth},
                gauge={
                    "axis": {"range": [-3, 10]},
                    "bar":  {"color": colour},
                    "steps": [
                        {"range": [-3, 1],   "color": "#ffcdd2"},
                        {"range": [1, 3.5],  "color": "#fff9c4"},
                        {"range": [3.5, 10], "color": "#c8e6c9"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3},
                                  "thickness": 0.75, "value": 3},
                },
                title={"text": "GDP Growth Gauge"},
                number={"suffix": "%"},
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if predicted_growth < 1:
                st.error("⚠️ High-stress scenario: recession risk elevated.")
            elif predicted_growth < 3:
                st.warning("🟡 Moderate growth — stabilisation underway.")
            else:
                st.success("✅ Strong growth trajectory — favourable conditions.")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: About ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.subheader("ℹ️ About This Project")
    st.markdown("""
### Pakistan Economic Analytics: GDP Growth Forecasting

**Problem Statement**  
Forecast Pakistan's annual GDP growth rate using 30+ macroeconomic indicators
spanning 2000–2025, and provide an interactive economic health dashboard for
policymakers, investors, and analysts.

**Technical Stack**
| Component            | Technology |
|----------------------|------------|
| Data pipeline        | pandas, scikit-learn |
| Feature engineering  | Domain features: lag, rolling, ratios, indices |
| Models               | Ridge, Lasso, RandomForest, GradientBoosting |
| Validation           | TimeSeriesSplit cross-validation (no leakage) |
| Dashboard            | Streamlit + Plotly |
| Reproducibility      | joblib, JSON manifests |

**Project Structure**
```
pakeco/
├── data/raw/               ← Original CSV (2000–2025)
├── data/processed/         ← Engineered features
├── src/data/               ← Loading & preprocessing
├── src/features/           ← Feature engineering (35 features)
├── src/models/             ← Training, evaluation, inference
├── src/visualization/      ← Plotting utilities
├── notebooks/01_EDA.ipynb  ← End-to-end analysis
├── models/saved/           ← best_model.pkl, feature_names.json
├── scripts/train_pipeline.py
├── train_demo.py           ← Lightweight demo model trainer
├── streamlit_app.py        ← This dashboard
├── tests/                  ← 35 unit tests
└── config.py               ← Central configuration
```

**Quick Start**
```bash
pip install -r requirements.txt
python scripts/train_pipeline.py  # Train model
streamlit run streamlit_app.py    # Launch dashboard
```

**Author** — Muhammad Farooq  
GitHub: [Muhammad-Farooq13](https://github.com/Muhammad-Farooq13/Pakistan-economic-analytics)
""")
