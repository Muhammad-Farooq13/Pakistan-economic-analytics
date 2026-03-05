"""src/visualization/plots.py
==============================
Publication-quality visualizations for Pakistan economic indicators.

All functions accept an optional ``ax`` parameter so they can be
composed into multi-panel figures, and an optional ``save_path``
for report generation.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

# apply a clean global style
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})
PAKISTAN_GREEN  = "#01411C"
PAKISTAN_WHITE  = "#FFFFFF"
ACCENT          = "#1565C0"
DANGER          = "#C62828"


# ─────────────────────────────────────────────────────────────────────────────
# 1. GDP & growth timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_gdp_timeline(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Dual-axis chart: GDP (USD bn) and GDP growth (%) over time."""
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.fill_between(df.index, df["gdp_usd_bn"], alpha=0.3, color=ACCENT)
    ax1.plot(df.index, df["gdp_usd_bn"], color=ACCENT, linewidth=2.5)
    ax1.set_ylabel("GDP (USD Billion)", color=ACCENT, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=ACCENT)

    ax2 = ax1.twinx()
    colors = [DANGER if g < 0 else PAKISTAN_GREEN for g in df["gdp_growth_pct"]]
    ax2.bar(df.index, df["gdp_growth_pct"], color=colors, alpha=0.7, width=0.6)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("GDP Growth (%)", color=PAKISTAN_GREEN, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=PAKISTAN_GREEN)

    ax1.set_title("Pakistan: GDP Size & Growth Rate (2000–2025)",
                  fontsize=15, fontweight="bold", pad=14)
    ax1.set_xlabel("Year")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame, cols: list[str] | None = None, save_path=None
) -> plt.Figure:
    """Annotated seaborn heatmap of Pearson correlations."""
    if cols:
        df = df[cols]
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.4,
        annot_kws={"size": 7}, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-indicator time series panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_macro_panel(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """4-panel chart of core macroeconomic indicators."""
    indicators = [
        ("inflation_cpi_pct",      "Inflation (CPI %)",       "#E53935"),
        ("pkr_per_usd",            "PKR / USD Exchange Rate",  "#6D4C41"),
        ("forex_reserves_usd_bn",  "FX Reserves (USD bn)",     "#00897B"),
        ("public_debt_gdp_pct",    "Public Debt (% GDP)",      "#5E35B1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.flatten()

    for ax, (col, label, color) in zip(axes, indicators):
        ax.plot(df.index, df[col], color=color, linewidth=2)
        ax.fill_between(df.index, df[col], alpha=0.15, color=color)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Year")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.suptitle("Pakistan Key Macroeconomic Indicators (2000–2025)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. External sector chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_external_sector(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Stacked view of exports, imports, remittances, and trade balance."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.fill_between(df.index, df["exports_usd_bn"], alpha=0.6,
                     color="#43A047", label="Exports")
    ax1.fill_between(df.index, df["imports_usd_bn"], alpha=0.4,
                     color="#E53935", label="Imports")
    ax1.fill_between(df.index, df["remittances_usd_bn"], alpha=0.7,
                     color="#1E88E5", label="Remittances")
    ax1.set_ylabel("USD Billion")
    ax1.set_title("Exports, Imports & Remittances", fontweight="bold")
    ax1.legend(loc="upper left")

    ax2.bar(df.index, df["trade_balance_usd_bn"],
            color=[DANGER if v < 0 else PAKISTAN_GREEN
                   for v in df["trade_balance_usd_bn"]],
            alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("USD Billion")
    ax2.set_title("Trade Balance", fontweight="bold")

    fig.suptitle("Pakistan External Sector (2000–2025)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Distribution plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(
    df: pd.DataFrame, cols: list[str], save_path=None
) -> plt.Figure:
    """Grid of KDE + histogram panels for selected columns."""
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i],
                     color=ACCENT, alpha=0.6)
        axes[i].set_title(col, fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
