"""src/models/evaluate.py
=========================
Model evaluation utilities: regression metrics, residual analysis,
feature importance, and SHAP-based explainability.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

def regression_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Return a comprehensive regression scorecard.

    Metrics
    -------
    MAE   – Mean Absolute Error (same units as target)
    RMSE  – Root Mean Squared Error (penalises large errors)
    MAPE  – Mean Absolute Percentage Error (unit-free)
    R²    – Coefficient of Determination (variance explained)
    """
    metrics = {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2":   r2_score(y_true, y_pred),
    }
    logger.info(
        "%s  MAE=%.3f  RMSE=%.3f  MAPE=%.1f%%  R²=%.3f",
        model_name, metrics["MAE"], metrics["RMSE"],
        metrics["MAPE"], metrics["R2"],
    )
    return metrics


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a dict of {model_name: metrics_dict}.
    """
    return pd.DataFrame(results).T.sort_values("RMSE")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(
    pipeline, feature_names: list[str], top_n: int = 15
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models inside a Pipeline.

    Falls back to permutation importance for linear models.
    """
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        raise ValueError("Model does not expose feature importance.")

    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False).head(top_n)
    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    save_path=None,
) -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df.sort_values("importance").plot.barh(
        x="feature", y="importance", ax=ax, color="steelblue", legend=False
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Residual analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true, y_pred, years=None, title="Actual vs Predicted GDP Growth (%)",
    save_path=None
) -> plt.Figure:
    """Line plot comparing actual and predicted values over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = years if years is not None else range(len(y_true))
    ax.plot(x, y_true, "o-", label="Actual",    color="#2196F3", linewidth=2)
    ax.plot(x, y_pred, "s--", label="Predicted", color="#FF5722", linewidth=2)
    ax.fill_between(
        x, np.array(y_true) - np.array(y_pred),
        alpha=0.1, color="gray", label="Residual"
    )
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
