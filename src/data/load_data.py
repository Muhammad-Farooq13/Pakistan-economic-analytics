"""src/data/load_data.py
======================
Responsible for loading and performing light-touch validation of the raw
Pakistan economic indicators dataset.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame and apply minimal type coercions.

    Parameters
    ----------
    filepath : path-like, optional
        Override the default raw data path defined in config.

    Returns
    -------
    pd.DataFrame
        Raw data with correct dtypes; ``year`` set as the index.
    """
    if filepath is None:
        from config import RAW_DATA_FILE
        filepath = RAW_DATA_FILE

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Loaded raw data: %s rows × %s cols", *df.shape)

    # ── Type coercions ──────────────────────────────────────────────────
    df["year"] = df["year"].astype(int)
    df["imf_program_active"] = df["imf_program_active"].astype(int)

    # Ordinal mapping for gdp_growth_category
    growth_order = {
        "Recession/Negative": 0,
        "Low Growth":         1,
        "Moderate Growth":    2,
        "High Growth":        3,
    }
    if "gdp_growth_category" in df.columns:
        df["gdp_growth_category"] = df["gdp_growth_category"].map(growth_order)

    # Ordinal mapping for inflation_category
    inflation_order = {
        "Low":             0,
        "Moderate":        1,
        "High":            2,
        "Hyperinflationary": 3,
    }
    if "inflation_category" in df.columns:
        df["inflation_category"] = df["inflation_category"].map(inflation_order)

    # Ordinal mapping for decade
    decade_map = {"2000s": 0, "2010s": 1, "2020s": 2}
    if "decade" in df.columns:
        df["decade"] = df["decade"].map(decade_map)

    df = df.set_index("year").sort_index()
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """Return a summary dict useful for quick sanity checks."""
    return {
        "shape":          df.shape,
        "missing_counts": df.isnull().sum().to_dict(),
        "dtypes":         df.dtypes.astype(str).to_dict(),
        "year_range":     (int(df.index.min()), int(df.index.max())),
    }
