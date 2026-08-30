"""
src/data.py — Data loading and validation.

Phase: 2 (pandas fundamentals)
Status: STUB — implementation added in Phase 2.

Responsibility:
    Load `observed_daily.csv` and `with_ground_truth.csv`, validate column
    types, check for missing values, and return clean DataFrames.

Rule: This module must NEVER read `true_demand`, `lost_sales`,
      `is_censored`, or `demand_mean` from the CSV and expose them as
      training features. Those columns exist only for evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns that are available in a real store system (no ground truth leakage).
OBSERVED_COLS: list[str] = [
    "date",
    "store",
    "product",
    "opening_stock",
    "sales",
    "closing_stock",
    "price",
    "promo",
    "holiday",
    "delivery_qty",
    "stockout_flag",
    "data_source",
]

# Columns that exist only because data is simulated.
# NEVER use these as model features.
GROUND_TRUTH_ONLY_COLS: list[str] = [
    "true_demand",
    "lost_sales",
    "is_censored",
    "demand_mean",
]


def load_observed(path: Path | str) -> pd.DataFrame:
    """Load the observable store data. No ground-truth columns included.

    Parameters
    ----------
    path:
        Path to ``observed_daily.csv``.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with ``date`` as datetime and correct dtypes.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If expected columns are missing.
    """
    # Phase 2: implement full loading + validation.
    raise NotImplementedError("Phase 2: implement load_observed()")


def load_with_ground_truth(path: Path | str) -> pd.DataFrame:
    """Load the full dataset including hidden true demand.

    Use ONLY for evaluation — never pass ground-truth columns to a model.

    Parameters
    ----------
    path:
        Path to ``with_ground_truth.csv``.

    Returns
    -------
    pd.DataFrame
        Full DataFrame with all columns.
    """
    # Phase 2: implement.
    raise NotImplementedError("Phase 2: implement load_with_ground_truth()")
