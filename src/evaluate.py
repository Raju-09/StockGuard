"""
src/evaluate.py — Evaluation metrics.

Phase: 7 (proper evaluation)
Status: STUB — implementation added in Phase 7.

Responsibility:
    Compute forecast accuracy metrics on a temporal hold-out set.
    All metrics must be interpretable in business terms, not just
    abstract statistics.

Metrics:
    MAE  — Mean Absolute Error.  Unit: units/day.
    RMSE — Root Mean Squared Error.  Unit: units/day.
    WAPE — Weighted Absolute Percentage Error.
           = sum(|actual - pred|) / sum(actual)
           Better than MAPE when actuals contain zeros (e.g. stockout days).

Rules:
    - Never evaluate on rows where true_demand is needed but unavailable.
    - Report metrics separately for:
        * all rows
        * censored rows only  (is_censored == 1)
        * non-censored rows   (is_censored == 0)
      That split is how we measure whether stockout-aware models help.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Mean Absolute Error."""
    # Phase 7: implement.
    raise NotImplementedError("Phase 7: implement mae()")


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Root Mean Squared Error."""
    # Phase 7: implement.
    raise NotImplementedError("Phase 7: implement rmse()")


def wape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    Returns a value in [0, ∞). Lower is better.
    Returns NaN if sum(actual) == 0 (degenerate case).
    """
    # Phase 7: implement.
    raise NotImplementedError("Phase 7: implement wape()")


def evaluation_report(
    df: pd.DataFrame,
    actual_col: str = "true_demand",
    predicted_col: str = "predicted_demand",
) -> dict[str, float]:
    """Return a dict with MAE, RMSE, WAPE for all / censored / non-censored rows.

    Parameters
    ----------
    df:
        Must contain ``actual_col``, ``predicted_col``, ``is_censored``.
    actual_col:
        Column name for ground-truth demand.
    predicted_col:
        Column name for model predictions.

    Returns
    -------
    dict
        Keys: ``mae_all``, ``rmse_all``, ``wape_all``,
              ``mae_censored``, ``rmse_censored``, ``wape_censored``,
              ``mae_uncensored``, ``rmse_uncensored``, ``wape_uncensored``.
    """
    # Phase 7: implement.
    raise NotImplementedError("Phase 7: implement evaluation_report()")
