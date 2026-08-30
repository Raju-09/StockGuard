"""
src/features.py — Feature engineering.

Phase: 6 (feature engineering)
Status: STUB — implementation added in Phase 6.

Responsibility:
    Take a clean observed DataFrame and produce a feature matrix X
    suitable for supervised learning. All features must be computable
    from information available BEFORE the prediction date.

Features planned (chronological availability guaranteed):
    - day_of_week         (from date)
    - month               (from date)
    - is_weekend          (from date)
    - is_holiday          (from holiday column)
    - promo               (from promo column)
    - lag_1               (sales yesterday)
    - lag_7               (sales 7 days ago)
    - rolling_mean_7      (mean of last 7 days sales)
    - rolling_mean_14     (mean of last 14 days sales)
    - opening_stock       (known before the day's trading begins)
    - price               (known before the day's trading begins)

NEVER include:
    - closing_stock of the SAME day (future leakage)
    - true_demand, lost_sales, is_censored, demand_mean (ground-truth leakage)
"""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the feature matrix from the observed DataFrame.

    Parameters
    ----------
    df:
        Output of ``data.load_observed()``. Must contain at least:
        ``date``, ``product``, ``sales``, ``price``, ``promo``,
        ``holiday``, ``opening_stock``.

    Returns
    -------
    pd.DataFrame
        Feature matrix X. Target column (``sales`` or imputed demand)
        is NOT included here — kept separate to avoid confusion.
    """
    # Phase 6: implement lag features, rolling means, calendar features.
    raise NotImplementedError("Phase 6: implement build_features()")
