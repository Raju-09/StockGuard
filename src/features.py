"""
src/features.py — Feature engineering for demand forecasting.

LEAKAGE RULES (enforced by _leakage_guard()):
    1. GROUND_TRUTH_ONLY_COLS (true_demand, lost_sales, is_censored, demand_mean)
       must NEVER appear in the feature matrix. They are not observable at
       prediction time in a real store.

    2. closing_stock of the CURRENT day must NOT be used as a feature
       when predicting that day's demand. (opening_stock is fine — it is
       known before the day's trading begins.)

    3. All lag and rolling features use shift(1) before aggregating, so
       the current day's sales are excluded. Predictions can only see the past.

Feature columns produced:
    Calendar:        day_of_week, month, is_weekend, is_holiday
    Price/promo:     price, promo
    Inventory:       opening_stock
    Lags (per SKU):  lag_1, lag_7
    Rolling (per SKU): rolling_mean_7, rolling_mean_14,
                        rolling_std_7, rolling_min_7, rolling_max_7

Rows with insufficient lag history (first 14 days per product) will have
NaN values; these are filled with the product's train-set mean during
training and dropped from validation/test evaluation.
"""

from __future__ import annotations

import pandas as pd

from src.data import GROUND_TRUTH_ONLY_COLS

# ── Constants ──────────────────────────────────────────────────────────────────

# Columns the model MAY use as features.
FEATURE_COLS: list[str] = [
    # Calendar
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    # Product attributes (known before trading starts)
    "price",
    "promo",
    "opening_stock",
    # Lag / rolling (past sales only, shifted by ≥1 day)
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "rolling_min_7",
    "rolling_max_7",
]

# Target column for demand forecasting models.
TARGET_COL: str = "sales"

# Columns that are NOT features but are useful for post-hoc analysis.
_META_COLS: list[str] = [
    "date",
    "store",
    "product",
    "closing_stock",
    "delivery_qty",
    "stockout_flag",
    "data_source",
]

# Leakage: columns forbidden from appearing as features in X.
# Note: closing_stock IS in observed_daily.csv (the store logs it at end-of-day),
# but must NOT be used as a feature for same-day demand prediction.
# The guard on raw df only checks ground-truth columns.
# The guard on the feature matrix X checks both.
_FORBIDDEN_IN_X: frozenset[str] = frozenset(
    GROUND_TRUTH_ONLY_COLS + ["closing_stock"]
)
# Only ground-truth columns are forbidden in the raw observed DataFrame.
_FORBIDDEN_IN_RAW: frozenset[str] = frozenset(GROUND_TRUTH_ONLY_COLS)


# ── Leakage guard ──────────────────────────────────────────────────────────────


def _leakage_guard_raw(df: pd.DataFrame, context: str = "") -> None:
    """Check that ground-truth-only columns are not in a raw observed DataFrame.

    closing_stock IS allowed in the raw DataFrame (it is logged by the store).
    It is only forbidden when extracted into a feature matrix X.
    """
    leaked = [c for c in _FORBIDDEN_IN_RAW if c in df.columns]
    if leaked:
        ctx = f" [{context}]" if context else ""
        raise ValueError(
            f"{ctx} Ground-truth leakage! Columns {leaked} must not appear "
            f"in observed store data — they require knowing the true demand."
        )


def _leakage_guard_features(cols: list[str], context: str = "") -> None:
    """Check that no forbidden column is included in a feature list."""
    leaked = [c for c in cols if c in _FORBIDDEN_IN_X]
    if leaked:
        ctx = f" [{context}]" if context else ""
        raise ValueError(
            f"{ctx} Feature leakage! Forbidden columns in feature list: {leaked}. "
            f"'closing_stock' cannot be a feature (same-day future info). "
            f"Ground-truth columns cannot be features."
        )


# ── Feature builder ────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns to an observed DataFrame.

    Input ``df`` must be the output of ``data.load_observed()`` (no ground-truth
    columns). All new columns are derived only from information available
    BEFORE the day's closing stock is known.

    Parameters
    ----------
    df:
        Observed DataFrame. Must contain at minimum:
        ``date``, ``product``, ``sales``, ``price``, ``promo``,
        ``holiday``, ``opening_stock``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with additional feature columns. Original columns
        are preserved. Rows are sorted by (product, date).

    Raises
    ------
    ValueError
        If ground-truth leakage columns are detected in the input DataFrame.
    """
    # Guard: observed DataFrame must not contain ground-truth columns.
    _leakage_guard_raw(df, context="build_features input")

    df = df.copy()
    df = df.sort_values(["product", "date"]).reset_index(drop=True)

    # ── Calendar features ──────────────────────────────────────────────────────
    df["day_of_week"] = df["date"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = df["holiday"].astype(int)

    # ── Lag features per product ───────────────────────────────────────────────
    # shift(1): yesterday's sales. Safe — no current-day information.
    df["lag_1"] = df.groupby("product")["sales"].transform(lambda x: x.shift(1))
    df["lag_7"] = df.groupby("product")["sales"].transform(lambda x: x.shift(7))

    # ── Rolling features per product ───────────────────────────────────────────
    # shift(1) before rolling: window covers [t-8, t-1] for mean_7.
    # min_periods=1 avoids NaN when history < window size.
    def _shifted_rolling(series: pd.Series, window: int, func: str) -> pd.Series:
        shifted = series.shift(1)
        rolled = shifted.rolling(window, min_periods=1)
        if func == "mean":
            return rolled.mean()
        if func == "std":
            return rolled.std().fillna(0.0)
        if func == "min":
            return rolled.min()
        if func == "max":
            return rolled.max()
        raise ValueError(f"Unknown rolling func: {func}")

    for col_name, window, func in [
        ("rolling_mean_7", 7, "mean"),
        ("rolling_mean_14", 14, "mean"),
        ("rolling_std_7", 7, "std"),
        ("rolling_min_7", 7, "min"),
        ("rolling_max_7", 7, "max"),
    ]:
        df[col_name] = df.groupby("product")["sales"].transform(
            lambda x, w=window, f=func: _shifted_rolling(x, w, f)
        )

    # Leakage guard on output feature list
    _leakage_guard_features(FEATURE_COLS, context="build_features output")

    return df


def get_X_y(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y from a featured DataFrame.

    Parameters
    ----------
    df:
        Output of ``build_features()``.
    feature_cols:
        Subset of FEATURE_COLS to use. Defaults to all of FEATURE_COLS.
    target_col:
        Target column name. Default is ``"sales"``.
    drop_na:
        If True, drop rows where any feature is NaN (e.g. first 14 rows
        per product that lack full lag history).

    Returns
    -------
    X : pd.DataFrame  (feature matrix)
    y : pd.Series     (target)

    Raises
    ------
    ValueError
        If forbidden columns appear in feature_cols.
    """
    cols = feature_cols if feature_cols is not None else FEATURE_COLS

    # Leakage guard on feature list
    _leakage_guard_features(cols, context="get_X_y")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature columns not found in DataFrame: {missing}. "
            f"Did you call build_features() first?"
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")

    if drop_na:
        mask = df[cols].notna().all(axis=1)
        df = df[mask]

    X = df[cols].copy()
    y = df[target_col].copy()
    return X, y


def add_stockout_future_label(
    df: pd.DataFrame,
    horizon: int = 3,
) -> pd.DataFrame:
    """Add a binary label: will a stockout occur within the next `horizon` days?

    This label is used to train the stockout classifier.
    It is computed retrospectively on historical data (so we can see the future
    during training) — at inference time the classifier predicts this label
    WITHOUT seeing the future.

    Uses ``stockout_flag`` (observable) not ``is_censored`` (ground-truth-only).

    Parameters
    ----------
    df:
        Observed DataFrame with ``stockout_flag`` column.
    horizon:
        Number of future days to look ahead.

    Returns
    -------
    pd.DataFrame with new column ``future_stockout_{horizon}d``.
    """
    df = df.copy().sort_values(["product", "date"])
    col = f"future_stockout_{horizon}d"
    df[col] = (
        df.groupby("product")["stockout_flag"]
        .transform(
            lambda x: x.shift(-1)
            .rolling(horizon, min_periods=1)
            .max()
        )
        .fillna(0)
        .astype(int)
    )
    return df


# ── Feature names helper ────────────────────────────────────────────────────────

def get_feature_cols() -> list[str]:
    """Return the list of valid feature column names."""
    return list(FEATURE_COLS)


def get_target_col() -> str:
    """Return the name of the demand forecast target column."""
    return TARGET_COL


def impute_train_means(
    train_df: pd.DataFrame,
    val_or_test_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaN feature values using training-set column means.

    Call this AFTER the temporal split, using only training data to compute
    fill values — never val/test data (that would be leakage).

    Returns
    -------
    (train_filled, val_or_test_filled)
    """
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    train_means = train_df[cols].mean()

    train_filled = train_df.copy()
    train_filled[cols] = train_filled[cols].fillna(train_means)

    vt_filled = val_or_test_df.copy()
    vt_filled[cols] = vt_filled[cols].fillna(train_means)

    return train_filled, vt_filled
