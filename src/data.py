"""
src/data.py — Data loading and validation.

Responsibility: Load CSVs, validate schema and dtypes, enforce the
leakage boundary between observable and ground-truth columns.

LEAKAGE RULE:
    GROUND_TRUTH_ONLY_COLS must NEVER appear in a model's feature matrix.
    They exist only in with_ground_truth.csv for evaluation purposes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Column contracts ──────────────────────────────────────────────────────────

# Columns a real store system could log (no ground-truth information).
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

# Columns that exist ONLY because data is simulated.
# These must NEVER be used as model input features.
GROUND_TRUTH_ONLY_COLS: list[str] = [
    "true_demand",
    "lost_sales",
    "is_censored",
    "demand_mean",
]

# Columns that must be non-negative (validated on load).
_NON_NEGATIVE_COLS: list[str] = [
    "opening_stock",
    "sales",
    "closing_stock",
    "price",
]

# Columns that are binary (0 or 1).
_BINARY_COLS: list[str] = ["promo", "holiday", "stockout_flag"]

# Default paths relative to the project root.
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OBSERVED_CSV = DEFAULT_RAW_DIR / "observed_daily.csv"
TRUTH_CSV = DEFAULT_RAW_DIR / "with_ground_truth.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_schema(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{label}] Missing required columns: {missing}. "
            f"Available: {df.columns.tolist()}"
        )


def _validate_non_negative(df: pd.DataFrame, label: str) -> None:
    """Raise ValueError if any non-negative column contains a negative value."""
    for col in _NON_NEGATIVE_COLS:
        if col not in df.columns:
            continue
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            raise ValueError(
                f"[{label}] Column '{col}' has {n_neg} negative values. "
                "Check for data corruption."
            )


def _validate_binary(df: pd.DataFrame, label: str) -> None:
    """Warn if binary columns contain values outside {0, 1}."""
    for col in _BINARY_COLS:
        if col not in df.columns:
            continue
        bad = ~df[col].isin([0, 1])
        if bad.any():
            n_bad = bad.sum()
            raise ValueError(
                f"[{label}] Column '{col}' has {n_bad} values outside {{0, 1}}."
            )


def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes after loading from CSV."""
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    int_cols = [
        "opening_stock", "sales", "closing_stock", "delivery_qty",
        "promo", "holiday", "stockout_flag",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "price" in df.columns:
        df["price"] = df["price"].astype(float)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_observed(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the observable store data.

    This file contains only columns a real store system could log.
    ``true_demand``, ``lost_sales``, ``is_censored``, and ``demand_mean``
    are NOT present — using them as features would be leakage.

    Parameters
    ----------
    path:
        Path to ``observed_daily.csv``. Defaults to
        ``data/raw/observed_daily.csv`` relative to the project root.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame sorted by ``product``, ``date``.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    p = Path(path) if path is not None else OBSERVED_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"observed_daily.csv not found at: {p}\n"
            "Run: python src/simulate.py"
        )
    df = pd.read_csv(p)
    df = _apply_dtypes(df)
    _validate_schema(df, OBSERVED_COLS, "observed")
    _validate_non_negative(df, "observed")
    _validate_binary(df, "observed")

    # Leakage guard: abort if ground-truth columns somehow ended up in this file.
    leaked = [c for c in GROUND_TRUTH_ONLY_COLS if c in df.columns]
    if leaked:
        raise ValueError(
            f"[observed] Ground-truth leakage detected: columns {leaked} are "
            "present in the observed file. These must never be model features."
        )

    return df.sort_values(["product", "date"]).reset_index(drop=True)


def load_with_ground_truth(path: Path | str | None = None) -> pd.DataFrame:
    """Load the full dataset including hidden true demand.

    Use ONLY for evaluation: measure how well models recover true demand.
    The columns ``true_demand``, ``lost_sales``, ``is_censored``,
    ``demand_mean`` must NEVER be passed to a model as features.

    Parameters
    ----------
    path:
        Path to ``with_ground_truth.csv``. Defaults to
        ``data/raw/with_ground_truth.csv``.

    Returns
    -------
    pd.DataFrame
        Full DataFrame sorted by ``product``, ``date``.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    p = Path(path) if path is not None else TRUTH_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"with_ground_truth.csv not found at: {p}\n"
            "Run: python src/simulate.py"
        )
    df = pd.read_csv(p)
    df = _apply_dtypes(df)
    all_cols = OBSERVED_COLS + GROUND_TRUTH_ONLY_COLS
    _validate_schema(df, all_cols, "ground_truth")
    _validate_non_negative(df, "ground_truth")

    return df.sort_values(["product", "date"]).reset_index(drop=True)


def get_product_list(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique product names."""
    return sorted(df["product"].unique().tolist())


def get_date_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date) from the DataFrame."""
    return df["date"].min(), df["date"].max()


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product summary: total sales, avg daily sales, stockout days.

    Safe to call on either the observed or ground-truth DataFrame.
    """
    agg = df.groupby("product", as_index=False).agg(
        days=("date", "count"),
        total_sales=("sales", "sum"),
        avg_daily_sales=("sales", "mean"),
        stockout_days=("stockout_flag", "sum"),
    )
    agg["avg_daily_sales"] = agg["avg_daily_sales"].round(2)
    agg["stockout_pct"] = (agg["stockout_days"] / agg["days"] * 100).round(1)
    return agg.sort_values("stockout_days", ascending=False).reset_index(drop=True)
