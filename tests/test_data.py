"""
tests/test_data.py — Unit tests for src/data.py and data leakage prevention.
"""

import pandas as pd

from src.data import (
    GROUND_TRUTH_ONLY_COLS,
    OBSERVED_COLS,
    load_observed,
    load_with_ground_truth,
    summary_stats,
)


def test_load_observed_schema():
    df = load_observed()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1200
    for col in OBSERVED_COLS:
        assert col in df.columns, f"Missing observed col {col}"


def test_observed_has_no_ground_truth_leakage():
    df = load_observed()
    for col in GROUND_TRUTH_ONLY_COLS:
        assert col not in df.columns, f"Leakage: ground truth column {col} in observed data!"


def test_load_with_ground_truth_schema():
    df = load_with_ground_truth()
    assert len(df) == 1200
    for col in GROUND_TRUTH_ONLY_COLS:
        assert col in df.columns, f"Missing ground truth col {col}"


def test_summary_stats():
    df = load_observed()
    stats = summary_stats(df)
    assert len(stats) == 10
    assert "product" in stats.columns
    assert "avg_daily_sales" in stats.columns
    assert "stockout_pct" in stats.columns
