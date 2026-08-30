"""
tests/test_features.py — Unit tests for feature engineering and leakage prevention.
"""

import pytest

from src.data import load_observed
from src.features import (
    FEATURE_COLS,
    _leakage_guard_features,
    add_stockout_future_label,
    build_features,
    get_X_y,
)


def test_build_features():
    df = load_observed()
    feat_df = build_features(df)
    for col in FEATURE_COLS:
        assert col in feat_df.columns, f"Missing feature {col}"
    # Lag 1 on first day of each product should be NaN
    assert feat_df.groupby("product")["lag_1"].nth(0).isna().all()


def test_leakage_guard_rejects_closing_stock():
    with pytest.raises(ValueError, match="Feature leakage"):
        _leakage_guard_features(["closing_stock", "price"])


def test_leakage_guard_rejects_true_demand():
    with pytest.raises(ValueError, match="Feature leakage"):
        _leakage_guard_features(["true_demand", "lag_1"])


def test_get_X_y_no_leakage():
    df = load_observed()
    feat_df = build_features(df)
    X, y = get_X_y(feat_df, drop_na=True)
    assert len(X) == len(y)
    assert "true_demand" not in X.columns
    assert "closing_stock" not in X.columns
    assert "is_censored" not in X.columns


def test_add_stockout_future_label():
    df = load_observed()
    df_labeled = add_stockout_future_label(df, horizon=3)
    assert "future_stockout_3d" in df_labeled.columns
    assert set(df_labeled["future_stockout_3d"].unique()).issubset({0, 1})
