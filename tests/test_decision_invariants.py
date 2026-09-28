"""
tests/test_decision_invariants.py - Failure-mode tests for inventory decisions,
leakage prevention, and probabilistic forecasts (Google Production ML guidance).
"""

import numpy as np
import pytest

from src.data import load_observed
from src.features import build_features
from src.predict import (
    PredictionRequest,
    _compute_recommended_order,
    predict,
)


def test_no_future_leakage_in_features():
    """Verify that build_features does not leak future data into lags."""
    df_obs = load_observed()
    milk = df_obs[df_obs["product"] == "Milk"].sort_values("date").copy()
    feat = build_features(milk)

    # lag_1 on row t must equal sales from row t-1, never row t or row t+1
    sales_arr = milk["sales"].values
    lag1_arr = feat["lag_1"].values

    for idx in range(1, len(lag1_arr)):
        if not np.isnan(lag1_arr[idx]):
            assert lag1_arr[idx] == sales_arr[idx - 1], (
                f"Leakage detected at index {idx}: lag_1={lag1_arr[idx]} vs sales={sales_arr[idx-1]}"
            )


def test_order_non_negativity_invariant():
    """Replenishment order quantity must never be negative, and must be zero when stock > S."""
    # For forecast=15, lead_time=2: lead_demand=30, safety_stock=22.5 => S ~ 53
    # Any inventory >= 53 must result in order = 0
    for excess_stock in [55, 100, 500, 10000]:
        order, _, _ = _compute_recommended_order(
            forecast_demand=15.0,
            current_inventory=excess_stock,
            lead_time_days=2,
        )
        assert order == 0, f"Excess inventory {excess_stock} generated non-zero order: {order}"


def test_probabilistic_ordering_invariant():
    """P10 <= P50 <= P90 must strictly hold under any forecast."""
    req = PredictionRequest(
        product="Bread",
        current_inventory=10,
        price=35.0,
        promo=0,
        lead_time_days=2,
    )
    res = predict(req)
    ex = res.explanation
    assert ex["p10"] <= ex["p50"] <= ex["p90"], (
        f"Quantile monotonicity violated: P10={ex['p10']}, P50={ex['p50']}, P90={ex['p90']}"
    )


def test_inventory_position_identity():
    """Inventory Position = On-Hand + On-Order - Reserved."""
    req = PredictionRequest(
        product="Eggs",
        current_inventory=40,
        price=60.0,
        promo=0,
        lead_time_days=1,
    )
    res = predict(req)
    ex = res.explanation
    expected_position = ex["current_inventory"] + ex["on_order"] - ex["reserved"]
    assert ex["inventory_position"] == expected_position


def test_order_zero_explanation_is_triggered():
    """When stockout risk is elevated but order=0, explanation must contain order_zero_reason."""
    req = PredictionRequest(
        product="Eggs",
        current_inventory=500,
        price=60.0,
        promo=0,
        lead_time_days=2,
    )
    res = predict(req)
    if res.recommended_order == 0 and res.stockout_risk >= 0.40:
        assert res.explanation["order_zero_reason"] is not None
        assert "covers the full stochastic lead-time demand" in res.explanation["order_zero_reason"]
