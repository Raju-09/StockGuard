"""
tests/test_baselines.py — Tests for statistical & intermittent baselines.
"""

import numpy as np
import pytest
from src.baselines import (
    CrostonSBAForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    compute_intermittent_profile,
    evaluate_baselines_on_sku,
)


def test_naive_forecaster():
    y = np.array([10.0, 15.0, 20.0])
    model = NaiveForecaster().fit(y)
    preds = model.predict(3)
    assert len(preds) == 3
    assert np.all(preds == 20.0)


def test_seasonal_naive_forecaster():
    # 7-day pattern
    pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y = np.tile(pattern, 3)  # 21 days
    model = SeasonalNaiveForecaster(seasonality=7).fit(y)
    preds = model.predict(7)
    np.testing.assert_array_equal(preds, pattern)


def test_moving_average_forecaster():
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    model = MovingAverageForecaster(window=3).fit(y)
    preds = model.predict(2)
    # window=3 on [30, 40, 50] -> mean = 40.0
    assert preds[0] == pytest.approx(40.0)
    assert preds[1] == pytest.approx(40.0)


def test_croston_sba_forecaster_intermittent():
    # Intermittent demand series: demand occurs every 3 days
    y = np.array([0, 0, 10, 0, 0, 12, 0, 0, 11, 0, 0, 9], dtype=float)
    model = CrostonSBAForecaster(alpha=0.1, method="sba").fit(y)
    preds = model.predict(5)
    assert len(preds) == 5
    # Rate should be positive and roughly demand / interval ~ 10 / 3 ~ 3.33 * (1 - 0.05) ~ 3.16
    assert 2.0 <= preds[0] <= 4.5
    assert np.all(preds == preds[0])


def test_compute_intermittent_profile():
    # Regular series
    reg = np.array([10, 12, 11, 14, 13, 15, 12])
    p_reg = compute_intermittent_profile(reg)
    assert p_reg["zero_rate"] == 0.0
    assert p_reg["is_intermittent"] is False

    # Intermittent series (>50% zeros)
    inter = np.array([0, 0, 5, 0, 0, 0, 8, 0, 0, 12, 0, 0])
    p_inter = compute_intermittent_profile(inter)
    assert p_inter["zero_rate"] > 0.5
    assert p_inter["is_intermittent"] is True
    assert p_inter["classification"] in ["intermittent", "lumpy"]


def test_evaluate_baselines_on_sku():
    y = np.random.normal(20, 3, 50)
    results = evaluate_baselines_on_sku(y, test_horizon=7)
    assert "naive" in results
    assert "seasonal_naive_7d" in results
    assert "moving_average_7d" in results
    assert "croston_sba" in results
    for _, metrics in results.items():
        assert "wape" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
