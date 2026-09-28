"""
tests/test_evaluate.py — Unit tests for evaluation metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import bias, evaluation_report, mae, rmse, wape


def test_mae_rmse_wape_bias_basic():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 30.0])

    assert mae(actual, predicted) == pytest.approx(4.0 / 3.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(8.0 / 3.0))
    assert wape(actual, predicted) == pytest.approx(4.0 / 60.0)
    assert bias(actual, predicted) == pytest.approx(0.0)


def test_wape_handles_zero_actuals_without_zerodivisionerror():
    actual = np.array([0.0, 0.0])
    predicted = np.array([2.0, 3.0])
    # Denominator is 0 -> returns nan without crashing
    val = wape(actual, predicted)
    assert np.isnan(val)


def test_evaluation_report_splits_censored():
    df = pd.DataFrame({
        "actual": [10, 20, 30, 40],
        "pred": [10, 15, 25, 40],
        "is_censored": [0, 1, 1, 0],
    })
    rep = evaluation_report(df, actual_col="actual", predicted_col="pred", censored_col="is_censored")
    assert rep["n_all"] == 4
    assert rep["n_censored"] == 2
    assert rep["n_uncensored"] == 2
    assert rep["mae_uncensored"] == 0.0
    assert rep["mae_censored"] == 5.0


def test_pinball_loss_and_coverage():
    from src.evaluate import interval_coverage, pinball_loss

    actual = np.array([10.0, 20.0, 30.0])
    p50 = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(actual, p50, alpha=0.5) == 0.0

    lower = np.array([8.0, 18.0, 25.0])
    upper = np.array([12.0, 22.0, 35.0])
    cov = interval_coverage(actual, lower, upper)
    assert cov["coverage"] == 1.0
    assert cov["n"] == 3


def test_walk_forward_cv_execution():
    from src.evaluate import walk_forward_cv
    from src.train import NaiveForecast

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "product": "Milk",
        "sales": np.random.randint(10, 30, size=30),
        "true_demand": np.random.randint(10, 30, size=30),
        "opening_stock": 50,
        "closing_stock": 25,
        "price": 50.0,
        "promo": 0,
        "holiday": 0,
        "stockout_flag": 0,
    })
    res = walk_forward_cv(df, model_factory=lambda: NaiveForecast(), n_folds=3, test_days=5)
    assert "per_fold" in res
    assert "aggregate" in res
    assert res["aggregate"]["n_folds"] > 0

