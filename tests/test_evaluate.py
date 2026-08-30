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
