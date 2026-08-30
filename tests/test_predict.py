"""
tests/test_predict.py — Unit tests for inference and business replenishment logic.
"""

import pytest

from src.predict import (
    PredictionRequest,
    PredictionResponse,
    _compute_days_until_stockout,
    _compute_recommended_order,
    _validate_request,
    predict,
)


def test_predict_validation_errors():
    with pytest.raises(ValueError, match="current_inventory"):
        _validate_request(
            PredictionRequest(product="Milk", current_inventory=-5, price=55.0, promo=0),
            known_products=["Milk"],
        )

    with pytest.raises(ValueError, match="price"):
        _validate_request(
            PredictionRequest(product="Milk", current_inventory=10, price=0.0, promo=0),
            known_products=["Milk"],
        )

    with pytest.raises(ValueError, match="Unknown product"):
        _validate_request(
            PredictionRequest(product="Pizza", current_inventory=10, price=50.0, promo=0),
            known_products=["Milk"],
        )


def test_compute_recommended_order():
    # Demand = 30, lead time = 2 -> lead time demand = 60
    # Safety stock (2 days) = 60
    # Current inventory = 25
    # Suggested order = 60 + 60 - 25 = 95
    order = _compute_recommended_order(forecast_demand=30, current_inventory=25, lead_time_days=2)
    assert order == 95

    # Excess stock -> order should be 0 (never negative)
    order_zero = _compute_recommended_order(forecast_demand=10, current_inventory=100, lead_time_days=2)
    assert order_zero == 0


def test_compute_days_until_stockout():
    assert _compute_days_until_stockout(50, 25.0) == 2.0
    assert _compute_days_until_stockout(0, 25.0) == 0.0


def test_full_predict_pipeline():
    req = PredictionRequest(
        product="Milk",
        current_inventory=20,
        price=55.0,
        promo=0,
        model_name="censoring_aware",
    )
    res = predict(req)
    assert isinstance(res, PredictionResponse)
    assert res.product == "Milk"
    assert res.forecast_demand > 0
    assert 0.0 <= res.stockout_risk <= 1.0
    assert res.recommended_order >= 0
    assert "explanation" in res.__dict__
    assert len(res.history["dates"]) > 0
