"""
tests/test_api.py — Integration tests for FastAPI endpoints.
"""

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_index_dashboard():
    res = client.get("/")
    assert res.status_code == 200
    assert "STOCKGUARD" in res.text
    assert "demand-chart" in res.text


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "models_available" in data


def test_products():
    res = client.get("/products")
    assert res.status_code == 200
    data = res.json()
    assert "products" in data
    assert len(data["products"]) == 10
    assert data["products"][0]["name"] in [
        "Biscuits", "Bread", "Eggs", "Milk", "Oil",
        "Rice", "Soap", "Sugar", "Tea", "Yogurt",
    ]


def test_simulation_results():
    res = client.get("/simulation-results")
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) >= 4


def test_predict_endpoint_valid():
    payload = {
        "product": "Milk",
        "current_inventory": 20,
        "price": 55.0,
        "promo": 0,
        "lead_time_days": 2,
        "model_name": "censoring_aware",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["product"] == "Milk"
    assert data["forecast_demand"] > 0
    assert 0.0 <= data["stockout_risk"] <= 1.0
    assert data["stockout_risk_label"] in ["LOW", "MEDIUM", "HIGH"]
    assert "history" in data
    assert "explanation" in data
    assert "financial_impact" in data
    assert "substitution_alert" in data
    assert "daily_holding_cost" in data["financial_impact"]


def test_predict_endpoint_validation_error():
    payload = {
        "product": "Milk",
        "current_inventory": -10,  # invalid negative inventory
        "price": 55.0,
        "promo": 0,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
