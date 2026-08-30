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


def test_cold_start_endpoint():
    payload = {
        "product_name": "Organic Almond Milk 1L",
        "category": "Dairy",
        "brand": "AlmondBreeze",
        "price": 85.0,
        "pack_size": 1,
        "unit": "Litre",
        "is_perishable": 1,
        "supplier_lead_time": 2,
        "moq": 24,
        "case_pack_size": 6,
    }
    res = client.post("/cold-start", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["product_name"] == "Organic Almond Milk 1L"
    assert data["status"] == "READY"
    assert "top_analogs" in data
    assert data["p50"] > 0
    assert data["recommended_pilot_order"] >= 24
    assert data["recommended_pilot_order"] % 6 == 0


def test_what_if_endpoint():
    payload = {
        "product": "Milk",
        "current_inventory": 30,
        "price": 55.0,
        "promo": 0,
        "lead_time_days": 2,
        "inventory_delta": -20,
        "lead_time_delta": 3,
        "promo_override": 1,
    }
    res = client.post("/simulate-what-if", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "baseline" in data
    assert "scenario" in data
    assert "deltas" in data
    assert data["scenario"]["inventory"] == 10
    assert data["scenario"]["lead_time"] == 5
    assert data["scenario"]["stockout_risk"] >= data["baseline"]["stockout_risk"]
