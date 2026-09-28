"""
tests/test_api.py — Integration tests for FastAPI endpoints.
"""

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_index_landing():
    res = client.get("/")
    assert res.status_code == 200
    assert "STOCKGUARD" in res.text
    assert "Open Dashboard" in res.text


def test_dashboard_ui():
    res = client.get("/dashboard")
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


# ── v1 Product Capabilities Tests ─────────────────────────────────────────────

def test_v1_upload_preview():
    csv_text = "Order Date,StockCode,Qty Sold,Price,Store Location\n2026-01-01,Milk,15,55.0,Store-A\n2026-01-02,Milk,12,55.0,Store-A"
    res = client.post("/api/v1/upload/preview", json={"csv_content": csv_text})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "detected_headers" in data
    assert "Order Date" in data["detected_headers"]
    assert "suggested_mapping" in data
    assert data["suggested_mapping"]["date"] == "Order Date"
    assert data["suggested_mapping"]["sku"] == "StockCode"
    assert len(data["preview_rows"]) == 2


def test_v1_upload_validate_and_ingest():
    csv_text = "Order Date,StockCode,Qty Sold,Price\n2026-01-01,Milk,15,55.0\n2026-01-02,Milk,12,55.0\n2026-01-03,Milk,18,55.0"
    mapping = {
        "date": "Order Date",
        "sku": "StockCode",
        "sales": "Qty Sold",
        "price": "Price",
    }
    res = client.post("/api/v1/upload/validate-and-ingest", json={"csv_content": csv_text, "mapping": mapping})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["report"]["total_rows"] == 3
    assert data["report"]["unique_skus"] == 1


def test_v1_load_sample_store():
    res = client.post("/api/v1/store/load-sample")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "Sample store dataset active" in data["message"]


def test_v1_actions_queue():
    res = client.get("/api/v1/actions")
    assert res.status_code == 200
    data = res.json()
    assert "actions" in data
    assert len(data["actions"]) >= 5
    action = data["actions"][0]
    assert "decision_id" in action
    assert "sku" in action
    assert "status" in action
    assert "runway_days" in action
    assert "why_breakdown" in action
    wb = action["why_breakdown"]
    assert "p10_daily" in wb
    assert "p50_daily" in wb
    assert "p90_daily" in wb
    assert "target_service_level" in wb


def test_v1_actions_execute_and_audit():
    # 1. Get an action to approve
    actions_res = client.get("/api/v1/actions")
    actions = actions_res.json()["actions"]
    assert len(actions) > 0
    target_action = actions[0]
    dec_id = target_action["decision_id"]

    # 2. Execute approval
    exec_payload = {
        "decision_id": dec_id,
        "action": "APPROVED",
        "approved_units": 45,
        "user_notes": "Integration test approved order",
        "unit_price": 55.0,
    }
    exec_res = client.post("/api/v1/actions/execute", json=exec_payload)
    assert exec_res.status_code == 200
    exec_data = exec_res.json()
    assert exec_data["success"] is True
    assert exec_data["action_recorded"] == "APPROVED"
    assert "purchase_order" in exec_data
    assert exec_data["purchase_order"]["total_units"] == 45

    # 3. Query audit history
    audit_res = client.get("/api/v1/audit/history")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert "decisions" in audit_data
    assert "purchase_orders" in audit_data
    assert len(audit_data["decisions"]) > 0
    assert len(audit_data["purchase_orders"]) > 0
