"""
tests/test_cold_start.py — Unit tests for new-SKU cold start forecasting.
"""

from src.cold_start import (
    NewProductRequest,
    compute_product_similarity,
    run_cold_start_forecast,
)


def test_cold_start_dairy_analog_match():
    # New Organic Whole Milk 1L should match closest to Milk
    req = NewProductRequest(
        product_name="Organic Whole Milk 1L",
        category="Dairy",
        brand="PureFarm",
        price=60.0,
        pack_size=1,
        unit="Litre",
        is_perishable=1,
        supplier_lead_time=2,
        moq=30,
        case_pack_size=6,
    )
    result = run_cold_start_forecast(req)
    assert result.status == "READY"
    assert result.confidence_level in ["HIGH", "MEDIUM"]
    assert result.max_similarity > 0.60
    assert result.top_analogs[0]["product"] in ["Milk", "Yogurt"]
    assert result.p10 < result.p50 <= result.p80 <= result.p90 <= result.p95
    # Verification of MOQ and pack size rounding
    assert result.recommended_pilot_order >= req.moq
    assert result.recommended_pilot_order % req.case_pack_size == 0


def test_cold_start_low_confidence_abstain():
    # Truly novel SKU with weak similarity to existing grocery catalog
    req = NewProductRequest(
        product_name="Solar Powered Drone Toy",
        category="Electronics",
        brand="TechGizmo",
        price=3500.0,
        pack_size=1,
        unit="Piece",
        is_perishable=0,
        supplier_lead_time=7,
        moq=10,
        case_pack_size=5,
    )
    result = run_cold_start_forecast(req)
    assert result.status == "LOW_CONFIDENCE_ABSTAIN"
    assert result.confidence_level == "LOW"
    assert "ABSTAINS" in result.explanation
    assert result.recommended_pilot_order % req.case_pack_size == 0


def test_compute_product_similarity_identical_attributes():
    req = NewProductRequest(
        product_name="Identical Milk Test",
        category="Dairy",
        brand="StoreBrand",
        price=55.0,
        pack_size=1,
        is_perishable=1,
    )
    milk_catalog = {
        "category": "Dairy",
        "brand": "StoreBrand",
        "price": 55.0,
        "pack_size": 1,
        "is_perishable": 1,
    }
    sim = compute_product_similarity(req, milk_catalog)
    assert sim == 1.0
