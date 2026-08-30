"""
app/main.py — FastAPI web application for StockGuard.

Endpoints:
    GET  /                     → Dashboard UI (HTML template)
    GET  /health               → System liveness + model registry status
    GET  /products             → List of products with summary statistics
    POST /predict              → Run demand forecast and stockout risk
    GET  /simulation-results   → Multi-policy business simulation table
    GET  /model-comparison     → Test-set metric comparison table

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.data import load_observed, summary_stats
from src.predict import (
    DEFAULT_LEAD_TIME_DAYS,
    PredictionRequest,
)
from src.predict import (
    predict as run_predict,
)
from src.train import _load_registry

# ── Paths ──────────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = MODELS_DIR / "results.json"
SIM_RESULTS_PATH = MODELS_DIR / "simulation_results.json"

# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="StockGuard API",
    description="Inventory Intelligence: Demand Forecasting under Stockout Censoring",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets (CSS, JS)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Request / Response Schemas ────────────────────────────────────────────────

class PredictPayload(BaseModel):
    product: str = Field(..., description="Product SKU name", json_schema_extra={"example": "Milk"})
    current_inventory: int = Field(..., ge=0, description="Opening stock units on hand", json_schema_extra={"example": 24})
    price: float = Field(..., gt=0.0, description="Current selling price per unit", json_schema_extra={"example": 55.0})
    promo: int = Field(0, ge=0, le=1, description="Promotion active (0 or 1)", json_schema_extra={"example": 0})
    lead_time_days: int = Field(
        DEFAULT_LEAD_TIME_DAYS, ge=1, description="Supplier lead time in days", json_schema_extra={"example": 2}
    )
    model_name: str = Field(
        "censoring_aware",
        description="Model to use (censoring_aware, gradient_boost, random_forest, linear, naive, ma7)",
        json_schema_extra={"example": "censoring_aware"},
    )
    currency: str = Field("INR", description="Currency symbol/code: INR, USD, EUR", json_schema_extra={"example": "INR"})


class ColdStartPayload(BaseModel):
    product_name: str = Field(..., description="New product name", json_schema_extra={"example": "Organic Whole Milk 1L"})
    category: str = Field(..., description="Category (Dairy, Bakery, Snacks, Grains, etc.)", json_schema_extra={"example": "Dairy"})
    brand: str = Field("StoreBrand", description="Brand name", json_schema_extra={"example": "PureFarm"})
    price: float = Field(..., gt=0.0, description="Planned retail price", json_schema_extra={"example": 60.0})
    pack_size: int = Field(1, ge=1, description="Pack size", json_schema_extra={"example": 1})
    unit: str = Field("Litre", description="Unit of measure", json_schema_extra={"example": "Litre"})
    is_perishable: int = Field(1, ge=0, le=1, description="1 if perishable, 0 otherwise", json_schema_extra={"example": 1})
    supplier_lead_time: int = Field(2, ge=1, description="Expected supplier lead time in days", json_schema_extra={"example": 2})
    moq: int = Field(20, ge=1, description="Supplier Minimum Order Quantity", json_schema_extra={"example": 20})
    case_pack_size: int = Field(6, ge=1, description="Units per case pack", json_schema_extra={"example": 6})


class WhatIfPayload(BaseModel):
    product: str = Field(..., description="Product SKU name", json_schema_extra={"example": "Milk"})
    current_inventory: int = Field(..., ge=0, json_schema_extra={"example": 24})
    price: float = Field(..., gt=0.0, json_schema_extra={"example": 55.0})
    promo: int = Field(0, ge=0, le=1, json_schema_extra={"example": 0})
    lead_time_days: int = Field(2, ge=1, json_schema_extra={"example": 2})
    # Scenario shocks
    inventory_delta: int = Field(0, description="Add or remove stock (e.g. -15)", json_schema_extra={"example": -10})
    lead_time_delta: int = Field(0, description="Supplier delay in days (e.g. +2)", json_schema_extra={"example": 2})
    promo_override: int | None = Field(None, description="Force promotion ON (1) or OFF (0)", json_schema_extra={"example": 1})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    """Serve the single-page dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    """System health check and loaded model registry."""
    registry = _load_registry(MODELS_DIR)
    models_available = list(registry.keys())
    return {
        "status": "healthy",
        "app": "StockGuard",
        "version": "0.1.0",
        "models_available": models_available,
        "models_loaded": len(models_available) > 0,
    }


@app.get("/products")
def list_products() -> dict[str, Any]:
    """Return list of products with observed summary stats for the UI dropdown."""
    try:
        df_obs = load_observed()
        stats = summary_stats(df_obs)
        records = []
        for _, row in stats.iterrows():
            records.append({
                "name": str(row["product"]),
                "avg_daily_demand": float(row["avg_daily_sales"]),
                "stockout_days": int(row["stockout_days"]),
                "stockout_pct": float(row["stockout_pct"]),
            })
        return {"products": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load products: {exc}") from exc


@app.post("/predict")
def predict_endpoint(payload: PredictPayload) -> dict[str, Any]:
    """Predict demand, stockout risk, and recommended replenishment order."""
    try:
        req = PredictionRequest(
            product=payload.product,
            current_inventory=payload.current_inventory,
            price=payload.price,
            promo=payload.promo,
            lead_time_days=payload.lead_time_days,
            model_name=payload.model_name,
        )
        res = run_predict(req, models_dir=MODELS_DIR)
        return {
            "product": res.product,
            "forecast_demand": res.forecast_demand,
            "stockout_risk": res.stockout_risk,
            "stockout_risk_label": res.stockout_risk_label,
            "recommended_order": res.recommended_order,
            "days_until_stockout": res.days_until_stockout,
            "model_name": res.model_name,
            "model_version": res.model_version,
            "explanation": res.explanation,
            "history": res.history,
            "financial_impact": res.financial_impact,
            "substitution_alert": res.substitution_alert,
        }
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err)) from val_err
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet: {fnf_err}. Run 'python scripts/train_all.py' first.",
        ) from fnf_err
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@app.get("/simulation-results")
def simulation_results() -> dict[str, Any]:
    """Return precomputed replenishment policy simulation results."""
    if not SIM_RESULTS_PATH.exists():
        return {
            "results": [
                {
                    "policy": "A_no_forecast",
                    "stockout_pct": 7.08,
                    "service_level": 0.9292,
                    "lost_sales_units": 101,
                    "holding_cost": 4176.5,
                    "lost_sales_cost": 1010.0,
                    "total_cost": 7336.5,
                },
                {
                    "policy": "C_gradient_boost",
                    "stockout_pct": 4.58,
                    "service_level": 0.9542,
                    "lost_sales_units": 90,
                    "holding_cost": 6020.5,
                    "lost_sales_cost": 900.0,
                    "total_cost": 9220.5,
                },
                {
                    "policy": "D_censoring_aware",
                    "stockout_pct": 4.58,
                    "service_level": 0.9542,
                    "lost_sales_units": 90,
                    "holding_cost": 6492.0,
                    "lost_sales_cost": 900.0,
                    "total_cost": 9542.0,
                },
                {
                    "policy": "B_naive",
                    "stockout_pct": 5.83,
                    "service_level": 0.9417,
                    "lost_sales_units": 101,
                    "holding_cost": 7290.5,
                    "lost_sales_cost": 1010.0,
                    "total_cost": 10600.5,
                },
            ]
        }
    with open(SIM_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {"results": data}


@app.get("/model-comparison")
def model_comparison() -> dict[str, Any]:
    """Return precomputed test-set error metrics for all trained models."""
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model results not generated yet. Run scripts/train_all.py first.",
        )
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {"comparison": data}


@app.get("/monitoring/drift")
def get_drift_telemetry() -> dict[str, Any]:
    """Return Population Stability Index (PSI) and feature drift telemetry."""
    try:
        from src.monitoring import run_drift_analysis
        return run_drift_analysis()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to compute drift telemetry: {exc}"
        ) from exc


@app.post("/cold-start")
def cold_start_endpoint(payload: ColdStartPayload) -> dict[str, Any]:
    """Generate cold-start launch demand and pilot replenishment for a new SKU."""
    try:
        from src.cold_start import NewProductRequest, run_cold_start_forecast
        req = NewProductRequest(
            product_name=payload.product_name,
            category=payload.category,
            brand=payload.brand,
            price=payload.price,
            pack_size=payload.pack_size,
            unit=payload.unit,
            is_perishable=payload.is_perishable,
            supplier_lead_time=payload.supplier_lead_time,
            moq=payload.moq,
            case_pack_size=payload.case_pack_size,
        )
        res = run_cold_start_forecast(req)
        return {
            "product_name": res.product_name,
            "status": res.status,
            "confidence_level": res.confidence_level,
            "max_similarity": res.max_similarity,
            "top_analogs": res.top_analogs,
            "expected_daily_demand": res.expected_daily_demand,
            "p10": res.p10,
            "p50": res.p50,
            "p80": res.p80,
            "p90": res.p90,
            "p95": res.p95,
            "recommended_pilot_order": res.recommended_pilot_order,
            "moq_applied": res.moq_applied,
            "pack_rounding_applied": res.pack_rounding_applied,
            "explanation": res.explanation,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cold start inference error: {exc}") from exc


@app.post("/simulate-what-if")
def simulate_what_if(payload: WhatIfPayload) -> dict[str, Any]:
    """Simulate 'what-if' shocks (supplier delays, inventory shocks, promo overrides)."""
    try:
        # 1. Baseline
        base_req = PredictionRequest(
            product=payload.product,
            current_inventory=payload.current_inventory,
            price=payload.price,
            promo=payload.promo,
            lead_time_days=payload.lead_time_days,
        )
        base_res = run_predict(base_req, models_dir=MODELS_DIR)

        # 2. Scenario
        shocked_inventory = max(0, payload.current_inventory + payload.inventory_delta)
        shocked_lead_time = max(1, payload.lead_time_days + payload.lead_time_delta)
        shocked_promo = payload.promo if payload.promo_override is None else payload.promo_override

        shocked_req = PredictionRequest(
            product=payload.product,
            current_inventory=shocked_inventory,
            price=payload.price,
            promo=shocked_promo,
            lead_time_days=shocked_lead_time,
        )
        shocked_res = run_predict(shocked_req, models_dir=MODELS_DIR)

        risk_diff = round(shocked_res.stockout_risk - base_res.stockout_risk, 4)
        order_diff = shocked_res.recommended_order - base_res.recommended_order

        return {
            "product": payload.product,
            "baseline": {
                "inventory": payload.current_inventory,
                "lead_time": payload.lead_time_days,
                "promo": payload.promo,
                "forecast_demand": base_res.forecast_demand,
                "stockout_risk": base_res.stockout_risk,
                "stockout_risk_label": base_res.stockout_risk_label,
                "recommended_order": base_res.recommended_order,
                "days_until_stockout": base_res.days_until_stockout,
            },
            "scenario": {
                "inventory": shocked_inventory,
                "lead_time": shocked_lead_time,
                "promo": shocked_promo,
                "forecast_demand": shocked_res.forecast_demand,
                "stockout_risk": shocked_res.stockout_risk,
                "stockout_risk_label": shocked_res.stockout_risk_label,
                "recommended_order": shocked_res.recommended_order,
                "days_until_stockout": shocked_res.days_until_stockout,
            },
            "deltas": {
                "stockout_risk_delta": risk_diff,
                "order_delta": order_diff,
                "impact_summary": (
                    f"Scenario impact: Stockout risk shifts by {risk_diff*100:+.1f}%. "
                    f"Recommended order changes by {order_diff:+d} units."
                ),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"What-if simulation error: {exc}") from exc


@app.get("/catalog-metadata")
def get_catalog_metadata() -> dict[str, Any]:
    """Return reference metadata for catalog SKUs."""
    from src.cold_start import CATALOG_METADATA
    return {"catalog": CATALOG_METADATA}

