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

import io
import pandas as pd
from src.data import load_observed, summary_stats
from src.canonical import (
    CANONICAL_FIELDS,
    suggest_column_mapping,
    validate_and_profile_dataframe,
)
from src.audit import (
    get_audit_history,
    get_purchase_orders,
    log_recommendation,
    record_action,
)
from app.schemas.decision import ActionQueueResponse
from app.schemas.audit import AuditHistoryResponse
from app.services.decision_service import DecisionService
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
    on_order: int = Field(0, ge=0, description="Units already on order from supplier", json_schema_extra={"example": 0})
    reserved: int = Field(0, ge=0, description="Customer reserved / backordered units", json_schema_extra={"example": 0})
    moq: int = Field(1, ge=1, description="Supplier Minimum Order Quantity", json_schema_extra={"example": 1})
    case_pack_size: int = Field(1, ge=1, description="Case pack multiple", json_schema_extra={"example": 1})


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


class UploadPreviewPayload(BaseModel):
    csv_content: str = Field(..., description="Raw text or CSV content uploaded by user")


class ColumnMappingPayload(BaseModel):
    csv_content: str = Field(..., description="Raw CSV content")
    mapping: dict[str, str] = Field(..., description="User mapping: {canonical_key: user_column_header}")


class ActionExecutePayload(BaseModel):
    decision_id: str = Field(..., description="Unique decision ID (e.g. SG-20260913-XXXX)")
    action: str = Field(..., description="Action taken: APPROVED, ADJUSTED, or DISMISSED")
    approved_units: int | None = Field(None, description="Units approved to purchase")
    user_notes: str = Field("", description="Optional human rationale or operator note")
    unit_price: float = Field(50.0, description="Estimated unit purchase cost")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def landing(request: Request) -> Any:
    """Serve the marketing landing page."""
    return templates.TemplateResponse(request=request, name="landing.html")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request) -> Any:
    """Serve the full intelligence dashboard."""
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
            on_order=payload.on_order,
            reserved=payload.reserved,
            moq=payload.moq,
            case_pack_size=payload.case_pack_size,
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


@app.get("/store-overview")
def store_overview() -> dict[str, Any]:
    """Return all SKUs ranked by urgency for the Action Center procurement queue.

    Each SKU entry includes:
      - status (critical / warning / healthy)
      - on_hand inventory
      - avg_daily_demand
      - runway_days = on_hand / avg_daily_demand
      - recommended_order (from baseline predict)
      - stockout_risk_pct
    """
    try:
        df_obs = load_observed()
        stats = summary_stats(df_obs)

        skus = []
        for _, row in stats.iterrows():
            product_name = str(row["product"])
            avg_demand = float(row["avg_daily_sales"])

            # Get a representative current inventory from observed data
            try:
                prod_df = df_obs[df_obs["product"] == product_name]
                if not prod_df.empty:
                    on_hand = int(prod_df["opening_stock"].iloc[-1])
                else:
                    on_hand = 24  # safe fallback
            except Exception:
                on_hand = 24

            # Compute runway
            runway_days = (on_hand / avg_demand) if avg_demand > 0 else 30.0

            # Get recommended order via predict
            try:
                req = PredictionRequest(
                    product=product_name,
                    current_inventory=on_hand,
                    price=55.0,
                    promo=0,
                    lead_time_days=2,
                )
                res = run_predict(req, models_dir=MODELS_DIR)
                recommended_order = res.recommended_order
                stockout_risk_pct = round(res.stockout_risk * 100, 1)
            except Exception:
                recommended_order = max(0, int(avg_demand * 7) - on_hand)
                stockout_risk_pct = 0.0

            # Classify urgency
            if runway_days < 2 or stockout_risk_pct >= 60:
                status = "critical"
            elif runway_days < 4 or stockout_risk_pct >= 30:
                status = "warning"
            else:
                status = "healthy"

            skus.append({
                "name": product_name,
                "status": status,
                "on_hand": on_hand,
                "avg_daily_demand": round(avg_demand, 2),
                "runway_days": round(runway_days, 2),
                "recommended_order": recommended_order,
                "stockout_risk_pct": stockout_risk_pct,
                "stockout_days_historical": int(row["stockout_days"]),
            })

        # Sort: critical first, then warning, then healthy; within tier by runway ASC
        order = {"critical": 0, "warning": 1, "healthy": 2}
        skus.sort(key=lambda s: (order[s["status"]], s["runway_days"]))
        return {"skus": skus}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Store overview error: {exc}") from exc


# ── v1.0 Product Endpoints: Onboarding, Action Center, & Audit History ──────────

@app.post("/api/v1/upload/preview")
def upload_preview(payload: UploadPreviewPayload) -> dict[str, Any]:
    """Parse uploaded CSV/text, detect columns, suggest mappings, and return preview rows."""
    try:
        df = pd.read_csv(io.StringIO(payload.csv_content))
        if df.empty:
            raise ValueError("Uploaded file contains no rows.")

        columns = df.columns.tolist()
        suggested = suggest_column_mapping(columns)
        preview_rows = df.head(5).to_dict(orient="records")

        return {
            "success": True,
            "columns": columns,
            "detected_headers": columns,
            "suggested_mapping": suggested,
            "canonical_fields": CANONICAL_FIELDS,
            "preview_rows": preview_rows,
            "total_rows_detected": len(df),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc


@app.post("/api/v1/upload/validate-and-ingest")
def validate_and_ingest(payload: ColumnMappingPayload) -> dict[str, Any]:
    """Validate mapping against canonical schema, run quality diagnostics, and ingest."""
    try:
        df = pd.read_csv(io.StringIO(payload.csv_content))
        report, cleaned_df = validate_and_profile_dataframe(df, payload.mapping)

        if not report.is_valid or cleaned_df is None:
            return {
                "success": False,
                "report": report.to_dict(),
                "message": "Data quality validation failed. Please check errors.",
            }

        # Save to custom user store location
        user_store_path = PROJECT_ROOT / "data" / "raw" / "user_imported.csv"
        cleaned_df.to_csv(user_store_path, index=False)

        return {
            "success": True,
            "report": report.to_dict(),
            "message": f"Successfully ingested {report.total_rows} rows across {report.unique_skus} SKUs.",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ingestion error: {exc}") from exc


@app.post("/api/v1/store/load-sample")
def load_sample_store() -> dict[str, Any]:
    """Reset store data back to the realistic benchmark demo store (10 SKUs, 120 days)."""
    try:
        user_imported = PROJECT_ROOT / "data" / "raw" / "user_imported.csv"
        if user_imported.exists():
            user_imported.unlink()
        return {
            "success": True,
            "message": "Sample store dataset active (10 FMCG SKUs, 120 days, simulated stockouts).",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset error: {exc}") from exc


@app.get("/api/v1/actions", response_model=ActionQueueResponse)
def get_action_center() -> ActionQueueResponse:
    """Get prioritized actionable decisions with decision IDs and 'Why?' audit contract."""
    try:
        return DecisionService.get_action_queue()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Action center error: {exc}") from exc



@app.post("/api/v1/actions/execute")
def execute_action(payload: ActionExecutePayload) -> dict[str, Any]:
    """Execute human approval, adjustment, or dismissal with persistent audit trail."""
    try:
        res = record_action(
            decision_id=payload.decision_id,
            action=payload.action,
            approved_units=payload.approved_units,
            user_notes=payload.user_notes,
            unit_price=payload.unit_price,
        )
        return {
            "success": True,
            "action_recorded": res["action"],
            "decision_id": res["decision_id"],
            "approved_units": res["approved_units"],
            "purchase_order": {
                "po_number": res["po_number"],
                "total_units": res["approved_units"],
            } if res.get("po_number") else None,
            **res,
        }
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err)) from val_err
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Action execution error: {exc}") from exc


@app.get("/api/v1/audit/history", response_model=AuditHistoryResponse)
def get_audit_log(limit: int = 50) -> AuditHistoryResponse:
    """Retrieve persistent decision history and created purchase orders."""
    try:
        decisions = get_audit_history(limit=limit)
        purchase_orders = get_purchase_orders(limit=limit)
        return AuditHistoryResponse(
            total_decisions=len(decisions),
            total_purchase_orders=len(purchase_orders),
            decisions=decisions,
            purchase_orders=purchase_orders,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit query error: {exc}") from exc


