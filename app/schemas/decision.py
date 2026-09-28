"""
app/schemas/decision.py — Pydantic Schemas for Inventory Decision Intelligence.
Strict typed request and response contracts for all decision endpoints.
"""

from typing import Any
from pydantic import BaseModel, Field


class DecisionApproveRequest(BaseModel):
    decision_id: str = Field(..., description="Unique UUID for this recommendation")
    ordered_qty: int = Field(..., ge=0, description="Approved purchase order quantity")
    unit_price: float = Field(default=50.0, ge=0.0, description="Estimated unit cost")


class DecisionOverrideRequest(BaseModel):
    decision_id: str = Field(..., description="Unique UUID for this recommendation")
    override_qty: int = Field(..., ge=0, description="User adjusted order quantity")
    override_reason: str = Field(..., min_length=3, description="Mandatory reason for planner override")
    unit_price: float = Field(default=50.0, ge=0.0, description="Estimated unit cost")


class WhyBreakdown(BaseModel):
    p10_daily: float
    p50_daily: float
    p90_daily: float
    on_hand: int
    on_order: int
    reserved: int
    inventory_position: int
    lead_time_days: float
    lead_time_variance: str
    p50_ltd: float
    p90_ltd: float
    target_service_level: str
    moq: int
    case_pack: int
    model_used: str
    why_narrative: str
    reasoning_chain: list[str] = Field(default_factory=list)


class ActionItemResponse(BaseModel):
    decision_id: str
    sku: str
    status: str  # critical, warning, overstock, healthy, uncertain
    runway_days: float
    stockout_risk_pct: float
    recommended_order: int
    recommendation: str
    potential_loss_7d: float
    unit_price: float
    why_breakdown: WhyBreakdown


class ActionQueueResponse(BaseModel):
    actions: list[ActionItemResponse]
    total_actions: int
    total_stockout_risk_exposure: float = 0.0
    total_excess_inventory_value: float = 0.0
    critical_count: int = 0
    warning_count: int = 0
    overstock_count: int = 0
    healthy_count: int = 0
