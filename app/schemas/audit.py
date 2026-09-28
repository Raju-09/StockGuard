"""
app/schemas/audit.py — Pydantic Schemas for Audit Trail & Purchase Orders.
"""

from typing import Any
from pydantic import BaseModel, Field


class DecisionLogEntry(BaseModel):
    decision_id: str
    created_at: str
    sku: str
    forecast_p50: float
    forecast_p90: float
    inventory_position: int
    recommended_order: int
    action_status: str  # PENDING, APPROVED, ADJUSTED, DISMISSED
    ordered_qty: int | None = None
    override_reason: str | None = None
    reason: str | None = None


class PurchaseOrderLine(BaseModel):
    po_number: str
    created_at: str
    sku: str
    ordered_units: int
    supplier: str
    status: str
    total_cost: float | None = None


class AuditHistoryResponse(BaseModel):
    total_decisions: int
    total_purchase_orders: int
    decisions: list[dict[str, Any]]
    purchase_orders: list[dict[str, Any]]
