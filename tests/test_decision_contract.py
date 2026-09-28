"""
tests/test_decision_contract.py — Invariant and unit tests for the deterministic decision engine.
Validates Prompt 2 and Section 16 minimum invariants.
"""

import pytest
from src.inventory.decision import (
    ActionType,
    DecisionEngine,
    DemandForecast,
    InventoryState,
)


def test_inventory_position_accounting():
    """Invariant: inventory_position == on_hand + on_order + in_transit - reserved."""
    state = InventoryState(
        sku="TEST01",
        on_hand=50,
        on_order=30,
        in_transit=20,
        reserved=10,
    )
    assert state.inventory_position == 90


def test_negative_values_rejected():
    """Negative stock or invalid constraints must be explicitly rejected."""
    with pytest.raises(ValueError, match="on_hand cannot be negative"):
        InventoryState(sku="ERR", on_hand=-5)

    with pytest.raises(ValueError, match="moq must be >= 1"):
        InventoryState(sku="ERR", on_hand=10, moq=0)

    with pytest.raises(ValueError, match="pack_size must be >= 1"):
        InventoryState(sku="ERR", on_hand=10, pack_size=0)


def test_quantile_monotonicity_enforced():
    """DemandForecast must satisfy P10 <= P50 <= P90."""
    with pytest.raises(ValueError, match="Quantile monotonicity"):
        DemandForecast(p10=30.0, p50=20.0, p90=40.0, daily_run_rate=20.0, daily_std=5.0)


def test_in_transit_stock_prevents_duplicate_purchasing():
    """If 50 units are in transit, system must not re-order the same 50 units."""
    forecast = DemandForecast(p10=8.0, p50=10.0, p90=14.0, daily_run_rate=10.0, daily_std=2.0)
    
    # Case A: 0 on-hand, 0 in-transit -> Orders required
    state_empty = InventoryState(sku="BIS01", on_hand=0, in_transit=0, lead_time_days=3, moq=1, pack_size=1)
    decision_empty = DecisionEngine.evaluate(state_empty, forecast)
    assert decision_empty.action == ActionType.BUY
    assert decision_empty.recommended_order > 30

    # Case B: 0 on-hand, but 45 in-transit -> In-transit covers target stock -> HOLD
    state_in_transit = InventoryState(sku="BIS01", on_hand=0, in_transit=45, lead_time_days=3, moq=1, pack_size=1)
    decision_in_transit = DecisionEngine.evaluate(state_in_transit, forecast)
    assert decision_in_transit.action == ActionType.HOLD
    assert decision_in_transit.recommended_order == 0


def test_moq_and_pack_size_invariants():
    """All positive orders must satisfy: order >= MOQ and order % pack_size == 0."""
    forecast = DemandForecast(p10=4.0, p50=5.0, p90=7.0, daily_run_rate=5.0, daily_std=1.0)
    
    # Target stock ~ 15 + safety buffer (~20 units total). On-hand = 2 -> Shortage ~ 18
    # MOQ = 50, Pack size = 12
    state = InventoryState(sku="EGG01", on_hand=2, lead_time_days=3, moq=50, pack_size=12)
    decision = DecisionEngine.evaluate(state, forecast)

    assert decision.action == ActionType.BUY
    order = decision.recommended_order

    # Invariant 1: Order is strictly positive
    assert order > 0
    # Invariant 2: Order satisfies supplier MOQ
    assert order >= 50
    # Invariant 3: Order is an exact multiple of case pack 12
    assert order % 12 == 0


def test_decision_contract_audit_reasons():
    """Contract must produce a multi-step human-readable explanation chain."""
    forecast = DemandForecast(p10=15.0, p50=20.0, p90=26.0, daily_run_rate=20.0, daily_std=4.0)
    state = InventoryState(sku="MILK01", on_hand=10, lead_time_days=2, lead_time_std=1.0, moq=20, pack_size=5)
    
    decision = DecisionEngine.evaluate(state, forecast)
    d = decision.to_dict()

    assert "sku" in d
    assert "action" in d
    assert "reasoning_chain" in d
    assert len(d["reasoning_chain"]) >= 3
    assert "constraints_applied" in d
    assert d["constraints_applied"]["moq"] == 20
    assert d["constraints_applied"]["pack_size"] == 5
