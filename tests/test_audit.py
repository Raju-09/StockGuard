"""
tests/test_audit.py — Tests for SQLite Decision Ledger and Purchase Order Lifecycle.
"""

import pytest
from src.audit import (
    get_audit_history,
    get_purchase_orders,
    log_recommendation,
    record_action,
)


def test_log_and_approve_decision(tmp_path):
    """Logging a decision and approving it creates an immutable record and a PO."""
    db_file = tmp_path / "test_audit.db"

    dec_id = log_recommendation(
        sku="Biscuits",
        forecast_p50=12.5,
        forecast_p90=18.0,
        inventory_position=8,
        recommended_order=20,
        reason="Stockout risk exceeds 80%",
        db_path=db_file,
    )
    assert dec_id.startswith("SG-")

    # Verify pending in ledger
    history = get_audit_history(db_path=db_file)
    assert len(history) == 1
    assert history[0]["decision_id"] == dec_id
    assert history[0]["action_status"] == "PENDING"

    # Human approves
    res = record_action(
        decision_id=dec_id,
        action="APPROVED",
        approved_units=20,
        user_notes="Confirmed by inventory manager",
        unit_price=35.0,
        db_path=db_file,
    )
    assert res["status"] == "success"
    assert res["po_number"] is not None
    assert res["po_number"].startswith("PO-")

    # Verify updated ledger & PO
    updated = get_audit_history(db_path=db_file)
    assert updated[0]["action_status"] == "APPROVED"
    assert updated[0]["user_notes"] == "Confirmed by inventory manager"

    pos = get_purchase_orders(db_path=db_file)
    assert len(pos) == 1
    assert pos[0]["sku"] == "Biscuits"
    assert pos[0]["ordered_units"] == 20
    assert pos[0]["total_cost"] == 700.0


def test_dismiss_decision(tmp_path):
    """Dismissing a decision records dismissal without creating a PO."""
    db_file = tmp_path / "test_audit.db"

    dec_id = log_recommendation(
        sku="Rice",
        forecast_p50=5.0,
        forecast_p90=8.0,
        inventory_position=40,
        recommended_order=0,
        reason="Overstock condition",
        db_path=db_file,
    )

    res = record_action(
        decision_id=dec_id,
        action="DISMISSED",
        user_notes="Supplier minimum order not cost-effective",
        db_path=db_file,
    )
    assert res["status"] == "success"
    assert res["po_number"] is None

    pos = get_purchase_orders(db_path=db_file)
    assert len(pos) == 0
