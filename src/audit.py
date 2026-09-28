"""
src/audit.py — Persistent Decision Audit Ledger & Purchase Order Tracker.

Every inventory recommendation, human override, and purchase order approval
is recorded in an SQLite database (data/audit_log.db).

Provides:
1. Traceable decision IDs (SG-YYYYMMDD-XXXX).
2. Human-in-the-loop action logging (APPROVED, ADJUSTED, DISMISSED).
3. Purchase Order status lifecycle (DRAFT -> APPROVED -> IN_TRANSIT -> RECEIVED).
4. Immutable historical query endpoints for compliance and accuracy post-mortems.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "audit_log.db"


def get_db_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Connect to SQLite database and ensure tables exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Initialize audit tables if not existing."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                sku TEXT NOT NULL,
                forecast_p50 REAL NOT NULL,
                forecast_p90 REAL NOT NULL,
                inventory_position INTEGER NOT NULL,
                recommended_order INTEGER NOT NULL,
                action_status TEXT NOT NULL,  -- PENDING, APPROVED, ADJUSTED, DISMISSED
                approved_order INTEGER,
                reason TEXT,
                model_used TEXT,
                user_notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS purchase_orders (
                po_number TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                sku TEXT NOT NULL,
                ordered_units INTEGER NOT NULL,
                supplier TEXT,
                status TEXT NOT NULL,         -- DRAFT, APPROVED, IN_TRANSIT, RECEIVED
                unit_price REAL,
                total_cost REAL,
                expected_delivery_date TEXT
            )
        """)


def log_recommendation(
    *,
    sku: str,
    forecast_p50: float,
    forecast_p90: float,
    inventory_position: int,
    recommended_order: int,
    reason: str,
    model_used: str = "censoring_aware_v2.1",
    db_path: Path | str = DB_PATH,
) -> str:
    """Log an initial recommendation and return a traceable decision ID."""
    decision_id = f"SG-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO decisions (
                decision_id, created_at, sku, forecast_p50, forecast_p90,
                inventory_position, recommended_order, action_status, approved_order,
                reason, model_used, user_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, '')
            """,
            (
                decision_id,
                datetime.utcnow().isoformat(),
                sku,
                forecast_p50,
                forecast_p90,
                inventory_position,
                recommended_order,
                recommended_order,
                reason,
                model_used,
            ),
        )
    conn.close()
    return decision_id


def record_action(
    *,
    decision_id: str,
    action: str,  # APPROVED, ADJUSTED, DISMISSED
    approved_units: int | None = None,
    user_notes: str = "",
    unit_price: float = 50.0,
    db_path: Path | str = DB_PATH,
) -> dict[str, Any]:
    """Record human approval, adjustment, or dismissal of a recommendation."""
    conn = get_db_connection(db_path)
    with conn:
        cursor = conn.execute(
            "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Decision ID '{decision_id}' not found.")

        sku = row["sku"]
        final_units = approved_units if approved_units is not None else row["recommended_order"]

        conn.execute(
            """
            UPDATE decisions
            SET action_status = ?, approved_order = ?, user_notes = ?
            WHERE decision_id = ?
            """,
            (action, final_units, user_notes, decision_id),
        )

        po_number = None
        if action in ("APPROVED", "ADJUSTED") and final_units > 0:
            po_number = f"PO-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
            total_cost = round(final_units * unit_price, 2)
            conn.execute(
                """
                INSERT INTO purchase_orders (
                    po_number, created_at, sku, ordered_units, supplier, status,
                    unit_price, total_cost, expected_delivery_date
                ) VALUES (?, ?, ?, ?, 'Primary Supplier', 'APPROVED', ?, ?, '')
                """,
                (
                    po_number,
                    datetime.utcnow().isoformat(),
                    sku,
                    final_units,
                    unit_price,
                    total_cost,
                ),
            )

    conn.close()
    return {
        "decision_id": decision_id,
        "sku": sku,
        "action": action,
        "approved_units": final_units,
        "po_number": po_number,
        "status": "success",
    }


def get_audit_history(limit: int = 50, db_path: Path | str = DB_PATH) -> list[dict[str, Any]]:
    """Query recent decisions and human approvals from the audit ledger."""
    conn = get_db_connection(db_path)
    cursor = conn.execute(
        "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_purchase_orders(limit: int = 50, db_path: Path | str = DB_PATH) -> list[dict[str, Any]]:
    """Query created purchase orders."""
    conn = get_db_connection(db_path)
    cursor = conn.execute(
        "SELECT * FROM purchase_orders ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
