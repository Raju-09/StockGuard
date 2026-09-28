"""
tests/test_canonical.py — Tests for the Canonical Ingestion & Data Quality Engine.
"""

import pandas as pd
import pytest

from src.canonical import (
    suggest_column_mapping,
    validate_and_profile_dataframe,
)


def test_suggest_column_mapping_real_world_names():
    """Messy real-world column names must match canonical schema correctly."""
    cols = ["Order Date", "StockCode", "Quantity Sold", "Unit Price", "Warehouse"]
    mapping = suggest_column_mapping(cols)

    assert mapping["date"] == "Order Date"
    assert mapping["sku"] == "StockCode"
    assert mapping["sales"] == "Quantity Sold"
    assert mapping["price"] == "Unit Price"
    assert mapping["store"] == "Warehouse"


def test_validate_and_profile_dataframe_clean():
    """Clean data should produce valid report with 0 errors."""
    df = pd.DataFrame({
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "product_id": ["EGG001", "EGG001", "EGG001"],
        "units": [12, 15, 0],
        "rate": [5.0, 5.0, 5.0],
    })
    mapping = {
        "date": "order_date",
        "sku": "product_id",
        "sales": "units",
        "price": "rate",
    }
    report, cleaned = validate_and_profile_dataframe(df, mapping)
    assert report.is_valid is True
    assert report.unique_skus == 1
    assert report.total_rows == 3
    assert cleaned is not None
    assert "product" in cleaned.columns
    assert "sales" in cleaned.columns


def test_validate_negative_sales_clipping():
    """Negative sales (returns) must be clipped to 0 and flagged as a warning."""
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "sku": ["MILK001", "MILK001"],
        "sales": [10, -5],
    })
    mapping = {"date": "date", "sku": "sku", "sales": "sales"}
    report, cleaned = validate_and_profile_dataframe(df, mapping)
    assert report.is_valid is True
    assert report.negative_sales_rows == 1
    assert len(report.warnings) > 0
    assert (cleaned["sales"] >= 0).all()


def test_validate_missing_required_column():
    """Missing required 'sku' or 'sales' must cause is_valid=False with clear error."""
    df = pd.DataFrame({
        "date": ["2026-01-01"],
        "random_col": [123],
    })
    mapping = {"date": "date"}
    report, cleaned = validate_and_profile_dataframe(df, mapping)
    assert report.is_valid is False
    assert cleaned is None
    assert any("sku" in err.lower() for err in report.errors)
