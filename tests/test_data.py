"""
tests/test_data.py — Unit tests for src/data.py and data leakage prevention.
"""

import pandas as pd

from src.data import (
    GROUND_TRUTH_ONLY_COLS,
    OBSERVED_COLS,
    load_observed,
    load_with_ground_truth,
    summary_stats,
)


def test_load_observed_schema():
    df = load_observed()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1200
    for col in OBSERVED_COLS:
        assert col in df.columns, f"Missing observed col {col}"


def test_observed_has_no_ground_truth_leakage():
    df = load_observed()
    for col in GROUND_TRUTH_ONLY_COLS:
        assert col not in df.columns, f"Leakage: ground truth column {col} in observed data!"


def test_load_with_ground_truth_schema():
    df = load_with_ground_truth()
    assert len(df) == 1200
    for col in GROUND_TRUTH_ONLY_COLS:
        assert col in df.columns, f"Missing ground truth col {col}"


def test_summary_stats():
    df = load_observed()
    stats = summary_stats(df)
    assert len(stats) == 10
    assert "product" in stats.columns
    assert "avg_daily_sales" in stats.columns
    assert "stockout_pct" in stats.columns


def test_load_uci_online_retail_adapter(tmp_path):
    from src.data import OBSERVED_COLS, load_uci_online_retail

    csv_path = tmp_path / "sample_uci.csv"
    sample_data = pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "536366"],
        "StockCode": ["85123A", "71053", "85123A"],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN", "WHITE HANGING HEART T-LIGHT HOLDER"],
        "Quantity": [6, 6, 12],
        "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 08:26:00", "2010-12-02 09:00:00"],
        "UnitPrice": [2.55, 3.39, 2.55],
        "CustomerID": [17850, 17850, 17850],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom"],
    })
    sample_data.to_csv(csv_path, index=False)

    df_uci = load_uci_online_retail(csv_path, top_n_skus=2)
    assert isinstance(df_uci, pd.DataFrame)
    for col in OBSERVED_COLS:
        assert col in df_uci.columns
    assert "WHITE HANGING HEART T-LIGHT HOLDER" in df_uci["product"].values

