"""
tests/test_simulate.py — Smoke tests for the data simulator.

These tests verify the mathematical invariants of the simulation,
NOT ML model accuracy. They run fast (< 2 s) with no external deps.

Invariants that must always hold (regardless of seed or parameters):
    1. Row count = n_products × n_days.
    2. All required columns are present.
    3. sales <= opening_stock for every row  (censoring rule).
    4. closing_stock == opening_stock - sales for every row.
    5. lost_sales == true_demand - sales for every row.
    6. stockout_flag == 1 iff closing_stock == 0.
    7. is_censored == 1 iff true_demand >= opening_stock.
    8. data_source == "simulated" for every row.
    9. No negative values in stock or sales columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow importing from src/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulate import N_DAYS, PRODUCTS, simulate_store


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Run the simulator once; reuse across all tests in this module."""
    return simulate_store(seed=42)


class TestShape:
    def test_row_count(self, df: pd.DataFrame) -> None:
        expected = len(PRODUCTS) * N_DAYS
        assert len(df) == expected, (
            f"Expected {expected} rows (products × days), got {len(df)}"
        )

    def test_product_count(self, df: pd.DataFrame) -> None:
        assert df["product"].nunique() == len(PRODUCTS)

    def test_date_count(self, df: pd.DataFrame) -> None:
        assert df["date"].nunique() == N_DAYS


class TestColumns:
    REQUIRED_COLS = [
        "date", "store", "product",
        "opening_stock", "sales", "closing_stock",
        "price", "promo", "holiday", "delivery_qty",
        "stockout_flag", "true_demand", "lost_sales",
        "is_censored", "demand_mean", "data_source",
    ]

    def test_required_columns_present(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        assert missing == [], f"Missing columns: {missing}"


class TestCensoringInvariants:
    """The core mathematical constraints of sales = min(demand, stock)."""

    def test_sales_le_opening_stock(self, df: pd.DataFrame) -> None:
        violations = df[df["sales"] > df["opening_stock"]]
        assert len(violations) == 0, (
            f"Censoring rule violated on {len(violations)} rows: "
            f"sales > opening_stock"
        )

    def test_closing_stock_identity(self, df: pd.DataFrame) -> None:
        expected = df["opening_stock"] - df["sales"]
        bad = df[df["closing_stock"] != expected]
        assert len(bad) == 0, (
            f"closing_stock != opening_stock - sales on {len(bad)} rows"
        )

    def test_lost_sales_identity(self, df: pd.DataFrame) -> None:
        expected = df["true_demand"] - df["sales"]
        bad = df[df["lost_sales"] != expected]
        assert len(bad) == 0, (
            f"lost_sales != true_demand - sales on {len(bad)} rows"
        )

    def test_stockout_flag_consistency(self, df: pd.DataFrame) -> None:
        expected_flag = (df["closing_stock"] == 0).astype(int)
        bad = df[df["stockout_flag"] != expected_flag]
        assert len(bad) == 0, (
            f"stockout_flag inconsistency on {len(bad)} rows"
        )

    def test_is_censored_consistency(self, df: pd.DataFrame) -> None:
        expected_censored = (df["true_demand"] >= df["opening_stock"]).astype(int)
        bad = df[df["is_censored"] != expected_censored]
        assert len(bad) == 0, (
            f"is_censored inconsistency on {len(bad)} rows"
        )


class TestNonNegativity:
    NON_NEGATIVE_COLS = [
        "opening_stock", "sales", "closing_stock",
        "true_demand", "lost_sales",
    ]

    @pytest.mark.parametrize("col", NON_NEGATIVE_COLS)
    def test_no_negative_values(self, df: pd.DataFrame, col: str) -> None:
        neg = df[df[col] < 0]
        assert len(neg) == 0, f"Negative values in '{col}': {len(neg)} rows"


class TestDataSource:
    def test_data_source_is_simulated(self, df: pd.DataFrame) -> None:
        unique_sources = df["data_source"].unique().tolist()
        assert unique_sources == ["simulated"], (
            f"Expected ['simulated'], got {unique_sources}"
        )


class TestCensoringPresent:
    """Verify that the simulator actually produces censored observations.

    If there are zero censored rows, something is wrong with the setup
    (products always have enough stock) — our research problem doesn't exist.
    """

    def test_censored_rows_exist(self, df: pd.DataFrame) -> None:
        n_censored = df["is_censored"].sum()
        assert n_censored > 0, (
            "No censored rows found. Simulation parameters may be too generous. "
            "The stockout-censoring research problem requires censoring to occur."
        )

    def test_lost_sales_nonzero(self, df: pd.DataFrame) -> None:
        total_lost = df["lost_sales"].sum()
        assert total_lost > 0, "Zero total lost sales — censoring is not occurring."


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        df1 = simulate_store(seed=99)
        df2 = simulate_store(seed=99)
        pd.testing.assert_frame_equal(df1, df2, check_like=False)

    def test_different_seeds_differ(self) -> None:
        df1 = simulate_store(seed=1)
        df2 = simulate_store(seed=2)
        # They might agree on some rows by chance, but the totals must differ.
        assert not df1["sales"].equals(df2["sales"])
