"""
tests/test_demand_recovery.py — Unit tests for latent demand reconstruction.
"""

import pandas as pd
import pytest
from src.demand_recovery import DemandRecoveryEngine, run_censoring_ablation


def test_naive_demand_recovery():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5),
        "product": ["Milk"] * 5,
        "sales": [10.0, 12.0, 0.0, 0.0, 15.0],
        "opening_stock": [20.0, 8.0, 0.0, 0.0, 25.0],
    })
    engine = DemandRecoveryEngine(method="naive")
    res = engine.fit_transform(df)
    assert "latent_demand" in res.columns
    assert "is_censored" in res.columns
    # On stockout days (opening_stock = 0), is_censored should be 1
    assert list(res["is_censored"]) == [0, 0, 1, 1, 0]
    # Naive sets latent = sales
    assert list(res["latent_demand"]) == [10.0, 12.0, 0.0, 0.0, 15.0]


def test_tobit_demand_recovery_imputes_above_zero():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=6),
        "product": ["Eggs"] * 6,
        "sales": [20.0, 22.0, 24.0, 0.0, 0.0, 21.0],
        "opening_stock": [50.0, 30.0, 8.0, 0.0, 0.0, 40.0],
    })
    engine = DemandRecoveryEngine(method="tobit")
    res = engine.fit_transform(df)

    # Days 3 and 4 are stockout days (sales=0, stock=0)
    # Tobit reconstruction must estimate positive latent demand based on historical velocity
    stockout_demand = res.loc[res["is_censored"] == 1, "latent_demand"].values
    assert len(stockout_demand) == 2
    for val in stockout_demand:
        assert val > 15.0, f"Expected Tobit to recover non-zero demand, got {val}"


def test_censoring_ablation_summary():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10),
        "product": ["Bread"] * 10,
        "sales": [30.0, 35.0, 0.0, 0.0, 32.0, 34.0, 0.0, 28.0, 31.0, 29.0],
        "opening_stock": [50.0, 20.0, 0.0, 0.0, 40.0, 10.0, 0.0, 45.0, 30.0, 15.0],
    })
    stats = run_censoring_ablation(df)
    assert stats["censored_stockout_days"] == 3
    assert stats["censored_rate_pct"] == 30.0
    assert stats["recovered_latent_demand_units"] > stats["recorded_sales_units"]
    assert stats["masked_demand_units"] > 0
    assert stats["systematic_underestimation_pct"] > 0
