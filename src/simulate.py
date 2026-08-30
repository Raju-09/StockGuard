"""
Simulate one grocery store so we can study inventory-censored demand.

THIS IS SIMULATED DATA.
It is useful for testing methods under known assumptions.
It does not prove those methods work in a real shop.

Core rule (the whole point of StockGuard):

    observed_sales = min(true_demand, opening_stock)

If the shelf is empty, sales can be 0 while customers still wanted the product.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 120
START_DATE = date(2025, 9, 1)
STORE = "store_01"

# Weekend / holiday / promo multiply the Poisson mean for that day.
WEEKEND_MULTIPLIER = 1.25
HOLIDAY_MULTIPLIER = 1.50
PROMO_MULTIPLIER = 1.80

# A few calendar holidays inside the 120-day window (not a full official list).
HOLIDAYS = {
    date(2025, 10, 2),   # Gandhi Jayanti (example)
    date(2025, 10, 20),  # Diwali-adjacent example
    date(2025, 12, 25),  # Christmas
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Tight inventory on purpose for some SKUs so stockouts actually happen.
# reorder_point and order_qty are a simple (s, Q) policy with lead time.
PRODUCTS = [
    # name, daily_lambda, price, initial_stock, reorder_point, order_qty, lead_time_days, promo_rate
    ("Milk", 22, 55, 40, 30, 50, 2, 0.08),
    ("Bread", 18, 40, 28, 24, 40, 1, 0.05),
    ("Eggs", 14, 72, 40, 20, 36, 2, 0.04),
    ("Yogurt", 10, 45, 12, 10, 20, 2, 0.10),  # tight → frequent censoring
    ("Biscuits", 8, 30, 8, 8, 16, 3, 0.12),   # tight → frequent censoring
    ("Rice", 6, 80, 50, 18, 40, 3, 0.03),
    ("Oil", 5, 140, 35, 12, 24, 3, 0.04),
    ("Sugar", 7, 50, 40, 16, 30, 3, 0.03),
    ("Tea", 4, 90, 30, 10, 20, 4, 0.06),
    ("Soap", 3, 35, 25, 8, 18, 4, 0.05),
]


def _demand_mean(base_lambda: float, the_date: date, promo: int) -> float:
    mean = float(base_lambda)
    if the_date.weekday() >= 5:
        mean *= WEEKEND_MULTIPLIER
    if the_date in HOLIDAYS:
        mean *= HOLIDAY_MULTIPLIER
    if promo:
        mean *= PROMO_MULTIPLIER
    return mean


def simulate_store(seed: int = SEED) -> pd.DataFrame:
    """Return one row per product-day, including hidden true_demand."""
    rng = np.random.default_rng(seed)
    dates = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]
    rows: list[dict] = []

    for name, lam, price, initial, reorder_point, order_qty, lead_time, promo_rate in PRODUCTS:
        on_hand = int(initial)
        # Incoming deliveries: day_index → units arriving that morning.
        inbound: dict[int, int] = {}

        for day_index, the_date in enumerate(dates):
            delivery_qty = int(inbound.pop(day_index, 0))
            opening_stock = on_hand + delivery_qty

            promo = int(rng.random() < promo_rate)
            mean = _demand_mean(lam, the_date, promo)
            true_demand = int(rng.poisson(mean))

            # Censoring: you cannot sell more than is on the shelf.
            sales = min(true_demand, opening_stock)
            closing_stock = opening_stock - sales
            lost_sales = true_demand - sales
            is_censored = true_demand >= opening_stock
            stockout_flag = int(closing_stock == 0)

            # Place a replenishment order if inventory (plus already-inbound) is low.
            pipeline = sum(inbound.values())
            if opening_stock + pipeline - sales <= reorder_point:
                arrival = day_index + lead_time
                inbound[arrival] = inbound.get(arrival, 0) + int(order_qty)

            on_hand = closing_stock

            rows.append(
                {
                    "date": the_date.isoformat(),
                    "store": STORE,
                    "product": name,
                    "opening_stock": opening_stock,
                    "sales": sales,
                    "closing_stock": closing_stock,
                    "price": price,
                    "promo": promo,
                    "holiday": int(the_date in HOLIDAYS),
                    "delivery_qty": delivery_qty,
                    "stockout_flag": stockout_flag,
                    "true_demand": true_demand,
                    "lost_sales": lost_sales,
                    "is_censored": int(is_censored),
                    "demand_mean": round(mean, 3),
                    "data_source": "simulated",
                }
            )

    return pd.DataFrame(rows)


def split_observed_and_truth(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    observed: what a real store system could log.
    truth: hidden demand, only for evaluation / research. Never a model feature.
    """
    observed_cols = [
        "date",
        "store",
        "product",
        "opening_stock",
        "sales",
        "closing_stock",
        "price",
        "promo",
        "holiday",
        "delivery_qty",
        "stockout_flag",
        "data_source",
    ]
    return df[observed_cols].copy(), df.copy()


def write_outputs(df: pd.DataFrame) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    observed, truth = split_observed_and_truth(df)

    observed_path = RAW_DIR / "observed_daily.csv"
    truth_path = RAW_DIR / "with_ground_truth.csv"
    meta_path = RAW_DIR / "simulation_metadata.json"

    observed.to_csv(observed_path, index=False)
    truth.to_csv(truth_path, index=False)

    metadata = {
        "data_source": "simulated",
        "warning": "Not real store data. Ground-truth demand is known only because we generated it.",
        "seed": SEED,
        "n_days": N_DAYS,
        "n_products": len(PRODUCTS),
        "n_rows": int(len(df)),
        "store": STORE,
        "start_date": START_DATE.isoformat(),
        "end_date": (START_DATE + timedelta(days=N_DAYS - 1)).isoformat(),
        "censoring_rule": "sales = min(true_demand, opening_stock)",
        "stockout_flag_definition": "closing_stock == 0 (observable; not the same as true lost demand)",
        "is_censored_definition": "true_demand >= opening_stock (requires ground truth)",
        "replenishment": "(s, Q) policy with lead time; some SKUs are intentionally under-stocked",
        "true_demand_usage": "evaluation only; do not use as a training feature",
        "n_censored_rows": int(df["is_censored"].sum()),
        "n_stockout_flag_rows": int(df["stockout_flag"].sum()),
        "total_lost_sales_units": int(df["lost_sales"].sum()),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"observed": observed_path, "truth": truth_path, "metadata": meta_path}


def main() -> None:
    df = simulate_store()
    paths = write_outputs(df)
    print(f"rows={len(df)} products={df['product'].nunique()} days={df['date'].nunique()}")
    print(f"censored_rows={int(df['is_censored'].sum())} stockout_flag={int(df['stockout_flag'].sum())}")
    print(f"lost_sales_units={int(df['lost_sales'].sum())}")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
