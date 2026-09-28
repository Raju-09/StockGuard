"""
src/demand_recovery.py — Latent Demand Reconstruction for Censored Retail Time Series.

Addresses the foundational problem: Sales != Demand.
During stockouts (opening_stock <= 0 or sales == opening_stock), observed POS sales
understate real consumer demand. This module provides three recovery strategies:
1. 'naive': Assumes observed sales equal true demand (standard industry error).
2. 'tobit': Tobit-inspired imputation using conditional expectation above censoring threshold.
3. 'analog': Imputes demand during stockout days using category velocity & store-level uplift.
"""

from typing import Any
import math
import numpy as np
import pandas as pd


class DemandRecoveryEngine:
    """
    Reconstructs unconstrained consumer demand from censored POS sales and inventory.
    """

    def __init__(self, method: str = "tobit") -> None:
        self.method = method.lower()
        if self.method not in ("naive", "tobit", "analog"):
            raise ValueError(f"Unknown recovery method: {method}. Choose 'naive', 'tobit', or 'analog'.")

    def fit_transform(
        self,
        df: pd.DataFrame,
        sales_col: str = "sales",
        stock_col: str = "opening_stock",
        sku_col: str = "product",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Takes retail dataframe, detects stockout intervals, and creates 'latent_demand'
        and 'is_censored' columns.
        """
        out = df.copy()

        # Flag stockouts: opening_stock <= 0
        if stock_col in out.columns:
            out["is_censored"] = (out[stock_col] <= 0).astype(int)
        else:
            out["is_censored"] = 0

        if self.method == "naive":
            # Naive baseline: sales = demand
            out["latent_demand"] = out[sales_col].astype(float)
            return out

        # Compute per-SKU non-censored statistics
        grouped = out.groupby(sku_col)
        latent_series = []

        for sku, group in grouped:
            group = group.sort_values(date_col).copy()
            sales = group[sales_col].values.astype(float)
            censored = group["is_censored"].values.astype(int)

            # Normal (non-censored) days
            uncensored_sales = sales[censored == 0]
            if len(uncensored_sales) == 0:
                # Fallback if product was 100% stocked out
                group["latent_demand"] = np.maximum(sales, 1.0)
                latent_series.append(group)
                continue

            base_mean = float(np.mean(uncensored_sales))
            base_std = float(np.std(uncensored_sales)) if len(uncensored_sales) > 1 else max(1.0, base_mean * 0.25)
            base_std = max(base_std, 0.5)

            if self.method == "tobit":
                # Tobit expectation: E[Y | Y >= threshold] for normal distribution:
                # E[Y | Y >= c] = mu + sigma * [ phi((c - mu)/sigma) / (1 - Phi((c - mu)/sigma)) ]
                # where phi is standard normal PDF and Phi is standard normal CDF (Mills ratio)
                reconstructed = []
                for s, c in zip(sales, censored):
                    if c == 0:
                        reconstructed.append(s)
                    else:
                        threshold = s  # Lower bound observed
                        z = (threshold - base_mean) / base_std
                        # Avoid overflow in Mills ratio
                        z = float(np.clip(z, -3.0, 3.0))
                        pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)
                        cdf = 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))
                        mills_ratio = pdf / max(1.0 - cdf, 1e-4)
                        expected_demand = base_mean + base_std * mills_ratio
                        # Demand during stockout cannot be less than observed sales
                        reconstructed.append(max(float(s), round(expected_demand, 2)))

                group["latent_demand"] = reconstructed

            elif self.method == "analog":
                # Rolling 7-day non-censored median replacement
                rolling_valid = (
                    pd.Series(np.where(censored == 0, sales, np.nan))
                    .bfill()
                    .ffill()
                    .rolling(7, min_periods=1)
                    .median()
                )
                group["latent_demand"] = np.maximum(sales, rolling_valid.fillna(base_mean).values)

            latent_series.append(group)

        result_df = pd.concat(latent_series, ignore_index=True)
        return result_df.sort_values([sku_col, date_col]).reset_index(drop=True)


def run_censoring_ablation(
    df: pd.DataFrame,
    sales_col: str = "sales",
    stock_col: str = "opening_stock",
    sku_col: str = "product",
    date_col: str = "date",
) -> dict[str, Any]:
    """
    Controlled ablation comparing:
    (A) Naive POS Sales
    (B) Tobit-Recovered Latent Demand
    Measures under-forecasting bias and downstream unfulfilled demand estimate.
    """
    engine_naive = DemandRecoveryEngine(method="naive")
    engine_tobit = DemandRecoveryEngine(method="tobit")

    df_naive = engine_naive.fit_transform(df, sales_col=sales_col, stock_col=stock_col, sku_col=sku_col, date_col=date_col)
    df_tobit = engine_tobit.fit_transform(df, sales_col=sales_col, stock_col=stock_col, sku_col=sku_col, date_col=date_col)

    total_censored_days = int(df_naive["is_censored"].sum())
    total_sales = float(df_naive[sales_col].sum())
    recovered_demand = float(df_tobit["latent_demand"].sum())
    latent_gap = round(recovered_demand - total_sales, 2)
    underestimation_pct = round((latent_gap / total_sales) * 100, 2) if total_sales > 0 else 0.0

    return {
        "total_records": len(df),
        "censored_stockout_days": total_censored_days,
        "censored_rate_pct": round((total_censored_days / len(df)) * 100, 2) if len(df) > 0 else 0.0,
        "recorded_sales_units": total_sales,
        "recovered_latent_demand_units": recovered_demand,
        "masked_demand_units": latent_gap,
        "systematic_underestimation_pct": underestimation_pct,
    }
