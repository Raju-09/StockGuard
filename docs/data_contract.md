# StockGuard — Canonical Data Contract & Leakage-Free Validation

## 1. Canonical Schema Definition

Any incoming retail data source (CSV upload, POS connector, Shopify/ERP export) must be parsed into the canonical schema before downstream consumption.

| Field Name | Type | Nullable | Valid Range | Business Definition |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `ISO-8601 Date` (`YYYY-MM-DD`) | No | $\ge 2015\text{-}01\text{-}01$ | The operational calendar date of the record. |
| `sku` | `String` | No | ASCII / Alphanumeric | Unique product identifier (Stock Keeping Unit). |
| `sales` | `Float` | No | $[0.0, 100,000.0]$ | Observed POS sales units recorded for the day. |
| `opening_stock`| `Float` | No | $[0.0, 1,000,000.0]$ | Available on-shelf inventory at start of day. |
| `price` | `Float` | No | $(0.0, 500,000.0]$ | Current retail selling price in store currency (INR/USD). |
| `store` | `String` | Yes | Default: `STORE-001` | Store or fulfillment center identifier. |
| `promo` | `Integer` | Yes | $\{0, 1\}$ | Binary promotional flag active on that date. |

### Operational Extension Fields (Inventory Position)
| Field Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `on_order` | `Float` | `0.0` | Open Purchase Orders confirmed with supplier but not shipped. |
| `in_transit` | `Float` | `0.0` | Purchase Orders shipped and currently in logistics transit. |
| `reserved` | `Float` | `0.0` | Inventory committed to customer orders but not yet fulfilled. |
| `lead_time_days` | `Float` | `3.0` | Expected supplier delivery time in calendar days ($\overline{LT}$). |
| `lead_time_std` | `Float` | `0.5` | Standard deviation of supplier delivery time ($\sigma_{LT}$). |
| `moq` | `Integer` | `1` | Supplier Minimum Order Quantity. |
| `case_pack_size`| `Integer` | `1` | Supplier shipping case pack multiple. |

---

## 2. Invariant Rules: Zero vs. Missing Values

A common source of retail data corruption is conflating zero with missing:

1. **Zero Sales with Opening Stock $> 0$:**
   $$\text{Recorded: } S_t = 0, \quad I_t > 0 \implies \text{True Zero Demand } (D_t = 0)$$
   *Action:* Retained as valid negative example for model training.

2. **Zero Sales with Opening Stock $\le 0$:**
   $$\text{Recorded: } S_t = 0, \quad I_t \le 0 \implies \text{Censored Demand } (D_t \ge 0, \text{ unobserved})$$
   *Action:* Annotated with `is_stockout = 1`. Passed to Latent Demand Recovery Layer. Must **NOT** be treated as a zero-demand day.

3. **Negative Sales Values:**
   $$\text{Recorded: } S_t < 0 \implies \text{Customer Return / Adjustment}$$
   *Action:* Clipped to $0$ for demand estimation. Customer returns must be routed to reverse-logistics reconciliation.

4. **Missing Dates:**
   *Action:* The time series must be reindexed to continuous calendar days. Gaps are filled with forward-filled inventory and zero sales only if store is verified open.

---

## 3. Walk-Forward Chronological Split Protocol

To eliminate temporal leakage in time-series validation:

```
  Fold 1: [--- Train 180d ---] [Gap 7d] [Test 14d]
  Fold 2: [------ Train 194d ------] [Gap 7d] [Test 14d]
  Fold 3: [--------- Train 208d ---------] [Gap 7d] [Test 14d]
  Fold 4: [------------ Train 222d ------------] [Gap 7d] [Test 14d]
  Fold 5: [--------------- Train 236d ---------------] [Gap 7d] [Test 14d]
```

### Formal Split Generator Contract
- Given temporal sequence $\mathcal{T} = \{t_1, t_2, \dots, t_N\}$:
  - $\text{Cutoff}_k = t_{N - (K - k) \cdot H}$
  - $\text{Train}_k = \{ t \in \mathcal{T} \mid t \le \text{Cutoff}_k \}$
  - $\text{Blackout Gap}_k = \{ t \in \mathcal{T} \mid \text{Cutoff}_k < t \le \text{Cutoff}_k + L \}$
  - $\text{Test}_k = \{ t \in \mathcal{T} \mid \text{Cutoff}_k + L < t \le \text{Cutoff}_k + L + H \}$
- **Lead-Time Blackout ($L$):** Set equal to supplier lead time (default: 7 days) to simulate the real-world operational delay where replenishment decisions must be committed $L$ days ahead of arrival.

---

## 4. Leakage-Prevention Invariants (CI Automated Gates)

The test suite enforces the following unit assertions:
1. **Feature Calculation Lag:**
   Every rolling aggregate (e.g. `sales_rolling_mean_7d`, `sales_std_14d`) must be strictly lagged:
   $$\text{Feature}_t = f(\{ x_\tau \mid \tau \le t - 1 \})$$
   Any rolling window that includes $t$ (contemporary) fails CI.
2. **Target Invariant:**
   Target variable $y_{t}$ represents future cumulative demand during lead-time horizon $(t, t + L]$ and must never appear in feature matrix $X_{t}$.
3. **Split Separation Invariant:**
   $$\max(\text{Dates}(\text{Train}_k)) < \min(\text{Dates}(\text{Test}_k)) \quad \forall k \in [1, K]$$
