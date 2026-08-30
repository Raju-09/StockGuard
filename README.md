# StockGuard

> **Estimate true product demand when observed sales are censored by inventory availability.**

[![CI](https://github.com/YOUR_USERNAME/StockGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/StockGuard/actions/workflows/ci.yml)
[![CodeRabbit](https://img.shields.io/badge/CodeRabbit-enabled-orange)](https://coderabbit.ai)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)

---

## Problem

A grocery store records **sales**, not **demand**.

When a shelf runs out of stock, zero sales does not mean zero demand — it means the store had nothing left to sell. A model trained directly on observed sales will learn that demand was low precisely on the days it was actually high but constrained.

```
Day  | Opening Stock | True Demand | Recorded Sales
-----|---------------|-------------|---------------
1    | 8             | 14          | 8   ← stock ran out
2    | 0             | 8           | 0   ← looks like zero demand
3    | 0             | 5           | 0   ← still looks like zero demand
```

A naive model sees `8, 0, 0` and concludes demand is falling. It did not fall.

This is **inventory-censored demand** — the core problem StockGuard addresses.

---

## Why existing approaches fail

| Common approach | Why it fails |
| --- | --- |
| Train on observed sales | Learns artificial demand drops during stockouts |
| Random train/test split | Future periods leak into training; unrealistic evaluation |
| Accuracy as metric | Classification metric on a regression problem |
| MAPE with zero actuals | Division by zero on stockout days |

---

## Research Questions

**H1:** Accounting for inventory-censored sales will reduce demand-forecast error during stockout-affected periods, compared with a model trained directly on observed sales.

**H2:** Improved demand estimates will reduce simulated stockout rate and/or total inventory cost under the same replenishment policy.

**H0 (null):** Stockout-aware modeling does not materially improve forecasting or inventory outcomes over the baseline.

We test all three. H0 is a valid result — we do not manipulate experiments to get a pretty answer.

---

## Dataset

| Item | Value |
| --- | --- |
| Source | **SIMULATED** (not a real shop) |
| Store | 1 |
| Products | 10 |
| Days | 120 (2025-09-01 → 2025-12-29) |
| Rows | 1,200 |
| Censoring rule | `observed_sales = min(true_demand, opening_stock)` |

Two files:

| File | Contents |
| --- | --- |
| `data/raw/observed_daily.csv` | What a store system could log. No ground-truth demand. |
| `data/raw/with_ground_truth.csv` | Same rows + hidden `true_demand` (evaluation only, never a model feature). |

Regenerate:

```powershell
.\venv\Scripts\python.exe src\simulate.py
```

---

## Data Collection Protocol

*(Real-store extension — Phase TODO)*

For each product, record every evening:
1. Opening stock (morning count)
2. Units sold
3. Closing stock
4. Selling price
5. Promotions active
6. Whether product was unavailable at any point during the day

Label data as `real` or `simulated` — never mix them silently.

---

## Exploratory Analysis

See [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb).

Key findings: *(filled in during Phase 3)*

---

## Baselines

See [`notebooks/03_baseline.ipynb`](notebooks/03_baseline.ipynb).

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| Naive (yesterday's sales) | — | — | — |
| 7-day moving average | — | — | — |
| 14-day moving average | — | — | — |

---

## Methodology

### Temporal validation

```
TRAIN      days   0 – 83   (70 %)
VALIDATION days  84 – 95   (10 %)
TEST       days  96 – 119  (20 %)
```

Future must remain future. No random splitting.

### Prediction-time information boundary

At the moment of prediction (e.g. Monday 8 PM), the model may use:
- Monday's opening stock, price, promotion, holiday flag
- Historical sales up to and including Monday
- Lag features (yesterday, 7 days ago)
- Rolling means of past sales

The model must **not** use:
- Tuesday's closing stock (future)
- `true_demand`, `lost_sales`, `is_censored` (ground truth)

---

## Experimental Design

Three models are compared on the **same test set**:

1. **Baseline** — best moving average
2. **Standard ML** — best of Linear Regression / Random Forest / XGBoost trained on observed sales
3. **Censoring-aware** — same ML architecture but demand imputation applied to stockout-affected training rows (Tobit regression or EM imputation, Phase 8)

Then the business simulator compares replenishment outcomes under each model.

---

## Model Comparison

See [`notebooks/04_model_comparison.ipynb`](notebooks/04_model_comparison.ipynb).

| Model | MAE (all) | MAE (censored) | MAE (uncensored) | WAPE | Stockout % | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | — | — | — | — | — | — |
| MA-7 | — | — | — | — | — | — |
| Linear Regression | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| Gradient Boosting | — | — | — | — | — | — |
| Censoring-aware | — | — | — | — | — | — |

*(Filled during Phases 5–8)*

---

## Stockout-Censoring Analysis

*(Phase 8)*

Algorithm used: *(Tobit regression / EM imputation — decided in Phase 8)*

Key question: does the model recover demand that was hidden by inventory constraints?

---

## Business Simulation

*(Phase 10)*

Compare four replenishment policies:

| Policy | Description |
| --- | --- |
| A | No forecasting (fixed order quantity) |
| B | Naive forecast |
| C | ML forecast (standard) |
| D | ML forecast + censoring-aware demand estimation |

Metrics: stockout %, holding cost, ordering cost, lost-sales cost, service level.

---

## Results

*(Filled after Phase 10)*

---

## Failure Analysis

*(Phase 14)*

| Situation | Error level | Hypothesis |
| --- | --- | --- |
| Stockout day | High | Demand censored in training |
| Promotion | High | Promotion multiplier underestimated |
| Holiday | Medium | Calendar feature insufficient |
| New product | High | Insufficient history |

---

## Deployment Architecture

*(Phase 11)*

```
Browser → HTTP → FastAPI → validate input → load model → predict → JSON → Browser
```

---

## API

*(Phase 11)*

```http
POST /predict
Content-Type: application/json

{
  "product": "Milk",
  "current_inventory": 25,
  "price": 55,
  "promotion": 0
}
```

Response:

```json
{
  "product": "Milk",
  "forecast_demand": 31.2,
  "stockout_risk": 0.87,
  "recommended_order": 55,
  "days_until_stockout": 0.8,
  "model_version": "v2",
  "explanation": {
    "lead_time_demand": 62.4,
    "safety_stock": 20,
    "current_inventory": 25
  }
}
```

---

## Frontend

*(Phase 12)* — HTML/CSS/JavaScript. No React.

---

## Monitoring

*(Phase 13)*

Track monthly: MAE, WAPE, stockout rate, prediction distribution, feature drift.

---

## Limitations

- Data is simulated. Conclusions about real grocery stores require validation against real data.
- The (s, Q) replenishment policy used in simulation is a simplification.
- Tobit regression assumes normally distributed demand — real demand may differ.
- 120 days is a short history for seasonal products.

---

## Reproducibility

```powershell
# 1. Clone and create venv
git clone https://github.com/YOUR_USERNAME/StockGuard.git
python -m venv venv
.\venv\Scripts\activate

# 2. Install
pip install -e ".[dev]"

# 3. Regenerate dataset
python src\simulate.py

# 4. Run tests
pytest tests/ -v
```

All randomness is seeded (`SEED = 42` in `src/simulate.py`).

---

## Future Work

- Extension A: Multi-store — does a cross-store model generalise better than per-store models?
- Extension B: Promotion-aware forecasting
- Extension C: Cold-start forecasting for new products

---

## Current Phase

**Phase 2** — pandas fundamentals on this dataset. No ML yet.

| Phase | Status |
| --- | --- |
| 0: Environment | ✅ Done |
| 1: Dataset + simulate.py | ✅ Done |
| 2: pandas / EDA | 🔄 In progress |
| 3: Baselines | ⬜ |
| 4: First ML model | ⬜ |
| 5: Temporal validation | ⬜ |
| 6: Feature engineering | ⬜ |
| 7: Evaluation metrics | ⬜ |
| 8: Censoring-aware model | ⬜ |
| 9: Stockout classifier | ⬜ |
| 10: Business simulator | ⬜ |
| 11: FastAPI | ⬜ |
| 12: Frontend | ⬜ |
| 13: Deploy + monitoring | ⬜ |
| 14: Error analysis + README | ⬜ |

---

## Environment

Python 3.11 in a local virtualenv.

```powershell
.\venv\Scripts\activate
python --version         # 3.11.x
pip install -e ".[dev]"  # installs all deps + dev tools
```
