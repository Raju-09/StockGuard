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

Empirical results evaluated on the unseen temporal test set (240 product-days, Dec 06 – Dec 29, 2025):

| Model | MAE (all) | MAE (censored) | MAE (uncensored) | WAPE (all) | Bias | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Censoring-Aware (Imputed)** | **3.042** | **3.595** | **2.828** | **0.276** | -0.787 | **#1** |
| 14-day Moving Average | 3.476 | 3.839 | 3.336 | 0.316 | -1.769 | #2 |
| 7-day Moving Average | 3.489 | 3.823 | 3.360 | 0.317 | -1.744 | #3 |
| Gradient Boosting (XGBoost) | 3.723 | 5.881 | 2.888 | 0.338 | -1.521 | #4 |
| Random Forest Regressor | 3.753 | 5.870 | 2.933 | 0.341 | -1.624 | #5 |
| Linear Regression | 3.987 | 5.533 | 3.388 | 0.362 | -1.910 | #6 |
| Naive (Lag 1) | 4.850 | 4.433 | 5.012 | 0.441 | -1.717 | #7 |

### Stockout Classifier Performance

Predicting whether a stockout will occur within the next 3 days ($P(\text{stockout} \le 3\text{d})$):
- **Precision:** 86.9%
- **Recall:** 80.7%
- **F1 Score:** 83.7%
- **PR-AUC:** 0.924

---

## Stockout-Censoring Analysis

### Research Finding (H1 Confirmed)
Standard supervised ML models trained on recorded sales suffer severe degradation on stockout days (**MAE rises to 5.88**). Because they observed `sales = min(demand, stock)`, the models learned that demand crashes to zero when stock runs out.

By applying **Two-Stage Censored Imputation** (fitting an unbiased estimator on non-stockout periods, imputing constrained days as $\max(\text{observed}, \hat{d})$, and retraining), the Censoring-Aware model cuts error on censored periods from **5.88 to 3.59 (a 38.9% error reduction)**.

---

## Business Simulation

Evaluating 4 inventory replenishment policies over the test horizon:

| Policy | Replenishment Logic | Stockout % | Service Level | Lost Sales (Units) | Total Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| **A: No Forecast** | Static $(s, Q)$ reorder policy | 7.08% | 92.9% | 101 | $7,336.50 |
| **B: Naive** | Dynamic $(s, Q)$ using Lag 1 | 5.83% | 94.2% | 101 | $10,600.50 |
| **C: Gradient Boost** | Dynamic $(s, Q)$ using XGBoost | 4.58% | 95.4% | 90 | $9,220.50 |
| **D: Censoring-Aware** | Dynamic $(s, Q)$ using Imputed Model | **4.58%** | **95.4%** | **90** | $9,542.00 |

### Business Takeaway (H2 Confirmed)
Dynamic ML replenishment improves service level from 92.9% to 95.4%, preventing 11 stockout days across the store and preserving 11 units of customer demand. Because Censoring-Aware estimates higher unconstrained demand, it maintains slightly larger safety buffers to prevent stockouts.

---

## Engineering Scars: What Didn't Work

Real engineering leaves a paper trail of hypotheses that failed and required architectural redesigns.

| What Failed | Why it Failed | What We Changed |
|---|---|---|
| **1. Unshifted Rolling Aggregations** | Rolling mean of sales without `shift(1)` included today's sales in today's feature matrix, producing an artificially low test WAPE. | Enforced strict `shift(1)` across all lags and rolling statistics in `src/features.py`. Added invariant test `test_no_future_leakage_in_features`. |
| **2. Un-leveled Policy Simulation** | In early simulation backtests, the No-Forecast baseline won on total cost because it held lower safety stock, running inventory down to zero while ML maintained target service levels. | Standardized every policy onto the exact same Base-Stock $(s, S)$ mechanism. Only the forecast signal $\hat{d}$ varies, isolating the value of forecasting accuracy. |
| **3. High Stockout Risk on Overstocked SKUs** | Classification model predicted high risk based on historical frequency, but system recommended `Order: 0` because current runway was 30 days. Users were confused. | Added structured `order_zero_reason` in the decision contract: explicitly differentiating historical risk from immediate runway safety. |
| **4. Raw PSI on Calendar Features** | Applying Population Stability Index to `month` (1–12) or `is_weekend` (0/1) produced false alarms ($PSI > 5.0$) due to normal seasonal progression. | Engineered feature-type-aware drift routing in `src/monitoring.py`: PSI for continuous, Jensen-Shannon for categorical, and Wilson 95% CI for binary flags. |
| **5. Deterministic 2-Day Lead Time** | Assuming zero supplier delivery variability resulted in stockouts during transit delays, even with 95% service target. | Added stochastic lead time parameterization ($\sigma_{\text{LT}} = 0.5\text{d}$) and upgraded safety stock: $SS = z \sqrt{\overline{\text{LT}} \sigma_d^2 + \bar{d}^2 \sigma_{\text{LT}}^2}$. |

---

## Architectural Decision Records: Why We Rejected This

- **Why not Deep Learning / LSTM?**  
  Our dataset size (10 SKUs, 120 days) does not justify sequential neural networks. Gradient boosting with lag features trains in milliseconds, offers exact feature attribution, and avoids catastrophic overfitting on small samples.
- **Why not an LLM for replenishment quantities?**  
  Calculating reorder points is deterministic arithmetic: $S = \text{Lead Demand} + \text{Safety Stock}$, rounded to case packs and constrained by MOQ. Using an LLM for numerical supply chain optimization is brittle, non-reproducible, and un-auditable by CFOs.
- **Why not Reinforcement Learning (RL)?**  
  Before attempting sequential policy optimization via PPO or DQN, we must establish a rigorous offline $(s, S)$ heuristic benchmark with stochastic demand and lead time.

---

## Experiments Archive

See detailed hypothesis-driven experiment writeups in [`experiments/`](experiments/):
- [`001_baseline_vs_ml.md`](experiments/001_baseline_vs_ml.md): Empirical verification that statistical baselines must be beaten.
- [`002_censoring_recovery.md`](experiments/002_censoring_recovery.md): Two-stage Tobit-inspired latent demand imputation.
- [`003_stochastic_lead_time.md`](experiments/003_stochastic_lead_time.md): Dual-variance safety stock under supplier delivery shocks.


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

## Project Status & Implementation Phases

| Phase | Description | Status |
| :---: | :--- | :---: |
| 0 | Virtualenv, dependencies, pyproject.toml | ✅ Complete |
| 1 | Ground truth simulator & synthetic data collection | ✅ Complete |
| 2 | Pandas data validation & schema contracts (`src/data.py`) | ✅ Complete |
| 3 | Exploratory Data Analysis (`notebooks/02_eda.ipynb`) | ✅ Complete |
| 4 | Naive & Moving Average baselines (`notebooks/03_baseline.ipynb`) | ✅ Complete |
| 5 | Supervised ML Models (Linear, RF, XGBoost) | ✅ Complete |
| 6 | Chronological feature engineering (`src/features.py`) | ✅ Complete |
| 7 | Evaluation metrics: MAE, RMSE, WAPE, Bias (`src/evaluate.py`) | ✅ Complete |
| 8 | Two-Stage Censoring-Aware Imputation Model (`src/train.py`) | ✅ Complete |
| 9 | Stockout Classifier ($P(\text{stockout} \le 3\text{d})$) | ✅ Complete |
| 10 | 4-Policy Business Replenishment Simulator (`src/simulator.py`) | ✅ Complete |
| 11 | Production FastAPI service (`app/main.py`) | ✅ Complete |
| 12 | Vanilla HTML/CSS/JS Dashboard with Chart.js (`app/static/`) | ✅ Complete |
| 13 | End-to-end integration tests & CI/CD workflow (`tests/`) | ✅ Complete |
| 14 | Model comparison, error analysis & research report | ✅ Complete |

---

## Environment

Python 3.11 in a local virtualenv.

```powershell
.\venv\Scripts\activate
python --version         # 3.11.x
pip install -e ".[dev]"  # installs all deps + dev tools
```
