# StockGuard — Research Question & Falsifiable Claim

## 1. Core Research Hypothesis ($H_1$)

> **When retail demand is censored by stockout periods, recovering latent demand prior to forecasting reduces downstream inventory decision costs more than improving the forecasting model itself.**

### Mathematical Formulation

Let:
- $D_t \in \mathbb{R}_{\ge 0}$ be the true, latent consumer demand at time $t$.
- $I_t \in \mathbb{R}_{\ge 0}$ be the on-shelf inventory available at the beginning of period $t$.
- $S_t = \min(D_t, I_t)$ be the observed sales recorded in POS / ERP.
- $M \in \{ \text{Simple}, \text{Complex} \}$ be the forecasting model family (e.g. Simple = Seasonal Naïve / Ridge Regression, Complex = LightGBM / XGBoost).
- $R \in \{ \text{Naive}, \text{Censoring-Aware} \}$ be the demand recovery function, where:
  - $R_{\text{Naive}}(S_t) = S_t$ (treating observed sales as unconstrained demand)
  - $R_{\text{Aware}}(S_t, I_t) = \hat{D}_t$ (reconstructing latent demand via Tobit / EM-analog when $I_t \le 0$)
- $\pi(\hat{y}, I_{\text{pos}}, \mathbf{\Omega})$ be the deterministic replenishment policy under supplier constraints $\mathbf{\Omega} = \{ \text{MOQ}, \text{case\_pack}, \overline{LT}, \sigma_{LT} \}$.
- $\mathcal{C}(\pi)$ be the total operational business cost:
  $$\mathcal{C}(\pi) = \sum_{t=1}^T \left( c_h \cdot \max(0, I_t) + c_u \cdot \max(0, D_t - I_t) + c_w \cdot \text{Waste}_t \right)$$
  where $c_h$ is holding cost per unit-day, $c_u$ is stockout penalty (gross margin + customer churn risk factor), and $c_w$ is perishable disposal cost.

### Formal Falsifiable Claim

$$\mathbb{E}\left[ \mathcal{C}\left(\pi\left( M_{\text{Simple}}(R_{\text{Aware}}), \cdot \right)\right) \right] < \mathbb{E}\left[ \mathcal{C}\left(\pi\left( M_{\text{Complex}}(R_{\text{Naive}}), \cdot \right)\right) \right]$$

**Null Hypothesis ($H_0$):**
$$\mathbb{E}\left[ \mathcal{C}\left(\pi\left( M_{\text{Simple}}(R_{\text{Aware}}), \cdot \right)\right) \right] \ge \mathbb{E}\left[ \mathcal{C}\left(\pi\left( M_{\text{Complex}}(R_{\text{Naive}}), \cdot \right)\right) \right]$$
*(The recovery method makes no statistically significant reduction in business inventory cost; model complexity dominates or neither helps.)*

---

## 2. Secondary Research Hypothesis ($H_2$)

> **A reduction in forecast error ($\Delta \text{WAPE} < 0$) does not monotonically translate into a reduction in operational inventory cost ($\Delta \mathcal{C} < 0$) once discrete supplier constraints (MOQ, case packs) and stochastic lead times are enforced.**

This directly addresses the gap between pure ML benchmarking (Kaggle-style WAPE/RMSE) and real supply-chain decision intelligence.

---

## 3. Evaluation Metric Families

We evaluate the system across three distinct, non-overlapping metric families:

### Family A: Forecast Accuracy & Calibration
1. **Weighted Absolute Percentage Error (WAPE):**
   $$\text{WAPE} = \frac{\sum_{i,t} |y_{i,t} - \hat{y}_{i,t}|}{\sum_{i,t} y_{i,t}}$$
2. **Quantile Pinball Loss ($\alpha \in \{0.10, 0.50, 0.90\}$):**
   $$L_\alpha(y, \hat{q}_\alpha) = \max\left(\alpha(y - \hat{q}_\alpha), (\alpha - 1)(y - \hat{q}_\alpha)\right)$$
3. **Empirical Prediction Interval Coverage ($\hat{C}_{80}$):**
   $$\hat{C}_{80} = \frac{1}{N} \sum_{i,t} \mathbb{I}\left( y_{i,t} \in [\hat{q}_{0.10}, \hat{q}_{0.90}] \right) \quad (\text{Target: } 80\% \pm 5\%)$$

### Family B: Inventory Operational Outcomes
1. **Service Level / Fill Rate:**
   $$\text{Fill Rate} = 1 - \frac{\sum \text{Stockout Units}}{\sum \text{Consumer Demand}}$$
2. **Stockout Incidence Rate:**
   $$\text{Stockout Days \%} = \frac{\sum \mathbb{I}(I_{i,t} \le 0)}{N \cdot T}$$
3. **Average Days of Cover:**
   $$\text{DOC}_{i} = \frac{I_{i,t}}{\hat{d}_{i, 7d}}$$

### Family C: Business Decision Cost ($\mathcal{C}$)
1. **Holding Cost:** Cumulative capital tied in warehouse storage ($c_h = 18\%\text{--}24\% \text{ annual rate} / 365$).
2. **Lost Margin (Stockout Cost):** Unfulfilled demand during stockouts multiplied by SKU gross margin percentage ($c_u = \text{price} - \text{cogs} + \text{goodwill penalty}$).
3. **Decision Error Cost (INR):** Direct financial delta between the policy recommendation and the hindsight-optimal cost.

---

## 4. Benchmark Baselines (Leaderboard Anchor)

To prevent ungrounded claims, every experiment must report against:
1. **Naïve Baseline:** Tomorrow's demand = Today's observed sales ($\hat{y}_{t+1} = S_t$).
2. **Seasonal Naïve:** Tomorrow's demand = Same weekday last week ($\hat{y}_{t+1} = S_{t-7}$).
3. **Moving Average (7-day / 28-day):** Rolling window mean of sales.
4. **Croston's Method & SBA (Syntetos-Boylan Approximation):** Dedicated baseline for intermittent demand series ($>50\%$ zero-demand days).
5. **Linear Ridge Regression:** Benchmark regularized regression with temporal features.
6. **Gradient Boosted Decision Trees (LightGBM / XGBoost):** Default non-linear tabular forecaster.

---

## 5. Statistical Rigor Protocol

- **Split Protocol:** Chronological rolling-origin walk-forward CV ($K=5$ folds) with a mandatory lead-time blackout window $L$ to ensure zero future information leakage.
- **Hypothesis Testing:** Paired Wilcoxon signed-rank test across all SKU time series comparing $\mathcal{C}(M_{\text{Simple}}(R_{\text{Aware}}))$ versus $\mathcal{C}(M_{\text{Complex}}(R_{\text{Naive}}))$ with significance threshold $\alpha = 0.05$.
- **Sub-segment Reporting:** All leaderboard results must be broken down by:
  - High-velocity regular demand ($\text{Zero-rate} < 20\%$)
  - Intermittent demand ($\text{Zero-rate} \ge 50\%$)
  - Highly seasonal / promotional SKUs
