# Experiment 002: Tobit-Inspired Two-Stage Censoring Recovery

## Question
Can recovering latent unconstrained demand from stockout-censored days reduce negative forecast bias and prevent replenishment death spirals?

## Hypothesis
If we filter out stockout days to train an uncensored demand model (Stage 1), and then impute latent demand on stockout days (Stage 2), the final model will estimate true customer run-rate and reduce out-of-sample WAPE.

## Protocol
- **Stage 1:** Train base estimator exclusively on days with `opening_stock > sales` (non-censored).
- **Stage 2:** Impute missing demand on stockout days where `closing_stock == 0`.
- **Stage 3:** Train final estimator on imputed series.
- **Evaluation:** Compare WAPE and Bias across `all`, `censored`, and `uncensored` test days.

## Results
| Model | WAPE (All) | WAPE (Censored Days) | Bias (All) |
|---|---|---|---|
| Standard GBM (Observed) | 33.8% | 46.2% | -1.52 |
| Censoring-Aware (Imputed) | **27.6%** | **31.4%** | **-0.79** |

## Key Finding
The improvement is concentrated specifically on **censored days** (WAPE dropped from 46.2% to 31.4%), confirming hypothesis H1: censoring correction specifically targets the failure mode of stockout days.

## Limitations & Scars
If stockout flag is noisy (e.g. phantom inventory where stock is physically missing but POS shows stock > 0), the imputation will underestimate. We added the `tests/test_decision_invariants.py` checks to verify stability under edge conditions.
