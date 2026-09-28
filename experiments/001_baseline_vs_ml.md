# Experiment 001: Benchmarking ML vs Statistical Baselines under Chronological Split

## Question
Does machine learning (Gradient Boosting, Random Forest) actually beat simple statistical baselines (yesterday's persistence, 7-day moving average) on retail demand forecasting?

## Hypothesis
Lagged and promotional features will allow non-linear models to outperform persistence, but simple 7-day moving average will remain competitive on stable staple goods.

## Protocol
- **Dataset:** 10 FMCG SKUs, 120 days of daily observations.
- **Split:** Strictly chronological (70% train: days 0-83, 10% val: days 84-95, 20% test: days 96-119). No shuffling.
- **Leakage Prevention:** All lag and rolling features computed using `sales.shift(1)` to exclude same-day sales.

## Results on Held-Out Test Window
| Model | Test WAPE | Mean Bias (units) | Comments |
|---|---|---|---|
| Naive (Yesterday) | 44.1% | -1.72 | Erratic on weekend demand spikes |
| Moving Average 7d | 31.7% | -1.74 | Strong, smooth baseline |
| Linear Regression | 35.2% | -1.80 | Underfits promo interaction |
| Random Forest | 34.1% | -1.62 | Overfits small training window |
| Gradient Boosting (GBM) | 33.8% | -1.52 | Captures promo multipliers, but suffers negative bias |

## What Failed
All models trained directly on observed sales exhibited a systematic **negative forecast bias** (-1.5 to -1.8 units/day). 

## Decision
Investigate root cause of negative bias: retail shelf stockouts censor sales to zero when actual customer demand was positive.
