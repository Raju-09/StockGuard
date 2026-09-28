# Experiment 003: Stochastic Lead Time & Safety Stock Buffering

## Question
Does modeling supplier delivery lead time as a random variable change optimal safety stock and prevent supply shocks compared to deterministic lead-time assumptions?

## Hypothesis
Using a constant 2-day lead time assumes zero supplier variability. When real transit times fluctuate (e.g., normal distribution with $\sigma_{\text{LT}} = 0.5$ days), a deterministic safety stock formula will experience stockouts due to early stock depletion before deliveries arrive.

## Formula Comparison
- **Deterministic formula:**
  $$SS = z \cdot \sqrt{\text{LT}} \cdot \sigma_d$$
- **Stochastic formula (Accounts for both demand & supplier variance):**
  $$SS = z \cdot \sqrt{\overline{\text{LT}} \cdot \sigma_d^2 + \bar{d}^2 \cdot \sigma_{\text{LT}}^2}$$

## Simulation Results (10 SKUs, 24 Test Days)
| Formulation | Service Level | Stockout Days | Total Holding Cost | Total Cost |
|---|---|---|---|---|
| Deterministic SS ($\sigma_{\text{LT}} = 0$) under random transit | 91.8% | 14 | ₹1,420 | ₹5,890 |
| Stochastic SS ($\sigma_{\text{LT}} = 0.5$) under random transit | **95.4%** | **4** | ₹1,610 | **₹4,250** |

## Engineering Scars & Takeaway
A slightly higher buffer holding cost (₹1,610 vs ₹1,420) completely eliminated 10 stockout days, dramatically lowering lost margin penalties and reducing total enterprise cost by over 27%.
