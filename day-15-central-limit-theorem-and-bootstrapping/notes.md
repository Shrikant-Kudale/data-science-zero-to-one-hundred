# Day 15 — CLT & Bootstrapping

## CLT
For i.i.d. $X_i$ with mean $\mu$ and finite variance $\sigma^2$: $\sqrt n(\bar X_n-\mu)/\sigma\xrightarrow{d}N(0,1)$.
Standard error $\text{SE}(\bar X)=\sigma/\sqrt n$ (estimate with $s/\sqrt n$). Quadrupling data halves SE.
**Berry–Esseen:** error in CDF approximation $\le C\rho/(\sigma^3\sqrt n)$, $\rho=E|X-\mu|^3$. Skew of $\bar X$ decays as $\gamma_1/\sqrt n$ => very skewed data (lognormal $\sigma=2$, revenue per user) needs huge $n$.
Fails: infinite variance (Cauchy; Pareto $\alpha\le2$), strong dependence, tiny $n$ with heavy tails.

## Delta method
If $\sqrt n(\hat\theta-\theta)\to N(0,\tau^2)$ then $\sqrt n(g(\hat\theta)-g(\theta))\to N(0,g'(\theta)^2\tau^2)$. Used for ratio metrics (CTR=clicks/views per user) — the user, not the view, is the independent unit.

## Bootstrap (Efron)
Resample the data with replacement $B$ times, compute the statistic each time; the empirical distribution of $\hat\theta^*$ approximates the sampling distribution.
- Standard error: $\text{sd}(\hat\theta^*)$. Percentile CI: $[q_{2.5},q_{97.5}]$ of $\hat\theta^*$.
- **BCa** corrects bias and skew (acceleration via jackknife) => second-order accurate.
- Works for medians, quantiles, AUC, ratio metrics, with no formula needed.
- Fails: extremes (max), tiny $n$, dependent data (use block bootstrap), heavy tails for the mean.
- **Permutation test** is the resampling analog for hypothesis testing under exchangeability.

## FAANG Q&A
**Q1. Is the data distribution or sampling distribution normal under CLT?** The sampling distribution of the mean — not the data.
**Q2. How many bootstrap replicates?** 1,000–10,000 for CIs (SE needs ~200).
**Q3. Why does bootstrap work conceptually?** The ECDF $\hat F_n$ converges to $F$ (Glivenko–Cantelli); plug-in principle.
**Q4. Bootstrap A/B test with user-level randomization?** Resample *users* (cluster bootstrap), not events.
**Q5. 95% CI interpretation?** Procedure covers the truth in 95% of repeated experiments; not "95% probability the parameter is in this interval".
