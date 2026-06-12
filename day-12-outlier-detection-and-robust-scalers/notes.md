# Day 12 — Outlier Detection & Robust Scalers

## Univariate rules
- **z-score:** $|z|>3$ with $z=(x-\bar x)/s$. Weakness: $\bar x$ and $s$ are themselves inflated by outliers (**masking**). Max possible $|z|$ in a sample of $n$ is $(n-1)/\sqrt n$ (small $n$ can never reach 3).
- **IQR (Tukey):** flag outside $[Q_1-1.5\,\text{IQR},\,Q_3+1.5\,\text{IQR}]$; breakdown point 25%.
- **Modified z (MAD):** $M_i=0.6745\,(x_i-\tilde x)/\text{MAD}$, flag $|M_i|>3.5$; breakdown 50%. $\text{MAD}\cdot1.4826\approx\sigma$ for Gaussian.

## Multivariate
- **Mahalanobis:** $d^2=(x-\mu)^\top\Sigma^{-1}(x-\mu)\sim\chi^2_p$ under Gaussianity; use a robust covariance (MCD) to avoid masking.
- **Isolation Forest:** outliers are isolated by fewer random splits; score $s=2^{-E[h(x)]/c(n)}$. No distribution assumptions, $O(n\log n)$.
- LOF (local density), DBSCAN, one-class SVM, autoencoder reconstruction error.

## Treat or not?
Outliers can be errors (remove/fix), rare-but-real (keep! fraud, high-value customers), or heavy-tailed behavior (transform: log, Box-Cox). Handling: winsorize/clip, transform, robust models (Huber, quantile regression, trees), separate model.

## Robust scaling
$x'=(x-\text{median})/\text{IQR}$ — insensitive to extremes; StandardScaler uses $(\mu,\sigma)$ and is dragged by them. `PowerTransformer` / `QuantileTransformer` for skew.

## FAANG Q&A
**Q1. Why can a single outlier break z-score detection?** Inflates $s$, shrinking everyone's $z$.
**Q2. Gaussian: what % falls outside 1.5 IQR?** ≈0.7%.
**Q3. Do trees need outlier removal?** Split-based => monotone invariance; far less sensitive. Linear/NN/KNN/k-means are.
**Q4. Outliers in a target for regression?** Huber/quantile loss, or log-target.
**Q5. Online anomaly detection in a metric?** Rolling median/MAD, EWMA control chart, seasonal decomposition residuals.
