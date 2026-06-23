# Day 23 — Feature Encoding

## Options for categoricals
| Encoding | Dim | Notes |
|---|---|---|
| One-hot | $k$ | linear models; blows up for high cardinality (sparse) |
| Ordinal | 1 | only for truly ordered levels (or trees) |
| Frequency/count | 1 | popularity signal; collisions between levels with equal counts |
| Target (mean) | 1 | powerful for high cardinality; **leakage-prone** |
| Hashing | $m$ fixed | collisions; no vocabulary; online-friendly |
| Embedding | $m$ | learned (NNs, entity embeddings) |
| Binary/Helmert/WoE | log k | WoE $=\ln\frac{P(x|y{=}1)}{P(x|y{=}0)}$ for credit scoring |

## Target encoding math & leakage
Naive: $\text{TE}(c)=\bar y_c$. For a rare level with $n_c=1$ the encoding *is the label* ⇒ the model memorizes → huge train/test gap.
**Smoothing** (empirical-Bayes shrinkage): $\text{TE}(c)=\frac{n_c\bar y_c+m\,\bar y}{n_c+m}$; $m$ = pseudo-count (prior strength). Equivalent to the posterior mean of Beta–Binomial.
**Out-of-fold (K-fold) encoding:** encode each row using statistics from other folds so a row's own label never enters its feature. At inference use full-train statistics. Add noise or leave-one-out (LOO still leaks via the shift of the mean — row's encoding is a deterministic function of its own label: $\frac{S_c-y_i}{n_c-1}$).
For time-dependent data use expanding (ordered) target statistics (CatBoost's ordered boosting).

## Practical rules
Fit all encoders inside the CV `Pipeline`; handle unseen categories (fallback to prior/“other”); group rare levels; regression targets can use mean/median; multiclass => per-class probabilities; combine frequency + target encoding.

## FAANG Q&A
**Q1. Why does target encoding overfit?** Label information from the row enters its own feature, strongest for small groups.
**Q2. Zip code with 40k levels?** Smoothed OOF target encoding + frequency + hierarchical backoff (zip→city→state).
**Q3. One-hot for tree models?** High-cardinality one-hot makes trees' splits inefficient; use native categorical support (LightGBM/CatBoost) or TE.
**Q4. Handle unseen category at inference?** Map to global prior/“unknown” bucket; monitor rate.
**Q5. Is frequency encoding leaky?** Not target-based, but can encode data-collection artifacts; compute on train only.
