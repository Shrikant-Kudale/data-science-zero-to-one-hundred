# Day 20 — Exploratory Data Analysis Frameworks

## A repeatable framework (use in every case study)
1. **Frame**: target, unit of analysis, time horizon, decision to be made, evaluation metric.
2. **Structure & quality**: shape, dtypes, duplicates, key uniqueness, constant/near-constant columns, impossible values, units, timezone/date coverage.
3. **Missingness**: % by column, patterns (heatmap), relation to target (MAR/MNAR clue).
4. **Univariate**: histograms/KDE, skew, outliers, cardinality, rare categories.
5. **Bivariate vs target**: group target rate by bin/category; correlation (Pearson/Spearman), mutual information, Cramér's V; **partial** relationships (confounders).
6. **Multivariate**: correlation clusters (VIF/multicollinearity), PCA, interaction hunts.
7. **Leakage audit**: features unavailable at prediction time, post-outcome fields, ID-like columns, near-perfect single-feature predictors, time-order violations.
8. **Temporal**: stability of target rate and feature distributions across time (drift), seasonality.
9. **Hypotheses & next steps**: feature ideas, data requests, modeling plan, baseline.

## Statistical care
- Correlation ≠ causation; Simpson's paradox; survivorship and selection bias.
- Look at **segments**: the average hides heterogeneity.
- Beware multiple-comparison "garden of forking paths" — EDA generates hypotheses, a holdout confirms them.
- Do EDA on **training data only** to avoid leakage into feature/model decisions.

## Tooling
`df.info()`, `describe(include="all")`, `ydata-profiling`/`sweetviz` for quick profiles, but write your own checks for domain rules (see script) — and be able to explain every number.

## FAANG Q&A
**Q1. You receive a new dataset; first 30 minutes?** Schema/grain/keys, row counts vs source, missingness, target definition/balance, time range, leakage scan.
**Q2. How do you detect target leakage in EDA?** Single-feature AUC≈1, feature populated only after the event, importance dominated by an ID/timestamp.
**Q3. Two features correlate at 0.95?** Redundant; drop one or combine; matters for linear-model coefficient stability, not for trees' accuracy.
**Q4. How to explore a high-cardinality categorical?** Frequency distribution, target rate with CI/shrinkage, grouping rare levels.
**Q5. EDA found a strong effect — what next?** Check confounding, replicate on holdout/time slice, then experiment if causal.
