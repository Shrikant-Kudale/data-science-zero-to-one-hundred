# Day 11 — Data Cleaning & Missing-Data Imputation

## Missingness mechanisms (Rubin)
- **MCAR:** $P(M\mid X_{obs},X_{mis})=P(M)$. Complete-case analysis is unbiased (just less power).
- **MAR:** $P(M\mid X)=P(M\mid X_{obs})$. Missingness explained by observed variables: valid with model-based/multiple imputation or weighting.
- **MNAR:** depends on the missing value itself (high earners skip income). Not fixable by imputation alone; need a selection model, sensitivity analysis, or domain mechanism.

## Imputation methods
| Method | Preserves mean | Preserves variance/correlation | Notes |
|---|---|---|---|
| Mean/median | (MCAR) | **No** — shrinks variance, attenuates correlation | baseline |
| Regression | yes | variance shrinks (no noise) | stochastic regression adds residual noise |
| KNN | approx | yes (local) | scale features; $O(n^2)$ |
| Iterative (MICE) | yes | yes | chained equations; `IterativeImputer` (+`sample_posterior`) |
| Multiple imputation | yes | yes + proper uncertainty | Rubin's rules: $T=\bar W+(1+\tfrac1m)B$ |
| Missing indicator | — | — | lets tree models learn informative missingness |

Mean imputation variance bias: if fraction $\pi$ missing, imputed variance $\approx(1-\pi)\sigma^2$.

## Good practice
1. Fit imputers **inside the CV pipeline** (fit on train only).
2. Add `missing` indicators when missingness may be informative.
3. Prefer models handling NaN natively (XGBoost, LightGBM, HistGradientBoosting).
4. Clean: dedupe, type-fix, unit checks, impossible values (age<0), consistent categories, parse dates, document every rule.

## FAANG Q&A
**Q1. 30% of a feature is missing. What do you do?** Diagnose mechanism; compare missing-vs-observed target rates; impute + indicator; or drop if uninformative; check downstream CV.
**Q2. Why is mean imputation dangerous?** Distorts distribution/covariances; underestimates SE.
**Q3. How to evaluate imputation?** Mask known values, impute, compare RMSE/distribution (KS), then check downstream-task performance.
**Q4. MAR vs MNAR — can you test?** Not from observed data alone; MAR vs MCAR testable (Little's test, logistic regression on M).
**Q5. Impute before or after train/test split?** After; fit on train only.
