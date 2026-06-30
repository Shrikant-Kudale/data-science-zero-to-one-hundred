# Day 30 — FAANG Data Science Interview Mock & Cheatsheet

## Loop structure (typical)
1. **Coding/SQL** (45 min): window functions, joins, pandas, algorithmic basics.
2. **Statistics & probability** (45 min): hypothesis tests, Bayes, distributions, estimators.
3. **Experimentation / product analytics** (45–60 min): metric design, A/B design, interpreting results, trade-offs.
4. **ML / modeling** (45 min): problem framing, features, evaluation, deployment, monitoring.
5. **Behavioral / cross-functional** (45 min): impact stories (STAR), disagreement, ambiguity.

## Mock questions with model answers
**1. Probability — two children, at least one is a boy. P(both boys)?** 1/3 (sample space BB, BG, GB). If "the older is a boy": 1/2.
**2. Bayes — disease prevalence 1%, test sens 99%, spec 95%. P(disease | +)?** $\frac{.99\cdot.01}{.99\cdot.01+.05\cdot.99}=16.7\%$.
**3. Why does Facebook-style DAU drop 2% on Monday after a launch?** Check logging/instrumentation, SRM, segment (platform/geo/new vs old), seasonality (weekday), release timeline, then causal analysis via holdout.
**4. Design a metric for "Stories" engagement.** North-star (e.g., daily story viewers per DAU), supporting (completion rate), guardrails (feed time, creator posts, latency); beware gaming/cannibalization.
**5. Explain p-value to a PM.** "If there were truly no effect, we'd see a difference this large or larger about p% of the time." Pair with effect size and CI.
**6. A/B test: +1% CTR (p=.04) but −0.5% revenue. Decision?** Check guardrail significance/CI, segment heterogeneity, long-term effects; decide by overall objective hierarchy, maybe iterate.
**7. Bias–variance.** $E[(y-\hat f)^2]=\text{Bias}^2+\text{Var}+\sigma^2$; regularization trades bias for variance.
**8. How do you handle a feature with 50% missing?** (Day 11) mechanism, indicator, model-native NaN, validate.
**9. Model has 95% accuracy on 5% prevalence data.** Meaningless alone; PR-AUC, recall/precision, cost.
**10. Explain regularization geometrically.** L1 diamond hits axes (sparse); L2 ball shrinks; Bayesian priors Laplace/Gaussian.
**11. SQL: second-highest salary per department.** `DENSE_RANK() OVER (PARTITION BY dept ORDER BY salary DESC) = 2`.
**12. How to detect fake reviews?** Graph/behavior features, text models, temporal bursts; evaluate with precision@k on audits; adversarial drift.

## Cheatsheet (verified numerically in `script.py`)
| Topic | Formula |
|---|---|
| SE of mean | $\sigma/\sqrt n$ |
| CI (mean) | $\bar x\pm t_{n-1,1-\alpha/2}\,s/\sqrt n$ |
| Sample size/arm (means) | $2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2/\delta^2$ |
| Sample size/arm (props) | $(z_{1-\alpha/2}+z_{1-\beta})^2[p_1q_1+p_2q_2]/\delta^2$ |
| Bayes | $P(A|B)=P(B|A)P(A)/P(B)$ |
| PCA variance | $\lambda_i=\sigma_i^2/(n-1)$ |
| OLS | $\hat w=(X^\top X)^{-1}X^\top y$, $\text{Var}=\sigma^2(X^\top X)^{-1}$ |
| Ridge | $\hat w=(X^\top X+\lambda I)^{-1}X^\top y$ |
| Logistic gradient | $X^\top(\sigma(Xw)-y)$ |
| F1 | $2PR/(P+R)$ |
| NDCG | $\text{DCG}/\text{IDCG}$ |
| BH-FDR | reject $p_{(i)}\le iq/m$ |
| PSI | $\sum(a-e)\ln(a/e)$ |
| LTV | $m/(1-r/(1+d))$ |

## Final prep strategy
Rehearse the 30 days' Q&A aloud; write SQL and pandas by hand; practice 3 end-to-end case studies; always state assumptions, define metrics first, discuss trade-offs and failure modes, and finish with "how I'd validate & monitor".
