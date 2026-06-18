# Day 18 — A/B Testing: Design, Sample Size & Power

## Design checklist
1. Hypothesis + **one primary metric** (+ guardrails: latency, unsubscribes, revenue). 2. Randomization unit (user > session > pageview; must match analysis unit). 3. MDE (minimum detectable effect) from business value. 4. Sample size & duration (whole weeks for weekday effects; novelty/primacy effects). 5. Pre-register stopping rule. 6. SRM check. 7. Analyze, report effect + CI. 8. Ship / iterate.

## Sample size (two-sided, equal allocation)
Proportions: $n_{\text{per arm}}=\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,[p_1(1-p_1)+p_2(1-p_2)]}{(p_2-p_1)^2}$.
Means: $n=\frac{2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2}{\delta^2}$.
Rule of thumb (α=.05, power=.8): $n\approx16\sigma^2/\delta^2$ per arm. Halving the MDE quadruples $n$.
Power $=\Phi\!\left(\frac{|\delta|}{\text{SE}}-z_{1-\alpha/2}\right)$.

## Peeking / optional stopping
Checking a fixed-horizon test repeatedly and stopping at the first $p<.05$ inflates Type I error (5 looks ≈ 14%, 10 looks ≈ 19%, continuous → 100%). Remedies: fixed horizon, group-sequential boundaries (O'Brien–Fleming, Pocock), alpha-spending, always-valid p-values (mSPRT), Bayesian with proper decision rules.

## Variance reduction: CUPED
$Y^{*}=Y-\theta(X-\bar X)$, $\theta=\frac{\text{Cov}(Y,X)}{\text{Var}(X)}$ with pre-experiment covariate $X$. Variance shrinks by factor $(1-\rho^2)$ => ρ=0.7 gives ~51% fewer users needed. Unbiased because $X$ is pre-treatment. Also: stratification, regression adjustment, trimming outliers.

## Other pitfalls
Network effects/interference (use cluster or switchback designs), multiple variants/metrics (correct), Simpson's paradox with unbalanced allocation changes mid-test, novelty effects, ratio metrics (delta method), winner's curse (true lift typically smaller than observed).

## FAANG Q&A
**Q1. Baseline CR 10%, want +1pp (relative 10%) lift; sample size?** ≈14.7k per arm (compute below).
**Q2. Test ran 3 days and p=.03. Ship?** No — pre-set duration, weekday coverage, peeking risk.
**Q3. Metric is flat but p is large; conclude no effect?** Absence of evidence: check CI vs MDE; power.
**Q4. Marketplace test with spillover?** Cluster/geo randomization or switchback; interference violates SUTVA.
**Q5. How to reduce required sample size?** CUPED, better metric (less noisy), larger MDE, one-sided only if justified, trigger analysis (only exposed users).
