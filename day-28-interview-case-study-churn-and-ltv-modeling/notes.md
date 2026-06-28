# Day 28 — Case Study: Churn & Customer Lifetime Value

## Prompt
"Subscription company loses 3% of customers monthly. Build a churn model and decide whom to target with a retention offer."

## Structure your answer (use this skeleton in interviews)
1. **Clarify:** definition of churn (no activity 30d? cancellation?), prediction horizon (90d), action (offer/call), cost per action, capacity, success metric (net profit, not AUC).
2. **Data/features:** tenure, usage trend (slope, recency, frequency), support tickets, payment failures, plan, price changes, NPS; all **as of the snapshot date** (no leakage; cohorted snapshots).
3. **Labeling:** snapshot at $t$, label = churn within $(t,t+90]$; time-based train/validation/test split; censoring for recent cohorts. Survival analysis (Cox, discrete-time hazard) handles censoring and gives time-to-churn.
4. **Model:** baseline logistic → gradient boosting; calibrate (probabilities feed expected value).
5. **Evaluation:** PR-AUC, lift/gains by decile, calibration, backtest across time; segment performance.
6. **Decision:** expected value of intervention, not "highest risk". Uplift matters: target customers whose churn is *reduced* by the offer (persuadables), not sure-things/lost causes.
7. **Rollout:** A/B test with holdout; monitor drift; feedback effects.

## LTV math
Monthly retention $r=1-c$, margin $m$ per month, discount $d$ per month: 
$$\text{LTV}=\sum_{t=0}^{\infty}m\left(\frac{r}{1+d}\right)^t=\frac{m}{1-\frac{r}{1+d}}\ \ (\text{start-of-month cash flows}).$$ Without discounting $=m/c$ (expected lifetime $1/c$ months). Converting horizon probability: $c_{month}=1-(1-p_{90})^{1/3}$.
Extensions: BG/NBD + Gamma-Gamma for non-contractual settings; cohort-based empirical LTV; CAC payback; LTV:CAC > 3 target.

## Campaign economics
Profit per targeted customer $=p_{churn}\cdot s\cdot\text{LTV}-\text{cost}$ where $s$ = save probability (uplift). Target those with positive expected profit; with capacity $k$, rank by this value.

## FAANG Q&A
**Q1. AUC 0.85 but campaign loses money. Why?** Targeting sure-churners/low-value customers; uplift ≠ risk; wrong cost assumptions.
**Q2. How do you validate that the offer works?** Randomized holdout, measure incremental retention and profit.
**Q3. Leakage examples in churn?** Cancellation-page visits, "account closed" flags, post-snapshot support tickets.
**Q4. Model decays in 2 months?** Retention actions alter behavior (feedback); drift; retrain cadence.
**Q5. Why calibrate?** Expected value computations need real probabilities.
