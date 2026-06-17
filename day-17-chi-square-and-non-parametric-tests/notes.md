# Day 17 — Chi-Square & Non-Parametric Tests

## Chi-square
- **Goodness of fit:** $\chi^2=\sum\frac{(O_i-E_i)^2}{E_i}\sim\chi^2_{k-1-m}$ ($m$ estimated params).
- **Independence** in $r\times c$ table: $E_{ij}=\frac{R_iC_j}{N}$, df $=(r-1)(c-1)$. Rule of thumb: all $E_{ij}\ge5$; otherwise **Fisher's exact** (hypergeometric).
- Effect size: Cramér's $V=\sqrt{\chi^2/(N\min(r-1,c-1))}$.
- Sample ratio mismatch (SRM) check in A/B tests is a chi-square GOF on assignment counts: p<0.001 => broken randomization; discard results.
- McNemar for paired binary outcomes.

## Non-parametric tests (rank-based; fewer assumptions)
| Parametric | Non-parametric |
|---|---|
| 2-sample t | Mann–Whitney U / Wilcoxon rank-sum |
| paired t | Wilcoxon signed-rank |
| ANOVA | Kruskal–Wallis |
| Pearson | Spearman / Kendall |
| normality | KS, Shapiro, Anderson-Darling |
Mann–Whitney tests $P(X>Y)\neq\frac12$ (stochastic dominance), not strictly medians. Rank tests lose ~5% efficiency vs t under normality ($3/\pi$ ARE ≈ 0.955) but can be far better under heavy tails.

## Permutation test
Under $H_0$ the labels are exchangeable: shuffle labels, recompute statistic $T^*$, $p=\frac{1+\#\{|T^*|\ge|T_{obs}|\}}{1+B}$. Exact, works for any statistic (difference of medians, AUC, correlation).

## Choosing
Normal-ish & large $n$: parametric. Heavy-tailed/ordinal/small $n$: rank or permutation. Revenue/user metrics: bootstrap or permutation on user level, or log/winsorize.

## FAANG Q&A
**Q1. Chi-square vs Fisher?** Fisher exact for sparse tables (E<5).
**Q2. CTR differs between 3 variants — which test?** Chi-square on 3×2 table, then pairwise with correction.
**Q3. Why Spearman over Pearson?** Monotonic, not just linear; robust to outliers/ordinal.
**Q4. SRM detected, what do you do?** Don't trust results; investigate bucketing, bots, logging, redirects.
**Q5. Do non-parametric tests have no assumptions?** Still need independence and (for location shift interpretation) similar shapes.
