# Day 16 — Hypothesis Testing: z, t, ANOVA

## Framework
$H_0$ vs $H_1$; test statistic $T$; **p-value** $=P(T\ge t_{obs}\mid H_0)$. Type I error $\alpha=P(\text{reject }H_0\mid H_0)$; Type II $\beta$; power $=1-\beta$. A p-value is *not* $P(H_0\mid\text{data})$ and says nothing about effect size — always report an effect size and CI.

## Tests
- **One-sample z** ($\sigma$ known / large $n$): $z=\frac{\bar x-\mu_0}{\sigma/\sqrt n}$.
- **One-sample t:** $t=\frac{\bar x-\mu_0}{s/\sqrt n}\sim t_{n-1}$.
- **Welch two-sample t:** $t=\frac{\bar x_1-\bar x_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}}$, df by Welch–Satterthwaite $\nu=\frac{(a+b)^2}{a^2/(n_1-1)+b^2/(n_2-1)}$, $a=s_1^2/n_1,b=s_2^2/n_2$. Default to Welch (no equal-variance assumption; costs almost nothing).
- **Paired t:** t-test on differences (removes between-subject variance).
- **Proportions:** $z=\frac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)(1/n_1+1/n_2)}}$.
- Effect size: Cohen's $d=\frac{\bar x_1-\bar x_2}{s_p}$.

## One-way ANOVA
$SS_T=SS_B+SS_W$; $F=\frac{SS_B/(k-1)}{SS_W/(N-k)}\sim F_{k-1,N-k}$. Assumptions: independence, normal residuals, equal variances (Levene; else Welch ANOVA / Kruskal). A significant $F$ says *some* means differ; follow with **post-hoc** (Tukey HSD).

## Multiple comparisons
$m$ independent tests: $P(\ge1\text{ false positive})=1-(1-\alpha)^m$ (20 tests => 64%). FWER control: Bonferroni ($\alpha/m$), Holm (step-down, uniformly better). FDR control: Benjamini–Hochberg: reject $p_{(i)}\le\frac im q$ — preferred when screening many features.

## FAANG Q&A
**Q1. Why t not z?** $s$ is estimated; heavier tails for small $n$. Converges to z as $n\to\infty$.
**Q2. Two-tailed vs one-tailed?** Decide before seeing data; one-tailed doubles power in one direction but cannot detect the opposite.
**Q3. p=0.049 vs 0.051?** Arbitrary threshold; examine effect size/CI/prior plausibility.
**Q4. Why not run many t-tests instead of ANOVA?** Inflated familywise error.
**Q5. Statistical vs practical significance?** With $n=10^7$, a 0.01% lift is "significant" but may be worthless.
