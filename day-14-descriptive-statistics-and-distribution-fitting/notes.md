# Day 14 — Descriptive Statistics & Distribution Fitting

## Summaries
- Location: mean, median, trimmed mean, mode. Spread: variance $s^2=\frac{1}{n-1}\sum(x-\bar x)^2$ (Bessel correction makes $E[s^2]=\sigma^2$), std, IQR, MAD, CV $=s/\bar x$.
- Shape: skewness $\gamma_1=E[(\tfrac{X-\mu}{\sigma})^3]$, excess kurtosis $\gamma_2=E[(\tfrac{X-\mu}{\sigma})^4]-3$.
- Mean $>$ median usually signals right skew. Heavy tails: report quantiles (p50/p90/p99), not just mean.

## Distribution zoo (what generates what)
| Dist | Process | Use |
|---|---|---|
| Normal | sum of many small effects | errors, averages |
| Lognormal | product of many factors | income, latency, sizes |
| Exponential | memoryless waiting time | inter-arrival |
| Gamma | sum of exponentials | wait to $k$th event, claims |
| Weibull | failure time, hazard $\propto t^{k-1}$ | reliability |
| Poisson / NegBin | counts / overdispersed counts | events/day |
| Beta | probability of probability | CTR priors |
| Pareto | power law | wealth, traffic |

## MLE
$\hat\theta=\arg\max\sum\log f(x_i;\theta)$. Lognormal MLE: $\hat\mu=\overline{\log x}$, $\hat\sigma^2=\text{Var}(\log x)$. Exponential: $\hat\lambda=1/\bar x$. Asymptotically efficient and normal: $\sqrt n(\hat\theta-\theta)\to N(0,I(\theta)^{-1})$.

## Goodness of fit
- **AIC** $=2k-2\ell$ (compare non-nested models; lower better); BIC $=k\ln n-2\ell$.
- **KS** statistic $D=\sup_x|F_n(x)-F(x)|$; p-values are invalid when parameters were estimated from the same data (use Lilliefors / parametric bootstrap).
- **QQ plot:** sample vs theoretical quantiles; straight line => good fit; curved ends => tail mismatch. Preferred over p-values for large $n$ (tests reject trivial deviations).

## FAANG Q&A
**Q1. Why divide by $n-1$?** One degree of freedom spent estimating the mean; the $n$ divisor is biased low by factor $(n-1)/n$.
**Q2. Mean vs median for latency SLOs?** Percentiles; latency is heavy-tailed/lognormal.
**Q3. Method of moments vs MLE?** MoM simpler, less efficient; MLE efficient and invariant to reparameterization.
**Q4. Overdispersed counts?** Poisson variance=mean; if variance>mean use NegBin or quasi-Poisson.
**Q5. Fit says "lognormal", Weibull AIC is 2 lower. Decide?** Differences <2 are negligible; prefer the process-justified, simpler model and validate tails.
