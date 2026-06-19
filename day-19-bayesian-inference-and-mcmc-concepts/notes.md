# Day 19 — Bayesian Inference & MCMC

## Bayes' rule
$p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)}\propto\text{likelihood}\times\text{prior}$. The evidence $p(D)=\int p(D\mid\theta)p(\theta)d\theta$ is usually intractable => conjugacy, MCMC, or variational inference.

## Conjugate example: Beta–Binomial
Prior $\theta\sim\text{Beta}(\alpha,\beta)$; data $k$ successes in $n$ trials ⇒ posterior $\text{Beta}(\alpha+k,\beta+n-k)$. Posterior mean $=\frac{\alpha+k}{\alpha+\beta+n}$ = weighted average of prior mean and MLE $k/n$ with prior weight $\alpha+\beta$ ("pseudo-counts"). Other pairs: Gamma–Poisson, Normal–Normal, Dirichlet–Multinomial.

## Bayesian A/B
$P(B>A\mid D)=\iint\mathbb 1[\theta_B>\theta_A]\,p\,d\theta$ (Monte-Carlo), **expected loss** $E[\max(\theta_A-\theta_B,0)]$ for decision-making, credible interval (HDI) — directly interpretable: "95% probability θ in this interval".
Frequentist vs Bayesian: p-values answer $P(D\mid H_0)$; posteriors answer $P(\theta\mid D)$. Priors encode prior knowledge (and regularize, equal to ridge/lasso for Gaussian/Laplace priors). Bayesian "peeking" is valid for posterior statements, but frequentist error rates of a stopping rule still matter.

## MCMC: Metropolis–Hastings
Goal: sample from $\pi(\theta)\propto p(D\mid\theta)p(\theta)$ without the normalizer.
1. Propose $\theta'\sim q(\cdot\mid\theta)$. 2. Accept with $a=\min\!\left(1,\frac{\pi(\theta')q(\theta\mid\theta')}{\pi(\theta)q(\theta'\mid\theta)}\right)$. Detailed balance ⇒ $\pi$ is stationary.
Gibbs: sample each conditional. HMC/NUTS: use gradients $\nabla\log\pi$ (Stan, PyMC) — scales to high dimensions.
Diagnostics: trace plots, **$\hat R<1.01$** (between-/within-chain variance), effective sample size, divergences, acceptance rate (~23% optimal for random walk in high-d, ~44% in 1-d). Burn-in/warm-up discard.

## FAANG Q&A
**Q1. Choose prior for CTR?** Beta fitted to historical CTRs (empirical Bayes): shrinks small-sample estimates.
**Q2. Ratings 5.0 with 3 reviews vs 4.7 with 3000?** Posterior mean/lower bound — Bayesian shrinkage.
**Q3. Why MCMC?** Posterior normalizer intractable in non-conjugate/high-dim models.
**Q4. MAP vs full posterior?** MAP = mode (= regularized MLE); posterior carries uncertainty.
**Q5. Prior sensitivity?** Large $n$ washes the prior; test with alternative priors.
