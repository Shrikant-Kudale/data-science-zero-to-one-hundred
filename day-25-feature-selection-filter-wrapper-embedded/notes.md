# Day 25 — Feature Selection

## Why
Reduce overfitting and variance ($p\gg n$), cost, latency, and improve interpretability. Not always accuracy gains for regularized/tree models.

## Three families
**Filter** (model-free, fast, univariate): variance threshold; correlation / ANOVA $F$; chi-square; **mutual information** $I(X;Y)=\sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$ (captures non-linear); mRMR (max relevance, min redundancy). Misses interactions/XOR effects.
**Wrapper** (search with the model): forward/backward selection, **RFE** (recursive feature elimination using coefficients/importances), genetic algorithms. Expensive ($O(p)$–$O(2^p)$ fits), high overfitting risk to the CV score.
**Embedded** (selection during training): **L1/Lasso** $\min\|y-Xw\|^2+\lambda\|w\|_1$ (sparsity via kink at 0; elastic net handles correlated groups), tree importances, SelectFromModel, stability selection (frequency of selection under subsampling).

## Importance caveats
- Impurity (MDI) importance is biased toward high-cardinality/continuous features.
- **Permutation importance** (drop in score when a feature is shuffled; compute on held-out data) is model-agnostic but dilutes among correlated features.
- SHAP: consistent additive attributions; global importance = mean $|\phi|$.
- Correlated features share importance; Lasso arbitrarily picks one.

## The selection-bias trap
Selecting features using **all** data (labels included) and *then* cross-validating gives optimistically biased scores — even on pure noise. Selection must be inside each CV fold (`Pipeline`), and hyperparameter search/selection counts as part of the model (nested CV).

## FAANG Q&A
**Q1. Lasso vs Ridge?** L1: sparse, selection; L2: shrinkage, keeps correlated groups. Elastic net blends.
**Q2. Why can MI beat correlation?** Detects non-monotonic dependence ($y=x^2$ has zero Pearson).
**Q3. Two perfectly correlated features, trees' importance?** Split between them; permutation importance halves each.
**Q4. 10k features, 500 samples?** Univariate filter within CV → L1/elastic net or tree-based with stability selection; strong regularization; nested CV.
**Q5. How do you validate the selected set?** Held-out set untouched by selection; stability across bootstrap resamples.
