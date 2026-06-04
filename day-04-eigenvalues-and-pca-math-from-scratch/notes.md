# Day 04 — Eigenvalues & PCA From Scratch

## Derivation (max-variance view)
Centered data $X\in\mathbb{R}^{n\times d}$, covariance $S=\frac{1}{n-1}X^\top X$. Find unit $w$ maximizing $\text{Var}(Xw)=w^\top Sw$.
Lagrangian: $\mathcal{L}=w^\top Sw-\lambda(w^\top w-1)$, $\nabla_w=0\Rightarrow Sw=\lambda w$. Variance $=\lambda$, so pick the top eigenvector; subsequent components are orthogonal eigenvectors in descending order.

## Min-reconstruction-error view (equivalent)
Projecting onto top-$k$ axes $W_k$: $\min\|X-XW_kW_k^\top\|_F^2$ = (up to a factor) $\sum_{i>k}\lambda_i$. Variance maximized = error minimized, because total variance $\text{tr}(S)=\sum\lambda_i$ is fixed.

## Computation
1. Center (and usually scale) features. 2. `eigh(S)` (symmetric => real, orthogonal) or SVD of $X$: $\lambda_i=\sigma_i^2/(n-1)$. 3. Explained variance ratio $\lambda_i/\sum\lambda$. 4. Scores $T=XW$.
SVD is preferred (no explicit $X^\top X$ => better conditioning). Whitening: $T\Lambda^{-1/2}$ gives identity covariance.

## Gotchas
- Scale matters: PCA on unscaled data is dominated by large-variance units. Standardize (correlation PCA) unless units are comparable.
- Signs of eigenvectors are arbitrary.
- PCA is linear, captures global variance, not class separation (use LDA) and not nonlinear manifolds (kernel PCA, t-SNE/UMAP).
- Choose $k$: cumulative variance (90-95%), scree elbow, Kaiser (>1 on correlation matrix), parallel analysis, CV reconstruction error.

## FAANG Q&A
**Q1. Why are principal components orthogonal?** Eigenvectors of a symmetric matrix with distinct eigenvalues are orthogonal (spectral theorem).
**Q2. $n<d$ (e.g., genomics)?** $S$ has rank $\le n-1$; use the $n\times n$ Gram matrix or SVD; at most $n-1$ non-zero components.
**Q3. Does PCA need Gaussian data?** No for the algebra; it is optimal for second-order structure, and fully sufficient only for Gaussians.
**Q4. Is PCA sensitive to outliers?** Yes (variance is squared) — use robust PCA / winsorize.
**Q5. How would you apply PCA in production?** Persist mean, scale, and $W_k$ from the training set; apply the same transform at inference.
