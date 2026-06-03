# Day 03 — Matrix Decompositions & SVD

## Decompositions cheat table
| Decomp | Form | Needs | Use |
|---|---|---|---|
| LU | $A=PLU$ | square | solving $Ax=b$ |
| Cholesky | $A=LL^\top$ | SPD | Gaussian sampling, fast solves, GP |
| QR | $A=QR$ | any | stable least squares |
| Eigen | $A=Q\Lambda Q^\top$ | symmetric | PCA, spectral clustering |
| **SVD** | $A=U\Sigma V^\top$ | **any** | PCA, LSA, pseudo-inverse, low-rank |

## SVD
For $A\in\mathbb{R}^{m\times n}$: $U\in\mathbb{R}^{m\times r}$ orthonormal columns, $\Sigma=\text{diag}(\sigma_1\ge\dots\ge\sigma_r>0)$, $V\in\mathbb{R}^{n\times r}$.
Link to eigen: $A^\top A=V\Sigma^2V^\top$, $AA^\top=U\Sigma^2U^\top$.

**Eckart–Young–Mirsky:** the best rank-$k$ approximation in Frobenius *and* spectral norm is $A_k=\sum_{i\le k}\sigma_iu_iv_i^\top$ with
$$\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2,\qquad \|A-A_k\|_2=\sigma_{k+1}.$$

## Least squares & conditioning
Minimize $\|Xw-y\|^2$. Normal equations: $w=(X^\top X)^{-1}X^\top y$. Condition number: $\kappa(X^\top X)=\kappa(X)^2$ — squaring loses digits. QR: $Rw=Q^\top y$; SVD: $w=V\Sigma^{-1}U^\top y=X^+y$ (and truncating tiny $\sigma$ gives ridge-like regularization: ridge multiplies components by $\sigma_i/(\sigma_i^2+\lambda)$).

## Cost
Full SVD $O(\min(mn^2,m^2n))$; randomized/truncated SVD $O(mnk)$ for top-$k$ (used by sklearn `TruncatedSVD`, `svds`).

## FAANG Q&A
**Q1. Relationship between SVD and PCA?** Center $X$; right singular vectors are principal axes; variances $=\sigma_i^2/(n-1)$.
**Q2. Why prefer QR/SVD over the normal equations?** $\kappa$ squares; near-collinear features give garbage coefficients.
**Q3. How do you compress an image/embedding matrix?** Truncated SVD; choose $k$ via energy $\sum_{i\le k}\sigma_i^2/\sum\sigma_i^2\ge 0.9$–$0.99$; storage $k(m+n+1)$ vs $mn$.
**Q4. What is the rank of $A$ numerically?** Number of $\sigma_i>\epsilon\,\sigma_1\max(m,n)$.
**Q5. Why does multicollinearity inflate variance?** $\text{Var}(\hat w)=\sigma^2 V\Sigma^{-2}V^\top$; small $\sigma_i$ => huge variance along $v_i$.
