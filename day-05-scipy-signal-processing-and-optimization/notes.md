# Day 05 — SciPy Signal Processing & Optimization

## Signals
- **Sampling:** Nyquist: to resolve frequency $f$ need $f_s>2f$; otherwise aliasing.
- **DFT:** $X_k=\sum_{n=0}^{N-1}x_ne^{-2\pi ikn/N}$; FFT does it in $O(N\log N)$. Frequency resolution $\Delta f=f_s/N$. Use `rfft` for real signals; windowing (Hann) reduces spectral leakage.
- **Butterworth low-pass:** $|H(j\omega)|^2=1/(1+(\omega/\omega_c)^{2n})$ — maximally flat passband. `sosfiltfilt` applies forward-backward => zero phase (non-causal: offline only!). Prefer SOS form over (b,a) for numerical stability.
- Peak finding: `find_peaks(height, prominence, distance)`.

## Optimization
Unconstrained smooth: BFGS / L-BFGS-B with analytic gradients (much faster and more reliable than finite differences). Constrained: SLSQP, `trust-constr`; LP: `linprog` (HiGHS).
**Logistic regression MLE:** $\ell(w)=\sum_i[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))]$, gradient $X^\top(\sigma(Xw)-y)$ — convex, so any local min is global. Hessian $X^\top\text{diag}(p(1-p))X$ (Newton/IRLS).
**Nonlinear least squares:** `curve_fit` (Levenberg–Marquardt); standard errors from $\sqrt{\text{diag}(J^\top J)^{-1}\hat\sigma^2}$.

## Trade-offs
Gradient accuracy vs effort; convex vs non-convex (multiple restarts, global methods like `differential_evolution`); scaling features before optimizing (conditioning governs convergence speed).

## FAANG Q&A
**Q1. Why `filtfilt` is unsuitable for real-time?** It uses future samples (non-causal); use `sosfilt` in streaming.
**Q2. Why does the optimizer converge slowly on unscaled features?** Ill-conditioned Hessian => zig-zagging; standardize or use second-order methods.
**Q3. How do you detect seasonality quickly?** Periodogram/FFT peak; confirm with ACF.
**Q4. Explain the convexity of logistic loss.** Hessian is $X^\top DX\succeq0$ since $D\succeq0$.
**Q5. Constrained portfolio problem?** Min $w^\top\Sigma w$ s.t. $\mathbf1^\top w=1$, $w\ge0$ -> SLSQP/QP.
