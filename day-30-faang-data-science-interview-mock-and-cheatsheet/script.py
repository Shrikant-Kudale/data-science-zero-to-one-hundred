import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

def savefig(name):
    path = os.path.join(OUT, name)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print(f"[saved] {path}")


import math
from scipy import stats

checks = []
def check(name, got, expected, tol=1e-6):
    ok = abs(got - expected) <= tol * max(1, abs(expected)); checks.append(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got:.6g} expected {expected:.6g}")

# ---- 1. Probability --------------------------------------------------------------------------------------------------------------------------------
sims = 400_000; kids = rng.integers(0, 2, (sims, 2)); atleast = kids.sum(1) >= 1; check("P(BB | at least one boy) ~ 1/3", ((kids.sum(1) == 2) & atleast).sum() / atleast.sum(), 1 / 3, 1e-2)
sens, spec, prev = .99, .95, .01; check("Bayes P(disease|+)", sens * prev / (sens * prev + (1 - spec) * (1 - prev)), 0.1666667, 1e-5)

# ---- 2. Estimation / CI coverage -------------------------------------------------------------------------------------------------------------------
n, mu, sd, cover = 25, 5.0, 2.0, 0
for _ in range(20000):
    x = rng.normal(mu, sd, n); h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / math.sqrt(n); cover += abs(x.mean() - mu) <= h
check("95% t-interval coverage", cover / 20000, .95, 5e-3)
x = rng.normal(0, 3, (20000, 5)); check("E[var] with ddof=1 is unbiased (sigma^2=9)", x.var(axis=1, ddof=1).mean(), 9.0, 2e-2)

# ---- 3. Sample size & power -------------------------------------------------------------------------------------------------------------------------
za, zb = stats.norm.ppf(.975), stats.norm.ppf(.8); sig, delta = 10.0, 2.0; n_arm = 2 * sig ** 2 * (za + zb) ** 2 / delta ** 2
a, b = rng.normal(0, sig, (6000, int(math.ceil(n_arm)))), rng.normal(delta, sig, (6000, int(math.ceil(n_arm))))
power = np.mean(stats.ttest_ind(a, b, axis=1).pvalue < .05); check(f"power at n={math.ceil(n_arm)}/arm ~ .80", power, .80, 2e-2)

# ---- 4. Linear algebra / ML --------------------------------------------------------------------------------------------------------------------------
X = rng.normal(size=(500, 4)); w = np.array([1, -2, 0.5, 3.]); yv = X @ w + rng.normal(0, 1, 500)
w_ols = np.linalg.solve(X.T @ X, X.T @ yv); check("OLS vs lstsq", np.linalg.norm(w_ols - np.linalg.lstsq(X, yv, rcond=None)[0]), 0.0, 1e-8)
lam = 5.0; w_r = np.linalg.solve(X.T @ X + lam * np.eye(4), X.T @ yv); U, s, Vt = np.linalg.svd(X, full_matrices=False); w_svd = Vt.T @ ((s / (s ** 2 + lam)) * (U.T @ yv))
check("ridge closed form == SVD filter factors", np.linalg.norm(w_r - w_svd), 0.0, 1e-8)
Xc = X - X.mean(0); check("PCA eigenvalues == s^2/(n-1)", np.sort(np.linalg.eigvalsh(np.cov(Xc.T)))[-1], (np.linalg.svd(Xc, compute_uv=False)[0] ** 2) / 499, 1e-8)

# bias-variance decomposition by simulation (polynomial degree vs noise)
def bv(deg, trials=400, n=30, sigma=.3, x0=.5):
    f = lambda t: np.sin(2 * np.pi * t); preds = []
    for _ in range(trials):
        xs = rng.random(n); ys = f(xs) + rng.normal(0, sigma, n); preds.append(np.polyval(np.polyfit(xs, ys, deg), x0))
    preds = np.array(preds); return (preds.mean() - f(x0)) ** 2, preds.var()
print("\nBias^2 / Variance by polynomial degree at x=0.5:")
for deg in (1, 3, 9): b2, v = bv(deg); print(f"  degree {deg}: bias^2={b2:.4f} variance={v:.4f}")

# ---- 5. Multiple testing / metrics ----------------------------------------------------------------------------------------------------------------------
check("P(>=1 false positive in 20 tests @ .05)", 1 - .95 ** 20, 0.641514, 1e-5)
P_, R_ = .6, .3; check("F1 harmonic mean", 2 * P_ * R_ / (P_ + R_), 0.4, 1e-9)
r, d = .97, .01; check("LTV closed form vs 3000-term series", 10 / (1 - r / (1 + d)), sum(10 * (r / (1 + d)) ** t for t in range(3000)), 1e-9)
print(f"\n{sum(checks)}/{len(checks)} checks passed")

# ---- 6. Mock-interview runner: print a rotating drill sheet ----------------------------------------------------------------------------------------
drills = ["Explain the CLT and when it fails.", "Design an A/B test for a new checkout button; give sample size.", "Write SQL: 7-day rolling retention.", "Why does SMOTE leak if applied before CV split?",
          "Interpret NDCG@10 dropping online while offline improves.", "Walk through churn model -> retention campaign decision."]
sheet = "\n".join(f"{i+1}. {q}" for i, q in enumerate(drills)); print("\nDRILL SHEET (answer aloud in 3 minutes each):\n" + sheet)
open(os.path.join(OUT, "drill_sheet.txt"), "w").write(sheet + "\n")
plt.figure(figsize=(5, 3.2)); degs = [1, 3, 5, 9]; B_, V_ = zip(*[bv(dg, 200) for dg in degs]); plt.plot(degs, B_, "o-", label="bias^2"); plt.plot(degs, V_, "s-", label="variance"); plt.legend(); plt.xlabel("polynomial degree"); plt.title("Bias-variance trade-off"); savefig("bias_variance.png")
assert all(checks), "cheatsheet self-check failed"
