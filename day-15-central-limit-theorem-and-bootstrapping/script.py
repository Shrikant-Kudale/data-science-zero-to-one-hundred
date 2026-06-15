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


from scipy import stats

# ---- 1. CLT on a skewed population (Exponential(1): skew=2) ---------------------------------------------------
ns = [1, 5, 30, 200]; reps = 8000
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
for a, n in zip(ax, ns):
    means = rng.exponential(1.0, size=(reps, n)).mean(1)
    print(f"n={n:4d} mean={means.mean():.3f} SE={means.std():.3f} (theory {1/np.sqrt(n):.3f}) skew={stats.skew(means):.2f} (theory {2/np.sqrt(n):.2f})")
    a.hist(means, bins=50, density=True, alpha=.6); a.set_title(f"n={n}")
    xs = np.linspace(means.min(), means.max(), 200); a.plot(xs, stats.norm.pdf(xs, 1, 1 / np.sqrt(n)), "r")
savefig("clt.png")

# ---- 2. Bootstrap CI for a median --------------------------------------------------------------------------------------
data = rng.lognormal(3, 1, 80); B = 5000
boot = np.median(data[rng.integers(0, len(data), (B, len(data)))], axis=1)
print(f"\nmedian={np.median(data):.2f} bootstrap SE={boot.std():.2f} percentile CI=({np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f})")
try:
    bca = stats.bootstrap((data,), np.median, n_resamples=2000, method="BCa", random_state=1)
    print("scipy BCa CI:", bca.confidence_interval.low.round(2), bca.confidence_interval.high.round(2))
except Exception as e:
    print("BCa unavailable:", e)

# ---- 3. Coverage experiment: t-interval vs percentile bootstrap on lognormal mean ------------------------------
true_mean = np.exp(0.5); trials, n, Bc = 600, 30, 400; cov_t = cov_b = 0; wid_t = wid_b = 0
for _ in range(trials):
    s = rng.lognormal(0, 1, n); m, se = s.mean(), s.std(ddof=1) / np.sqrt(n); tcrit = stats.t.ppf(.975, n - 1)
    lo, hi = m - tcrit * se, m + tcrit * se; cov_t += lo <= true_mean <= hi; wid_t += hi - lo
    bm = s[rng.integers(0, n, (Bc, n))].mean(1); lo, hi = np.percentile(bm, [2.5, 97.5]); cov_b += lo <= true_mean <= hi; wid_b += hi - lo
print(f"\ncoverage of nominal 95% (lognormal, n={n}): t-interval={cov_t/trials:.3f} (width {wid_t/trials:.2f}) | percentile bootstrap={cov_b/trials:.3f} (width {wid_b/trials:.2f})")

# ---- 4. Cluster bootstrap for a ratio metric (CTR) ----------------------------------------------------------------
users = 400; views = rng.poisson(20, users) + 1; clicks = rng.binomial(views, np.clip(rng.beta(2, 18, users), 0, 1))
ctr = clicks.sum() / views.sum(); bs = []
for _ in range(3000):
    i = rng.integers(0, users, users); bs.append(clicks[i].sum() / views[i].sum())
naive_se = np.sqrt(ctr * (1 - ctr) / views.sum())
print(f"CTR={ctr:.4f} | cluster-bootstrap SE={np.std(bs):.5f} vs naive binomial SE={naive_se:.5f} (naive too small when users differ)")
