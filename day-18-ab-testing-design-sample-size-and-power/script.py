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

def n_prop(p1, p2, alpha=.05, power=.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2))
def power_prop(p1, p2, n, alpha=.05):
    se = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n); return stats.norm.cdf(abs(p2 - p1) / se - stats.norm.ppf(1 - alpha / 2))

# ---- 1. Sample size & power ------------------------------------------------------------------------------------------------
n = n_prop(.10, .11); print(f"n per arm for 10% -> 11%: {n:,} | power at that n: {power_prop(.10, .11, n):.3f}")
for mde in (.02, .01, .005): print(f"  MDE {mde:.3f} abs -> n/arm = {n_prop(.10, .10 + mde):,}")
try:
    from statsmodels.stats.power import NormalIndPower; from statsmodels.stats.proportion import proportion_effectsize
    print("statsmodels check:", int(np.ceil(NormalIndPower().solve_power(proportion_effectsize(.11, .10), power=.8, alpha=.05))))
except ImportError: pass

# ---- 2. Monte-Carlo power verification ------------------------------------------------------------------------------------------
sims, hits = 3000, 0
for _ in range(sims):
    a, b = rng.binomial(n, .10), rng.binomial(n, .11); pp = (a + b) / (2 * n)
    z = (b / n - a / n) / np.sqrt(2 * pp * (1 - pp) / n); hits += abs(z) > 1.96
print(f"simulated power={hits/sims:.3f} (theory .80)")

# ---- 3. Peeking inflation: A/A tests ----------------------------------------------------------------------------------------------
sims, looks, step, p0 = 4000, 10, 500, .10
ca = rng.binomial(step, p0, (sims, looks)).cumsum(1); cb = rng.binomial(step, p0, (sims, looks)).cumsum(1)
nn = step * np.arange(1, looks + 1); pp = (ca + cb) / (2 * nn); z = (cb - ca) / nn / np.sqrt(2 * pp * (1 - pp) / nn)
sig = np.abs(z) > 1.96
fixed = sig[:, -1].mean(); peek = sig.any(1).mean()
print(f"A/A false positive: fixed horizon={fixed:.3f} | stop at first significant of {looks} looks={peek:.3f}")
bonf = (np.abs(z) > stats.norm.isf(.05 / looks / 2)).any(1).mean(); print(f"Bonferroni-corrected looks={bonf:.3f} (conservative)")

# ---- 4. CUPED -------------------------------------------------------------------------------------------------------------------------------
N = 20000; pre = rng.gamma(3, 10, N); y0 = 0.8 * pre + rng.normal(0, 10, N); treat = rng.random(N) < .5; y = y0 + 1.0 * treat
theta = np.cov(y, pre)[0, 1] / pre.var(ddof=1); ycv = y - theta * (pre - pre.mean())
def diff_se(v): d = v[treat].mean() - v[~treat].mean(); se = np.sqrt(v[treat].var(ddof=1) / treat.sum() + v[~treat].var(ddof=1) / (~treat).sum()); return d, se
for nm, v in [("raw", y), ("CUPED", ycv)]:
    d, se = diff_se(v); print(f"{nm:6s} lift={d:.3f} SE={se:.3f} 95% CI=({d-1.96*se:.2f}, {d+1.96*se:.2f})")
rho = np.corrcoef(y, pre)[0, 1]; print(f"rho={rho:.2f} -> theoretical var reduction {rho**2:.1%}, empirical {1 - ycv.var()/y.var():.1%}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); ns = np.arange(2000, 30000, 500)
ax[0].plot(ns, [power_prop(.10, .11, k) for k in ns]); ax[0].axhline(.8, ls="--", c="gray"); ax[0].set_title("Power vs n/arm (10%->11%)")
ax[1].bar(["fixed horizon", "peek x10", "Bonferroni"], [fixed, peek, bonf], color=["#27ae60", "#c0392b", "#2980b9"]); ax[1].axhline(.05, ls="--", c="k"); ax[1].set_title("A/A false-positive rate")
savefig("ab_power_peeking.png")
