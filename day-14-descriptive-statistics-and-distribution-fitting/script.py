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


import pandas as pd
from scipy import stats

# ---- 1. Latency-like data: lognormal body + Pareto tail ------------------------------------------------
x = np.r_[rng.lognormal(3.0, 0.5, 4700), (rng.pareto(2.5, 300) + 1) * 60]
s = pd.Series(x)
print(s.describe(percentiles=[.5, .9, .99]).round(2).to_string())
print(f"skew={stats.skew(x):.2f} excess kurtosis={stats.kurtosis(x):.2f} mean/median={x.mean()/np.median(x):.2f} CV={x.std(ddof=1)/x.mean():.2f}")
print(f"bessel check: biased var={x.var():.2f} unbiased={x.var(ddof=1):.2f}")

# ---- 2. Fit candidate distributions via MLE -----------------------------------------------------------------
cands = {"norm": stats.norm, "lognorm": stats.lognorm, "gamma": stats.gamma, "weibull_min": stats.weibull_min, "expon": stats.expon, "pareto-lomax": stats.lomax}
rows, fits = [], {}
for name, dist in cands.items():
    params = dist.fit(x) if name == "norm" else dist.fit(x, floc=0)
    ll = dist.logpdf(x, *params).sum(); k = len(params) - (0 if name == "norm" else 1)
    ks = stats.kstest(x, dist.cdf, args=params).statistic
    rows.append((name, ll, 2 * k - 2 * ll, k * np.log(len(x)) - 2 * ll, ks)); fits[name] = params
res = pd.DataFrame(rows, columns=["dist", "loglik", "AIC", "BIC", "KS_D"]).set_index("dist").sort_values("AIC")
print("\n", res.round(1)); best = res.index[0]; print("best by AIC:", best)

# ---- 3. Closed-form MLE check for lognormal -------------------------------------------------------------------------
print(f"lognormal closed form mu={np.log(x).mean():.4f} sigma={np.log(x).std():.4f} | scipy sigma={fits['lognorm'][0]:.4f} scale->mu={np.log(fits['lognorm'][2]):.4f}")

# ---- 4. Parametric bootstrap for KS p-value (params estimated from data) ----------------------------------------
d_obs = stats.kstest(x, "lognorm", args=fits["lognorm"]).statistic; sig, _, scale = fits["lognorm"]; cnt = 0
for _ in range(100):
    sim = stats.lognorm.rvs(sig, 0, scale, size=len(x), random_state=rng); p = stats.lognorm.fit(sim, floc=0)
    cnt += stats.kstest(sim, "lognorm", args=p).statistic >= d_obs
print(f"KS D={d_obs:.4f}, parametric-bootstrap p ~ {(cnt+1)/101:.2f} (tail mismatch => reject)")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8)); grid = np.linspace(0.5, np.percentile(x, 99), 400)
ax[0].hist(x, bins=100, range=(0, np.percentile(x, 99)), density=True, alpha=.4)
for nm in ["lognorm", "gamma", "weibull_min"]: ax[0].plot(grid, cands[nm].pdf(grid, *fits[nm]), label=nm)
ax[0].legend(); ax[0].set_title("Fitted densities")
for a, nm in zip(ax[1:], ["lognorm", "gamma"]):
    stats.probplot(x, sparams=fits[nm], dist=cands[nm], plot=a); a.set_title(f"QQ vs {nm}")
savefig("distribution_fits.png")
