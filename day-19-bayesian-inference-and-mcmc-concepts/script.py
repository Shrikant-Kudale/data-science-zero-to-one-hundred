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

# ---- 1. Bayesian A/B with Beta-Binomial ------------------------------------------------------------------------------------
nA, kA, nB, kB = 1000, 120, 1000, 145
prior = (1, 1); postA = stats.beta(prior[0] + kA, prior[1] + nA - kA); postB = stats.beta(prior[0] + kB, prior[1] + nB - kB)
sa, sb = postA.rvs(200_000, random_state=1), postB.rvs(200_000, random_state=2)
print(f"posterior mean A={postA.mean():.4f} B={postB.mean():.4f} | P(B>A)={np.mean(sb > sa):.3f} | expected loss choosing B={np.mean(np.maximum(sa - sb, 0)):.5f}")
print("95% credible interval of relative lift:", np.percentile((sb - sa) / sa, [2.5, 97.5]).round(3))
print(f"frequentist z-test p for comparison: {2*stats.norm.sf(abs((kB/nB-kA/nA)/np.sqrt((kA+kB)/(nA+nB)*(1-(kA+kB)/(nA+nB))*2/nA))):.3f}")

# ---- 2. Shrinkage: prior strength --------------------------------------------------------------------------------------------------
for a, b in [(1, 1), (12, 88), (120, 880)]:
    print(f"prior Beta({a},{b}) -> posterior mean for 3/10 successes: {(a+3)/(a+b+10):.3f} (MLE 0.300)")

# ---- 3. Metropolis-Hastings for one chain posterior (no conjugacy used) -------------------------------------------------------------
def log_post(th):
    if not 0 < th < 1: return -np.inf
    return stats.binom.logpmf(kB, nB, th) + stats.beta.logpdf(th, 2, 18)        # informative prior Beta(2,18)
def mh(n_iter, start, step, seed):
    r = np.random.default_rng(seed); th, lp, out, acc = start, log_post(start), [], 0
    for _ in range(n_iter):
        prop = th + r.normal(0, step); lq = log_post(prop)
        if np.log(r.random()) < lq - lp: th, lp, acc = prop, lq, acc + 1
        out.append(th)
    return np.array(out), acc / n_iter
chains = [mh(6000, s, 0.02, i) for i, s in enumerate([0.05, 0.1, 0.3, 0.6])]
draws = np.array([c[0][1000:] for c in chains]); print("acceptance rates:", [round(c[1], 2) for c in chains])
n_, m_ = draws.shape[1], draws.shape[0]; W = draws.var(axis=1, ddof=1).mean(); Bv = n_ * draws.mean(1).var(ddof=1)
rhat = np.sqrt(((n_ - 1) / n_ * W + Bv / n_) / W)
exact = stats.beta(2 + kB, 18 + nB - kB)
print(f"R-hat={rhat:.4f} | MCMC mean={draws.mean():.4f} sd={draws.std():.4f} | analytic mean={exact.mean():.4f} sd={exact.std():.4f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5)); g = np.linspace(.08, .2, 400)
ax[0].plot(g, postA.pdf(g), label="A"); ax[0].plot(g, postB.pdf(g), label="B"); ax[0].legend(); ax[0].set_title("Posteriors")
for c in chains: ax[1].plot(c[0][:1500], lw=.5)
ax[1].set_title("MH traces (4 dispersed starts)")
ax[2].hist(draws.ravel(), bins=60, density=True, alpha=.5); ax[2].plot(g, exact.pdf(g), "r"); ax[2].set_title("MCMC vs analytic")
savefig("bayesian_mcmc.png")
