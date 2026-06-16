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

# ---- 1. One-sample z and t ------------------------------------------------------------------------------------------
x = rng.normal(102, 15, 50)
z = (x.mean() - 100) / (15 / np.sqrt(len(x))); print(f"z-test: z={z:.3f} p={2*stats.norm.sf(abs(z)):.4f}")
print("t-test:", stats.ttest_1samp(x, 100))

# ---- 2. Welch vs Student, unequal variance & sizes ---------------------------------------------------------------------
a = rng.normal(10, 1, 20); b = rng.normal(10, 4, 200)                        # null true, variances differ
fp = {"student": 0, "welch": 0}
for _ in range(2000):
    a_ = rng.normal(10, 1, 20); b_ = rng.normal(10, 4, 200)
    fp["student"] += stats.ttest_ind(a_, b_, equal_var=True).pvalue < .05; fp["welch"] += stats.ttest_ind(a_, b_, equal_var=False).pvalue < .05
print("false-positive rate under H0 with unequal var/size -> student:", fp["student"] / 2000, "welch:", fp["welch"] / 2000, "(nominal .05)")

# ---- 3. Paired vs unpaired ----------------------------------------------------------------------------------------------------
subj = rng.normal(0, 10, 30); before = 50 + subj + rng.normal(0, 2, 30); after = before + 1.5 + rng.normal(0, 2, 30)
print(f"paired p={stats.ttest_rel(before, after).pvalue:.5f} vs wrongly unpaired p={stats.ttest_ind(before, after).pvalue:.3f}")
sp = np.sqrt((a.var(ddof=1) + x[:20].var(ddof=1)) / 2); print("Cohen's d (a vs x[:20]):", round((a.mean() - x[:20].mean()) / sp, 2))

# ---- 4. ANOVA from scratch -----------------------------------------------------------------------------------------------------------
g = [rng.normal(m, 5, 40) for m in (50, 52, 56)]; allv = np.concatenate(g); N, k = len(allv), len(g)
ssb = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in g); ssw = sum(((v - v.mean()) ** 2).sum() for v in g)
F = (ssb / (k - 1)) / (ssw / (N - k)); p = stats.f.sf(F, k - 1, N - k)
print(f"ANOVA scratch F={F:.3f} p={p:.2e} | scipy: {stats.f_oneway(*g)} | eta^2={ssb/(ssb+ssw):.3f}")
print("Levene (equal var):", round(stats.levene(*g).pvalue, 3), "| Shapiro on residuals:", round(stats.shapiro(np.concatenate([v - v.mean() for v in g])).pvalue, 3))
try: print(stats.tukey_hsd(*g))
except Exception as e: print("tukey_hsd unavailable:", e)

# ---- 5. Multiple testing: 1000 tests, 100 true effects -----------------------------------------------------------------------
m = 1000; true = np.r_[np.ones(100, bool), np.zeros(900, bool)]
pv = np.array([stats.ttest_ind(rng.normal(0.8 * t, 1, 25), rng.normal(0, 1, 25)).pvalue for t in true])
def bh(p, q=.05):
    o = np.argsort(p); thr = q * np.arange(1, len(p) + 1) / len(p); ok = np.where(p[o] <= thr)[0]
    rej = np.zeros(len(p), bool)
    if len(ok): rej[o[:ok.max() + 1]] = True
    return rej
def holm(p, a=.05):
    o = np.argsort(p); rej = np.zeros(len(p), bool)
    for i, idx in enumerate(o):
        if p[idx] <= a / (len(p) - i): rej[idx] = True
        else: break
    return rej
for name, rej in [("uncorrected", pv < .05), ("Bonferroni", pv < .05 / m), ("Holm", holm(pv)), ("BH (FDR)", bh(pv))]:
    print(f"{name:12s} rejections={rej.sum():4d} false pos={(rej & ~true).sum():3d} power={(rej & true).sum()/true.sum():.2f}")

pd.DataFrame({f"g{i}": v for i, v in enumerate(g)}).boxplot(figsize=(5, 3.5)); plt.title("ANOVA groups"); savefig("anova_groups.png")
