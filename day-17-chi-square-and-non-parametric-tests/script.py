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

# ---- 1. Chi-square independence: device vs conversion -----------------------------------------------------------------
tbl = np.array([[320, 1680], [410, 1590], [150, 1850]])              # rows: mobile, desktop, tablet ; cols: convert, no
chi2, p, dof, exp = stats.chi2_contingency(tbl)
manual = ((tbl - exp) ** 2 / exp).sum(); V = np.sqrt(chi2 / (tbl.sum() * (min(tbl.shape) - 1)))
print(f"chi2={chi2:.2f} dof={dof} p={p:.2e} manual={manual:.2f} Cramer's V={V:.3f} | min expected={exp.min():.0f}")
print("standardized residuals:\n", ((tbl - exp) / np.sqrt(exp)).round(2))

# ---- 2. Fisher exact for sparse 2x2 and SRM check -------------------------------------------------------------------------
print("Fisher exact [[3,1],[1,3]]:", stats.fisher_exact([[3, 1], [1, 3]]))
obs = np.array([50_400, 49_100]); print("SRM check 50/50 split:", stats.chisquare(obs, f_exp=[obs.sum() / 2] * 2), "(tiny p => SRM!)")

# ---- 3. Mann-Whitney vs t-test on heavy tails -----------------------------------------------------------------------------------
rej_t = rej_u = 0
for _ in range(1000):
    a = rng.standard_t(2, 40); b = rng.standard_t(2, 40) + 1.0
    rej_t += stats.ttest_ind(a, b, equal_var=False).pvalue < .05; rej_u += stats.mannwhitneyu(a, b).pvalue < .05
print(f"power with t(2) shift=1: Welch t={rej_t/1000:.2f}  Mann-Whitney={rej_u/1000:.2f}")

# ---- 4. Kruskal, Wilcoxon, Spearman --------------------------------------------------------------------------------------------------
g1, g2, g3 = rng.exponential(1, 50), rng.exponential(1.3, 50), rng.exponential(2.0, 50)
print("Kruskal:", stats.kruskal(g1, g2, g3)); bef = rng.exponential(5, 40); print("Wilcoxon signed-rank:", stats.wilcoxon(bef, bef * 0.8 + rng.normal(0, .3, 40)))
xx = rng.uniform(0, 5, 300); yy = np.exp(xx) + rng.normal(0, 3, 300)
print(f"monotone nonlinear: Pearson={stats.pearsonr(xx, yy)[0]:.2f} Spearman={stats.spearmanr(xx, yy)[0]:.2f} Kendall={stats.kendalltau(xx, yy)[0]:.2f}")

# ---- 5. Permutation test for difference in medians --------------------------------------------------------------------------------------
A, Bv = rng.lognormal(1.0, 0.8, 60), rng.lognormal(1.3, 0.8, 70); obs_d = np.median(Bv) - np.median(A); pool = np.r_[A, Bv]; perm = []
for _ in range(10000):
    rng.shuffle(pool); perm.append(np.median(pool[60:]) - np.median(pool[:60]))
perm = np.array(perm); pval = (1 + (np.abs(perm) >= abs(obs_d)).sum()) / (1 + len(perm))
print(f"median diff={obs_d:.3f} permutation p={pval:.4f}")
plt.figure(figsize=(6, 3.5)); plt.hist(perm, bins=60, alpha=.7); plt.axvline(obs_d, c="r"); plt.axvline(-obs_d, c="r", ls="--"); plt.title("Permutation null: diff of medians"); savefig("permutation_null.png")
