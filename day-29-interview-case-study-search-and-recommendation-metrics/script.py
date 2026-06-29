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

# ---- 1. Metric implementations ---------------------------------------------------------------------------------------------------------------------------
def precision_at_k(rel, k): return np.mean(np.asarray(rel[:k]) > 0)
def recall_at_k(rel, k, total=None): total = total if total is not None else (np.asarray(rel) > 0).sum(); return (np.asarray(rel[:k]) > 0).sum() / max(total, 1)
def mrr(rel): hit = np.where(np.asarray(rel) > 0)[0]; return 1 / (hit[0] + 1) if len(hit) else 0.0
def ap(rel):
    r = np.asarray(rel) > 0; return 0.0 if r.sum() == 0 else np.mean([r[:i + 1].mean() for i in np.where(r)[0]])
def dcg(rel, k): rel = np.asarray(rel[:k], float); return ((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2))).sum()
def ndcg(rel, k, all_rel=None): ideal = dcg(sorted(all_rel if all_rel is not None else rel, reverse=True), k); return dcg(rel, k) / ideal if ideal > 0 else 0.0

# sanity checks against hand-computed values
print("hand example rel=[1,0,1,0,0]: P@3=%.3f R@3=%.3f MRR=%.3f AP=%.3f NDCG@5=%.3f" % (precision_at_k([1, 0, 1, 0, 0], 3), recall_at_k([1, 0, 1, 0, 0], 3), mrr([1, 0, 1, 0, 0]), ap([1, 0, 1, 0, 0]), ndcg([1, 0, 1, 0, 0], 5)))

# ---- 2. Simulate search: 300 queries x 30 candidates, graded relevance 0-3 ------------------------------------------------------------------------------
Q, D = 300, 30; relev = rng.choice([0, 1, 2, 3], (Q, D), p=[.6, .2, .13, .07])
def evaluate(noise, k=10):
    out = []
    for q in range(Q):
        s = relev[q] + rng.normal(0, noise, D); order = np.argsort(-s); r = relev[q][order]
        out.append((precision_at_k(r, k), recall_at_k(r, k), mrr(r), ap(r), ndcg(r, k)))
    return np.mean(out, 0)
res = pd.DataFrame({f"noise={s}": evaluate(s) for s in (0.1, 0.8, 1.5, 5.0)}, index=["P@10", "R@10", "MRR", "MAP", "NDCG@10"]).T; print("\nranker quality vs noise:\n", res.round(3))

# ---- 3. Position bias & IPW ---------------------------------------------------------------------------------------------------------------------------------
pos = np.arange(1, 11); exam = 1 / pos ** 0.8; true_rel = rng.beta(2, 5, 10)                           # true relevance probabilities
imps = 200_000; shown = rng.integers(0, 10, imps)                                                       # randomized positions (propensity estimation)
click = rng.random(imps) < exam[shown] * true_rel[shown]
naive_ctr = np.array([click[shown == i].mean() for i in range(10)]); ipw = naive_ctr / exam
print("\nitem  true_rel  naive_CTR  IPW_est"); print(pd.DataFrame({"true_rel": true_rel, "naive_CTR": naive_ctr, "IPW_est": ipw}).round(3).head(5).to_string())
print("corr(naive CTR, true_rel) at fixed order = position-confounded; corr(IPW, true_rel) =", round(np.corrcoef(ipw, true_rel)[0, 1], 3))

# ---- 4. Recommender: popularity vs SVD matrix factorization, hit-rate@10 on leave-one-out ------------------------------------------------------------------
U, I, f = 1500, 400, 8; P = rng.normal(0, 1, (U, f)); Qm = rng.normal(0, 1, (I, f)); pop = np.exp(rng.normal(0, 1, I)); prob = 1 / (1 + np.exp(-(P @ Qm.T / 2 - 4 + 0.8 * np.log(pop))))
R = (rng.random((U, I)) < prob).astype(float); has = np.where(R.sum(1) >= 5)[0]
test_item = np.array([rng.choice(np.where(R[u])[0]) for u in has]); Rtr = R.copy(); Rtr[has, test_item] = 0
def hitrate(score):
    score = np.where(Rtr[has] > 0, -np.inf, score[has]); top = np.argpartition(-score, 10, axis=1)[:, :10]; return np.mean([test_item[i] in top[i] for i in range(len(has))])
popscore = np.tile(Rtr.sum(0), (U, 1)); u_, s_, vt_ = np.linalg.svd(Rtr, full_matrices=False)
hr = {"popularity": float(hitrate(popscore))}
for kk in (5, 20, 50): hr[f"SVD k={kk}"] = float(hitrate((u_[:, :kk] * s_[:kk]) @ vt_[:kk]))
print("\nhit-rate@10 (leave-one-out):", {k: round(v, 3) for k, v in hr.items()})
top_svd = np.argpartition(-np.where(Rtr[has] > 0, -np.inf, (u_[has, :20] * s_[:20]) @ vt_[:20]), 10, axis=1)[:, :10]
cov = len(set(top_svd.ravel())) / I
print(f"catalog coverage of SVD(20) top-10 lists: {cov:.1%} (popularity baseline covers ~{10/I:.1%})")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6)); res[["P@10", "MAP", "NDCG@10"]].plot(ax=ax[0], marker="o"); ax[0].set_title("Metrics vs ranker noise")
ax[1].bar(range(1, 11), naive_ctr, alpha=.5, label="naive CTR"); ax[1].plot(range(1, 11), true_rel, "ro-", label="true relevance"); ax[1].plot(range(1, 11), ipw, "g^--", label="IPW est."); ax[1].legend(fontsize=7); ax[1].set_title("Position bias correction"); savefig("ranking_metrics.png")
