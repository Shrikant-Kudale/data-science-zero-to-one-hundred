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


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, recall_score, precision_score

X, y = make_classification(n_samples=40000, n_features=20, n_informative=6, n_redundant=4, weights=[.99], flip_y=.005, class_sep=.9, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)
print(f"prevalence train={ytr.mean():.4f} test={yte.mean():.4f} | baseline PR-AUC = prevalence = {yte.mean():.4f}")

def smote(X, y, k=5, ratio=1.0, seed=0):
    r = np.random.default_rng(seed); Xp = X[y == 1]; nn = NearestNeighbors(n_neighbors=k + 1).fit(Xp); nbr = nn.kneighbors(Xp, return_distance=False)[:, 1:]
    n_new = int(ratio * ((y == 0).sum() - (y == 1).sum())); i = r.integers(0, len(Xp), n_new); j = nbr[i, r.integers(0, k, n_new)]
    lam = r.random((n_new, 1)); Xn = Xp[i] + lam * (Xp[j] - Xp[i])
    return np.vstack([X, Xn]), np.r_[y, np.ones(n_new, int)]

def report(name, proba, thr=.5):
    pred = proba >= thr; print(f"{name:32s} ROC-AUC={roc_auc_score(yte, proba):.3f} PR-AUC={average_precision_score(yte, proba):.3f} recall={recall_score(yte, pred):.2f} precision={precision_score(yte, pred, zero_division=0):.2f} @thr={thr:.2f}")
    return precision_recall_curve(yte, proba)
curves = {}
lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr); p_base = lr.predict_proba(Xte)[:, 1]; curves["baseline"] = report("LogReg baseline", p_base)
lw = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr); p_w = lw.predict_proba(Xte)[:, 1]; curves["class_weight"] = report("LogReg class_weight=balanced", p_w)
Xs, ys = smote(Xtr, ytr); ls = LogisticRegression(max_iter=1000).fit(Xs, ys); p_s = ls.predict_proba(Xte)[:, 1]; curves["SMOTE"] = report("LogReg + own SMOTE (train only)", p_s)
print(f"   SMOTE train size {len(Xtr)} -> {len(Xs)} | mean predicted prob: baseline {p_base.mean():.3f}, balanced {p_w.mean():.3f}, smote {p_s.mean():.3f} (true {yte.mean():.3f}) => calibration distorted")
gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=.08, class_weight="balanced", random_state=0).fit(Xtr, ytr); p_g = gb.predict_proba(Xte)[:, 1]; curves["GBM balanced"] = report("HistGB class_weight=balanced", p_g)
try:
    from imblearn.over_sampling import SMOTE
    Xi, yi = SMOTE(random_state=0).fit_resample(Xtr, ytr); report("LogReg + imblearn SMOTE", LogisticRegression(max_iter=1000).fit(Xi, yi).predict_proba(Xte)[:, 1])
except ImportError: pass

# ---- cost-sensitive threshold on the (un-distorted) baseline model, chosen on validation split ---------------------------------------
C_FN, C_FP = 200.0, 5.0; print(f"\ntheoretical Bayes threshold for calibrated probs = {C_FP/(C_FP+C_FN):.3f}")
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=.3, stratify=ytr, random_state=1); m = HistGradientBoostingClassifier(max_depth=4, random_state=0).fit(Xa, ya)
pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xte)[:, 1]; grid = np.linspace(.005, .9, 180)
cost = lambda yt, pr, t: C_FN * ((pr < t) & (yt == 1)).sum() + C_FP * ((pr >= t) & (yt == 0)).sum()
tstar = grid[np.argmin([cost(yv, pv, t) for t in grid])]
print(f"validation-optimal threshold={tstar:.3f} | test cost: default 0.5 -> {cost(yte, pt, .5):,.0f} | tuned -> {cost(yte, pt, tstar):,.0f} | predict-all-negative -> {C_FN*yte.sum():,.0f}")

plt.figure(figsize=(7, 4.5))
for k, (p, r, _) in curves.items(): plt.plot(r, p, label=k)
plt.axhline(yte.mean(), ls="--", c="gray"); plt.xlabel("recall"); plt.ylabel("precision"); plt.legend(); plt.title("Precision-Recall curves (1% prevalence)"); savefig("pr_curves.png")
