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
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ---- 1. Mixed dataset: 6 informative, 6 redundant, 38 noise -------------------------------------------------------------------------
X, y = make_classification(n_samples=2500, n_features=50, n_informative=6, n_redundant=6, n_repeated=0, shuffle=False, random_state=0)
names = [f"f{i:02d}" for i in range(50)]; Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
print("ground truth: f00-f05 informative, f06-f11 linear combos of them, f12-f49 noise")
sel = {}
sel["filter F-test"] = np.argsort(-SelectKBest(f_classif, k=6).fit(Xtr, ytr).scores_)[:6]
sel["filter MI"] = np.argsort(-mutual_info_classif(Xtr, ytr, random_state=0))[:6]
sel["wrapper RFE (LogReg)"] = np.where(RFE(LogisticRegression(max_iter=500), n_features_to_select=6).fit(StandardScaler().fit_transform(Xtr), ytr).support_)[0]
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=.05).fit(StandardScaler().fit_transform(Xtr), ytr); sel["embedded L1"] = np.where(l1.coef_[0] != 0)[0]
rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(Xtr, ytr); sel["embedded RF importance"] = np.argsort(-rf.feature_importances_)[:6]
pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0, n_jobs=-1); sel["permutation (held-out)"] = np.argsort(-pi.importances_mean)[:6]
for k, v in sel.items(): print(f"{k:24s} -> {sorted(names[i] for i in v)[:12]}  | hits in f00-f11: {(np.array(v) < 12).sum()}/{len(v)}")

# ---- 2. Downstream accuracy with each selected set ----------------------------------------------------------------------------------------
print()
for k, v in sel.items():
    if len(v) == 0: continue
    s = cross_val_score(RandomForestClassifier(150, random_state=0, n_jobs=-1), Xtr[:, v], ytr, cv=3, scoring="roc_auc").mean(); print(f"{k:24s} CV AUC (RF on selected)={s:.3f}")
print(f"{'all 50 features':24s} CV AUC={cross_val_score(RandomForestClassifier(150, random_state=0, n_jobs=-1), Xtr, ytr, cv=3, scoring='roc_auc').mean():.3f}")

# ---- 3. Selection-bias leakage on PURE NOISE ----------------------------------------------------------------------------------------------------
Xn = rng.normal(size=(100, 2000)); yn = rng.integers(0, 2, 100); skf = StratifiedKFold(5, shuffle=True, random_state=0)
Xsel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)                                           # WRONG: uses all labels
wrong = cross_val_score(LogisticRegression(max_iter=500), Xsel, yn, cv=skf, scoring="roc_auc").mean()
right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=500)), Xn, yn, cv=skf, scoring="roc_auc").mean()
print(f"\npure-noise data (true AUC=0.5): select-then-CV (leaky) AUC={wrong:.3f} | selection inside pipeline AUC={right:.3f}")

# ---- 4. Stability selection ----------------------------------------------------------------------------------------------------------------------------------
freq = np.zeros(50); Z = StandardScaler().fit_transform(X)
for b in range(60):
    i = rng.choice(len(Z), len(Z) // 2, replace=False); m = LogisticRegression(penalty="l1", solver="liblinear", C=.05).fit(Z[i], y[i]); freq += (m.coef_[0] != 0)
freq /= 60; print("stability selection (freq>=0.8):", [names[i] for i in np.where(freq >= .8)[0]])

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6)); ax[0].bar(range(50), rf.feature_importances_); ax[0].set_title("RF impurity importance (f00-f11 are signal)")
ax[1].bar(range(50), freq, color="#e67e22"); ax[1].axhline(.8, ls="--", c="k"); ax[1].set_title("L1 stability selection frequency"); savefig("feature_selection.png")
