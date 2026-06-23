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
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ---- 1. High-cardinality category with weak true signal --------------------------------------------------------------------------
n, K = 20000, 3000
zipc = rng.zipf(1.3, n) % K; level_eff = rng.normal(0, 0.5, K)
signal = rng.normal(size=n); logit = -2 + level_eff[zipc] + 0.8 * signal
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"zip": zipc.astype(str), "signal": signal, "y": y}); tr, te = train_test_split(df, test_size=.3, random_state=0)
print(f"levels in train={tr.zip.nunique()} | share of rows in levels with <=3 obs: {(tr.zip.map(tr.zip.value_counts()) <= 3).mean():.1%} | unseen in test: {(~te.zip.isin(tr.zip)).mean():.1%}")

prior = tr.y.mean()
def te_stats(d, m): g = d.groupby("zip").y.agg(["sum", "count"]); return (g["sum"] + m * prior) / (g["count"] + m)
def encode(train, test, m, oof=True, k=5):
    full = te_stats(train, m); test_enc = test.zip.map(full).fillna(prior)
    if not oof: return train.zip.map(full).values, test_enc.values
    tr_enc = np.zeros(len(train))
    for a, b in KFold(k, shuffle=True, random_state=1).split(train):
        tr_enc[b] = train.iloc[b].zip.map(te_stats(train.iloc[a], m)).fillna(prior).values
    return tr_enc, test_enc.values
freq = tr.zip.value_counts(normalize=True)

def evaluate(name, Xtr, Xte):
    mdl = LogisticRegression(max_iter=500).fit(Xtr, tr.y); a, b = roc_auc_score(tr.y, mdl.predict_proba(Xtr)[:, 1]), roc_auc_score(te.y, mdl.predict_proba(Xte)[:, 1])
    print(f"{name:34s} train AUC={a:.3f}  test AUC={b:.3f}  gap={a-b:+.3f}"); return b
res = {}
tr_n, te_n = encode(tr, te, m=0, oof=False); res["naive TE (m=0)"] = evaluate("naive target enc (leaky)", np.c_[tr_n, tr.signal], np.c_[te_n, te.signal])
tr_s, te_s = encode(tr, te, m=20, oof=False); res["smoothed TE (m=20)"] = evaluate("smoothed TE, in-sample", np.c_[tr_s, tr.signal], np.c_[te_s, te.signal])
tr_o, te_o = encode(tr, te, m=20, oof=True); res["OOF smoothed TE"] = evaluate("OOF smoothed TE (correct)", np.c_[tr_o, tr.signal], np.c_[te_o, te.signal])
res["frequency"] = evaluate("frequency encoding", np.c_[tr.zip.map(freq), tr.signal], np.c_[te.zip.map(freq).fillna(0), te.signal])
res["signal only"] = evaluate("no category (baseline)", tr[["signal"]], te[["signal"]])
# one-hot (sparse) for comparison
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(handle_unknown="ignore").fit(tr[["zip"]]); from scipy import sparse
res["one-hot + L2"] = evaluate("one-hot (C=0.3)", sparse.hstack([oh.transform(tr[["zip"]]), sparse.csr_matrix(tr[["signal"]].values)]).tocsr(), sparse.hstack([oh.transform(te[["zip"]]), sparse.csr_matrix(te[["signal"]].values)]).tocsr())

# ---- 2. Show shrinkage on rare levels ----------------------------------------------------------------------------------------------------
cnt = tr.zip.value_counts(); rare = cnt[cnt == 1].index[:3]
print("rare levels (n=1): raw mean ->", [float(tr[tr.zip == r].y.mean()) for r in rare], "| smoothed (m=20) ->", [round(float(te_stats(tr, 20)[r]), 3) for r in rare], "| prior", round(prior, 3))

plt.figure(figsize=(8, 3.8)); plt.barh(list(res.keys()), list(res.values()), color="#2980b9"); plt.xlim(.55, max(res.values()) + .02); plt.xlabel("test AUC"); plt.title("Categorical encodings (3000-level zip)"); savefig("encoding_auc.png")
