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

def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref); a = np.histogram(cur, edges)[0] / len(cur)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None); return float(((a - e) * np.log(a / e)).sum())
def psi_cat(ref, cur):
    cats = sorted(set(ref) | set(cur)); e = np.array([(ref == c).mean() for c in cats]).clip(1e-4); a = np.array([(cur == c).mean() for c in cats]).clip(1e-4); return float(((a - e) * np.log(a / e)).sum())

# ---- 1. Reference + 8 weeks of progressively drifting data --------------------------------------------------------------------------------------
ref = pd.DataFrame({"amount": rng.lognormal(3, .6, 20000), "age": rng.normal(40, 10, 20000), "channel": rng.choice(["web", "app", "store"], 20000, p=[.5, .35, .15])})
weeks = []
for w in range(8):
    n = 5000; d = pd.DataFrame({"amount": rng.lognormal(3 + .04 * w, .6 + .02 * w, n), "age": rng.normal(40, 10, n),            # amount drifts, age stable
                                "channel": rng.choice(["web", "app", "store"], n, p=[.5 - .03 * w, .35 + .03 * w, .15])})
    weeks.append(d)
rows = []
for w, d in enumerate(weeks):
    ks = stats.ks_2samp(ref.amount, d.amount); ct = pd.crosstab(pd.concat([pd.Series("ref", index=ref.index[:len(d)]), pd.Series("cur", index=d.index)]).values, pd.concat([ref.channel[:len(d)], d.channel]).values)
    rows.append(dict(week=w + 1, psi_amount=psi(ref.amount, d.amount), ks_amount=ks.statistic, ks_p=ks.pvalue, psi_age=psi(ref.age, d.age), psi_channel=psi_cat(ref.channel.values, d.channel.values), chi2_p=stats.chi2_contingency(ct)[1]))
mon = pd.DataFrame(rows).set_index("week"); print(mon.round(4))
status = lambda v: "STABLE" if v < .1 else ("MODERATE" if v < .25 else "ALERT"); print("amount drift status by week:", [status(v) for v in mon.psi_amount])

# ---- 2. Multivariate drift: domain classifier --------------------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier; from sklearn.model_selection import cross_val_score
for w in (0, 7):
    Z = pd.concat([ref.sample(5000, random_state=0).assign(y=0), weeks[w].assign(y=1)]); Z["channel"] = Z.channel.astype("category").cat.codes
    print(f"domain-classifier AUC reference vs week {w+1}: {cross_val_score(RandomForestClassifier(100, max_depth=4, random_state=0, n_jobs=-1), Z.drop(columns='y'), Z.y, cv=3, scoring='roc_auc').mean():.3f}")

# ---- 3. Point-in-time join: leakage demo --------------------------------------------------------------------------------------------------------------------
ids = np.arange(300); t0 = pd.Timestamp("2026-01-01")
feat = pd.concat([pd.DataFrame({"id": i, "ts": t0 + pd.to_timedelta(np.sort(rng.integers(0, 60, 6)), unit="D")}).assign(avg_spend=lambda d: np.cumsum(rng.normal(5, 3, 6))) for i in ids])
labels = pd.DataFrame({"id": ids, "ts": t0 + pd.to_timedelta(rng.integers(20, 40, len(ids)), unit="D")})
labels["y"] = (rng.random(len(ids)) < .5).astype(int)
pit = pd.merge_asof(labels.sort_values("ts"), feat.sort_values("ts"), on="ts", by="id", direction="backward", suffixes=("", "_f"))
latest = feat.sort_values("ts").groupby("id").tail(1)[["id", "avg_spend"]]; leaky = labels.merge(latest, on="id")
print(f"PIT-correct rows with no prior feature: {pit.avg_spend.isna().sum()} | leaky 'latest' join uses future values for {(leaky.merge(feat.groupby('id').ts.max().rename('last'), on='id').pipe(lambda d: d['last'] > d['ts'])).mean():.0%} of rows")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6)); mon[["psi_amount", "psi_age", "psi_channel"]].plot(ax=ax[0], marker="o"); ax[0].axhline(.1, ls="--", c="orange"); ax[0].axhline(.25, ls="--", c="r"); ax[0].set_title("PSI by week")
ax[1].hist(ref.amount, bins=60, alpha=.5, density=True, range=(0, 120), label="ref"); ax[1].hist(weeks[-1].amount, bins=60, alpha=.5, density=True, range=(0, 120), label="week 8"); ax[1].legend(); ax[1].set_title("amount drift")
savefig("drift_monitoring.png")
