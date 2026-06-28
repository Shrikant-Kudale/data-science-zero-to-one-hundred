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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

# ---- 1. Synthetic snapshot of 30k subscribers ---------------------------------------------------------------------------------------------------
n = 30000
df = pd.DataFrame({"tenure_m": rng.integers(1, 60, n), "monthly_fee": rng.choice([9, 15, 29, 49], n, p=[.4, .3, .2, .1]), "usage_trend": rng.normal(0, 1, n),
                   "days_since_login": rng.exponential(10, n).round(), "tickets_90d": rng.poisson(1.0, n), "payment_fail": rng.binomial(1, .06, n)})
logit = (-3.0 - 0.03 * df.tenure_m - 0.9 * df.usage_trend + 0.04 * df.days_since_login + 0.25 * df.tickets_90d + 1.1 * df.payment_fail + 0.012 * df.monthly_fee
         + 0.4 * np.sin(df.tenure_m / 6))
df["churn90"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int); print(f"90-day churn rate = {df.churn90.mean():.3%}")
split = int(.7 * n); feats = [c for c in df.columns if c != "churn90"]; tr, te = df.iloc[:split], df.iloc[split:]          # (rows are exchangeable; use time-split in real data)

lr = LogisticRegression(max_iter=1000).fit(tr[feats], tr.churn90); gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=.07, max_iter=250, random_state=0).fit(tr[feats], tr.churn90)
for nm, m in [("logistic", lr), ("gradient boosting", gb)]:
    p = m.predict_proba(te[feats])[:, 1]; print(f"{nm:18s} ROC-AUC={roc_auc_score(te.churn90, p):.3f} PR-AUC={average_precision_score(te.churn90, p):.3f} Brier={brier_score_loss(te.churn90, p):.4f}")
p = gb.predict_proba(te[feats])[:, 1]; te = te.assign(p=p)

# ---- 2. Lift / gains table --------------------------------------------------------------------------------------------------------------------------------
te["decile"] = pd.qcut(te.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = te.groupby("decile", observed=True).agg(n=("p", "size"), churn_rate=("churn90", "mean"), avg_pred=("p", "mean")); lift["lift"] = lift.churn_rate / te.churn90.mean()
lift["cum_capture"] = (lift.churn_rate * lift.n).cumsum() / te.churn90.sum(); print("\n", lift.round(3))

# ---- 3. LTV per customer --------------------------------------------------------------------------------------------------------------------------------------
d_m = 0.01; c_m = (1 - (1 - te.p.clip(1e-4, .99)) ** (1 / 3)); margin = te.monthly_fee * 0.7
te["ltv"] = margin / (1 - (1 - c_m) / (1 + d_m)); print(f"\nLTV mean=${te.ltv.mean():,.0f} median=${te.ltv.median():,.0f} | top-decile-risk LTV=${te[te.decile==1].ltv.mean():,.0f} vs bottom ${te[te.decile==10].ltv.mean():,.0f}")
c0 = 0.03; print(f"check: m/c without discount for c=3% -> {1/c0:.1f} months expected lifetime")

# ---- 4. Campaign economics: offer cost $12, 20% of would-be churners saved (uplift assumption) ------------------------------------------------
cost, save = 12.0, 0.20; te["exp_profit"] = te.p * save * te.ltv - cost
by_risk = te.sort_values("p", ascending=False).assign(cum=lambda d: d.exp_profit.cumsum()); by_ev = te.sort_values("exp_profit", ascending=False).assign(cum=lambda d: d.exp_profit.cumsum())
print(f"target all: ${te.exp_profit.sum():,.0f} | best-by-risk-rank peak: ${by_risk.cum.max():,.0f} at k={int(by_risk.cum.values.argmax())+1} | best-by-expected-value peak: ${by_ev.cum.max():,.0f} at k={int(by_ev.cum.values.argmax())+1}")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); ax[0].bar(lift.index.astype(int), lift.lift, color="#2980b9"); ax[0].axhline(1, c="k", ls="--"); ax[0].set_title("Lift by risk decile")
fr, mp = calibration_curve(te.churn90, te.p, n_bins=10, strategy="quantile"); ax[1].plot(mp, fr, "o-"); ax[1].plot([0, mp.max()], [0, mp.max()], "k--"); ax[1].set_title("Calibration")
ax[2].plot(by_risk.cum.values, label="rank by risk"); ax[2].plot(by_ev.cum.values, label="rank by expected profit"); ax[2].legend(); ax[2].set_title("Cumulative profit vs customers targeted"); savefig("churn_ltv.png")
