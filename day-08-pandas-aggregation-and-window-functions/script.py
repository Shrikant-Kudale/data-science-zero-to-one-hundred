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

# ---- 1. Synthetic daily revenue per product -----------------------------------------------------
days = pd.date_range("2026-01-01", periods=120, freq="D"); prods = ["A", "B", "C"]
df = pd.DataFrame([(d, p) for p in prods for d in days], columns=["date", "product"])
df["revenue"] = (100 + df["product"].map({"A": 0, "B": 40, "C": -20}) + 0.3 * df.groupby("product").cumcount()
                 + 15 * np.sin(2 * np.pi * df["date"].dt.dayofweek / 7) + rng.normal(0, 8, len(df))).round(2)
df.loc[rng.choice(len(df), 12, replace=False), "revenue"] = np.nan

# ---- 2. agg / named aggregation -----------------------------------------------------------------
summary = df.groupby("product").agg(total=("revenue", "sum"), mean=("revenue", "mean"), p95=("revenue", lambda s: s.quantile(.95)),
                                    missing=("revenue", lambda s: s.isna().sum()))
print(summary.round(1))

# ---- 3. transform: group z-score and group-mean imputation ---------------------------------------
df["z_in_group"] = df.groupby("product")["revenue"].transform(lambda s: (s - s.mean()) / s.std())
df["rev_filled"] = df["revenue"].fillna(df.groupby("product")["revenue"].transform("mean"))
print("z-score per group mean ~0:", df.groupby("product")["z_in_group"].mean().abs().max().round(10))

# ---- 4. windows: trailing only (no leakage) ---------------------------------------------------------
df = df.sort_values(["product", "date"])
g = df.groupby("product")["rev_filled"]
df["ma7"] = g.transform(lambda s: s.rolling(7, min_periods=3).mean())
df["ewm"] = g.transform(lambda s: s.ewm(halflife=5).mean())
df["lag1_feature"] = g.shift(1)                                   # safe feature for day t
df["wow_pct"] = g.pct_change(7)
df["cum_max"] = g.cummax()
df["rank_in_day"] = df.groupby("date")["rev_filled"].rank(ascending=False)
print(df[df["product"] == "A"][["date", "rev_filled", "ma7", "ewm", "lag1_feature", "wow_pct"]].tail(4).round(2))

# ---- 5. Top-N per group and pivot --------------------------------------------------------------------
top2 = df.sort_values("rev_filled", ascending=False).groupby("product").head(2)[["product", "date", "rev_filled"]]
print("\ntop-2 revenue days per product:\n", top2.sort_values(["product", "rev_filled"], ascending=[True, False]).round(1))
pt = df.assign(month=df["date"].dt.to_period("M")).pivot_table(index="month", columns="product", values="rev_filled", aggfunc="mean")
print("\nmonthly mean pivot:\n", pt.round(1))

# ---- 6. Simpson's paradox mini-demo ----------------------------------------------------------------------
sp = pd.DataFrame({"group": ["easy"] * 2 + ["hard"] * 2, "treat": ["A", "B", "A", "B"], "n": [90, 10, 10, 90], "succ": [81, 9.5, 3, 40]})
print("\nSimpson: within-group rates", (sp.succ / sp.n).round(2).tolist(), "| pooled A:",
      round(sp[sp.treat == "A"].succ.sum() / sp[sp.treat == "A"].n.sum(), 3), "pooled B:", round(sp[sp.treat == "B"].succ.sum() / sp[sp.treat == "B"].n.sum(), 3))

a = df[df["product"] == "A"]
plt.figure(figsize=(9, 3.6)); plt.plot(a["date"], a["rev_filled"], alpha=.4, label="daily"); plt.plot(a["date"], a["ma7"], label="7d MA"); plt.plot(a["date"], a["ewm"], label="EWM hl=5")
plt.legend(); plt.title("Product A revenue: rolling vs EWM"); savefig("rolling_windows.png")
