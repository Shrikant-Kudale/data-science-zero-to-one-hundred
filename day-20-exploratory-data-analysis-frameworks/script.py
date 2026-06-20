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


import json
import pandas as pd

# ---- 1. Synthetic customer table with planted problems -----------------------------------------------------------------------
n = 5000
df = pd.DataFrame({"customer_id": np.arange(n), "age": rng.normal(40, 12, n).round(), "income": rng.lognormal(10.5, 0.6, n).round(),
                   "tenure_m": rng.integers(1, 72, n), "plan": rng.choice(["basic", "plus", "pro"], n, p=[.6, .3, .1]),
                   "city": rng.choice([f"c{i}" for i in range(300)], n), "const_flag": 1})
logit = -2 + 0.02 * (50 - df.tenure_m) + 0.5 * (df.plan == "basic") - 0.00001 * df.income + rng.normal(0, .5, n)
df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df["cancel_reason_len"] = np.where(df.churned == 1, rng.integers(5, 80, n), np.nan)       # LEAK: only exists after churn
df["income_k"] = df.income / 1000 + rng.normal(0, 1, n)                                     # near-duplicate of income
df.loc[rng.choice(n, 400, replace=False), "age"] = np.nan; df.loc[rng.choice(n, 25, replace=False), "age"] = -1   # missing + impossible
df = pd.concat([df, df.sample(40, random_state=1)], ignore_index=True)                                            # duplicates

# ---- 2. Reusable EDA function ----------------------------------------------------------------------------------------------------------
def eda(df, target, id_cols=()):
    rep = {"shape": df.shape, "duplicate_rows": int(df.duplicated().sum()), "duplicate_ids": {c: int(df[c].duplicated().sum()) for c in id_cols}}
    cols = [c for c in df.columns if c != target]
    rep["missing_pct"] = (df[cols].isna().mean() * 100).round(1)[lambda s: s > 0].to_dict()
    rep["constant_cols"] = [c for c in cols if df[c].nunique(dropna=False) <= 1]
    rep["high_cardinality"] = {c: int(df[c].nunique()) for c in cols if df[c].dtype == object and df[c].nunique() > 50}
    num = df[cols].select_dtypes("number").drop(columns=list(id_cols) + rep["constant_cols"], errors="ignore")
    rep["skewed(|skew|>1)"] = num.skew()[lambda s: s.abs() > 1].round(2).to_dict()
    rep["impossible_negative"] = {c: int((num[c] < 0).sum()) for c in num if (num[c] < 0).any() and c in ("age", "income", "tenure_m")}
    corr = num.corr(); pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    rep["high_corr_pairs"] = {f"{a}~{b}": round(v, 3) for (a, b), v in pairs[pairs.abs() > .9].items()}
    leak = {}
    for c in cols:                                                                               # leakage heuristics
        if c in id_cols: continue
        s = df[c]
        if s.isna().mean() > 0 and abs(s.isna().astype(int).corr(df[target])) > .9: leak[c] = "missingness == target"
        elif pd.api.types.is_numeric_dtype(s) and s.nunique() > 1 and abs(s.fillna(s.median()).corr(df[target])) > .9: leak[c] = "|corr|>0.9"
    rep["leakage_suspects"] = leak
    rep["target_rate"] = round(float(df[target].mean()), 4)
    rep["target_rate_by_plan"] = df.groupby("plan")[target].agg(["mean", "size"]).round(3).to_dict("index")
    return rep, corr
report, corr = eda(df, "churned", id_cols=("customer_id",))
print(json.dumps(report, indent=2, default=str))
with open(os.path.join(OUT, "eda_report.json"), "w") as f: json.dump(report, f, indent=2, default=str)

# ---- 3. Bivariate: churn rate by tenure bin with Wilson CI ------------------------------------------------------------------------------
df["tenure_bin"] = pd.cut(df.tenure_m, [0, 6, 12, 24, 48, 72]); g = df.groupby("tenure_bin", observed=True)["churned"].agg(["mean", "size"])
z = 1.96; p, nn = g["mean"], g["size"]; g["lo"] = (p + z**2 / (2 * nn) - z * np.sqrt(p * (1 - p) / nn + z**2 / (4 * nn**2))) / (1 + z**2 / nn)
g["hi"] = (p + z**2 / (2 * nn) + z * np.sqrt(p * (1 - p) / nn + z**2 / (4 * nn**2))) / (1 + z**2 / nn); print("\nchurn by tenure (Wilson CI):\n", g.round(3))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
(df.isna().mean() * 100).sort_values(ascending=False).head(6).plot.barh(ax=ax[0], title="% missing")
im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); ax[1].set_xticks(range(len(corr))); ax[1].set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax[1].set_yticks(range(len(corr))); ax[1].set_yticklabels(corr.columns, fontsize=7); ax[1].set_title("Correlation")
ax[2].errorbar(range(len(g)), g["mean"], yerr=[g["mean"] - g["lo"], g["hi"] - g["mean"]], fmt="o-"); ax[2].set_xticks(range(len(g))); ax[2].set_xticklabels(g.index.astype(str), fontsize=7); ax[2].set_title("Churn by tenure")
savefig("eda_overview.png")
