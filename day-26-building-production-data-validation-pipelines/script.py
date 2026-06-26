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


import json, hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

@dataclass
class Rule:
    column: str; dtype: str                            # "number" | "string" | "datetime"
    nullable: bool = False; min: Optional[float] = None; max: Optional[float] = None
    allowed: Optional[Sequence] = None; unique: bool = False

class ContractError(Exception): pass

def validate(df, rules, max_bad_rate=0.05):
    report = {"rows_in": len(df), "schema_hash": hashlib.md5(json.dumps([r.__dict__ for r in rules], default=str).encode()).hexdigest()[:8], "violations": {}}
    missing = [r.column for r in rules if r.column not in df.columns]
    if missing: raise ContractError(f"missing columns {missing}")                                           # hard fail
    bad = pd.Series(False, index=df.index); reasons = pd.Series("", index=df.index)
    def flag(mask, why):
        nonlocal bad, reasons; mask = mask.fillna(False); report["violations"][why] = int(mask.sum()); reasons[mask & ~bad] = why; bad |= mask
    for r in rules:
        s = df[r.column]
        if r.dtype == "number": s = pd.to_numeric(s, errors="coerce"); flag(df[r.column].notna() & s.isna(), f"{r.column}:not_numeric")
        if r.dtype == "datetime": s = pd.to_datetime(s, errors="coerce"); flag(df[r.column].notna() & s.isna(), f"{r.column}:bad_date")
        if not r.nullable: flag(s.isna(), f"{r.column}:null")
        if r.min is not None: flag(s < r.min, f"{r.column}:below_min")
        if r.max is not None: flag(s > r.max, f"{r.column}:above_max")
        if r.allowed is not None: flag(~s.isin(r.allowed) & s.notna(), f"{r.column}:not_allowed")
        if r.unique: flag(s.duplicated(keep="first"), f"{r.column}:duplicate")
    report["violations"] = {k: v for k, v in report["violations"].items() if v}
    report["bad_rate"] = float(bad.mean()); report["rows_good"] = int((~bad).sum())
    if report["bad_rate"] > max_bad_rate: raise ContractError(f"bad rate {report['bad_rate']:.1%} > {max_bad_rate:.0%}: {report['violations']}")
    return df[~bad].copy(), df[bad].assign(reason=reasons[bad]), report

def batch_checks(df, ref_stats, tol=4.0):
    alerts = []
    for c, (mu, sd) in ref_stats.items():
        z = abs(df[c].mean() - mu) / (sd / np.sqrt(len(df)))
        if z > tol: alerts.append(f"mean shift in {c}: z={z:.1f}")
    return alerts

# ---- 1. Build a reference batch + a dirty batch ------------------------------------------------------------------------------------------
def make(n, dirty=False):
    d = pd.DataFrame({"user_id": np.arange(n), "age": rng.normal(38, 11, n).round().clip(18, 90), "income": rng.lognormal(10.3, .5, n),
                      "plan": rng.choice(["free", "plus", "pro"], n, p=[.6, .3, .1]), "signup": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 500, n), unit="D")})
    d["churn"] = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + (50 - d.age) / 25 + (d.plan == "free") * .6)))).astype(int)
    if dirty:
        d.loc[rng.choice(n, 30, replace=False), "age"] = -5; d.loc[rng.choice(n, 25, replace=False), "plan"] = "enterprise"
        d.loc[rng.choice(n, 20, replace=False), "income"] = np.nan; d.loc[5, "user_id"] = 4
    return d
ref, new = make(5000), make(2000, dirty=True)
rules = [Rule("user_id", "number", unique=True), Rule("age", "number", min=0, max=120), Rule("income", "number", min=0), Rule("plan", "string", allowed=["free", "plus", "pro"]),
         Rule("signup", "datetime"), Rule("churn", "number", min=0, max=1)]
good, quarantine, rep = validate(new, rules, max_bad_rate=.05); print(json.dumps(rep, indent=2)); print("quarantine sample:\n", quarantine[["user_id", "reason"]].head(4).to_string(index=False))
try: validate(new.drop(columns=["plan"]), rules)
except ContractError as e: print("hard-fail on schema break ->", e)
try: validate(new.assign(age=-1), rules)
except ContractError as e: print("fail-fast on high bad-rate ->", str(e)[:70], "...")
print("batch alerts (clean):", batch_checks(good, {"age": (ref.age.mean(), ref.age.std())}), "| (shifted):", batch_checks(good.assign(age=good.age + 3), {"age": (ref.age.mean(), ref.age.std())}))
quarantine.to_csv(os.path.join(OUT, "quarantine.csv"), index=False); json.dump(rep, open(os.path.join(OUT, "validation_report.json"), "w"), indent=2)

# ---- 2. One transformation path for train and serve ---------------------------------------------------------------------------------------------
num, cat = ["age", "income"], ["plan"]
pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))]).fit(ref[num + cat], ref.churn)
p = model.predict_proba(good[num + cat])[:, 1]; print(f"scored {len(p)} validated rows; mean risk={p.mean():.3f}; unseen-category rows safe (handle_unknown=ignore): {model.predict_proba(new[num+cat].head(3))[:,1].round(3)}")

pd.Series(rep["violations"]).sort_values().plot.barh(figsize=(7, 3.6), color="#c0392b", title="Validation violations by rule"); savefig("violations.png")
