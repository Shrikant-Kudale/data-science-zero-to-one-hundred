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
try:
    pd.options.mode.copy_on_write = True
except Exception:
    pass
print("pandas", pd.__version__)

# ---- 1. Synthetic transaction table ---------------------------------------------------
n = 400_000
cities = np.array(["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata"])
df = pd.DataFrame({
    "user_id": rng.integers(0, 50_000, n),
    "city": cities[rng.integers(0, len(cities), n)].astype(object),
    "amount": rng.gamma(2.0, 50.0, n),
    "qty": rng.integers(1, 10, n),
    "flag": rng.random(n) > 0.5,
})
before = df.memory_usage(deep=True)
df2 = df.assign(city=df["city"].astype("category"), user_id=pd.to_numeric(df["user_id"], downcast="unsigned"),
                amount=df["amount"].astype("float32"), qty=pd.to_numeric(df["qty"], downcast="unsigned"))
after = df2.memory_usage(deep=True)
print(f"memory: {before.sum()/1e6:.1f} MB -> {after.sum()/1e6:.1f} MB ({1-after.sum()/before.sum():.0%} saved)")
print(pd.DataFrame({"before_MB": before / 1e6, "after_MB": after / 1e6}).round(2))

# ---- 2. apply(axis=1) vs vectorized ------------------------------------------------------
sub = df.head(100_000)
t = time.perf_counter(); r1 = sub.apply(lambda r: r["amount"] * r["qty"], axis=1); t_apply = time.perf_counter() - t
t = time.perf_counter(); r2 = sub["amount"] * sub["qty"]; t_vec = time.perf_counter() - t
print(f"apply(axis=1) {t_apply:.3f}s vs vectorized {t_vec*1e3:.2f} ms -> {t_apply/t_vec:,.0f}x slower | equal={np.allclose(r1, r2)}")

# ---- 3. Index alignment -----------------------------------------------------------------
a = pd.Series([1, 2, 3], index=list("abc")); b = pd.Series([10, 20, 30], index=list("bcd"))
print("a + b =\n", (a + b).to_dict(), "\na.add(b, fill_value=0) =", a.add(b, fill_value=0).to_dict())

# ---- 4. Copy semantics -------------------------------------------------------------------
d = pd.DataFrame({"a": [1, -1, 2], "b": [0, 0, 0]})
d[d["a"] > 0]["b"] = 99          # chained assignment: no effect under CoW
print("after chained assignment:", d["b"].tolist())
d.loc[d["a"] > 0, "b"] = 99
print("after .loc assignment   :", d["b"].tolist())

# ---- 5. dtype pitfalls: nullable ints, categorical ordering -------------------------------
s = pd.Series([1, 2, None]); print("int with None ->", s.dtype, "| nullable:", pd.Series([1, 2, None], dtype="Int64").dtype)
size = pd.Categorical(["S", "L", "M", "S"], categories=["S", "M", "L"], ordered=True)
print("ordered categorical max:", size.max(), "| codes:", size.codes.tolist())

plt.figure(figsize=(6, 3.5)); plt.bar(["before", "after"], [before.sum() / 1e6, after.sum() / 1e6], color=["#c0392b", "#27ae60"])
plt.ylabel("MB"); plt.title("DataFrame memory after dtype optimization"); savefig("memory_savings.png")
