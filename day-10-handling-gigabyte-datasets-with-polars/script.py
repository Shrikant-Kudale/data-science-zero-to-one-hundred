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


import tempfile
import pandas as pd

N = 1_000_000
tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "events.csv")
keys = np.array([f"k{i:03d}" for i in range(200)])
pd.DataFrame({"key": keys[rng.integers(0, 200, N)], "value": rng.gamma(2, 10, N), "qty": rng.integers(1, 20, N),
              "ts": rng.integers(0, 10_000, N)}).to_csv(path, index=False)
print(f"wrote {N:,} rows -> {os.path.getsize(path)/1e6:.1f} MB CSV")

# ---- A. pandas chunked decomposable aggregation ----------------------------------------------------
t = time.perf_counter(); parts = []
for ch in pd.read_csv(path, chunksize=200_000, usecols=["key", "value"], dtype={"key": "category"}):
    ch = ch[ch["value"] > 5]
    parts.append(ch.groupby("key", observed=True)["value"].agg(["count", "sum"]))
tot = pd.concat(parts).groupby(level=0).sum(); tot["mean"] = tot["sum"] / tot["count"]
t_pd = time.perf_counter() - t
print(f"pandas chunked: {t_pd:.2f}s | top key: {tot['mean'].idxmax()} mean={tot['mean'].max():.3f}")

# ---- B. Polars lazy (falls back gracefully if not installed) ---------------------------------------
try:
    import polars as pl
    t = time.perf_counter()
    lazy = (pl.scan_csv(path).filter(pl.col("value") > 5).group_by("key")
            .agg(pl.col("value").mean().alias("mean"), pl.len().alias("count")).sort("key"))
    print("\n--- optimized plan ---\n", lazy.explain()[:400], "\n----------------------")
    res = lazy.collect().to_pandas().set_index("key"); t_pl = time.perf_counter() - t
    print(f"polars lazy: {t_pl:.2f}s | matches pandas: {np.allclose(res['mean'].sort_index().values, tot['mean'].sort_index().values)}")
    # window expression: z-score per group without apply
    z = pl.read_csv(path).with_columns(((pl.col("value") - pl.col("value").mean().over("key")) / pl.col("value").std().over("key")).alias("z"))
    print("window expr z mean ~", round(float(z["z"].mean()), 6))
    pq = os.path.join(tmp, "events.parquet"); pl.read_csv(path).write_parquet(pq)
    t = time.perf_counter(); pl.read_parquet(pq, columns=["key", "value"]); t_pq = time.perf_counter() - t
    print(f"parquet size {os.path.getsize(pq)/1e6:.1f} MB (csv {os.path.getsize(path)/1e6:.1f} MB) | 2-col read {t_pq*1e3:.0f} ms")
except ImportError:
    t_pl = float("nan"); print("polars not installed -> `pip install polars` to run the lazy section")

# ---- C. Decomposable statistics: Chan's parallel variance --------------------------------------------
def merge(a, b):
    na, ma, m2a = a; nb, mb, m2b = b; d = mb - ma; n = na + nb
    return n, ma + d * nb / n, m2a + m2b + d * d * na * nb / n
x = rng.normal(5, 3, 100_000); chunks = np.array_split(x, 7)
state = (0, 0.0, 0.0)
for c in chunks:
    state = merge(state, (len(c), c.mean(), ((c - c.mean()) ** 2).sum())) if state[0] else (len(c), c.mean(), ((c - c.mean()) ** 2).sum())
print("merged variance:", state[2] / (state[0] - 1), "| numpy:", x.var(ddof=1))

plt.figure(figsize=(5, 3.5)); plt.bar(["pandas chunked", "polars lazy"], [t_pd, 0 if np.isnan(t_pl) else t_pl], color=["#c0392b", "#27ae60"]); plt.ylabel("seconds")
plt.title("Group aggregation on 1M rows"); savefig("engine_timing.png"); shutil_rm = __import__("shutil").rmtree(tmp, ignore_errors=True)
