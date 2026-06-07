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

# ---- 1. MultiIndex on regional sales --------------------------------------------------------
regions = ["West", "East", "South"]; stores = ["S1", "S2"]; months = pd.period_range("2026-01", periods=4, freq="M")
idx = pd.MultiIndex.from_product([regions, stores, months], names=["region", "store", "month"])
sales = pd.DataFrame({"revenue": rng.gamma(5, 100, len(idx)).round(1), "units": rng.integers(10, 100, len(idx))}, index=idx).sort_index()
print(sales.head(5))
print("\n.loc[('West','S1')] rows:", len(sales.loc[("West", "S1")]), "| xs month 2026-02 (all stores):", sales.xs(pd.Period("2026-02"), level="month").shape)
wide = sales["revenue"].unstack("month"); print("\nunstack(month):\n", wide.round(0))
print("\nrevenue by region (groupby level):\n", sales.groupby(level="region")["revenue"].sum().round(0))

# ---- 2. Join cardinality & the row-explosion bug -------------------------------------------
customers = pd.DataFrame({"cust_id": range(1, 6), "segment": ["A", "B", "A", "C", "B"]})
orders = pd.DataFrame({"order_id": range(100, 108), "cust_id": [1, 1, 2, 3, 3, 3, 9, 5], "amount": rng.integers(10, 99, 8)})
rows = {}
for how in ("inner", "left", "right", "outer"):
    rows[how] = len(orders.merge(customers, on="cust_id", how=how))
print("\nrow counts per join type:", rows)
dup_customers = pd.concat([customers, customers.iloc[[0]]])             # duplicated key in dimension table
exploded = orders.merge(dup_customers, on="cust_id", how="left")
print(f"orders={len(orders)} -> after join with dup key: {len(exploded)} rows (revenue double counted: {exploded.amount.sum()} vs {orders.amount.sum()})")
try:
    orders.merge(dup_customers, on="cust_id", how="left", validate="many_to_one")
except Exception as e:
    print("validate caught it ->", type(e).__name__)

# ---- 3. Anti-join with indicator ------------------------------------------------------------
m = customers.merge(orders[["cust_id"]].drop_duplicates(), on="cust_id", how="left", indicator=True)
print("customers with no orders:", m.loc[m["_merge"] == "left_only", "cust_id"].tolist())

# ---- 4. merge_asof: point-in-time features ---------------------------------------------------
trades = pd.DataFrame({"time": pd.to_datetime("2026-06-01 09:30") + pd.to_timedelta(np.sort(rng.integers(0, 600, 8)), unit="s"),
                       "ticker": "ABC", "qty": rng.integers(1, 50, 8)})
quotes = pd.DataFrame({"time": pd.to_datetime("2026-06-01 09:30") + pd.to_timedelta(np.arange(0, 600, 100), unit="s"),
                       "ticker": "ABC", "price": 100 + np.arange(6) * 0.5})
asof = pd.merge_asof(trades.sort_values("time"), quotes.sort_values("time"), on="time", by="ticker", direction="backward")
print("\nmerge_asof (backward):\n", asof[["time", "qty", "price"]])
assert (asof["time"] >= asof["time"].min()).all()

plt.figure(figsize=(5.5, 3.5)); plt.bar(rows.keys(), rows.values(), color="#2980b9"); plt.title("Rows returned by join type"); plt.ylabel("rows")
savefig("join_row_counts.png")
