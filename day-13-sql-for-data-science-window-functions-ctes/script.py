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


import sqlite3
import pandas as pd

print("SQLite version:", sqlite3.sqlite_version, "(window functions need >= 3.25)")
con = sqlite3.connect(":memory:")
n_users, n_orders = 600, 6000
users = pd.DataFrame({"user_id": range(n_users), "country": rng.choice(["IN", "US", "UK", "DE"], n_users, p=[.4, .3, .2, .1])})
start = pd.Timestamp("2025-01-01")
signup = start + pd.to_timedelta(rng.integers(0, 240, n_users), unit="D")
uid = rng.integers(0, n_users, n_orders)
dates = signup[uid] + pd.to_timedelta(rng.exponential(60, n_orders).astype(int), unit="D")
orders = pd.DataFrame({"order_id": range(n_orders), "user_id": uid, "order_date": dates.strftime("%Y-%m-%d"),
                       "category": rng.choice(["books", "toys", "tech", "home"], n_orders), "amount": rng.gamma(3, 25, n_orders).round(2)})
orders = orders[orders.order_date < "2026-01-01"]
users.to_sql("users", con, index=False); orders.to_sql("orders", con, index=False)
q = lambda sql: pd.read_sql_query(sql, con)

print("\n[1] Top-2 users per category by spend (ROW_NUMBER)")
print(q("""WITH spend AS (SELECT category, user_id, SUM(amount) AS total FROM orders GROUP BY category, user_id),
ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY total DESC) AS rn FROM spend)
SELECT category, user_id, ROUND(total,1) AS total, rn FROM ranked WHERE rn <= 2 ORDER BY category, rn"""))

print("\n[2] Month-over-month revenue growth (LAG) + running total")
mom = q("""WITH m AS (SELECT substr(order_date,1,7) AS month, SUM(amount) AS rev FROM orders GROUP BY 1)
SELECT month, ROUND(rev,0) AS rev, ROUND(100.0*(rev - LAG(rev) OVER (ORDER BY month)) / LAG(rev) OVER (ORDER BY month),1) AS mom_pct,
ROUND(SUM(rev) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),0) AS running_total FROM m ORDER BY month""")
print(mom.head(6))

print("\n[3] Share of category revenue within country (window over join)")
print(q("""SELECT u.country, o.category, ROUND(SUM(o.amount),0) AS rev,
ROUND(100.0*SUM(o.amount)/SUM(SUM(o.amount)) OVER (PARTITION BY u.country),1) AS pct_of_country
FROM orders o JOIN users u USING(user_id) GROUP BY u.country, o.category ORDER BY u.country, rev DESC LIMIT 6"""))

print("\n[4] Gaps-and-islands: longest consecutive-month streak per user (top 3)")
print(q("""WITH act AS (SELECT DISTINCT user_id, CAST(substr(order_date,1,4) AS INT)*12 + CAST(substr(order_date,6,2) AS INT) AS m FROM orders),
grp AS (SELECT user_id, m, m - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY m) AS g FROM act)
SELECT user_id, COUNT(*) AS streak FROM grp GROUP BY user_id, g ORDER BY streak DESC LIMIT 3"""))

print("\n[5] Cohort retention (CTEs)")
coh = q("""WITH first AS (SELECT user_id, MIN(substr(order_date,1,7)) AS cohort FROM orders GROUP BY user_id),
act AS (SELECT DISTINCT user_id, substr(order_date,1,7) AS m FROM orders),
j AS (SELECT f.cohort, a.user_id, (CAST(substr(a.m,1,4) AS INT)-CAST(substr(f.cohort,1,4) AS INT))*12 + CAST(substr(a.m,6,2) AS INT)-CAST(substr(f.cohort,6,2) AS INT) AS age
      FROM first f JOIN act a USING(user_id))
SELECT cohort, age, COUNT(DISTINCT user_id) AS users FROM j GROUP BY cohort, age""")
ret = coh.pivot(index="cohort", columns="age", values="users"); ret = ret.div(ret[0], axis=0)
print(ret.iloc[:5, :6].round(2))

print("\n[6] NULL trap: NOT IN with NULL returns nothing ->",
      q("SELECT COUNT(*) AS n FROM users WHERE user_id NOT IN (SELECT 1 UNION SELECT NULL)").iloc[0, 0], "rows")

plt.figure(figsize=(7, 4)); plt.imshow(ret.iloc[:, :9].values, aspect="auto", cmap="Blues", vmin=0, vmax=1); plt.colorbar(label="retention")
plt.yticks(range(len(ret)), ret.index, fontsize=7); plt.xlabel("months since first order"); plt.title("Cohort retention heatmap"); savefig("cohort_retention.png")
