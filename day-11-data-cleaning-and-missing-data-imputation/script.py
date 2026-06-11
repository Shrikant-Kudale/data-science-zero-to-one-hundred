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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# ---- 1. Correlated data and a MAR mechanism --------------------------------------------------------
n = 3000; cov = np.array([[1, .7, .5, .3], [.7, 1, .4, .2], [.5, .4, 1, .6], [.3, .2, .6, 1]])
X = rng.multivariate_normal([10, 20, 30, 40], cov * 9, size=n)
p_miss = 1 / (1 + np.exp(-(X[:, 1] - 20) / 1.5 - 1.5))               # col0 missing more when col1 is high (MAR)
M = rng.random(n) < p_miss * 0.6
Xm = X.copy(); Xm[M, 0] = np.nan
print(f"missing in col0: {M.mean():.1%} | mean of col1 among missing {X[M,1].mean():.2f} vs observed {X[~M,1].mean():.2f} (-> not MCAR)")

# ---- 2. Benchmark imputers on the masked cells ----------------------------------------------------------
models = {"mean": SimpleImputer(strategy="mean"), "median": SimpleImputer(strategy="median"),
          "knn(k=10)": KNNImputer(n_neighbors=10), "iterative(BayesRidge)": IterativeImputer(max_iter=10, random_state=0)}
rows = []
for name, imp in models.items():
    Xi = imp.fit_transform(Xm); err = Xi[M, 0] - X[M, 0]
    rows.append((name, np.sqrt((err ** 2).mean()), Xi[:, 0].mean(), Xi[:, 0].std(), np.corrcoef(Xi[:, 0], Xi[:, 1])[0, 1]))
res = pd.DataFrame(rows, columns=["method", "RMSE_masked", "mean_col0", "std_col0", "corr(c0,c1)"]).set_index("method")
print(res.round(3)); print(f"truth: mean={X[:,0].mean():.3f} std={X[:,0].std():.3f} corr={np.corrcoef(X[:,0], X[:,1])[0,1]:.3f}")
print("complete-case mean (biased under MAR):", Xm[~M, 0].mean().round(3))

# ---- 3. Stochastic regression imputation keeps variance --------------------------------------------------
A = np.c_[np.ones((~M).sum()), X[~M][:, 1:]]; beta = np.linalg.lstsq(A, X[~M, 0], rcond=None)[0]
sd = (X[~M, 0] - A @ beta).std(); B = np.c_[np.ones(M.sum()), X[M][:, 1:]]
det = B @ beta; sto = det + rng.normal(0, sd, M.sum())
print(f"std of imputed cells: deterministic={det.std():.3f} stochastic={sto.std():.3f} true={X[M,0].std():.3f}")

# ---- 4. Cleaning checklist on a dirty frame -----------------------------------------------------------------
dirty = pd.DataFrame({"age": [25, -3, 41, 41, 230, np.nan], "city": [" pune", "Pune", "PUNE ", "Mumbai", "mumbai", "Pune"],
                      "joined": ["2026-01-05", "05/01/2026", "2026-02-30", "2026-03-01", "2026-03-01", None]})
clean = dirty.assign(city=dirty.city.str.strip().str.title(), age=dirty.age.where(dirty.age.between(0, 110)),
                     joined=pd.to_datetime(dirty.joined, errors="coerce", format="mixed")).drop_duplicates()
print("\ncleaned:\n", clean)

res["RMSE_masked"].plot.bar(figsize=(6, 3.5), color="#2980b9", title="Imputation RMSE (MAR, col0)"); savefig("imputation_rmse.png")
