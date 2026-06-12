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


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler

# ---- 1. Gaussian bulk + injected outliers ---------------------------------------------------------------
n_in, n_out = 970, 30
x = np.r_[rng.normal(50, 5, n_in), rng.normal(50, 5, n_out) + rng.choice([-1, 1], n_out) * rng.uniform(25, 60, n_out)]
truth = np.r_[np.zeros(n_in, bool), np.ones(n_out, bool)]
def score(flag, name):
    tp = (flag & truth).sum(); print(f"{name:18s} flagged={flag.sum():4d} precision={tp/max(flag.sum(),1):.2f} recall={tp/truth.sum():.2f}")
z = (x - x.mean()) / x.std(); score(np.abs(z) > 3, "z-score |z|>3")
q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1; score((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr), "IQR 1.5")
med = np.median(x); mad = np.median(np.abs(x - med)); mz = 0.6745 * (x - med) / mad; score(np.abs(mz) > 3.5, "modified z (MAD)")
iso = IsolationForest(contamination=0.03, random_state=0).fit(x[:, None]); score(iso.predict(x[:, None]) == -1, "IsolationForest")
print(f"masking: std with outliers={x.std():.2f} vs clean={x[~truth].std():.2f} | 1.4826*MAD={1.4826*mad:.2f}")

# ---- 2. Multivariate: Mahalanobis (classical vs robust MCD) -------------------------------------------------
from sklearn.covariance import MinCovDet, EmpiricalCovariance
Xm = rng.multivariate_normal([0, 0], [[1, .9], [.9, 1]], 500); Xm[:25] = rng.multivariate_normal([2, -2], 0.05 * np.eye(2), 25)  # off-axis cluster
for name, est in [("classical", EmpiricalCovariance()), ("robust MCD", MinCovDet(random_state=0))]:
    d2 = est.fit(Xm).mahalanobis(Xm); thr = 9.21           # chi2(2) 99%
    print(f"{name:11s} flagged {int((d2 > thr).sum()):3d} / 25 planted outliers in first 25 rows: {int((d2[:25] > thr).sum())}")

# ---- 3. Scalers -----------------------------------------------------------------------------------------------
X1 = x[:, None]; ss, rs = StandardScaler().fit(X1), RobustScaler().fit(X1)
print(f"StandardScaler mean/scale: {ss.mean_[0]:.2f}/{ss.scale_[0]:.2f} | RobustScaler center/scale: {rs.center_[0]:.2f}/{rs.scale_[0]:.2f}")
print("share of inliers within [-3,3] after scaling: standard", (np.abs(ss.transform(X1)[~truth]) < 3).mean().round(3), "robust", (np.abs(rs.transform(X1)[~truth]) < 3).mean().round(3))

# ---- 4. Winsorization -------------------------------------------------------------------------------------------
lo, hi = np.percentile(x, [1, 99]); xw = np.clip(x, lo, hi); print(f"winsorized std {xw.std():.2f} (orig {x.std():.2f})")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(x, bins=60, color="#7f8c8d"); ax[0].axvline(med - 3.5 * mad / 0.6745, c="r", ls="--"); ax[0].axvline(med + 3.5 * mad / 0.6745, c="r", ls="--"); ax[0].set_title("Distribution + MAD fences")
ax[1].scatter(Xm[:, 0], Xm[:, 1], s=6, c=(np.arange(500) < 25), cmap="coolwarm"); ax[1].set_title("Multivariate outliers invisible marginally")
savefig("outliers.png")
