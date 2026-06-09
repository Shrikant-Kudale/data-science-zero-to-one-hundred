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

# ---- 1. Synthetic daily series: trend + weekly + yearly seasonality + noise ------------------------
idx = pd.date_range("2023-01-01", "2025-12-31", freq="D"); n = len(idx); tt = np.arange(n)
y = pd.Series(200 + 0.15 * tt + 25 * np.sin(2 * np.pi * idx.dayofweek / 7) + 30 * np.sin(2 * np.pi * tt / 365.25)
              + rng.normal(0, 8, n), index=idx, name="sales")

# ---- 2. Resampling ------------------------------------------------------------------------------------
print("weekly sum:\n", y.resample("W").sum().head(3).round(0).to_string())
print("monthly mean:\n", y.resample("MS").mean().head(3).round(1).to_string())
gappy = y.drop(y.index[10:15]); print("missing timestamps detected:", gappy.asfreq("D").isna().sum(), "| interpolated NaNs left:", gappy.asfreq("D").interpolate().isna().sum())

# ---- 3. Classical weekly decomposition ----------------------------------------------------------------------
trend = y.rolling(7, center=True).mean()
seasonal_week = (y - trend).groupby(y.index.dayofweek).mean(); seasonal = pd.Series(y.index.dayofweek.map(seasonal_week), index=y.index)
resid = y - trend - seasonal
print("weekday effect:", seasonal_week.round(1).to_dict(), "| residual std:", round(resid.std(), 2))

# ---- 4. ACF and differencing ---------------------------------------------------------------------------------
acf = [y.autocorr(k) for k in range(1, 15)]
print("ACF lag 1/7/14:", round(acf[0], 3), round(acf[6], 3), round(acf[13], 3))
d1 = y.diff().dropna(); d7 = y.diff(7).dropna()
print(f"std raw={y.std():.1f}  diff1={d1.std():.1f}  seasonal-diff7={d7.std():.1f}")
try:
    from statsmodels.tsa.stattools import adfuller
    print("ADF p-value raw:", round(adfuller(y)[1], 3), "| after diff(7)+diff(1):", round(adfuller(y.diff(7).diff().dropna())[1], 5))
except ImportError:
    print("statsmodels not installed; skipping ADF")

# ---- 5. Honest forecast evaluation with rolling-origin ---------------------------------------------------------
h = 28; errs = {"naive": [], "seasonal_naive": [], "seas_naive+drift": []}
for cut in range(n - 8 * h, n - h + 1, h):
    train, test = y.iloc[:cut], y.iloc[cut:cut + h]
    errs["naive"].append(np.abs(test.values - train.iloc[-1]).mean())
    sn = np.array([train.iloc[-7 + (i % 7)] for i in range(h)]); errs["seasonal_naive"].append(np.abs(test.values - sn).mean())
    drift = (train.iloc[-1] - train.iloc[-29]) / 28 * np.arange(1, h + 1)
    errs["seas_naive+drift"].append(np.abs(test.values - (sn + drift)).mean())
print("rolling-origin MAE:", {k: round(float(np.mean(v)), 2) for k, v in errs.items()})

fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for a, s, t in zip(ax, [y, trend, seasonal, resid], ["observed", "trend", "weekly seasonal", "residual"]):
    a.plot(s.iloc[:200]); a.set_ylabel(t, fontsize=8)
savefig("decomposition.png")
