# Day 09 — Time-Series Analysis & Resampling

## Decomposition
Additive: $y_t=T_t+S_t+R_t$. Multiplicative: $y_t=T_tS_tR_t$ (log-transform to make additive). Classical method: trend = centered moving average of length = period; seasonal = mean of detrended values per phase; residual = remainder.

## Stationarity
Weak stationarity: constant mean, variance, autocovariance depending only on lag. Fix by differencing ($\nabla y_t=y_t-y_{t-1}$, seasonal $\nabla_sy_t=y_t-y_{t-s}$), log, detrend. Tests: ADF (H0: unit root), KPSS (H0: stationary).
ACF: $\rho_k=\text{Cov}(y_t,y_{t-k})/\text{Var}(y_t)$. AR($p$): PACF cuts off at $p$; MA($q$): ACF cuts off at $q$.
ARIMA$(p,d,q)$: $\phi(B)(1-B)^dy_t=\theta(B)\varepsilon_t$.

## Resampling
Downsample (aggregate: sum for flows, mean/last for stocks/levels), upsample (ffill/interpolate). Choose `closed`/`label` carefully; timezone/DST traps; use `asfreq` to expose missing timestamps.

## Evaluation (very commonly botched)
- **Never shuffle**. Use a time-ordered split / rolling-origin (expanding window) CV (`TimeSeriesSplit`).
- Always beat the baselines: naive ($\hat y_{t+1}=y_t$) and seasonal-naive ($\hat y_{t+h}=y_{t+h-s}$).
- Metrics: MAE, RMSE, MAPE (fails near 0), sMAPE, MASE (scale-free vs naive).
- Features: lags, rolling stats (shifted!), calendar, holidays, exogenous covariates available at forecast time.

## FAANG Q&A
**Q1. Why is random k-fold wrong for time series?** Leakage from the future; autocorrelation makes validation optimistic.
**Q2. Sales spike every Black Friday; how to model?** Holiday regressors / dummy variables, or Prophet/GBM with calendar features; avoid treating as outlier.
**Q3. Differencing vs detrending?** Differencing handles stochastic trends (unit root); regression detrending handles deterministic trends.
**Q4. How to forecast 1000 SKUs?** Global model (LightGBM / DeepAR) with item embeddings + hierarchical reconciliation.
**Q5. Detect anomaly in metric?** Forecast + prediction interval (e.g., STL residual $>3$ MAD), or CUSUM for shifts.
