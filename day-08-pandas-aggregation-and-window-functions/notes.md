# Day 08 — Aggregation & Window Functions

## Split–Apply–Combine
`groupby` splits by key; then **agg** (reduces: one row per group), **transform** (same shape as input: e.g., group z-score, group mean fill), **filter** (keep/drop whole groups), **apply** (anything; slow).
Named aggregation: `df.groupby("g").agg(total=("x","sum"), p95=("x", lambda s: s.quantile(.95)))`.

## Windows
- `rolling(w)`: fixed width; `rolling("7D")` time-based. `min_periods` controls warm-up NaNs.
- `expanding()`: cumulative from start.
- `ewm(alpha=)`: $y_t=\alpha x_t+(1-\alpha)y_{t-1}$; half-life $h$: $\alpha=1-2^{-1/h}$. Effective window $\approx(2-\alpha)/\alpha$.
- `shift`, `diff`, `pct_change`, `rank`, `cumsum/cummax`.
- Rolling statistic over window $w$: $O(nw)$ naive, pandas uses online updates for mean/sum/var ($O(n)$).

## Statistical cautions
- **Leakage:** a centered rolling window or group statistics computed on the whole dataset leak the future. Use trailing windows and `shift(1)` for features at time $t$.
- **Simpson's paradox:** aggregated trend can reverse within groups — always slice.
- Weighted means need `np.average(weights=)`; mean of means != overall mean when group sizes differ.

## SQL mapping
`transform("mean")` = `AVG() OVER (PARTITION BY g)`; `rank` = `RANK()`; `shift` = `LAG`; `rolling` = `ROWS BETWEEN`.

## FAANG Q&A
**Q1. Top-3 products per category?** `df.sort_values("rev", ascending=False).groupby("cat").head(3)` or `nlargest` in agg.
**Q2. 7-day moving average of DAU with gaps in dates?** `set_index("date").asfreq("D")` then `rolling("7D")` or reindex first.
**Q3. Difference between `agg("mean")` and `transform("mean")`?** Shape: per-group vs broadcast back to rows.
**Q4. Why can `groupby` drop NaN keys?** `dropna=True` default; set `dropna=False` to keep.
**Q5. Compute customer's running spend share of lifetime total?** `cumsum / transform("sum")` (note: uses future total — fine for analysis, leaks for modeling).
