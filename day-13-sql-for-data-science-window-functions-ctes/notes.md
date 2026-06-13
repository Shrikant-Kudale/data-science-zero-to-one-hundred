# Day 13 — SQL: Window Functions & CTEs

## Logical execution order
`FROM/JOIN → WHERE → GROUP BY → HAVING → WINDOW → SELECT → DISTINCT → ORDER BY → LIMIT`. Consequence: you cannot filter on a window result in `WHERE`; wrap in a CTE/subquery.

## Window anatomy
`fn(expr) OVER (PARTITION BY p ORDER BY o ROWS|RANGE BETWEEN a AND b)`
- Ranking: `ROW_NUMBER` (unique), `RANK` (gaps on ties), `DENSE_RANK` (no gaps), `NTILE(k)`.
- Navigation: `LAG/LEAD`, `FIRST_VALUE/LAST_VALUE` (watch default frame = up to current row!).
- Aggregates over windows: running sums, moving averages, share-of-total `x / SUM(x) OVER ()`.
- **ROWS vs RANGE:** ROWS counts physical rows; RANGE groups peers with equal ORDER BY value.

## CTEs
Readability, reuse, recursion (`WITH RECURSIVE` for hierarchies / date spines). Often inlined by the optimizer (engine-dependent: Postgres ≥12 inlines; some warehouses materialize).

## Classic interview patterns
1. Top-N per group → `ROW_NUMBER() OVER (PARTITION BY g ORDER BY m DESC)` then filter.
2. MoM growth → `LAG`.
3. Sessionization → flag gap > 30 min with `LAG`, cumulative `SUM(flag)`.
4. Retention/cohorts → first activity month + activity months join.
5. Gaps-and-islands → `date - ROW_NUMBER()` constant within consecutive streaks.
6. Dedup → `ROW_NUMBER() ... = 1`.
7. Median → `PERCENTILE_CONT(.5)` (Postgres) or NTILE/row-number trick.

## Performance
Index join/filter columns; avoid functions on indexed columns in `WHERE` (non-sargable); `EXISTS` vs `IN`; `COUNT(DISTINCT)` is costly; filter early; check `EXPLAIN`.

## FAANG Q&A
**Q1. `RANK` vs `DENSE_RANK` for [100,100,90]?** 1,1,3 vs 1,1,2.
**Q2. `COUNT(*)` vs `COUNT(col)`?** Rows vs non-NULLs. NULL semantics: `NULL = NULL` is unknown; use `IS NULL`; `NOT IN` with NULLs returns empty.
**Q3. WHERE vs HAVING?** Row filter before grouping vs group filter after.
**Q4. 7-day rolling active users?** Distinct counts can't be windowed additively — build a daily-user table and use a range self-join or `COUNT(DISTINCT)` per date over date-spine.
**Q5. Why is `LAST_VALUE` surprising?** Default frame ends at CURRENT ROW; specify `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`.
