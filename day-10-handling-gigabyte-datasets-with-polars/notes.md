# Day 10 — Gigabyte-Scale Data with Polars

## Why Polars is fast
1. **Columnar Apache Arrow memory** — cache/SIMD friendly, zero-copy interchange.
2. **Rust, multi-threaded** — operations parallelized across cores (pandas is mostly single-threaded).
3. **Lazy evaluation + query optimizer** — `scan_csv/scan_parquet` build a logical plan; the optimizer applies *predicate pushdown* (filter early), *projection pushdown* (read only needed columns), common-subexpression elimination, then executes. `.collect(engine="streaming")` processes data in batches => larger-than-RAM.
4. **Expressions**: `pl.col("x").mean().over("g")` = window function; no Python UDF per row.

## Scaling ladder
| Size | Tool |
|---|---|
| < 1 GB | pandas |
| 1–100 GB (single machine) | Polars / DuckDB / pandas+chunks / Parquet |
| > 100 GB / multi-node | Spark, Dask, BigQuery, Snowflake |
Parquet (columnar, compressed, typed, row-group statistics) beats CSV by 5-50x on read time and size.

## Chunked aggregation math
Mean, sum, count, min, max, variance are **decomposable**: keep partial `(count, sum, sum of squares)` per chunk and merge. Median/quantiles are *not* — need sketches (t-digest) or two-pass.
Merge variance via Chan's formula: $M_2=M_{2,a}+M_{2,b}+\delta^2\frac{n_an_b}{n_a+n_b}$.

## FAANG Q&A
**Q1. How would you compute group-wise mean over a 50 GB file on 16 GB RAM?** Parquet + lazy scan with projection pushdown + streaming engine, or chunked partial sums.
**Q2. Why can't you chunk a median?** Not decomposable; use approximate quantile sketches.
**Q3. Polars vs pandas semantic differences?** No index, strict dtypes, expressions, null vs NaN distinction, lazy API.
**Q4. When still use pandas?** Ecosystem integration (sklearn, statsmodels), small data, rich time-series tooling.
**Q5. What is predicate pushdown?** Applying filters at the scan so fewer rows/row-groups are read at all.
