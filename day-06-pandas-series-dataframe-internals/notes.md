# Day 06 — Pandas Series & DataFrame Internals

## Mental model
`DataFrame` = column-oriented container of typed arrays (historically grouped into 2-D "blocks" by dtype by the BlockManager) + row `Index` + column `Index`. A `Series` = 1-D array + Index. Operations **align on index labels** first, then compute (missing => NaN).

## Memory
- `object` dtype stores Python pointers: strings cost ~50-100 B each. Use `category` (integer codes + small dictionary) for low-cardinality text: memory $\approx n\cdot\lceil\log_{256}k\rceil+k\cdot\text{len}$.
- Down-cast numerics (`pd.to_numeric(downcast=...)`), use nullable dtypes (`Int64`, `boolean`) or Arrow-backed dtypes (`string[pyarrow]`) when needed.
- `df.memory_usage(deep=True)` — `deep=True` is required to include object payloads.

## Performance ladder (fast -> slow)
vectorized column ops / built-in methods > `.str`/`.dt` accessors > `np.where`/`map` > `apply(axis=1)` > `iterrows`. `apply(axis=1)` boxes each row into a Series: ~100-1000x slower than the vectorized equivalent.

## Copy semantics
Chained assignment `df[df.a>0]["b"]=1` may modify a temporary copy (the infamous `SettingWithCopyWarning`). With **Copy-on-Write** (default in pandas 3), any derived object behaves as a copy lazily; chained assignment never updates the parent. Use `df.loc[mask, "b"] = 1`.

## Index alignment pitfalls
`s1 + s2` with different labels => union index with NaN. Use `.add(..., fill_value=0)` when appropriate. Duplicate index labels => reindexing errors and merge blow-ups.

## FAANG Q&A
**Q1. Why is `groupby().apply(lambda)` slow?** Python-level function per group; prefer `agg` with built-ins, `transform`, or vectorized logic.
**Q2. How do you load a 10 GB CSV on a 16 GB laptop?** `usecols`, `dtype` mapping, `category`, `chunksize`, or switch to Parquet/Polars/DuckDB.
**Q3. `loc` vs `iloc` vs `at`?** Label-based (inclusive slice) / position-based (exclusive) / scalar fast access.
**Q4. What is a view in pandas?** Historically unpredictable; CoW removes this ambiguity — treat everything as a copy.
**Q5. Why is a sorted/unique index faster?** Enables binary search / hash lookup; `is_monotonic_increasing` allows slice lookups.
