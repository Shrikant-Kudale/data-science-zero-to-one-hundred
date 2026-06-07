# Day 07 — Hierarchical Indexing & Merging

## MultiIndex
Index with multiple levels (a tuple key). Key operations: `set_index([a,b])`, `.loc[(a,b)]`, `.xs(key, level=)`, `unstack` (long->wide), `stack`, `swaplevel`, `reorder_levels`, `groupby(level=)`. **Sort the index** (`sort_index`) before slicing, otherwise `PerformanceWarning`/`UnsortedIndexError`.

## Relational join semantics (relational algebra)
Let $L$ have $n_L$ rows, $R$ have $n_R$. For key $k$ with $a_k$ left and $b_k$ right matches, an inner join returns $\sum_k a_kb_k$ rows. **Duplicated keys multiply rows** — the #1 cause of silently wrong metrics (double counting revenue).
- inner / left / right / outer / cross; semi-join (`isin`) and anti-join (`~isin`, or `indicator=True`).
- Always use `validate="one_to_one" | "one_to_many" | "many_to_one"` and check row counts.
- `merge_asof`: nearest-key (backward/forward/nearest) join on sorted key, e.g. trades with the latest prior quote; supports `by=` and `tolerance=`. The point-in-time-correct join for feature stores.
- `concat` (stack axes) vs `merge` (key-based) vs `join` (index-based).

## Complexity
Hash join $O(n_L+n_R)$ average; sort-merge join $O((n_L+n_R)\log)$; merge on an index can skip hashing.

## FAANG Q&A
**Q1. Left join of orders (100k) with customers returns 130k rows. Why?** Duplicate keys in the right table. Fix: dedupe/aggregate first, `validate="many_to_one"`.
**Q2. How to find customers with no orders?** Left join with `indicator=True` and filter `_merge=="left_only"`, or `~customers.id.isin(orders.cust_id)`.
**Q3. Prevent data leakage in a join of events and features?** Use `merge_asof` with `direction="backward"` so only past feature values attach.
**Q4. `unstack` vs `pivot_table`?** `unstack` needs a unique index; `pivot_table` aggregates duplicates.
**Q5. Join on float keys?** Never; round/convert to integer or use tolerance in `merge_asof`.
