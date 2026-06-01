# Day 01 — NumPy Vectorization & Memory Layouts

## Why this matters
Every downstream library (pandas, scikit-learn, PyTorch) sits on the `ndarray`. Interviewers use it to probe whether you understand *why* code is slow, not just that it is.

## Core model
An `ndarray` = one contiguous byte buffer + metadata `(dtype, shape, strides)`.
Element address: $\text{addr}(i_1,\dots,i_k)=\text{base}+\sum_j i_j\,s_j$.
- C-order (row-major) strides for shape $(m,n)$, itemsize $b$: $(n b,\; b)$.
- F-order (column-major): $(b,\; m b)$.
- Transpose, slicing with a step, `reshape` (when possible), `np.newaxis` are **views**: only metadata changes.
- Fancy (integer-array) indexing and boolean masks **copy**.

## Why vectorization is fast
1. The Python interpreter loop costs ~50-100 ns per bytecode op; a C loop costs ~1 ns per element.
2. Contiguous memory => cache-line friendly (64B = 8 float64) and SIMD (AVX2/AVX-512).
3. BLAS (`np.dot`, `@`) adds blocking, multithreading: $O(n^3)$ matmul at near peak FLOPs.

Traversal cost: summing along the contiguous axis touches memory sequentially; summing along the strided axis causes cache misses. Same $O(mn)$ work, different constants (often 2-10x).

## Memory arithmetic
`float64` = 8B, `float32` = 4B, `float16` = 2B. A $10^8$-row float64 column = 800 MB. Down-casting is the cheapest "scaling" trick, but mind precision: float32 has ~7 decimal digits, so summing $10^7$ float32 numbers naively loses accuracy (use `dtype=np.float64` accumulators or pairwise summation, which NumPy uses for `sum`).

## Trade-offs
| Choice | Pro | Con |
|---|---|---|
| Views | zero-copy | silent aliasing bugs on in-place writes |
| `np.ascontiguousarray` | fast downstream ops | one-time copy |
| Temporary arrays in chained expressions (`a*b+c`) | readable | extra allocations; use `out=`, `numexpr`, or `np.einsum` |
| float32 | 2x memory/bandwidth | precision loss |

## FAANG-style Q&A
**Q1. `a[::2]` vs `a[[0,2,4]]` — what is the difference?** First is a view (stride doubled), second is a copy via fancy indexing. Mutating the view mutates `a`.

**Q2. Why can `A.sum(axis=0)` be slower than `A.sum(axis=1)` for a C-ordered array?** Axis 1 walks contiguous memory; axis 0 jumps by `n*itemsize` bytes per step => poor spatial locality. NumPy mitigates by iterating in memory order, but reductions over the outer axis still tend to be slower for huge arrays.

**Q3. How would you compute a rolling mean over 10M points without a Python loop?** `np.lib.stride_tricks.sliding_window_view` (view, no copy) + `.mean(-1)` — but this is $O(nw)$; the $O(n)$ way is a cumulative sum difference: $(c_{i+w}-c_i)/w$.

**Q4. What does `np.shares_memory` vs `np.may_share_memory` do?** The first solves exactly (can be expensive); the second is a fast bounds check that may yield false positives.

**Q5. Is `x += y` the same as `x = x + y`?** No: in-place writes into the existing buffer (no new allocation, and it affects other views), and dtype casting rules differ (`int_array += float` raises).
