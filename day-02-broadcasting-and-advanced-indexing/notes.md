# Day 02 — Broadcasting & Advanced Indexing

## Broadcasting rules (memorize)
Align shapes from the **trailing** dimension. Two dims are compatible iff they are equal or one is 1. Missing leading dims are treated as 1. The result dim is the max. Broadcasting never copies; it sets stride 0 along expanded axes.

Example: $(500,1,8) - (1,10,8) \to (500,10,8)$.

## Pairwise squared Euclidean distance (classic interview question)
Naive broadcast: $D_{ij}=\sum_k (x_{ik}-c_{jk})^2$ allocates an $n\times m\times d$ tensor.
Algebraic form: $\|x-c\|^2=\|x\|^2+\|c\|^2-2x^\top c$ => $D = \mathbf{x_2}\mathbf{1}^\top+\mathbf{1}\mathbf{c_2}^\top-2XC^\top$, memory $O(nm)$ and uses BLAS. Caveat: catastrophic cancellation can yield tiny negatives => `np.maximum(D, 0)`.

## Advanced indexing
- Integer-array indexing: result shape = index shape; **copy**.
- Boolean masks: 1-D result for full-shape masks; **copy**.
- Mixing slices and arrays: adjacent advanced indices => their dims placed in position; separated by a slice => advanced dims go **first** (famous gotcha).
- `np.ix_` builds an open mesh for cross-product selection; `np.take_along_axis` pairs with `argsort/argpartition`.
- `argpartition(D, k)` is $O(n)$ average vs $O(n\log n)$ for full sort — top-k retrieval.

## Trade-offs
Memory vs readability: a 3-D broadcast is the clearest, the Gram-matrix trick is the fastest at scale. For $n=10^6$, neither fits => chunk, or use ANN (FAISS/ScaNN).

## FAANG Q&A
**Q1. Shape of `(5,1) + (3,)`?** `(5,3)`.
**Q2. How do you standardize columns with no loops?** `(X - X.mean(0)) / X.std(0)`; mean has shape `(d,)` which broadcasts across rows.
**Q3. kNN brute-force complexity and trick?** $O(nmd)$ compute; use Gram trick + `argpartition` for top-k to avoid $O(m\log m)$ sorting.
**Q4. `a[[0,1],[1,2]]` vs `a[np.ix_([0,1],[1,2])]`?** First returns 2 elements (pairs); second returns a 2x2 block.
**Q5. When does assignment through fancy indexing misbehave?** Duplicate indices: `a[[0,0]] += 1` increments once; use `np.add.at` or `np.bincount`.
