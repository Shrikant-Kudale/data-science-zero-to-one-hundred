# Day 22 — t-SNE & UMAP

## Curse of dimensionality
In $d$ dims distances concentrate: $\frac{d_{max}-d_{min}}{d_{min}}\to0$; volume of the unit ball $\to0$; neighbors become uninformative. Intrinsic dimension is usually far below ambient $d$.

## t-SNE (van der Maaten & Hinton)
High-D affinities: $p_{j|i}=\frac{\exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\|x_i-x_k\|^2/2\sigma_i^2)}$, $\sigma_i$ set so perplexity $2^{H(P_i)}$ equals a target (≈ effective neighbors, 5–50). Symmetrize $p_{ij}=\frac{p_{j|i}+p_{i|j}}{2n}$.
Low-D affinities use Student-t (1 dof): $q_{ij}=\frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}}$ — heavy tails relieve the "crowding problem".
Minimize $KL(P\|Q)=\sum p_{ij}\log\frac{p_{ij}}{q_{ij}}$ by gradient descent: $\frac{\partial C}{\partial y_i}=4\sum_j(p_{ij}-q_{ij})(y_i-y_j)(1+\|y_i-y_j\|^2)^{-1}$. Cost $O(n\log n)$ with Barnes–Hut / FFT.
KL is asymmetric: penalizes placing neighbors far apart (preserves **local** structure) but tolerates false neighbors.

## UMAP
Builds a fuzzy kNN graph (local metric scaled by distance to nearest neighbor), then optimizes a low-D layout by minimizing cross-entropy between fuzzy simplicial sets. Parameters: `n_neighbors` (local vs global), `min_dist` (cluster tightness). Faster, better global structure than t-SNE (esp. with PCA/spectral init), supports `transform` for new points and supervised mode.

## What you may NOT conclude from a t-SNE/UMAP plot
- Cluster sizes and between-cluster distances are largely meaningless. Different perplexity ⇒ different picture; random seeds differ. Not a clustering algorithm; do clustering in original/PCA space. Use for exploration, QA of embeddings, label-noise detection — not as model features (non-parametric/unstable).
- Always: standardize, PCA to ~50 dims first, run multiple perplexities, check **trustworthiness** (neighborhood preservation).

## PCA vs t-SNE vs UMAP
| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Linear | yes | no | no |
| Global structure | good | poor | medium |
| Speed | fastest | slow | fast |
| New data | yes | no | yes |
| Deterministic | yes | no | no |

## FAANG Q&A
**Q1. Why Student-t in low-D?** Heavy tails allow moderately distant points to sit far apart => resolves crowding.
**Q2. Can I use t-SNE embeddings for a classifier?** Not recommended; no out-of-sample map, unstable.
**Q3. Perplexity too low/high?** Too low: fragmented clusters; too high: merges clusters.
**Q4. How to evaluate a DR method?** Trustworthiness/continuity, kNN-recall, downstream task, stability across seeds.
**Q5. Why PCA first?** Denoise, speed (distances in 50-D instead of 1000-D).
