# Day 29 — Case Study: Search & Recommendation Metrics

## Offline ranking metrics
For a ranked list with binary/graded relevance $rel_i$ at position $i$:
- **Precision@k** $=\frac1k\sum_{i\le k}\mathbb 1[rel_i>0]$; **Recall@k** $=\frac{\#\text{rel in top }k}{\#\text{rel total}}$; **HitRate@k**.
- **MRR** $=\frac1{|Q|}\sum_q\frac1{\text{rank of first relevant}}$ (navigational queries).
- **AP** $=\frac1R\sum_kP@k\cdot\mathbb 1[rel_k]$, **MAP** = mean over queries.
- **DCG@k** $=\sum_{i\le k}\frac{2^{rel_i}-1}{\log_2(i+1)}$; **NDCG@k** $=\text{DCG@k}/\text{IDCG@k}\in[0,1]$ (graded relevance, position discount, comparable across queries). Gain exponent 2^rel emphasizes highly relevant items.
- Beyond accuracy: **coverage**, diversity (intra-list distance), novelty/serendipity, popularity bias, fairness exposure, latency.

## Online metrics & experimentation
CTR, conversion, dwell time, successful-session rate, abandonment, queries per session, revenue; guardrails: latency, ad load. Offline NDCG gains do not always translate online → A/B test; interleaving (team-draft) is 10–100x more sensitive for comparing rankers.

## Biases in click data
**Position bias:** users click higher results regardless of relevance: $P(click)=P(examine|pos)\,P(rel)$. Debias with **inverse propensity weighting** $w=1/P(examine|pos)$ (estimate propensities via randomization/intervention harvesting), click models (cascade/DBN), counterfactual LTR. Also selection bias (only shown items get feedback), popularity bias, feedback loops.

## Recommender approaches
Popularity baseline; item-item CF; **matrix factorization** $\min\sum_{(u,i)}(r_{ui}-p_u^\top q_i)^2+\lambda(\|p_u\|^2+\|q_i\|^2)$ (ALS/SGD; implicit feedback uses confidence weights); two-tower retrieval + ANN → ranker (GBDT/DNN) → re-ranking (diversity, business rules). Cold-start: content features, bandits (exploration). Evaluate with time-based splits and leave-last-out.

## FAANG Q&A
**Q1. NDCG vs MAP?** NDCG handles graded relevance and discount; MAP binary.
**Q2. Offline metric improves, online flat. Why?** Metric mismatch, position bias in labels, novelty effects, distribution shift, latency.
**Q3. Why is recall@k key for retrieval and NDCG for ranking?** Retrieval must not lose candidates; ranking orders them.
**Q4. How to handle new users/items?** Content-based features, popularity prior, exploration.
**Q5. Evaluate search without labels?** Click-model-derived relevance, human raters (graded), interleaving.
