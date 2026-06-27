# Day 27 — Feature Stores & Drift Monitoring

## Feature store essentials
Components: offline store (historical, batch, Parquet/warehouse), online store (low-latency KV: Redis/DynamoDB), registry (definitions, owners, lineage), transformation engine, and **point-in-time (PIT) correct joins**.
Benefits: reuse, consistent training/serving features (kill skew), governance, backfills. Costs: infra and ownership overhead. (Feast, Tecton, Vertex, SageMaker FS.)
**PIT join:** for each training label at time $t$, attach feature values with the latest timestamp $\le t$ (optionally minus ingestion lag/TTL). Using "latest overall" leaks the future (label leakage) → inflated offline metrics, bad production.

## Types of drift
- **Covariate shift** $P(X)$ changes, $P(Y|X)$ stable. **Prior/label shift** $P(Y)$ changes. **Concept drift** $P(Y|X)$ changes (fraud tactics, seasonality). Plus **upstream data bugs** (schema/units).
- Speed: sudden, gradual, incremental, seasonal/recurring.

## Detection statistics
- **PSI** $=\sum_i(a_i-e_i)\ln\frac{a_i}{e_i}$ over bins (symmetric KL). Rule of thumb: <0.1 stable, 0.1–0.25 moderate, >0.25 significant. Bins from reference quantiles; smooth zero bins.
- **KS test** (continuous), **chi-square** (categorical), Jensen–Shannon, Wasserstein distance, domain-classifier AUC (train a classifier to separate reference vs current; AUC≈0.5 ⇒ no drift) — multivariate.
- With large $n$ every test is "significant": pair with effect size (PSI/Wasserstein) and business-impact thresholds.
- Performance monitoring needs labels (often delayed) → use proxy metrics: prediction distribution, feature drift, calibration of delayed labels, business KPIs.

## Responses
Alert → diagnose (data bug vs real shift) → retrain on recent window / reweight → champion–challenger → rollback; schedule retraining vs trigger-based.

## FAANG Q&A
**Q1. AUC offline 0.92, online 0.74. Why?** Training/serving skew, PIT leakage, label delay, drift, feedback loops.
**Q2. PSI is high only for one feature — retrain?** Check its importance and upstream cause first.
**Q3. Why not rely only on p-values for drift?** Huge $n$ ⇒ trivial shifts significant; use magnitude thresholds.
**Q4. How to monitor without labels?** Input/prediction drift, uncertainty, proxy metrics, sampled human labeling.
**Q5. Online vs offline store consistency?** Same definitions, materialization jobs, TTLs, backfill tests comparing both.
