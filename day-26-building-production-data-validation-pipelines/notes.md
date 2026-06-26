# Day 26 — Production Data-Validation Pipelines

## Why
Models fail silently on bad data: schema changes, unit changes, null explosions, duplicates, stale partitions. "Garbage in, garbage out" at scale — most ML incidents are data incidents.

## Layers of validation
1. **Schema/contract:** column names, dtypes, nullability, allowed categories, ranges, uniqueness, primary key, referential integrity. (Great Expectations, Pandera, Pydantic, TFX Data Validation, dbt tests.)
2. **Row-level rules:** domain constraints (age∈[0,120], end≥start).
3. **Batch-level statistical checks:** row count vs trailing mean, null rate, mean/std/quantile shifts, category frequency, freshness (max timestamp lag).
4. **Cross-table/lineage checks:** join cardinality, reconciliation to source-of-truth totals.
5. **Model I/O checks:** feature vector shape/range, prediction distribution.

## Design patterns
- **Fail-fast vs quarantine:** hard-fail on contract breaks (missing column); quarantine bad rows with reason codes and continue if the bad-rate stays below a threshold (e.g., <2%); alert otherwise.
- **Idempotent, versioned** transformations; deterministic seeds; schema version hash stored with the artifact.
- **Training/serving parity:** one transformation code path (sklearn `Pipeline`/`ColumnTransformer`, feature store) to avoid skew.
- **Observability:** emit validation metrics; dashboards, alerts, data SLAs; keep a quarantine table for triage.
- **Testing pyramid:** unit tests on transforms, property-based tests, integration tests with golden datasets.

## Orchestration
Airflow/Dagster/Prefect DAG: ingest → validate → transform → train/score → validate outputs → publish. Validation tasks block downstream tasks.

## FAANG Q&A
**Q1. A feature's null rate jumps 0→40% overnight. What do you do?** Check upstream deploy/schema, quarantine, fall back to last good snapshot or default, alert owner, add a null-rate gate.
**Q2. Train/serve skew?** Shared transformation code, feature store, log served features and compare to training.
**Q3. Where should validation live?** At ingestion (contracts with producers), before training, and at serving.
**Q4. Tradeoff of strict validation?** False alarms → alert fatigue/blocked pipelines; tune thresholds, severity levels.
**Q5. How to test a pipeline?** Unit tests with edge cases (empty, nulls, new categories), regression tests on golden data, shadow runs.
