# data-science-zero-to-one-hundred

A **30-day, interview-oriented curriculum** (June 01 – June 30, 2026) that takes a Data Science & AI student from NumPy memory layouts to FAANG-style case studies. Every day ships:

- `notes.md` — derivations, statistical intuition, trade-offs, and FAANG-style Q&A
- `script.py` — self-contained, runnable code: it generates dummy data, processes it, prints numerical results, and saves figures to `outputs/` next to the script

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd day-01-numpy-vectorization-and-memory-layouts
python script.py                                        # figures land in ./outputs/
```
Scripts use a headless matplotlib backend (`Agg`), fixed seeds, and degrade gracefully when optional libraries (Polars, UMAP, seaborn, Plotly, imbalanced-learn, statsmodels) are missing. Day 10 needs Polars for its lazy-engine section; Day 22 uses UMAP if installed.

## How to study each day (≈2–3 hours)
1. Read `notes.md` (30 min); derive the key formula on paper without looking.
2. Run `script.py`; predict each printed output before executing (20 min).
3. Modify one assumption (sample size, noise, prior) and explain the change (30 min).
4. Answer the Q&A aloud in under 2 minutes each (record yourself) (30 min).
5. Commit with the message listed in `metadata.json`.

## 30-Day Matrix
| Day | Date | Topic | Files |
|---|---|---|---|
| 01 | Jun 01, 2026 | [NumPy Vectorization & Memory Layouts](day-01-numpy-vectorization-and-memory-layouts/) | [notes](day-01-numpy-vectorization-and-memory-layouts/notes.md) · [script](day-01-numpy-vectorization-and-memory-layouts/script.py) |
| 02 | Jun 02, 2026 | [Broadcasting & Advanced Indexing](day-02-broadcasting-and-advanced-indexing/) | [notes](day-02-broadcasting-and-advanced-indexing/notes.md) · [script](day-02-broadcasting-and-advanced-indexing/script.py) |
| 03 | Jun 03, 2026 | [Linear Algebra: Matrix Decompositions & SVD](day-03-linear-algebra-matrix-decompositions-svd/) | [notes](day-03-linear-algebra-matrix-decompositions-svd/notes.md) · [script](day-03-linear-algebra-matrix-decompositions-svd/script.py) |
| 04 | Jun 04, 2026 | [Eigenvalues & PCA Math From Scratch](day-04-eigenvalues-and-pca-math-from-scratch/) | [notes](day-04-eigenvalues-and-pca-math-from-scratch/notes.md) · [script](day-04-eigenvalues-and-pca-math-from-scratch/script.py) |
| 05 | Jun 05, 2026 | [SciPy: Signal Processing & Optimization](day-05-scipy-signal-processing-and-optimization/) | [notes](day-05-scipy-signal-processing-and-optimization/notes.md) · [script](day-05-scipy-signal-processing-and-optimization/script.py) |
| 06 | Jun 06, 2026 | [Pandas Series & DataFrame Internals](day-06-pandas-series-dataframe-internals/) | [notes](day-06-pandas-series-dataframe-internals/notes.md) · [script](day-06-pandas-series-dataframe-internals/script.py) |
| 07 | Jun 07, 2026 | [Pandas Hierarchical Indexing & Merging](day-07-pandas-hierarchical-indexing-and-merging/) | [notes](day-07-pandas-hierarchical-indexing-and-merging/notes.md) · [script](day-07-pandas-hierarchical-indexing-and-merging/script.py) |
| 08 | Jun 08, 2026 | [Pandas Aggregation & Window Functions](day-08-pandas-aggregation-and-window-functions/) | [notes](day-08-pandas-aggregation-and-window-functions/notes.md) · [script](day-08-pandas-aggregation-and-window-functions/script.py) |
| 09 | Jun 09, 2026 | [Time-Series Analysis & Resampling](day-09-time-series-analysis-and-resampling/) | [notes](day-09-time-series-analysis-and-resampling/notes.md) · [script](day-09-time-series-analysis-and-resampling/script.py) |
| 10 | Jun 10, 2026 | [Handling Gigabyte Datasets with Polars](day-10-handling-gigabyte-datasets-with-polars/) | [notes](day-10-handling-gigabyte-datasets-with-polars/notes.md) · [script](day-10-handling-gigabyte-datasets-with-polars/script.py) |
| 11 | Jun 11, 2026 | [Data Cleaning & Missing-Data Imputation](day-11-data-cleaning-and-missing-data-imputation/) | [notes](day-11-data-cleaning-and-missing-data-imputation/notes.md) · [script](day-11-data-cleaning-and-missing-data-imputation/script.py) |
| 12 | Jun 12, 2026 | [Outlier Detection & Robust Scalers](day-12-outlier-detection-and-robust-scalers/) | [notes](day-12-outlier-detection-and-robust-scalers/notes.md) · [script](day-12-outlier-detection-and-robust-scalers/script.py) |
| 13 | Jun 13, 2026 | [SQL for Data Science: Window Functions & CTEs](day-13-sql-for-data-science-window-functions-ctes/) | [notes](day-13-sql-for-data-science-window-functions-ctes/notes.md) · [script](day-13-sql-for-data-science-window-functions-ctes/script.py) |
| 14 | Jun 14, 2026 | [Descriptive Statistics & Distribution Fitting](day-14-descriptive-statistics-and-distribution-fitting/) | [notes](day-14-descriptive-statistics-and-distribution-fitting/notes.md) · [script](day-14-descriptive-statistics-and-distribution-fitting/script.py) |
| 15 | Jun 15, 2026 | [Central Limit Theorem & Bootstrapping](day-15-central-limit-theorem-and-bootstrapping/) | [notes](day-15-central-limit-theorem-and-bootstrapping/notes.md) · [script](day-15-central-limit-theorem-and-bootstrapping/script.py) |
| 16 | Jun 16, 2026 | [Hypothesis Testing: Z-test, T-test, ANOVA](day-16-hypothesis-testing-z-test-t-test-anova/) | [notes](day-16-hypothesis-testing-z-test-t-test-anova/notes.md) · [script](day-16-hypothesis-testing-z-test-t-test-anova/script.py) |
| 17 | Jun 17, 2026 | [Chi-Square & Non-Parametric Tests](day-17-chi-square-and-non-parametric-tests/) | [notes](day-17-chi-square-and-non-parametric-tests/notes.md) · [script](day-17-chi-square-and-non-parametric-tests/script.py) |
| 18 | Jun 18, 2026 | [A/B Testing Design: Sample Size & Power](day-18-ab-testing-design-sample-size-and-power/) | [notes](day-18-ab-testing-design-sample-size-and-power/notes.md) · [script](day-18-ab-testing-design-sample-size-and-power/script.py) |
| 19 | Jun 19, 2026 | [Bayesian Inference & MCMC Concepts](day-19-bayesian-inference-and-mcmc-concepts/) | [notes](day-19-bayesian-inference-and-mcmc-concepts/notes.md) · [script](day-19-bayesian-inference-and-mcmc-concepts/script.py) |
| 20 | Jun 20, 2026 | [Exploratory Data Analysis Frameworks](day-20-exploratory-data-analysis-frameworks/) | [notes](day-20-exploratory-data-analysis-frameworks/notes.md) · [script](day-20-exploratory-data-analysis-frameworks/script.py) |
| 21 | Jun 21, 2026 | [Publication-Ready Dataviz: Seaborn & Plotly](day-21-publication-ready-dataviz-seaborn-plotly/) | [notes](day-21-publication-ready-dataviz-seaborn-plotly/notes.md) · [script](day-21-publication-ready-dataviz-seaborn-plotly/script.py) |
| 22 | Jun 22, 2026 | [Dimensionality Reduction: t-SNE & UMAP](day-22-dimensionality-reduction-tsne-and-umap/) | [notes](day-22-dimensionality-reduction-tsne-and-umap/notes.md) · [script](day-22-dimensionality-reduction-tsne-and-umap/script.py) |
| 23 | Jun 23, 2026 | [Feature Encoding: Target & Frequency Encoding](day-23-feature-encoding-target-and-frequency-encoding/) | [notes](day-23-feature-encoding-target-and-frequency-encoding/notes.md) · [script](day-23-feature-encoding-target-and-frequency-encoding/script.py) |
| 24 | Jun 24, 2026 | [Handling Extreme Imbalance: SMOTE & Cost-Sensitive Learning](day-24-handling-extreme-imbalance-smote-cost-sensitive/) | [notes](day-24-handling-extreme-imbalance-smote-cost-sensitive/notes.md) · [script](day-24-handling-extreme-imbalance-smote-cost-sensitive/script.py) |
| 25 | Jun 25, 2026 | [Feature Selection: Filter, Wrapper, Embedded](day-25-feature-selection-filter-wrapper-embedded/) | [notes](day-25-feature-selection-filter-wrapper-embedded/notes.md) · [script](day-25-feature-selection-filter-wrapper-embedded/script.py) |
| 26 | Jun 26, 2026 | [Building Production Data-Validation Pipelines](day-26-building-production-data-validation-pipelines/) | [notes](day-26-building-production-data-validation-pipelines/notes.md) · [script](day-26-building-production-data-validation-pipelines/script.py) |
| 27 | Jun 27, 2026 | [Feature Stores & Data Drift Monitoring](day-27-feature-stores-and-data-drift-monitoring/) | [notes](day-27-feature-stores-and-data-drift-monitoring/notes.md) · [script](day-27-feature-stores-and-data-drift-monitoring/script.py) |
| 28 | Jun 28, 2026 | [Interview Case Study: Churn & LTV Modeling](day-28-interview-case-study-churn-and-ltv-modeling/) | [notes](day-28-interview-case-study-churn-and-ltv-modeling/notes.md) · [script](day-28-interview-case-study-churn-and-ltv-modeling/script.py) |
| 29 | Jun 29, 2026 | [Interview Case Study: Search & Recommendation Metrics](day-29-interview-case-study-search-and-recommendation-metrics/) | [notes](day-29-interview-case-study-search-and-recommendation-metrics/notes.md) · [script](day-29-interview-case-study-search-and-recommendation-metrics/script.py) |
| 30 | Jun 30, 2026 | [FAANG Data Science Interview Mock & Cheatsheet](day-30-faang-data-science-interview-mock-and-cheatsheet/) | [notes](day-30-faang-data-science-interview-mock-and-cheatsheet/notes.md) · [script](day-30-faang-data-science-interview-mock-and-cheatsheet/script.py) |

## Phases
| Days | Phase | Outcome |
|---|---|---|
| 01–05 | Numerical foundations | Vectorized thinking, linear algebra, PCA/SVD, SciPy |
| 06–13 | Data wrangling at scale | pandas internals, joins, windows, time series, Polars, imputation, outliers, SQL |
| 14–19 | Statistics & inference | Distributions, CLT/bootstrap, hypothesis tests, nonparametrics, A/B power, Bayesian/MCMC |
| 20–25 | Analysis & feature engineering | EDA, dataviz, t-SNE/UMAP, encoding, imbalance, feature selection |
| 26–27 | Production data quality | Validation pipelines, feature stores, drift monitoring |
| 28–30 | Interview simulation | Churn/LTV, search/recsys metrics, full mock + cheatsheet |

## Interview focus areas
- **Statistics rigor:** p-values vs effect sizes, power/sample size, multiple testing, CLT limits, bootstrap, Bayesian decision-making.
- **Experimentation:** SRM, peeking, CUPED, interference, guardrails, metric design.
- **ML judgement:** leakage (target encoding, selection bias, PIT joins), imbalance, calibration, cost-sensitive decisions, drift.
- **Data engineering fluency:** memory layouts, join cardinality, SQL windows, lazy engines, validation contracts.
- **Communication:** state assumptions, define metrics first, discuss trade-offs and failure modes, close with validation and monitoring.

## Repository layout
```
data-science-zero-to-one-hundred/
├── README.md
├── requirements.txt
├── metadata.json
└── day-NN-topic-slug/
    ├── notes.md
    └── script.py        # writes figures to ./outputs/
```
Commit-message convention: `feat(day-NN): <summary>` (see `metadata.json`).
