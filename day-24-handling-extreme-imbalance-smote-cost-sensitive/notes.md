# Day 24 — Extreme Class Imbalance

## Why accuracy lies
With 1% positives, "always negative" has 99% accuracy and 0 recall. Use **PR-AUC (average precision)**, recall@precision, F-β, MCC, cost/expected-profit. ROC-AUC can look great because FPR is dominated by the huge negative class; PR curves reveal precision collapse. Baseline PR-AUC = prevalence.

## Remedies (in order of preference)
1. **Better data/features**, more positives, label quality.
2. **Cost-sensitive learning**: class weights $w_+=\frac{n}{2n_+}$, or sample weights, focal loss. Equivalent in effect to shifting the decision boundary.
3. **Threshold moving** on calibrated probabilities (cheap, powerful). With costs $C_{FN},C_{FP}$, Bayes-optimal rule: predict positive if $p>\frac{C_{FP}}{C_{FP}+C_{FN}}$.
4. **Resampling**: random under-sampling (lose info), over-sampling (overfit duplicates), **SMOTE**: for minority $x_i$ pick a $k$-NN neighbor $x_j$ and create $x_{new}=x_i+\lambda(x_j-x_i),\lambda\sim U(0,1)$. Variants: Borderline-SMOTE, ADASYN, SMOTE-Tomek.
5. Anomaly-detection framing if positives are extremely rare/heterogeneous.
6. Ensembles with balanced bagging (`BalancedRandomForest`).

## Critical rules
- **Resample only the training fold** (inside the CV pipeline — `imblearn.Pipeline`). Resampling before splitting leaks synthetic copies of validation points.
- Resampling/weighting **distorts probabilities** (calibration broken): recalibrate (Platt/isotonic) or correct priors: $\text{odds}_{true}=\text{odds}_{model}\cdot\frac{\pi_+/(1-\pi_+)}{\pi'_+/(1-\pi'_+)}$.
- Evaluate on the natural (imbalanced) distribution. SMOTE struggles in high-D, with categoricals (SMOTE-NC), and overlapping classes.

## FAANG Q&A
**Q1. Fraud at 0.1%: approach?** Stratified split, gradient boosting with class weights, PR-AUC, threshold from cost matrix, calibrate, monitor drift & alert volume (precision@k analyst capacity).
**Q2. Why not SMOTE everything?** Noise amplification, no new information, leakage if misused; weighting/thresholding often as good.
**Q3. Interpret PR-AUC 0.30 at 1% prevalence?** 30x better than random (0.01) — may still be useful.
**Q4. Probabilities after class_weight='balanced'?** Inflated; recalibrate before using as risk scores.
**Q5. Choose threshold given review capacity of 500 cases/day?** Pick top-500 scores (precision@k) rather than fixed probability.
