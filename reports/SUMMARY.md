# NeuroDefect — results summary for the course report

All numbers are held-out test metrics (stratified 70/30 split), mean over seeds 42, 7, 13, from the scripts in `scripts/`. Raw CSVs are in this folder.

## 1. System
GA feature selection → softmax BPN (2-unit softmax, weighted cross-entropy) → Mamdani fuzzy risk engine (optionally GA-tuned membership triangles) → Streamlit dashboard + `neurodefect` CLI (radon scanner for Python code).

Data: synthetic CM1-like sample (345 rows, bundled) and real NASA PROMISE CM1 (498), PC1 (1109), JM1 (13204). PC5 was not obtainable.

## 2. Softmax BPN vs. original sigmoid BPN and baselines (`scripts/evaluate.py`)

| Dataset | LR F1 | RF F1 | SVM F1 | BPN F1 before | BPN F1 softmax | BPN MCC before → softmax |
|---|---|---|---|---|---|---|
| Synthetic | .670 | .702 | .710 | .375 | .698 | .204 → .653 |
| CM1 | .377 | .352 | .344 | .192 | .316 | .001 → .223 |
| PC1 | .349 | .393 | .362 | .212 | .314 | .137 → .262 |
| JM1 | .444 | .452 | .448 | .411 | .427 | .221 → .202 |

The softmax BPN reaches baseline level on the synthetic sample and improves strongly on CM1/PC1, but stays below the best baseline on real data.

## 3. Ablation (`reports/ablation.md`, BPN on GA-selected features)

| Dataset | full F1 | no-GA F1 | no-SMOTE F1 | accuracy-floor F1 (acc) |
|---|---|---|---|---|
| Synthetic | .711 | .695 | .663 | .711 (.901) |
| CM1 | .325 | .321 | .360 | .325 (.787) |
| PC1 | .313 | .357 | .247 | .313 (.780) |
| JM1 | .413 (acc .465) | .428 | .451 | .440 (acc .647) |

- The GA keeps ~4–8 of 21 features with about the same F1 (smaller, more explainable model), but does not raise F1 on real data.
- SMOTE helps on the synthetic sample and PC1 (recall .683 vs .190 without it) and hurts on CM1/JM1.
- The accuracy-floor threshold fixes the JM1 accuracy drop (.465 → .647) and raises F1 (.413 → .440) and MCC (.175 → .237). On the other datasets it changes nothing because the F1-optimal threshold already meets the floor.

## 4. Fixed-quantile vs. GA-tuned fuzzy memberships (fuzzy tier ≥ HIGH as the defect decision)

GA fitness uses 3-fold out-of-fold BPN probabilities on the training split. These numbers were computed with the original 14-rule base, before the rule fix in §11; §9 has fuzzy CV results with the new rules.

| Dataset | F1 fixed | F1 GA-tuned | MCC fixed | MCC GA-tuned | risk-label corr fixed → tuned |
|---|---|---|---|---|---|
| Synthetic | .368 | .594 | .294 | .528 | .300 → .615 |
| CM1 | .241 | .225 | .129 | .135 | .144 → .166 |
| PC1 | .194 | .215 | .149 | .154 | .156 → .201 |
| JM1 | .413 | .356 | .180 | .177 | .196 → .196 |

GA tuning trades recall for precision. On the real datasets the effect is small or negative, so the default is `fuzzy.ga_tune: false`. Figures: `reports/figures/*_fuzzy_ga_fitness.png`.

## 5. Full NASA features vs. scanner-compatible shared subset (`reports/shared_vs_full.md`)

| Dataset | F1 full (21 cols) | F1 shared (15 cols) | ROC-AUC full → shared |
|---|---|---|---|
| CM1 | .325 | .242 | .774 → .680 |
| PC1 | .313 | .319 | .808 → .798 |
| JM1 | .413 | .407 | .687 → .672 |

## 6. GA diagnostics
`reports/figures/<dataset>_ga_fitness.png` (best fitness per generation, per seed and mean) and `<dataset>_feature_frequency.png` (share of GA individuals that included each feature). They are also shown live on the dashboard.

## 7. Real-code demo
`neurodefect scan src` → 187 functions in 47 files (`reports/demo/src_metrics.csv`); `neurodefect predict` with the CM1 shared model → `reports/demo/predict_output.txt`, `src_predictions.csv/json`.

## 9. Repeated stratified cross-validation (`reports/cv_results.md`, `scripts/cv_eval.py`, `scripts/cv_report.py`)

5×5 repeated stratified CV for synthetic/CM1/PC1; **JM1 uses 2×5 (10 folds) for time**. Every fold refits imputer/scaler, SMOTE, GA, BPN early stopping, threshold, ensemble weight and fuzzy sets on its training part only. Values are mean ± std over test folds.

| Dataset | Hybrid F1 | BPN F1 | Ensemble F1 | LR F1 | RF F1 | SVM F1 | Friedman p (F1) | Best mean rank |
|---|---|---|---|---|---|---|---|---|
| Synthetic | .710±.084 | .690±.100 | .706±.108 | .733±.095 | .732±.108 | .763±.084 | .0077 | SVM |
| CM1 | .335±.081 | .289±.116 | .335±.103 | .372±.091 | .242±.098 | .345±.084 | .00031 | LR |
| PC1 | .310±.055 | .354±.076 | .286±.085 | .360±.055 | .361±.054 | .329±.054 | .00040 | RF |
| JM1 (2×5) | .404±.011 | .436±.010 | .428±.022 | .440±.022 | .446±.017 | .447±.018 | .00026 | RF |

Wilcoxon signed-rank p-values (F1, paired over folds), Hybrid vs LR / RF / SVM:
- Synthetic: .187 / .307 / .0042 (worse than SVM)
- CM1: .120 / **.0039 (better than RF)** / .475
- PC1: .0003 / .0028 / .134 (worse than LR and RF)
- JM1: .0039 / .0020 / .0020 (worse than all)

The Ensemble beats Hybrid only on JM1 (ΔF1 +.024, p = .0098; ΔMCC +.074, p = .0020). It ties on synthetic and CM1 and is worse on PC1 (−.024, p = .24). **Default: ensemble off** (`training.ensemble: false`). The full per-comparison and MCC tables are in `cv_results.md`. Caveat: repeated-CV folds overlap, so the p-values are optimistic.

Fuzzy tier ≥ HIGH (new guarded rules, Hybrid probabilities) F1: synthetic .680, CM1 .302, PC1 .322, JM1 .410.

## 10. Feature selectors in CV (Hybrid BPN; synthetic/CM1/PC1 5×5, JM1 1×5)

| Dataset | none | GA (logistic probe) | GA (small BPN) | binary PSO |
|---|---|---|---|---|
| Synthetic F1 | .733 | .707 | .727 | .730 |
| CM1 F1 | .260 | **.338** | .292 | .333 |
| PC1 F1 | **.358** | .297 | .330 | .334 |
| JM1 F1 | **.429** | .415 | .411 | .408 |
| selection s/fold (CM1) | 0 | 1.8 | 18.8 | 1.9 |

- GA with the small-BPN fitness is better than the logistic probe on PC1 (p = .041), worse on CM1 (p = .018), and about the same elsewhere, at roughly 10× the cost. **Default stays `ga.fitness_model: logistic`.**
- PSO vs GA (logistic): PSO is better on PC1 (p = .0074) and not significantly different elsewhere. The GA stays the default selector (`ga.method: ga`); PSO is available with `ga.method: pso`.
- Feature selection itself only clearly helps on CM1 (GA vs none p = .0105). On PC1 using all features is better (p = .0013). Its main benefit is a ~4× smaller, more explainable input set (5–8 of 21 features).

## 11. Fuzzy rule fix and scanner demo
Rules R01, R02, R08, R09 and R12 (metric-only escalation to HIGH/CRITICAL) now also require `defect_prob IS AT_LEAST_POSSIBLE` (fuzzy OR of POSSIBLE and LIKELY). A new rule R15 maps `complexity HIGH AND defect_prob UNLIKELY → MEDIUM`.

Scanner demo on `src/` (187 functions):
- Before: CRITICAL 44 / HIGH 36 / MEDIUM 38 / LOW 48, on 166 functions with older code. CRITICAL included modules with p ≈ 0.001.
- After: **CRITICAL 18 / HIGH 52 / MEDIUM 60 / LOW 57**. Every CRITICAL module is now also predicted defective by the BPN.
- The CM1 shared model flags 124/187 functions as defective (threshold .275), so on Python code it is over-sensitive.

## 12. SHAP
KernelExplainer on P(defect), with a background of 50 sampled training rows stored with each run. It's shown on the Predictions and Scan & Predict pages next to the fired rules (on demand, about 0.5 s per module). If `shap` is not installed, the app falls back to showing the rules only.

## 13. Threats to validity
- Language gap (C vs. Python) and metric counting differences (see README).
- Small minority classes (CM1: 49 defects) → high variance across seeds.
- Derived, not ground-truth, severity.
- Single-split tables (sections 2–5) use three seeds; sections 9–10 use repeated CV with Wilcoxon/Friedman tests. Those p-values are optimistic because CV folds overlap.
