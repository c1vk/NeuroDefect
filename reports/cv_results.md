# Repeated stratified cross-validation results

All preprocessing, SMOTE, feature selection, BPN early stopping, threshold tuning, ensemble weighting and fuzzy membership fitting happen inside each training fold. Metrics: mean ± std over test folds. Wilcoxon signed-rank tests pair the same folds; Friedman ranks models per fold. Folds of repeated CV share training data, so p-values are optimistic and should be read as indicative.

- **Hybrid** = softmax BPN on GA-selected features (the app's Neuro-Fuzzy-GA classifier).
- **BPN** = softmax BPN on all features. **Ensemble** = w·BPN + (1−w)·RF on GA features, w and threshold tuned on held-out real training rows.
- **Fuzzy>=HIGH (x)** = derived risk tier ≥ HIGH used as the defect decision, with probabilities from x.

## 1–2. Models and significance tests

### synthetic_sample (5×5 CV, 25 folds)

| Model | accuracy | precision | recall | f1 | roc_auc | mcc |
|---|---|---|---|---|---|---|
| Hybrid | 0.897 ± 0.037 | 0.615 ± 0.110 | 0.863 ± 0.110 | 0.710 ± 0.084 | 0.953 ± 0.035 | 0.671 ± 0.096 |
| BPN | 0.904 ± 0.039 | 0.685 ± 0.150 | 0.736 ± 0.156 | 0.690 ± 0.100 | 0.955 ± 0.029 | 0.648 ± 0.116 |
| Ensemble | 0.913 ± 0.028 | 0.713 ± 0.136 | 0.754 ± 0.182 | 0.706 ± 0.108 | 0.954 ± 0.032 | 0.674 ± 0.105 |
| LR | 0.908 ± 0.038 | 0.644 ± 0.118 | 0.867 ± 0.096 | 0.733 ± 0.095 | 0.951 ± 0.030 | 0.695 ± 0.111 |
| RF | 0.918 ± 0.034 | 0.697 ± 0.118 | 0.780 ± 0.130 | 0.732 ± 0.108 | 0.952 ± 0.036 | 0.689 ± 0.128 |
| SVM | 0.926 ± 0.031 | 0.722 ± 0.109 | 0.822 ± 0.098 | 0.763 ± 0.084 | 0.948 ± 0.038 | 0.726 ± 0.098 |
| Fuzzy>=HIGH (Hybrid) | — | 0.568 ± 0.098 | 0.866 ± 0.098 | 0.680 ± 0.078 | — | 0.638 ± 0.093 |
| Fuzzy>=HIGH (Ensemble) | — | 0.557 ± 0.107 | 0.891 ± 0.076 | 0.680 ± 0.084 | — | 0.639 ± 0.096 |

Ensemble BPN weight chosen on train-only validation: mean 0.53 (min 0.0, max 1.0).

| Comparison (paired over folds) | Δ mean F1 | Wilcoxon p (F1) | Δ mean MCC | Wilcoxon p (MCC) |
|---|---|---|---|---|
| Hybrid vs LR | -0.024 | 0.1873 | -0.024 | 0.2818 |
| Hybrid vs RF | -0.022 | 0.3065 | -0.018 | 0.4675 |
| Hybrid vs SVM | -0.053 | 0.0042 | -0.055 | 0.0056 |
| BPN vs LR | -0.044 | 0.0578 | -0.046 | 0.0755 |
| BPN vs RF | -0.042 | 0.0278 | -0.040 | 0.0556 |
| BPN vs SVM | -0.073 | 0.0010 | -0.077 | 0.0012 |
| Ensemble vs LR | -0.028 | 0.4979 | -0.021 | 0.5901 |
| Ensemble vs RF | -0.026 | 0.2540 | -0.015 | 0.3993 |
| Ensemble vs SVM | -0.057 | 0.0425 | -0.052 | 0.0520 |
| Ensemble vs Hybrid | -0.004 | 0.3382 | +0.003 | 0.4290 |

Friedman (F1, 6 models): χ² = 15.73, p = 0.00765. Mean rank (1 = best): SVM 2.56, RF 3.10, LR 3.36, Ensemble 3.56, Hybrid 4.02, BPN 4.40

Friedman (MCC, 6 models): χ² = 13.58, p = 0.0185. Mean rank (1 = best): SVM 2.64, RF 3.12, LR 3.36, Ensemble 3.56, Hybrid 3.96, BPN 4.36

### CM1 (5×5 CV, 25 folds)

| Model | accuracy | precision | recall | f1 | roc_auc | mcc |
|---|---|---|---|---|---|---|
| Hybrid | 0.759 ± 0.063 | 0.246 ± 0.065 | 0.555 ± 0.168 | 0.335 ± 0.081 | 0.690 ± 0.078 | 0.247 ± 0.108 |
| BPN | 0.790 ± 0.089 | 0.246 ± 0.112 | 0.404 ± 0.217 | 0.289 ± 0.116 | 0.713 ± 0.105 | 0.200 ± 0.134 |
| Ensemble | 0.796 ± 0.064 | 0.273 ± 0.086 | 0.488 ± 0.205 | 0.335 ± 0.103 | 0.723 ± 0.071 | 0.252 ± 0.123 |
| LR | 0.767 ± 0.049 | 0.266 ± 0.071 | 0.630 ± 0.153 | 0.372 ± 0.091 | 0.762 ± 0.089 | 0.295 ± 0.121 |
| RF | 0.790 ± 0.037 | 0.199 ± 0.081 | 0.320 ± 0.146 | 0.242 ± 0.098 | 0.709 ± 0.060 | 0.135 ± 0.110 |
| SVM | 0.767 ± 0.037 | 0.249 ± 0.060 | 0.573 ± 0.167 | 0.345 ± 0.084 | 0.752 ± 0.070 | 0.261 ± 0.113 |
| Fuzzy>=HIGH (Hybrid) | — | 0.218 ± 0.064 | 0.516 ± 0.181 | 0.302 ± 0.086 | — | 0.205 ± 0.117 |
| Fuzzy>=HIGH (Ensemble) | — | 0.204 ± 0.054 | 0.604 ± 0.138 | 0.302 ± 0.068 | — | 0.206 ± 0.098 |

Ensemble BPN weight chosen on train-only validation: mean 0.62 (min 0.0, max 1.0).

| Comparison (paired over folds) | Δ mean F1 | Wilcoxon p (F1) | Δ mean MCC | Wilcoxon p (MCC) |
|---|---|---|---|---|
| Hybrid vs LR | -0.037 | 0.1199 | -0.048 | 0.1135 |
| Hybrid vs RF | +0.093 | 0.0039 | +0.112 | 0.0025 |
| Hybrid vs SVM | -0.010 | 0.4751 | -0.014 | 0.4576 |
| BPN vs LR | -0.083 | 0.0318 | -0.096 | 0.0255 |
| BPN vs RF | +0.047 | 0.0900 | +0.064 | 0.0483 |
| BPN vs SVM | -0.056 | 0.1073 | -0.061 | 0.1073 |
| Ensemble vs LR | -0.038 | 0.1283 | -0.043 | 0.1300 |
| Ensemble vs RF | +0.093 | 0.0033 | +0.117 | 0.0025 |
| Ensemble vs SVM | -0.011 | 0.7751 | -0.009 | 0.8639 |
| Ensemble vs Hybrid | -0.000 | 0.7112 | +0.005 | 0.6150 |

Friedman (F1, 6 models): χ² = 23.20, p = 0.00031. Mean rank (1 = best): LR 2.44, Ensemble 3.18, SVM 3.30, Hybrid 3.34, BPN 3.92, RF 4.82

Friedman (MCC, 6 models): χ² = 24.30, p = 0.00019. Mean rank (1 = best): LR 2.36, Ensemble 3.24, SVM 3.26, Hybrid 3.42, BPN 3.88, RF 4.84

### PC1 (5×5 CV, 25 folds)

| Model | accuracy | precision | recall | f1 | roc_auc | mcc |
|---|---|---|---|---|---|---|
| Hybrid | 0.796 ± 0.053 | 0.213 ± 0.049 | 0.620 ± 0.164 | 0.310 ± 0.055 | 0.797 ± 0.057 | 0.274 ± 0.071 |
| BPN | 0.871 ± 0.035 | 0.296 ± 0.084 | 0.480 ± 0.149 | 0.354 ± 0.076 | 0.818 ± 0.054 | 0.307 ± 0.090 |
| Ensemble | 0.856 ± 0.050 | 0.256 ± 0.107 | 0.414 ± 0.204 | 0.286 ± 0.085 | 0.821 ± 0.042 | 0.241 ± 0.095 |
| LR | 0.813 ± 0.029 | 0.242 ± 0.039 | 0.714 ± 0.128 | 0.360 ± 0.055 | 0.840 ± 0.046 | 0.339 ± 0.072 |
| RF | 0.865 ± 0.024 | 0.283 ± 0.049 | 0.523 ± 0.135 | 0.361 ± 0.054 | 0.840 ± 0.035 | 0.315 ± 0.067 |
| SVM | 0.802 ± 0.030 | 0.221 ± 0.038 | 0.657 ± 0.128 | 0.329 ± 0.054 | 0.819 ± 0.038 | 0.297 ± 0.073 |
| Fuzzy>=HIGH (Hybrid) | — | 0.230 ± 0.056 | 0.569 ± 0.115 | 0.322 ± 0.062 | — | 0.278 ± 0.076 |
| Fuzzy>=HIGH (Ensemble) | — | 0.204 ± 0.037 | 0.634 ± 0.093 | 0.307 ± 0.049 | — | 0.271 ± 0.062 |

Ensemble BPN weight chosen on train-only validation: mean 0.63 (min 0.0, max 1.0).

| Comparison (paired over folds) | Δ mean F1 | Wilcoxon p (F1) | Δ mean MCC | Wilcoxon p (MCC) |
|---|---|---|---|---|
| Hybrid vs LR | -0.050 | 0.0003 | -0.064 | 0.0004 |
| Hybrid vs RF | -0.051 | 0.0028 | -0.041 | 0.0710 |
| Hybrid vs SVM | -0.018 | 0.1336 | -0.023 | 0.1073 |
| BPN vs LR | -0.005 | 0.6528 | -0.032 | 0.0588 |
| BPN vs RF | -0.007 | 0.6915 | -0.009 | 0.6915 |
| BPN vs SVM | +0.026 | 0.1908 | +0.009 | 0.5782 |
| Ensemble vs LR | -0.073 | 0.0020 | -0.098 | 0.0002 |
| Ensemble vs RF | -0.075 | 0.0020 | -0.074 | 0.0020 |
| Ensemble vs SVM | -0.042 | 0.0487 | -0.056 | 0.0223 |
| Ensemble vs Hybrid | -0.024 | 0.2414 | -0.033 | 0.1409 |

Friedman (F1, 6 models): χ² = 22.61, p = 0.000401. Mean rank (1 = best): RF 2.62, LR 2.94, BPN 3.00, SVM 3.54, Hybrid 4.38, Ensemble 4.52

Friedman (MCC, 6 models): χ² = 15.96, p = 0.00696. Mean rank (1 = best): LR 2.72, RF 3.08, BPN 3.28, SVM 3.34, Hybrid 4.04, Ensemble 4.54

### JM1 (2×5 CV, 10 folds)

| Model | accuracy | precision | recall | f1 | roc_auc | mcc |
|---|---|---|---|---|---|---|
| Hybrid | 0.440 ± 0.057 | 0.267 ± 0.015 | 0.841 ± 0.059 | 0.404 ± 0.011 | 0.677 ± 0.021 | 0.153 ± 0.028 |
| BPN | 0.574 ± 0.040 | 0.312 ± 0.017 | 0.731 ± 0.053 | 0.436 ± 0.010 | 0.684 ± 0.010 | 0.218 ± 0.020 |
| Ensemble | 0.674 ± 0.047 | 0.360 ± 0.034 | 0.542 ± 0.081 | 0.428 ± 0.022 | 0.678 ± 0.022 | 0.227 ± 0.033 |
| LR | 0.708 ± 0.013 | 0.387 ± 0.020 | 0.510 ± 0.026 | 0.440 ± 0.022 | 0.684 ± 0.020 | 0.252 ± 0.030 |
| RF | 0.721 ± 0.008 | 0.403 ± 0.013 | 0.500 ± 0.028 | 0.446 ± 0.017 | 0.700 ± 0.016 | 0.265 ± 0.021 |
| SVM | 0.673 ± 0.011 | 0.361 ± 0.014 | 0.588 ± 0.033 | 0.447 ± 0.018 | 0.693 ± 0.019 | 0.247 ± 0.026 |
| Fuzzy>=HIGH (Hybrid) | — | 0.304 ± 0.016 | 0.631 ± 0.036 | 0.410 ± 0.022 | — | 0.177 ± 0.036 |
| Fuzzy>=HIGH (Ensemble) | — | 0.300 ± 0.015 | 0.637 ± 0.033 | 0.407 ± 0.020 | — | 0.171 ± 0.034 |

Ensemble BPN weight chosen on train-only validation: mean 0.64 (min 0.1, max 1.0).

| Comparison (paired over folds) | Δ mean F1 | Wilcoxon p (F1) | Δ mean MCC | Wilcoxon p (MCC) |
|---|---|---|---|---|
| Hybrid vs LR | -0.036 | 0.0039 | -0.098 | 0.0020 |
| Hybrid vs RF | -0.042 | 0.0020 | -0.112 | 0.0020 |
| Hybrid vs SVM | -0.043 | 0.0020 | -0.094 | 0.0020 |
| BPN vs LR | -0.004 | 0.3750 | -0.033 | 0.0137 |
| BPN vs RF | -0.010 | 0.1934 | -0.047 | 0.0020 |
| BPN vs SVM | -0.011 | 0.0645 | -0.029 | 0.0039 |
| Ensemble vs LR | -0.012 | 0.0645 | -0.025 | 0.0137 |
| Ensemble vs RF | -0.019 | 0.0273 | -0.038 | 0.0039 |
| Ensemble vs SVM | -0.019 | 0.0195 | -0.020 | 0.0840 |
| Ensemble vs Hybrid | +0.024 | 0.0098 | +0.074 | 0.0020 |

Friedman (F1, 6 models): χ² = 23.60, p = 0.000259. Mean rank (1 = best): RF 2.20, SVM 2.40, LR 3.10, BPN 3.50, Ensemble 4.10, Hybrid 5.70

Friedman (MCC, 6 models): χ² = 36.91, p = 6.23e-07. Mean rank (1 = best): RF 1.40, LR 2.60, SVM 2.70, Ensemble 3.80, BPN 4.60, Hybrid 5.90

## 3–4. Feature selection: none vs GA (logistic probe) vs GA (small BPN) vs binary PSO

### synthetic_sample (5×5 CV) — Hybrid BPN with different feature selectors

| Selector | F1 | MCC | ROC-AUC | Recall | #features | selection s/fold |
|---|---|---|---|---|---|---|
| none | 0.733 ± 0.098 | 0.694 ± 0.119 | 0.958 ± 0.027 | 0.756 ± 0.126 | 21.0 | 0.0 |
| ga_logistic | 0.707 ± 0.106 | 0.667 ± 0.123 | 0.953 ± 0.046 | 0.827 ± 0.143 | 4.7 | 1.8 |
| ga_bpn | 0.727 ± 0.101 | 0.688 ± 0.116 | 0.950 ± 0.035 | 0.863 ± 0.094 | 5.3 | 17.4 |
| pso | 0.730 ± 0.115 | 0.693 ± 0.131 | 0.953 ± 0.049 | 0.859 ± 0.123 | 5.3 | 2.0 |

| Comparison | Δ F1 | p (F1) | Δ MCC | p (MCC) |
|---|---|---|---|---|
| ga_logistic vs none | -0.026 | 0.1353 | -0.027 | 0.1266 |
| ga_bpn vs ga_logistic | +0.020 | 0.3993 | +0.022 | 0.4237 |
| ga_bpn vs none | -0.005 | 0.8076 | -0.005 | 0.9353 |
| pso vs ga_logistic | +0.023 | 0.3194 | +0.026 | 0.3254 |
| pso vs none | -0.003 | 0.8791 | -0.001 | 0.9032 |

### CM1 (5×5 CV) — Hybrid BPN with different feature selectors

| Selector | F1 | MCC | ROC-AUC | Recall | #features | selection s/fold |
|---|---|---|---|---|---|---|
| none | 0.260 ± 0.135 | 0.178 ± 0.157 | 0.697 ± 0.099 | 0.309 ± 0.225 | 21.0 | 0.0 |
| ga_logistic | 0.338 ± 0.087 | 0.253 ± 0.114 | 0.705 ± 0.069 | 0.597 ± 0.162 | 6.1 | 1.8 |
| ga_bpn | 0.292 ± 0.089 | 0.192 ± 0.116 | 0.683 ± 0.096 | 0.470 ± 0.173 | 7.1 | 18.8 |
| pso | 0.333 ± 0.097 | 0.244 ± 0.125 | 0.703 ± 0.113 | 0.548 ± 0.179 | 7.1 | 1.9 |

| Comparison | Δ F1 | p (F1) | Δ MCC | p (MCC) |
|---|---|---|---|---|
| ga_logistic vs none | +0.079 | 0.0105 | +0.076 | 0.0203 |
| ga_bpn vs ga_logistic | -0.047 | 0.0177 | -0.062 | 0.0101 |
| ga_bpn vs none | +0.032 | 0.3327 | +0.014 | 0.7510 |
| pso vs ga_logistic | -0.005 | 0.7982 | -0.009 | 0.6528 |
| pso vs none | +0.073 | 0.0203 | +0.066 | 0.0667 |

### PC1 (5×5 CV) — Hybrid BPN with different feature selectors

| Selector | F1 | MCC | ROC-AUC | Recall | #features | selection s/fold |
|---|---|---|---|---|---|---|
| none | 0.358 ± 0.078 | 0.310 ± 0.089 | 0.804 ± 0.058 | 0.471 ± 0.148 | 21.0 | 0.0 |
| ga_logistic | 0.297 ± 0.044 | 0.255 ± 0.060 | 0.782 ± 0.050 | 0.589 ± 0.133 | 5.8 | 2.2 |
| ga_bpn | 0.330 ± 0.069 | 0.283 ± 0.082 | 0.812 ± 0.052 | 0.543 ± 0.129 | 7.8 | 27.8 |
| pso | 0.334 ± 0.076 | 0.297 ± 0.092 | 0.810 ± 0.057 | 0.617 ± 0.151 | 6.6 | 2.4 |

| Comparison | Δ F1 | p (F1) | Δ MCC | p (MCC) |
|---|---|---|---|---|
| ga_logistic vs none | -0.061 | 0.0013 | -0.055 | 0.0067 |
| ga_bpn vs ga_logistic | +0.033 | 0.0409 | +0.029 | 0.0957 |
| ga_bpn vs none | -0.028 | 0.0588 | -0.027 | 0.1135 |
| pso vs ga_logistic | +0.037 | 0.0074 | +0.043 | 0.0088 |
| pso vs none | -0.024 | 0.0865 | -0.012 | 0.3957 |

### JM1 (1×5 CV) — Hybrid BPN with different feature selectors

| Selector | F1 | MCC | ROC-AUC | Recall | #features | selection s/fold |
|---|---|---|---|---|---|---|
| none | 0.429 ± 0.013 | 0.203 ± 0.024 | 0.682 ± 0.007 | 0.726 ± 0.032 | 21.0 | 0.0 |
| ga_logistic | 0.415 ± 0.013 | 0.179 ± 0.027 | 0.684 ± 0.018 | 0.836 ± 0.047 | 5.0 | 6.2 |
| ga_bpn | 0.411 ± 0.009 | 0.171 ± 0.020 | 0.676 ± 0.011 | 0.833 ± 0.038 | 5.8 | 25.9 |
| pso | 0.408 ± 0.009 | 0.163 ± 0.021 | 0.677 ± 0.018 | 0.846 ± 0.031 | 5.4 | 12.3 |

| Comparison | Δ F1 | p (F1) | Δ MCC | p (MCC) |
|---|---|---|---|---|
| ga_logistic vs none | -0.014 | 0.3125 | -0.025 | 0.3125 |
| ga_bpn vs ga_logistic | -0.003 | 0.8125 | -0.008 | 0.8125 |
| ga_bpn vs none | -0.017 | 0.0625 | -0.033 | 0.0625 |
| pso vs ga_logistic | -0.007 | 0.1875 | -0.015 | 0.3125 |
| pso vs none | -0.021 | 0.1250 | -0.040 | 0.1250 |
