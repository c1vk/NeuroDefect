# NeuroDefect ablation study

Mean ± std over seeds 42, 7, 13. Stratified 70/30 split per seed; preprocessing, SMOTE, GA feature selection, BPN early stopping/threshold and fuzzy-membership GA use training data only (fuzzy-GA fitness uses 3-fold out-of-fold BPN probabilities on the training split). Metrics are on the untouched held-out test split.

- **BPN (softmax)** rows: classifier decision (softmax class-1 probability ≥ validation-tuned threshold).
- **no-fuzzy** = BPN decision alone (the `full` BPN row). **Fuzzy** rows: decision = derived risk tier ≥ HIGH; `risk_corr` = Pearson correlation of the 0–100 risk score with the defect label.
- `acc_floor`: threshold maximises validation F1 subject to validation accuracy ≥ 0.65.

## synthetic_sample

| Variant | Component | accuracy | precision | recall | f1 | roc_auc | mcc | risk_corr |
|---|---|---|---|---|---|---|---|---|
| full | BPN (softmax) | 0.901 ± 0.024 | 0.619 ± 0.075 | 0.844 ± 0.083 | 0.711 ± 0.065 | 0.944 ± 0.020 | 0.667 ± 0.078 | — |
| full | Fuzzy tier>=HIGH (fixed quantile) | — | 0.228 ± 0.006 | 0.956 ± 0.031 | 0.368 ± 0.009 | — | 0.294 ± 0.020 | 0.300 ± 0.024 |
| full | Fuzzy tier>=HIGH (GA-tuned) | — | 0.563 ± 0.037 | 0.644 ± 0.113 | 0.594 ± 0.039 | — | 0.528 ± 0.046 | 0.615 ± 0.030 |
| no_ga | BPN (softmax) | 0.901 ± 0.016 | 0.635 ± 0.071 | 0.778 ± 0.031 | 0.695 ± 0.028 | 0.948 ± 0.016 | 0.644 ± 0.032 | — |
| no_smote | BPN (softmax) | 0.901 ± 0.018 | 0.654 ± 0.077 | 0.689 ± 0.126 | 0.663 ± 0.066 | 0.939 ± 0.020 | 0.611 ± 0.080 | — |
| acc_floor | BPN (softmax) | 0.901 ± 0.024 | 0.619 ± 0.075 | 0.844 ± 0.083 | 0.711 ± 0.065 | 0.944 ± 0.020 | 0.667 ± 0.078 | — |

## CM1

| Variant | Component | accuracy | precision | recall | f1 | roc_auc | mcc | risk_corr |
|---|---|---|---|---|---|---|---|---|
| full | BPN (softmax) | 0.787 ± 0.050 | 0.251 ± 0.037 | 0.524 ± 0.221 | 0.325 ± 0.083 | 0.774 ± 0.078 | 0.251 ± 0.096 | — |
| full | Fuzzy tier>=HIGH (fixed quantile) | — | 0.148 ± 0.011 | 0.643 ± 0.058 | 0.241 ± 0.019 | — | 0.129 ± 0.035 | 0.144 ± 0.049 |
| full | Fuzzy tier>=HIGH (GA-tuned) | — | 0.211 ± 0.039 | 0.286 ± 0.154 | 0.225 ± 0.061 | — | 0.135 ± 0.061 | 0.166 ± 0.073 |
| no_ga | BPN (softmax) | 0.807 ± 0.032 | 0.259 ± 0.075 | 0.429 ± 0.101 | 0.321 ± 0.083 | 0.765 ± 0.061 | 0.228 ± 0.100 | — |
| no_smote | BPN (softmax) | 0.799 ± 0.037 | 0.279 ± 0.016 | 0.548 ± 0.147 | 0.360 ± 0.027 | 0.772 ± 0.021 | 0.285 ± 0.036 | — |
| acc_floor | BPN (softmax) | 0.787 ± 0.050 | 0.251 ± 0.037 | 0.524 ± 0.221 | 0.325 ± 0.083 | 0.774 ± 0.078 | 0.251 ± 0.096 | — |

## PC1

| Variant | Component | accuracy | precision | recall | f1 | roc_auc | mcc | risk_corr |
|---|---|---|---|---|---|---|---|---|
| full | BPN (softmax) | 0.780 ± 0.018 | 0.204 ± 0.019 | 0.683 ± 0.059 | 0.313 ± 0.026 | 0.808 ± 0.028 | 0.285 ± 0.036 | — |
| full | Fuzzy tier>=HIGH (fixed quantile) | — | 0.111 ± 0.004 | 0.778 ± 0.045 | 0.194 ± 0.007 | — | 0.149 ± 0.017 | 0.156 ± 0.005 |
| full | Fuzzy tier>=HIGH (GA-tuned) | — | 0.185 ± 0.030 | 0.286 ± 0.140 | 0.215 ± 0.072 | — | 0.154 ± 0.067 | 0.201 ± 0.038 |
| no_ga | BPN (softmax) | 0.872 ± 0.019 | 0.288 ± 0.025 | 0.492 ± 0.112 | 0.357 ± 0.039 | 0.815 ± 0.035 | 0.309 ± 0.045 | — |
| no_smote | BPN (softmax) | 0.915 ± 0.014 | 0.387 ± 0.133 | 0.190 ± 0.067 | 0.247 ± 0.078 | 0.836 ± 0.032 | 0.226 ± 0.086 | — |
| acc_floor | BPN (softmax) | 0.780 ± 0.018 | 0.204 ± 0.019 | 0.683 ± 0.059 | 0.313 ± 0.026 | 0.808 ± 0.028 | 0.285 ± 0.036 | — |

## JM1

| Variant | Component | accuracy | precision | recall | f1 | roc_auc | mcc | risk_corr |
|---|---|---|---|---|---|---|---|---|
| full | BPN (softmax) | 0.465 ± 0.014 | 0.274 ± 0.006 | 0.838 ± 0.012 | 0.413 ± 0.006 | 0.687 ± 0.010 | 0.175 ± 0.014 | — |
| full | Fuzzy tier>=HIGH (fixed quantile) | — | 0.302 ± 0.009 | 0.651 ± 0.020 | 0.413 ± 0.012 | — | 0.180 ± 0.021 | 0.196 ± 0.017 |
| full | Fuzzy tier>=HIGH (GA-tuned) | — | 0.368 ± 0.028 | 0.356 ± 0.064 | 0.356 ± 0.021 | — | 0.177 ± 0.010 | 0.196 ± 0.036 |
| no_ga | BPN (softmax) | 0.546 ± 0.043 | 0.300 ± 0.016 | 0.752 ± 0.044 | 0.428 ± 0.011 | 0.676 ± 0.008 | 0.202 ± 0.020 | — |
| no_smote | BPN (softmax) | 0.630 ± 0.062 | 0.344 ± 0.032 | 0.671 ± 0.075 | 0.451 ± 0.014 | 0.693 ± 0.006 | 0.247 ± 0.027 | — |
| acc_floor | BPN (softmax) | 0.647 ± 0.071 | 0.353 ± 0.042 | 0.613 ± 0.095 | 0.440 ± 0.012 | 0.687 ± 0.010 | 0.237 ± 0.030 | — |
