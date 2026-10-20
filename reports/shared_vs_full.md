# Full NASA features vs scanner-compatible shared subset

Softmax BPN on GA-selected features, mean ± std over seeds [42, 7, 13]; held-out 30% test split.
Shared subset: loc, v(g), n, v, l, d, i, e, b, t, uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount.

| Dataset | Features | accuracy | precision | recall | f1 | roc_auc | mcc |
|---|---|---|---|---|---|---|---|
| CM1 | full (21 NASA cols) | 0.787 ± 0.050 | 0.251 ± 0.037 | 0.524 ± 0.221 | 0.325 ± 0.083 | 0.774 ± 0.078 | 0.251 ± 0.096 |
| CM1 | shared (15 scanner cols) | 0.586 ± 0.177 | 0.163 ± 0.026 | 0.619 ± 0.243 | 0.242 ± 0.023 | 0.680 ± 0.085 | 0.138 ± 0.033 |
| PC1 | full (21 NASA cols) | 0.780 ± 0.018 | 0.204 ± 0.019 | 0.683 ± 0.059 | 0.313 ± 0.026 | 0.808 ± 0.028 | 0.285 ± 0.036 |
| PC1 | shared (15 scanner cols) | 0.780 ± 0.030 | 0.209 ± 0.019 | 0.683 ± 0.059 | 0.319 ± 0.017 | 0.798 ± 0.021 | 0.289 ± 0.016 |
| JM1 | full (21 NASA cols) | 0.465 ± 0.014 | 0.274 ± 0.006 | 0.838 ± 0.012 | 0.413 ± 0.006 | 0.687 ± 0.010 | 0.175 ± 0.014 |
| JM1 | shared (15 scanner cols) | 0.417 ± 0.100 | 0.269 ± 0.021 | 0.862 ± 0.088 | 0.407 ± 0.017 | 0.672 ± 0.008 | 0.143 ± 0.044 |
