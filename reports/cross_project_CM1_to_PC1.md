# Cross-project check: train CM1, test PC1

Seed 42. The model is the dashboard pipeline fit on CM1. Same-project numbers are that run's held-out split. The PC1 numbers score every row of the other dataset.

| Set | Rows | Defective | Precision | Recall | F1 | MCC | Defects caught in top 10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| CM1 held-out | 133 | 14 | 0.222 | 0.571 | 0.320 | 0.232 | 35.7% |
| PC1 (unseen project) | 954 | 70 | 0.000 | 0.000 | 0.000 | 0.000 | 5.7% |

A drop from the first row to the second is the failure a tester hits when the training project is not the code under test.
