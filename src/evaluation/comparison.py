from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import ClassificationMetrics


def metrics_table(results: dict[str, ClassificationMetrics]) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "Classifier Model Architecture": name,
                "Accuracy": metrics.accuracy,
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "F1-Score": metrics.f1,
                "Matthews CC": metrics.mcc,
                "ROC-AUC": metrics.roc_auc,
                "PR-AUC": metrics.pr_auc,
                "G-Mean": metrics.gmean,
            }
        )
    return pd.DataFrame(rows)
