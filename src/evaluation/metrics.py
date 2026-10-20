from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    mcc: float
    gmean: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def classification_metrics(y_true, y_pred, y_score=None) -> ClassificationMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_score is None:
        y_score = y_pred.astype(float)
    y_score = np.asarray(y_score, dtype=float)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    spec = tn / max(tn + fp, 1)
    rec = recall_score(y_true, y_pred, zero_division=0)
    gmean = float(np.sqrt(max(rec, 0.0) * max(spec, 0.0)))
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(rec),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=_safe_auc(y_true, y_score),
        pr_auc=float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        mcc=float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        gmean=gmean,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def roc_points(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr
