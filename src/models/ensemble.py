"""Weighted BPN + Random-Forest probability ensemble (weight and threshold tuned on train-only validation)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.bpn import BPNClassifier, _tune_threshold


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=1)


class BPNRFEnsemble:
    """p = w * p_BPN + (1 - w) * p_RF. ``w`` and the decision threshold are chosen on validation data."""

    def __init__(self, bpn: BPNClassifier, rf: RandomForestClassifier, weight: float = 0.5, threshold: float = 0.5):
        self.bpn = bpn
        self.rf = rf
        self.weight = float(weight)
        self.decision_threshold_ = float(threshold)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        p = self.weight * self.bpn.predict_proba(X)[:, 1] + (1 - self.weight) * self.rf.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold_).astype(int)

    @classmethod
    def fit_weight(cls, bpn: BPNClassifier, rf: RandomForestClassifier, X_val, y_val, threshold_mode: str = "f1",
                   grid=tuple(np.round(np.linspace(0, 1, 11), 2))) -> "BPNRFEnsemble":
        """``bpn`` and ``rf`` must already be fitted on data disjoint from (X_val, y_val)."""
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val).astype(int)
        pb = bpn.predict_proba(X_val)[:, 1]
        pr = rf.predict_proba(X_val)[:, 1]
        best = (-1.0, 0.5, 0.5)
        for w in grid:
            p = w * pb + (1 - w) * pr
            t, info = _tune_threshold(y_val, p, mode=threshold_mode)
            f1 = info.get("val_f1", 0.0)
            if f1 > best[0] + 1e-12:
                best = (f1, float(w), float(t))
        return cls(bpn, rf, weight=best[1], threshold=best[2])
