"""SHAP attributions for the defect probability, with a graceful fallback when `shap` is absent.

KernelExplainer is model-agnostic, so it works for the softmax BPN and for the BPN+RF ensemble
(any object with ``predict_proba``). A small sampled background (<= 50 training rows, already
scaled and restricted to the selected features) keeps it interactive.
"""
from __future__ import annotations

import numpy as np

try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    shap = None
    HAS_SHAP = False


def sample_background(X: np.ndarray, k: int = 50, seed: int = 0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if len(X) <= k:
        return X.copy()
    idx = np.random.default_rng(seed).choice(len(X), size=k, replace=False)
    return X[idx]


def shap_values(model, x_row, background, feature_names: list[str], nsamples: int = 200) -> list[tuple[str, float]] | None:
    """Return [(feature, shap value)] sorted by |value| for P(defect), or None if SHAP is unavailable."""
    if not HAS_SHAP or background is None:
        return None
    background = np.asarray(background, dtype=np.float32)
    x = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    if background.ndim != 2 or background.shape[1] != x.shape[1]:
        return None

    def f(Z):
        return model.predict_proba(np.asarray(Z, dtype=np.float32))[:, 1]

    explainer = shap.KernelExplainer(f, background)
    vals = np.asarray(explainer.shap_values(x, nsamples=nsamples, silent=True)).reshape(-1)
    pairs = [(n, float(v)) for n, v in zip(feature_names, vals)]
    return sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)


def expected_value(model, background) -> float:
    return float(np.mean(model.predict_proba(np.asarray(background, dtype=np.float32))[:, 1]))
