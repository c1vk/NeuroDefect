from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.fuzzy.risk_engine import RiskAssessment


def prediction_confidence(probability: float) -> float:
    """Distance from the 0.5 decision boundary. Not a statistical confidence interval."""
    return float(min(1.0, abs(probability - 0.5) * 2.0))


def ga_feature_ranks(X: pd.DataFrame, y: pd.Series, selected: list[str]) -> list[tuple[str, float]]:
    if not selected:
        return []
    model = LogisticRegression(max_iter=400, class_weight="balanced")
    model.fit(X[selected], y)
    coef = np.abs(model.coef_.reshape(-1))
    if coef.max() > 0:
        coef = coef / coef.max()
    order = np.argsort(-coef)
    return [(selected[i], float(coef[i])) for i in order]


def ablation_contributions(
    model,
    x_row: np.ndarray,
    feature_names: list[str],
    baseline: np.ndarray,
) -> list[tuple[str, float]]:
    base = float(model.predict_proba(x_row.reshape(1, -1))[0, 1])
    contribs = []
    for i, name in enumerate(feature_names):
        tweaked = x_row.copy()
        tweaked[i] = baseline[i]
        p = float(model.predict_proba(tweaked.reshape(1, -1))[0, 1])
        contribs.append((name, base - p))
    contribs.sort(key=lambda item: abs(item[1]), reverse=True)
    return contribs


def explain_prediction(
    probability: float,
    label: int,
    risk: RiskAssessment,
    feature_contribs: list[tuple[str, float]],
) -> dict:
    return {
        "prediction": "DEFECTIVE" if label == 1 else "NON-DEFECTIVE",
        "defect_probability": probability,
        "prediction_confidence": prediction_confidence(probability),
        "risk_severity": risk.severity,
        "risk_score": risk.risk_score,
        "risk_factors": feature_contribs[:5],
        "fuzzy_rules": risk.fired_rules[:8],
        "testing_recommendation": risk.recommendation,
        "note": (
            "Risk severity is derived from defect probability, selected software metrics, "
            "and fuzzy inference. It is not a NASA ground-truth bug-severity label."
        ),
    }
