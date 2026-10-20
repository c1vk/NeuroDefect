from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.explainability.explainer import ablation_contributions, explain_prediction, prediction_confidence
from src.fuzzy.risk_engine import RiskAssessment
from src.pipeline.trainer import TrainingResult
from src.utils.exceptions import PredictionError


@dataclass
class ModulePrediction:
    module_id: str
    probability: float
    label: str
    confidence: float
    severity: str
    risk: RiskAssessment
    feature_contribs: list[tuple[str, float]]
    explanation: dict
    inspection_date: str


def predict_from_raw_row(
    result: TrainingResult,
    raw_row: dict[str, float],
    module_id: str = "manual",
) -> ModulePrediction:
    if result.bpn is None:
        raise PredictionError("Please train a model before running predictions.")
    frame = pd.DataFrame([raw_row])
    for column in result.all_features:
        if column not in frame.columns:
            frame[column] = np.nan
    try:
        scaled = result.preprocessor.transform(frame[result.all_features])
    except Exception as exc:
        raise PredictionError("Invalid manual input. Numeric software metrics are required.") from exc
    selected = scaled[result.selected_features]
    x = selected.to_numpy(dtype=np.float32)[0]
    if not np.isfinite(x).all():
        raise PredictionError("Input contains NaN or infinite values after preprocessing.")
    scorer = getattr(result, "ensemble", None) or result.bpn
    prob = float(scorer.predict_proba(x.reshape(1, -1))[0, 1])
    thr = float(getattr(scorer, "decision_threshold_", 0.5))
    label = int(prob >= thr)
    baseline = np.zeros_like(x)
    contribs = ablation_contributions(
        scorer, x, result.selected_features, baseline
    )
    raw_metrics = {k: _to_float(v) for k, v in raw_row.items()}
    risk = result.fuzzy.assess(raw_metrics, defect_prob=prob)
    explanation = explain_prediction(prob, label, risk, contribs)
    return ModulePrediction(
        module_id=module_id,
        probability=prob,
        label="DEFECTIVE" if label else "NON-DEFECTIVE",
        confidence=prediction_confidence(prob),
        severity=risk.severity,
        risk=risk,
        feature_contribs=contribs,
        explanation=explanation,
        inspection_date=date.today().isoformat(),
    )


def predict_module(result: TrainingResult, test_row_index: int) -> ModulePrediction:
    if test_row_index < 0 or test_row_index >= len(result.X_test_raw):
        raise PredictionError("Selected module is outside the held-out evaluation set.")
    raw = result.X_test_raw.iloc[test_row_index].to_dict()
    module_id = result.display_ids[test_row_index]
    return predict_from_raw_row(result, raw, module_id=module_id)


def recent_predictions(result: TrainingResult, limit: int = 5) -> list[ModulePrediction]:
    scores = result.scores.get("Neuro-Fuzzy-GA")
    if scores is None:
        return []
    order = np.argsort(-np.abs(scores - 0.5))
    chosen = list(order[:limit])
    return [predict_module(result, int(i)) for i in chosen]


def _to_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
