from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.feature_map import map_fuzzy_inputs
from src.fuzzy.membership import (
    PROBABILITY_SETS,
    RISK_SETS,
    Triangle,
    quantile_triangles,
)
from src.fuzzy.rules import FuzzyRule, default_rule_base


@dataclass
class FiredRule:
    rule_id: str
    description: str
    consequent: str
    strength: float
    rationale: str


@dataclass
class RiskAssessment:
    risk_score: float
    severity: str
    fired_rules: list[FiredRule]
    memberships: dict[str, dict[str, float]]
    available_inputs: list[str]
    missing_inputs: list[str]
    contributions: list[tuple[str, float]]
    recommendation: str


class FuzzyRiskEngine:
    """Mamdani inference over a small interpretable software-metric rule base.

    Membership boundaries for metric inputs are fit on training data quantiles.
    Defect probability uses fixed linguistic sets documented below:
    Unlikely (0-0.35), Possible (0.20-0.75), Likely (0.55-1.0).
    """

    def __init__(self, risk_bounds: dict[str, list[float]] | None = None) -> None:
        self.rules = default_rule_base()
        self.sets: dict[str, list[Triangle]] = {"defect_prob": list(PROBABILITY_SETS)}
        self.column_map: dict[str, str] = {}
        self.risk_bounds = risk_bounds or {
            "LOW": [0, 25],
            "MEDIUM": [25, 50],
            "HIGH": [50, 75],
            "CRITICAL": [75, 100],
        }

    def fit(
        self,
        X_raw: pd.DataFrame,
        custom_sets: dict[str, list[Triangle]] | None = None,
    ) -> "FuzzyRiskEngine":
        self.column_map = map_fuzzy_inputs(list(X_raw.columns))
        if custom_sets:
            # Keep only concepts we can map (plus defect_prob).
            self.sets = {"defect_prob": list(PROBABILITY_SETS)}
            for concept, column in self.column_map.items():
                if concept in custom_sets:
                    self.sets[concept] = list(custom_sets[concept])
                else:
                    self.sets[concept] = quantile_triangles(X_raw[column].to_numpy())
            if "defect_prob" in custom_sets:
                self.sets["defect_prob"] = list(custom_sets["defect_prob"])
        else:
            for concept, column in self.column_map.items():
                self.sets[concept] = quantile_triangles(X_raw[column].to_numpy())
        return self

    def available_concepts(self) -> list[str]:
        return list(self.column_map.keys()) + ["defect_prob"]

    def assess(self, raw_row: dict[str, float], defect_prob: float) -> RiskAssessment:
        memberships: dict[str, dict[str, float]] = {}
        values: dict[str, float] = {"defect_prob": float(np.clip(defect_prob, 0.0, 1.0))}
        missing: list[str] = []
        for concept, triangles in self.sets.items():
            if concept == "defect_prob":
                x = values["defect_prob"]
            else:
                column = self.column_map.get(concept)
                if column is None or column not in raw_row or raw_row[column] is None:
                    missing.append(concept)
                    continue
                try:
                    x = float(raw_row[column])
                except (TypeError, ValueError):
                    missing.append(concept)
                    continue
                if not np.isfinite(x):
                    missing.append(concept)
                    continue
                values[concept] = x
            memberships[concept] = {tri.label: tri.membership(x) for tri in triangles}
            if concept == "defect_prob":
                # Fuzzy OR (max) of POSSIBLE and LIKELY: metric-driven escalation rules
                # require the neural model to see at least a possible defect.
                m = memberships[concept]
                m["AT_LEAST_POSSIBLE"] = max(m.get("POSSIBLE", 0.0), m.get("LIKELY", 0.0))

        fired: list[FiredRule] = []
        clipped: dict[str, float] = {s.label: 0.0 for s in RISK_SETS}
        for rule in self.rules:
            strength = _rule_strength(rule, memberships)
            if strength <= 0.02:
                continue
            fired.append(
                FiredRule(
                    rule_id=rule.rule_id,
                    description=rule.describe(),
                    consequent=rule.consequent,
                    strength=float(strength),
                    rationale=rule.rationale,
                )
            )
            clipped[rule.consequent] = max(clipped[rule.consequent], strength)

        fired.sort(key=lambda item: item.strength, reverse=True)
        score = _centroid(clipped)
        severity = self._severity(score)
        contrib = _feature_contributions(fired)
        recommendation = _recommendation(severity, fired, defect_prob)
        return RiskAssessment(
            risk_score=score,
            severity=severity,
            fired_rules=fired,
            memberships=memberships,
            available_inputs=list(memberships.keys()),
            missing_inputs=missing,
            contributions=contrib,
            recommendation=recommendation,
        )

    def _severity(self, score: float) -> str:
        for label, (lo, hi) in self.risk_bounds.items():
            if lo <= score < hi:
                return label
        return "CRITICAL" if score >= 75 else "LOW"

    def to_dict(self) -> dict:
        return {
            "column_map": self.column_map,
            "sets": {
                key: [(t.a, t.b, t.c, t.label) for t in tris] for key, tris in self.sets.items()
            },
            "risk_bounds": self.risk_bounds,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FuzzyRiskEngine":
        engine = cls(risk_bounds=payload.get("risk_bounds"))
        engine.column_map = payload.get("column_map", {})
        engine.sets = {
            key: [Triangle(*vals) for vals in tris]
            for key, tris in payload.get("sets", {}).items()
        }
        if "defect_prob" not in engine.sets:
            engine.sets["defect_prob"] = list(PROBABILITY_SETS)
        return engine


def _rule_strength(rule: FuzzyRule, memberships: dict[str, dict[str, float]]) -> float:
    degrees: list[float] = []
    for variable, label in rule.antecedents:
        if variable not in memberships:
            return 0.0
        degrees.append(float(memberships[variable].get(label, 0.0)))
    return float(min(degrees)) if degrees else 0.0


_XS = np.linspace(0, 100, 201)
_RISK_VECS: dict[str, np.ndarray] = {}


def _risk_vec(tri: Triangle) -> np.ndarray:
    vec = _RISK_VECS.get(tri.label)
    if vec is None:
        vec = _tri_vec(_XS, tri)
        _RISK_VECS[tri.label] = vec
    return vec


def _centroid(clipped: dict[str, float]) -> float:
    xs = _XS
    agg = np.zeros_like(xs)
    for tri in RISK_SETS:
        mu = np.minimum(clipped.get(tri.label, 0.0), _risk_vec(tri))
        agg = np.maximum(agg, mu)
    mass = float(agg.sum())
    if mass <= 1e-9:
        return 0.0
    return float(np.sum(xs * agg) / mass)


def _tri_vec(xs: np.ndarray, tri: Triangle) -> np.ndarray:
    return np.array([tri.membership(float(x)) for x in xs])


def _feature_contributions(fired: list[FiredRule]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rule in fired:
        sign = 1.0 if rule.consequent in {"HIGH", "CRITICAL"} else -0.4 if rule.consequent == "LOW" else 0.2
        tokens = rule.description.replace("IF ", "").split(" THEN ")[0].split(" AND ")
        for token in tokens:
            feature = token.split(" IS ")[0].strip()
            if feature == "defect_prob":
                continue
            scores[feature] = scores.get(feature, 0.0) + sign * rule.strength
    ranked = sorted(scores.items(), key=lambda item: abs(item[1]), reverse=True)
    return ranked


def _recommendation(severity: str, fired: list[FiredRule], defect_prob: float) -> str:
    top = fired[0] if fired else None
    if severity == "CRITICAL":
        detail = top.rationale if top else "This module sits in the highest derived risk band."
        return (
            "IMMEDIATE CODE REVIEW REQUIRED. "
            f"{detail} Prioritize this module for manual testing and inspection."
        )
    if severity == "HIGH":
        return (
            "Prioritize this module for manual testing and code review. "
            f"Predicted defect probability is {defect_prob:.1%} with HIGH derived risk severity."
        )
    if severity == "MEDIUM":
        return (
            "Schedule a targeted review of the highlighted metrics. "
            "Derived risk is medium — not ground-truth bug severity."
        )
    return (
        "Routine testing is sufficient based on low derived risk and "
        f"a defect probability of {defect_prob:.1%}."
    )
