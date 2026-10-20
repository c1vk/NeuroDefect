from __future__ import annotations

from src.fuzzy.membership import Triangle, quantile_triangles
from src.fuzzy.risk_engine import FuzzyRiskEngine


def test_membership_range():
    tri = Triangle(0, 5, 10, "MEDIUM")
    assert tri.membership(-1) == 0
    assert 0.9 <= tri.membership(5) <= 1.0
    assert 0 <= tri.membership(7) <= 1
    assert tri.membership(11) == 0


def test_quantile_triangles_cover_range():
    values = [1, 2, 3, 4, 10, 20, 30]
    tris = quantile_triangles(values)
    assert len(tris) == 3
    assert tris[0].label == "LOW"


def test_rule_evaluation_and_risk_range(tiny_frame):
    engine = FuzzyRiskEngine()
    engine.fit(tiny_frame.drop(columns=["defects"]))
    row = tiny_frame.iloc[0].to_dict()
    low = engine.assess(row, defect_prob=0.05)
    high = engine.assess(row, defect_prob=0.95)
    assert 0 <= low.risk_score <= 100
    assert 0 <= high.risk_score <= 100
    assert low.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert high.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert all(0 <= rule.strength <= 1 for rule in high.fired_rules)
    assert "complexity" in engine.available_concepts() or "loc" in engine.available_concepts()
