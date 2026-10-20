from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cv import FoldOptions, repeated_cv
from src.explainability import shap_explainer
from src.feature_selection.ga_selector import GAFeatureSelector
from src.feature_selection.pso_selector import PSOFeatureSelector
from src.fuzzy.risk_engine import FuzzyRiskEngine
from src.models.bpn import BPNClassifier
from src.models.ensemble import BPNRFEnsemble, make_rf

FAST_BPN = dict(hidden_sizes=(8, 4), dropout=0.0, learning_rate=0.01, optimizer="adam", epochs=10, batch_size=32, patience=3)
FAST_GA = dict(population_size=6, generations=2, min_features=2, lambda_penalty=0.15, mutation_probability=0.05)


def _toy(n=300, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (X[:, 0] + X[:, 2] + rng.normal(0, 0.6, n) > 0.9).astype(int)
    return X, y, [f"f{i}" for i in range(p)]


def test_pso_selector_valid_and_monotone():
    X, y, names = _toy()
    r = PSOFeatureSelector(swarm_size=8, iterations=5, random_state=1, min_features=2).fit(X, y, names)
    assert r.mask.sum() >= 2 and len(r.selected_features) == r.mask.sum()
    assert all(b >= a for a, b in zip(r.history, r.history[1:]))
    assert set(r.metadata["feature_frequency"]) == set(names)
    assert "f0" in r.selected_features or "f2" in r.selected_features


def test_ga_bpn_fitness_option():
    X, y, names = _toy()
    r = GAFeatureSelector(population_size=4, generations=1, random_state=0, min_features=2, fitness_model="bpn").fit(X, y, names)
    assert r.metadata["fitness_model"] == "bpn" and r.mask.sum() >= 2
    assert -1.0 <= r.fitness <= 1.0


def test_ensemble_weight_and_proba():
    X, y, _ = _toy(400)
    bpn = BPNClassifier(**FAST_BPN, random_state=0).fit(X[:300], y[:300])
    rf = make_rf(0).fit(X[:300], y[:300])
    ens = BPNRFEnsemble.fit_weight(bpn, rf, X[300:], y[300:])
    assert 0.0 <= ens.weight <= 1.0 and 0.0 < ens.decision_threshold_ < 1.0
    p = ens.predict_proba(X[300:])
    assert p.shape == (100, 2) and np.allclose(p.sum(1), 1.0)
    w = ens.weight
    expected = w * bpn.predict_proba(X[300:])[:, 1] + (1 - w) * rf.predict_proba(X[300:])[:, 1]
    assert np.allclose(p[:, 1], np.clip(expected, 1e-6, 1 - 1e-6))


@pytest.mark.parametrize("selector", ["none", "ga_logistic", "pso"])
def test_repeated_cv_rows(tiny_frame, selector):
    opts = FoldOptions(selector=selector, models=("Hybrid", "Ensemble", "LR"), bpn=FAST_BPN, ga=FAST_GA)
    df = repeated_cv(tiny_frame, "defects", opts, n_splits=2, n_repeats=1, seed=0, log=None)
    assert set(df["fold"]) == {0, 1}
    cls = df[~df["model"].str.startswith("Fuzzy")]
    assert set(cls["model"]) == {"Hybrid", "Ensemble", "LR"}
    assert cls["f1"].between(0, 1).all() and cls["roc_auc"].between(0, 1).all()
    assert df["model"].str.startswith("Fuzzy>=HIGH").any()


def test_escalation_rules_need_possible_probability():
    X = pd.DataFrame({"v(g)": np.arange(1, 101, dtype=float), "branchCount": np.arange(1, 101, dtype=float),
                      "loc": np.arange(1, 101, dtype=float), "e": np.arange(1, 101, dtype=float) * 50})
    eng = FuzzyRiskEngine().fit(X)
    row = {"v(g)": 90.0, "branchCount": 90.0, "loc": 90.0, "e": 4500.0}
    low = eng.assess(row, defect_prob=0.01)
    assert "R02" not in {r.rule_id for r in low.fired_rules}
    assert low.severity in {"LOW", "MEDIUM"}
    high = eng.assess(row, defect_prob=0.9)
    assert "R02" in {r.rule_id for r in high.fired_rules} and high.severity in {"HIGH", "CRITICAL"}
    assert low.memberships["defect_prob"]["AT_LEAST_POSSIBLE"] == pytest.approx(0.0)


def test_shap_values_and_fallback(monkeypatch):
    X, y, names = _toy(200, 4)
    model = BPNClassifier(**FAST_BPN, random_state=0).fit(X, y)
    bg = shap_explainer.sample_background(X, k=20)
    pairs = shap_explainer.shap_values(model, X[0], bg, names[:4], nsamples=100)
    assert pairs is not None and {n for n, _ in pairs} == set(names[:4])
    total = sum(v for _, v in pairs)
    fx = model.predict_proba(X[:1])[0, 1]
    assert total == pytest.approx(fx - shap_explainer.expected_value(model, bg), abs=1e-3)
    assert shap_explainer.shap_values(model, X[0], bg[:, :3], names[:3]) is None  # shape mismatch
    monkeypatch.setattr(shap_explainer, "HAS_SHAP", False)
    assert shap_explainer.shap_values(model, X[0], bg, names[:4]) is None
