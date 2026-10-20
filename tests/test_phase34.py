from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.fuzzy.ga_tuner import FuzzyMembershipGA, decode_unit_genes, triangles_valid
from src.models.bpn import _tune_threshold
from src.pipeline.trainer import train_pipeline
from src.utils.config import load_config


@pytest.mark.parametrize("genes", [[0.9, 0.1, 0.5, 0.3], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1], [2, -1, 0.3, 7]])
def test_decoded_triangles_ordered_and_valid(genes):
    tris = decode_unit_genes(np.array(genes, float), 0.0, 10.0)
    assert triangles_valid(tris)
    assert [t.label for t in tris] == ["LOW", "MEDIUM", "HIGH"]
    assert tris[0].b <= tris[1].b <= tris[2].b
    assert tris[0].a == 0.0 and tris[2].c >= 10.0 - 1e-9


def test_fuzzy_ga_is_train_only_and_returns_valid_sets(tiny_frame):
    X = tiny_frame.drop(columns=["module_id", "defects"])
    y = tiny_frame["defects"].to_numpy()
    probs = np.clip(0.2 + 0.6 * y + np.random.default_rng(0).normal(0, 0.1, len(y)), 0, 1)
    res = FuzzyMembershipGA(population_size=6, generations=3, random_state=0).fit(X, y, probs)
    assert len(res.history) == 4
    assert all(b >= a for a, b in zip(res.history, res.history[1:]))  # elitist best-so-far
    assert all(triangles_valid(t) for t in res.sets.values())
    assert "defect_prob" in res.sets


def test_threshold_modes():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.15).astype(int)
    s = np.clip(0.3 * y + rng.normal(0.35, 0.15, 400), 0, 1)
    t_f1, _ = _tune_threshold(y, s, "f1")
    t_fl, info = _tune_threshold(y, s, "accuracy_floor", accuracy_floor=0.8)
    assert info["val_accuracy"] >= 0.8
    assert t_fl >= t_f1 - 1e-9 or info["val_accuracy"] >= 0.8
    t_c, _ = _tune_threshold(y, s, "cost_sensitive", fn_cost=10.0)
    t_c1, _ = _tune_threshold(y, s, "cost_sensitive", fn_cost=0.5)
    assert t_c <= t_c1  # costlier misses -> lower threshold


@pytest.mark.parametrize("variant", ["full", "no_ga", "no_smote", "no_fuzzy"])
def test_pipeline_variants(tiny_csv, variant):
    cfg = load_config().with_overrides(
        training={"epochs": 6, "patience": 3, "random_state": 0},
        ga={"population_size": 6, "generations": 2, "random_state": 0, "min_features": 2},
        fuzzy={"ga_tune": True, "ga": {"population_size": 4, "generations": 2}},
    )
    r = train_pipeline(load_dataset(tiny_csv), cfg, variant=variant, with_baselines=False)
    g = r.leakage_guards
    assert set(g["train_index"]).isdisjoint(g["test_index"])
    if variant == "no_ga":
        assert r.selected_features == r.all_features and r.ga_history == []
    else:
        assert len(r.ga_history) == 3 and set(r.ga_feature_frequency) == set(r.all_features)
        assert all(0 <= v <= 1 for v in r.ga_feature_frequency.values())
    if variant == "no_smote":
        assert r.smote.applied is False
    if variant == "no_fuzzy":
        assert r.fuzzy_mode == "disabled" and r.fuzzy_metrics_test == {}
    else:
        assert r.fuzzy_mode == "ga_tuned" and 0 <= r.fuzzy_metrics_test["high_f1"] <= 1
        assert "high_f1" in r.fuzzy_metrics_fixed_test


def test_plots_written(tmp_path):
    from src.evaluation.plots import plot_feature_frequency, plot_ga_fitness
    a = plot_ga_fitness({"1": [0.1, 0.2, 0.3]}, tmp_path / "f.png", "t")
    b = plot_feature_frequency({"loc": 0.8, "v(g)": 0.2}, tmp_path / "q.png", "t")
    assert a.stat().st_size > 1000 and b.stat().st_size > 1000
