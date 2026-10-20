from __future__ import annotations

import numpy as np

from src.feature_selection.ga_selector import GAFeatureSelector, _cv_score


def test_chromosome_and_selected_features(tiny_frame):
    X = tiny_frame.drop(columns=["module_id", "defects"]).to_numpy()
    y = tiny_frame["defects"].to_numpy().astype(int)
    names = [c for c in tiny_frame.columns if c not in {"module_id", "defects"}]
    selector = GAFeatureSelector(
        population_size=8,
        generations=3,
        random_state=0,
        min_features=2,
        max_features=5,
    )
    result = selector.fit(X, y, names)
    assert result.mask.sum() >= 2
    assert result.mask.sum() <= 5
    assert set(result.selected_features).issubset(set(names))
    assert len(result.mask) == X.shape[1]
    assert len(result.history) == 4  # gen 0 plus 3


def test_fitness_is_finite(tiny_frame):
    X = tiny_frame.drop(columns=["module_id", "defects"]).to_numpy()
    y = tiny_frame["defects"].to_numpy().astype(int)
    score = _cv_score(X, y, "f1", 0)
    assert np.isfinite(score)
    assert score >= 0.0
