from __future__ import annotations

import json

import numpy as np
import pytest

from src.models import bpn as bpn_mod
from src.models.bpn import BPNClassifier, _best_f1_threshold, _softmax


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n) > 0.8).astype(int)
    return X, y


def test_softmax_rows_sum_to_one():
    p = _softmax(np.array([[1.0, 2.0], [1000.0, -1000.0], [0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[2], [0.5, 0.5])
    assert np.isfinite(p).all()


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_bpn_softmax_outputs(backend, monkeypatch):
    if backend == "torch" and not bpn_mod.HAS_TORCH:
        pytest.skip("torch not installed")
    if backend == "numpy":
        monkeypatch.setattr(bpn_mod, "HAS_TORCH", False)
    X, y = _data()
    model = BPNClassifier(hidden_sizes=(16, 8), epochs=30, patience=10, random_state=1).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert model.backend_ == backend
    assert 0.0 < model.decision_threshold_ < 1.0
    pred = model.predict(X)
    assert np.array_equal(pred, (proba[:, 1] >= model.decision_threshold_).astype(int))
    # learns something on an easy separable-ish problem
    assert (pred == y).mean() > 0.75
    # weights round-trip (output layer has 2 units)
    clone = BPNClassifier.from_weights(model.save_weights())
    assert np.allclose(clone.predict_proba(X), proba, atol=1e-5)
    assert clone.decision_threshold_ == pytest.approx(model.decision_threshold_)


def test_numpy_softmax_gradient_matches_finite_difference(monkeypatch):
    """Check the analytic softmax-CE backprop on the output layer numerically."""
    monkeypatch.setattr(bpn_mod, "HAS_TORCH", False)
    X, y = _data(40, seed=3)
    m = BPNClassifier(hidden_sizes=(4, 3), dropout=0.0, epochs=1, random_state=0, tune_threshold=False)
    m.fit(X, y, X, y)
    W1, b1, W2, b2, W3, b3 = [w.astype(np.float64) for w in m.weights_]
    cw = bpn_mod._class_weights(y).astype(np.float64)

    def loss(b3v):
        a2 = np.maximum(np.maximum(X @ W1 + b1, 0) @ W2 + b2, 0)
        p = _softmax(a2 @ W3 + b3v).astype(np.float64)
        return np.mean(-cw[y] * np.log(p[np.arange(len(y)), y]))

    a2 = np.maximum(np.maximum(X @ W1 + b1, 0) @ W2 + b2, 0)
    p = _softmax(a2 @ W3 + b3).astype(np.float64)
    oh = np.eye(2)[y]
    analytic = ((cw[y][:, None] * (p - oh)) / len(y)).sum(axis=0)
    eps = 1e-4
    numeric = np.array([(loss(b3 + eps * np.eye(2)[i]) - loss(b3 - eps * np.eye(2)[i])) / (2 * eps) for i in range(2)])
    assert np.allclose(analytic, numeric, atol=1e-3)


def test_threshold_tuning_improves_or_matches_f1():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.45])
    t, f1 = _best_f1_threshold(y, s)
    assert f1 == pytest.approx(1.0)
    assert 0.35 < t <= 0.4


def test_torch_state_dict_loads_without_torch(monkeypatch):
    monkeypatch.setattr(bpn_mod, "HAS_TORCH", False)
    rng = np.random.default_rng(0)
    n, h0, h1 = 5, 4, 3
    state = {
        "net.0.weight": rng.normal(size=(h0, n)).astype(np.float32),
        "net.0.bias": rng.normal(size=(h0,)).astype(np.float32),
        "net.3.weight": rng.normal(size=(h1, h0)).astype(np.float32),
        "net.3.bias": rng.normal(size=(h1,)).astype(np.float32),
        "net.6.weight": rng.normal(size=(2, h1)).astype(np.float32),
        "net.6.bias": rng.normal(size=(2,)).astype(np.float32),
    }
    model = BPNClassifier.from_weights(
        {
            "n_features": n,
            "hidden": (h0, h1),
            "dropout": 0.0,
            "decision_threshold": 0.42,
            "backend": "torch",
            "state_dict": state,
        }
    )
    assert model.backend_ == "numpy"
    X = rng.normal(size=(6, n)).astype(np.float32)
    a1 = np.maximum(X @ state["net.0.weight"].T + state["net.0.bias"], 0)
    a2 = np.maximum(a1 @ state["net.3.weight"].T + state["net.3.bias"], 0)
    expected = _softmax(a2 @ state["net.6.weight"].T + state["net.6.bias"])
    expected = np.clip(expected, 1e-6, 1 - 1e-6)
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(model.predict_proba(X), expected, atol=1e-5)
    assert model.decision_threshold_ == pytest.approx(0.42)


def test_saved_torch_bundle_scores_without_torch():
    """The checked-in shared CM1 run was trained with torch. The UI must still open it."""
    from src.pipeline.inference import load_model, predict_frame
    from src.utils.config import ROOT

    run = ROOT / "models" / "shared" / "cm1_seed42"
    if not (run / "bundle.joblib").exists():
        pytest.skip("shared CM1 bundle is not in this checkout")
    model = load_model(run)
    frame = pd_frame(model.all_features)
    table = predict_frame(model, frame)
    assert len(table) == 1
    assert 0.0 <= float(table.loc[0, "defect_probability"]) <= 1.0
    assert table.loc[0, "risk_tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def pd_frame(columns):
    import pandas as pd

    return pd.DataFrame([{name: 1.0 for name in columns}])


def test_load_latest_handles_windows_pointer(tmp_path):
    from src.pipeline.artifacts import load_latest, save_run

    X, y = _data(60)
    model = BPNClassifier(hidden_sizes=(4, 3), epochs=2, random_state=0).fit(X, y)
    run = tmp_path / "run_x"
    save_run(run, {"bpn": model, "models": {}, "metrics": {}})
    assert json.loads((tmp_path / "latest.json").read_text())["run_dir"] == "run_x"
    (tmp_path / "latest.json").write_text(json.dumps({"run_dir": "D:\\\\Projects\\\\NeuroDefect\\\\models\\\\run_x"}))
    bundle = load_latest(tmp_path)
    assert bundle is not None and bundle["bpn"].decision_threshold_ == pytest.approx(model.decision_threshold_)
    (tmp_path / "latest.json").write_text("{not json")
    assert load_latest(tmp_path) is None
