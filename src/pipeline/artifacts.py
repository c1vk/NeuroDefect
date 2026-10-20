from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import joblib.numpy_pickle as _numpy_pickle
import numpy as np

from src.models.bpn import HAS_TORCH, BPNClassifier
from src.utils.config import ROOT

_LOAD_LOCK = threading.Lock()


class _TorchlessStub:
    """Stand-in for torch classes when a saved bundle is opened without torch."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs) -> None:
        return None

    def __setstate__(self, state) -> None:
        self._state = state

    def __call__(self, *args, **kwargs):
        return _TorchlessStub()


class _TorchlessUnpickler(_numpy_pickle.NumpyUnpickler):
    def find_class(self, module, name):
        if str(module).startswith("torch"):
            return _TorchlessStub
        found = super().find_class(module, name)
        # A missing optional class is stored as None in its module (see _TorchBPN).
        if found is None:
            return _TorchlessStub
        return found


def load_joblib(path: Path | str):
    """Load a joblib artifact. Torch modules are skipped when torch is not installed."""
    if HAS_TORCH:
        return joblib.load(path)
    with _LOAD_LOCK:
        original = _numpy_pickle.NumpyUnpickler
        _numpy_pickle.NumpyUnpickler = _TorchlessUnpickler
        try:
            return joblib.load(path)
        finally:
            _numpy_pickle.NumpyUnpickler = original


def new_run_dir(base: Path | None = None) -> Path:
    root = base or (ROOT / "models")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"run_{stamp}"
    suffix = 2
    while path.exists():
        path = root / f"run_{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_run(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    models = serializable.pop("models", {})
    joblib.dump(serializable, run_dir / "bundle.joblib")
    joblib.dump(models, run_dir / "sklearn_models.joblib")
    bpn = payload.get("bpn")
    if bpn is not None:
        joblib.dump(bpn.save_weights(), run_dir / "bpn_weights.joblib")
    metrics = {
        name: values.as_dict() if hasattr(values, "as_dict") else values
        for name, values in payload.get("metrics", {}).items()
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    meta = {
        "dataset": payload.get("dataset_name"),
        "seed": payload.get("seed"),
        "selected_features": payload.get("selected_features"),
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": payload.get("mode"),
        "training_seconds": payload.get("duration"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    latest = run_dir.parent / "latest.json"
    # Store the run directory relative to the models folder so the pointer is
    # portable across machines and operating systems.
    latest.write_text(json.dumps({"run_dir": run_dir.name}), encoding="utf-8")


def _resolve_run_dir(raw: str, root: Path) -> Path | None:
    """Resolve a latest.json pointer that may be relative, POSIX or Windows absolute."""
    if not raw:
        return None
    candidates = [Path(raw)]
    # Windows/POSIX absolute paths from another machine: fall back to the folder name.
    name = re.split(r"[\\/]", raw.rstrip("\\/"))[-1]
    candidates += [root / raw, root / name]
    for cand in candidates:
        try:
            if (cand / "bundle.joblib").exists():
                return cand
        except OSError:
            continue
    return None


def load_latest(base: Path | None = None) -> dict[str, Any] | None:
    root = base or (ROOT / "models")
    pointer = root / "latest.json"
    if not pointer.exists():
        return None
    try:
        raw = str(json.loads(pointer.read_text(encoding="utf-8")).get("run_dir", ""))
    except (OSError, ValueError, AttributeError):
        return None
    run_dir = _resolve_run_dir(raw, root)
    if run_dir is None:
        return None
    try:
        bundle = load_joblib(run_dir / "bundle.joblib")
        bundle["models"] = load_joblib(run_dir / "sklearn_models.joblib")
        weights_path = run_dir / "bpn_weights.joblib"
        if weights_path.exists():
            bundle["bpn"] = BPNClassifier.from_weights(load_joblib(weights_path))
    except Exception:
        # Corrupt or incompatible (e.g. pre-softmax) artifacts: treat as no saved run.
        return None
    bundle["run_dir"] = str(run_dir)
    return bundle


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
