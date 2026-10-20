from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 80
    loc = rng.normal(40, 20, n).clip(1, 200)
    vg = (loc / 10 + rng.normal(0, 2, n)).clip(1, 30)
    effort = (loc * vg * 8 + rng.normal(0, 50, n)).clip(0, 20000)
    volume = (loc * 6 + rng.normal(0, 20, n)).clip(0, 2000)
    branch = (vg * 1.5 + rng.normal(0, 1, n)).clip(1, 40)
    logit = -1.2 + 0.12 * (vg - 5) + 0.01 * (loc - 40)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    if y.sum() == 0:
        y[:8] = 1
    if y.sum() == n:
        y[:8] = 0
    return pd.DataFrame(
        {
            "module_id": [f"M_{i:03d}" for i in range(n)],
            "loc": loc,
            "v(g)": vg,
            "e": effort,
            "v": volume,
            "branchCount": branch,
            "d": rng.normal(10, 3, n).clip(0.5, 40),
            "uniq_Op": rng.normal(12, 4, n).clip(1, 40),
            "defects": y,
        }
    )


@pytest.fixture
def tiny_csv(tmp_path: Path, tiny_frame: pd.DataFrame) -> Path:
    path = tmp_path / "tiny.csv"
    tiny_frame.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path_factory, monkeypatch):
    """Keep test training runs out of the repo's models/ folder."""
    monkeypatch.setenv("NEURODEFECT_MODELS_DIR", str(tmp_path_factory.mktemp("models")))
