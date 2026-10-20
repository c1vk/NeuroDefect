from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.cli.main import main
from src.data.columns import canonicalize_nasa_columns, drop_derived_halstead, shared_subset
from src.scanner.radon_scanner import MODEL_COLUMNS, SHARED_COLUMNS, UNMAPPABLE, scan_folder, token_halstead

ROOT = Path(__file__).resolve().parents[1]

CODE = '''
def simple(a, b):
    return a + b


class Box:
    def branchy(self, x):
        # decide
        if x > 1:
            return 1
        elif x < -1:
            return -1
        for i in range(3):
            if i == x:
                return i
        return 0
'''


@pytest.fixture
def code_dir(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text(CODE)
    (tmp_path / "pkg" / "bad.py").write_text("def broken(:\n")
    return tmp_path


def test_scan_rows_and_mapping(code_dir):
    df = scan_folder(code_dir)
    assert set(SHARED_COLUMNS) <= set(df.columns)
    assert not any(c in df.columns for c in UNMAPPABLE)
    rows = df.set_index("function")
    assert set(rows.index) == {"simple", "Box.branchy"}  # syntax-error file skipped
    assert rows.loc["simple", "v(g)"] == 1 and rows.loc["simple", "branchCount"] == 1
    b = rows.loc["Box.branchy"]
    assert b["v(g)"] == 5 and b["branchCount"] == 9
    assert b["lOComment"] == 1 and b["loc"] == 10
    assert b["uniq_Op"] > 0 and b["total_Opnd"] >= b["uniq_Opnd"] and b["v"] > 0


def test_token_halstead_known_values():
    h = token_halstead("x = a + b")
    # operators: '=', '+'; operands: x, a, b
    assert (h["uniq_Op"], h["uniq_Opnd"], h["total_Op"], h["total_Opnd"]) == (2, 3, 2, 3)
    assert h["n"] == 5 and h["d"] == pytest.approx(1.0)


def test_derived_halstead_columns_drop_only_when_parents_exist():
    frame = pd.DataFrame({"v": [1.0], "b": [0.1], "e": [2.0], "t": [0.2], "loc": [3.0]})
    reduced, dropped = drop_derived_halstead(frame)
    assert dropped == ["b", "t"]
    assert "loc" in reduced.columns and "n" not in dropped


def test_canonicalize_and_shared_subset():
    f = pd.DataFrame({"LOC": [1], "iv(G)": [2], "V": [3], "defects": [0]})
    assert list(canonicalize_nasa_columns(f).columns) == ["loc", "iv(g)", "v", "defects"]
    full = pd.read_csv(ROOT / "data/sample/NASA_CM1_sample.csv")
    sub = shared_subset(full, "defects")
    assert list(sub.columns) == MODEL_COLUMNS + ["defects"]
    with pytest.raises(ValueError):
        shared_subset(f, "defects")


def test_cli_scan_train_predict(code_dir, tmp_path, capsys):
    metrics = tmp_path / "metrics.csv"
    assert main(["scan", str(code_dir), "-o", str(metrics)]) == 0
    model_dir = tmp_path / "shared"
    assert main(["train-shared", "--dataset", str(ROOT / "data/sample/NASA_CM1_sample.csv"),
                 "--out", str(model_dir), "--no-baselines"]) == 0
    out_csv, out_json = tmp_path / "p.csv", tmp_path / "p.json"
    assert main(["predict", str(metrics), "--model", str(model_dir), "--csv-out", str(out_csv),
                 "--json-out", str(out_json)]) == 0
    text = capsys.readouterr().out
    assert "Tiers:" in text and "Box.branchy" in text
    table = pd.read_csv(out_csv)
    assert list(table["rank"]) == [1, 2]
    assert set(table["risk_tier"]) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert table["defect_probability"].between(0, 1).all()
    assert len(json.loads(out_json.read_text())) == 2


def test_python_m_entrypoint_help():
    r = subprocess.run([sys.executable, "-m", "neurodefect", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "scan" in r.stdout and "predict" in r.stdout
