from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.cli.main import main
from src.data.loader import describe_dataset_path
from src.review.formats import format_markdown, format_sarif, format_text
from src.review.report import ReviewReport, ReviewRow, columns_outside, review_path
from src.review.scope import ChangeSet, overlaps, parse_unified_diff
from src.utils.config import ROOT

CODE = """
def simple(a, b):
    return a + b


class Box:
    def branchy(self, x):
        if x > 1:
            return 1
        elif x < -1:
            return -1
        for i in range(3):
            if i == x:
                return i
        return 0
"""


@pytest.fixture(scope="module")
def shared_model(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("shared")
    assert main([
        "train-shared",
        "--dataset",
        str(ROOT / "data/sample/NASA_CM1_sample.csv"),
        "--out",
        str(model_dir),
        "--no-baselines",
    ]) == 0
    return model_dir


@pytest.fixture
def code_dir(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text(CODE)
    return tmp_path


def _row(**kwargs) -> ReviewRow:
    base = dict(
        rank=1, module_id="pkg/m.py::simple", file="pkg/m.py", function="simple", lineno=2,
        call="clean", defect_probability=0.2, risk_score=10, risk_tier="LOW",
        top_rule="R04", top_rule_text="IF complexity IS LOW THEN risk IS LOW",
        recommendation="Schedule a routine review.", outside=[],
    )
    base.update(kwargs)
    return ReviewRow(**base)


def _report(*rows: ReviewRow, **kwargs) -> ReviewReport:
    base = dict(
        rows=list(rows), scanned_count=2, matched_count=len(rows),
        tier_counts={"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": len(rows)},
        model_dir="/tmp/model", dataset_name="cm1_shared", threshold=0.275,
        scope="tree", warnings=[], honesty=["Model trained on cm1_shared.", "Risk severity is a derived testing priority."],
    )
    base.update(kwargs)
    return ReviewReport(**base)


def test_columns_outside_uses_training_percentile():
    assert columns_outside({"loc": 100, "v(g)": 3}, {"loc": 40}) == ["loc"]
    assert columns_outside({"loc": 40}, {"loc": 40}) == []
    assert columns_outside({"loc": 1}, {}) == []


def test_parse_diff_and_overlap(tmp_path):
    found = parse_unified_diff("+++ b/pkg/m.py\n@@ -3 +3 @@\n")
    assert found == {"pkg/m.py": {3}}
    (tmp_path / "pkg").mkdir()
    changes = ChangeSet(lines=found)
    assert overlaps(tmp_path, tmp_path, "pkg/m.py", 2, 2, changes)
    assert not overlaps(tmp_path, tmp_path, "pkg/m.py", 8, 6, changes)
    changes.whole_files.add("pkg/new.py")
    assert overlaps(tmp_path, tmp_path, "pkg/new.py", 1, 4, changes)


def test_dataset_labels():
    sample = ROOT / "data/sample/NASA_CM1_sample.csv"
    nasa = ROOT / "data/raw/cm1.csv"
    assert "Practice sample (not official NASA)" in describe_dataset_path(sample, ROOT)
    assert describe_dataset_path(nasa, ROOT).endswith("NASA CM1")


def test_formats_include_location_and_honesty():
    report = _report(_row())
    text = format_text(report)
    assert "pkg/m.py:2" in text and "derived" in text
    md = format_markdown(report)
    assert "pkg/m.py:2" in md and "cm1_shared" in md
    sarif = json.loads(format_sarif(report))
    result = sarif["runs"][0]["results"][0]
    assert result["locations"][0]["physicalLocation"]["region"]["startLine"] == 2
    assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "pkg/m.py"
    assert "derived" in sarif["runs"][0]["invocations"][0]["toolExecutionNotifications"][0]["message"]["text"]


def test_fail_on_exits_only_when_tier_is_present(monkeypatch, capsys):
    quiet = _report(_row())
    loud = _report(_row(risk_tier="CRITICAL"), tier_counts={"CRITICAL": 1, "HIGH": 0, "MEDIUM": 0, "LOW": 0})
    monkeypatch.setattr("src.review.report.review_path", lambda *args, **kwargs: quiet)
    assert main(["review", "anywhere", "--fail-on", "CRITICAL"]) == 0
    monkeypatch.setattr("src.review.report.review_path", lambda *args, **kwargs: loud)
    assert main(["review", "anywhere", "--fail-on", "CRITICAL"]) == 1
    capsys.readouterr()


def test_review_path_reports_location_and_range(code_dir, shared_model):
    report = review_path(code_dir, model=shared_model, top=0)
    assert report.scanned_count == 2
    assert {row.function for row in report.rows} == {"simple", "Box.branchy"}
    assert all(row.lineno > 0 and ":" in row.location for row in report.rows)
    assert all(row.risk_tier in {"LOW", "MEDIUM", "HIGH", "CRITICAL"} for row in report.rows)
    assert all(0 <= row.defect_probability <= 1 for row in report.rows)
    assert any("derived" in line for line in report.honesty)
    assert "sample" in report.dataset_name
    meta = json.loads((shared_model / json.loads((shared_model / "latest.json").read_text())["run_dir"] / "metadata.json").read_text())
    assert meta["feature_high"]
    name, limit = next(iter(meta["feature_high"].items()))
    assert columns_outside({name: float(limit) + 1}, meta["feature_high"]) == [name]


def test_changed_scope_drops_untouched_function(code_dir, shared_model):
    subprocess.run(["git", "init"], cwd=code_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=code_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "init"],
        cwd=code_dir, check=True, capture_output=True,
    )
    file = code_dir / "pkg" / "m.py"
    file.write_text(file.read_text().replace("return a + b", "return a + b + 1"))
    report = review_path(code_dir, model=shared_model, changed_only=True, top=0)
    assert report.scope == "diff"
    assert {row.function for row in report.rows} == {"simple"}


def test_missing_git_scores_the_whole_tree(code_dir, shared_model, monkeypatch):
    monkeypatch.setattr("src.review.report.collect_changes", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.review.report.git_root", lambda *args, **kwargs: None)
    report = review_path(code_dir, model=shared_model, changed_only=True, top=0)
    assert report.scope == "tree"
    assert any("No git diff" in warning for warning in report.warnings)
    assert {row.function for row in report.rows} == {"simple", "Box.branchy"}


def test_cli_review_prints_location(code_dir, shared_model, capsys):
    assert main(["review", str(code_dir), "--model", str(shared_model), "--top", "10"]) == 0
    text = capsys.readouterr().out
    assert "pkg/m.py:" in text and "derived" in text
    assert main(["review", str(code_dir), "--model", str(shared_model), "--format", "md"]) == 0
    assert "pkg/m.py:" in capsys.readouterr().out
    assert main(["review", str(code_dir), "--model", str(shared_model), "--format", "sarif"]) == 0
    payload = json.loads(capsys.readouterr().out)
    blob = json.dumps(payload)
    assert "pkg/m.py" in blob and "startLine" in blob
    empty = code_dir / "empty"
    empty.mkdir()
    assert main(["review", str(empty), "--model", str(shared_model)]) == 1
