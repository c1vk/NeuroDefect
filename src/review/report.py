"""Scan a Python tree and rank functions with a saved NeuroDefect model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.pipeline.inference import SavedModel, load_model, predict_frame
from src.review.scope import collect_changes, git_root, overlaps
from src.scanner.radon_scanner import scan_folder


@dataclass
class ReviewRow:
    rank: int
    module_id: str
    file: str
    function: str
    lineno: int
    call: str
    defect_probability: float
    risk_score: float
    risk_tier: str
    top_rule: str
    top_rule_text: str
    recommendation: str
    outside: list[str] = field(default_factory=list)

    @property
    def location(self) -> str:
        return f"{self.file}:{self.lineno}"


@dataclass
class ReviewReport:
    rows: list[ReviewRow]
    scanned_count: int
    matched_count: int
    tier_counts: dict[str, int]
    model_dir: str
    dataset_name: str
    threshold: float
    scope: str
    warnings: list[str]
    honesty: list[str]


def honesty_lines(model: SavedModel) -> list[str]:
    dataset = str(model.meta.get("dataset") or "the saved training set")
    return [
        f"Model trained on {dataset}.",
        "Risk severity is a derived testing priority, not a NASA bug-severity label.",
        "Python scores are structural analogues of C/C++ metrics, not measured Python defects.",
        (
            f"Decision threshold {model.threshold:.3f}. "
            "A probability at or above this cutoff is labeled defective."
        ),
    ]


def columns_outside(values: dict, limits: dict) -> list[str]:
    """Feature names whose raw value sits above the training 95th percentile."""
    flagged: list[str] = []
    for name, limit in limits.items():
        raw = values.get(name)
        if raw is None or pd.isna(raw):
            continue
        try:
            if float(raw) > float(limit):
                flagged.append(str(name))
        except (TypeError, ValueError):
            continue
    return flagged


def review_path(
    path: str | Path,
    *,
    model: str | Path | None = None,
    top: int = 10,
    changed_only: bool = False,
    base: str = "HEAD",
    halstead: str = "token",
) -> ReviewReport:
    """Score ``path`` and return the top rows a reviewer can open."""
    target = Path(path)
    saved = load_model(model)
    frame = scan_folder(target, halstead=halstead)
    scan_root = target.resolve().parent if target.is_file() else target.resolve()
    warnings: list[str] = []
    scope = "tree"
    matched = frame
    if changed_only:
        changes = collect_changes(target, base)
        repo = git_root(target)
        if changes is None or repo is None:
            warnings.append("No git diff; scored the whole tree.")
        else:
            scope = "diff"
            keep = []
            for idx, row in frame.iterrows():
                length = row["loc"]
                length = 1 if pd.isna(length) else int(length)
                if overlaps(scan_root, repo, str(row["file"]), int(row["lineno"]), length, changes):
                    keep.append(idx)
            matched = frame.loc[keep].reset_index(drop=True)
            if matched.empty:
                warnings.append("No changed Python functions.")

    limits = saved.meta.get("feature_high") or {}
    if not isinstance(limits, dict):
        limits = {}
    rows = _rank(saved, matched, limits)
    shown = rows if top <= 0 else rows[:top]
    counts = {tier: sum(row.risk_tier == tier for row in rows) for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    return ReviewReport(
        rows=shown,
        scanned_count=int(len(frame)),
        matched_count=int(len(rows)),
        tier_counts=counts,
        model_dir=str(saved.run_dir),
        dataset_name=str(saved.meta.get("dataset") or ""),
        threshold=float(saved.threshold),
        scope=scope,
        warnings=warnings,
        honesty=honesty_lines(saved),
    )


def _rank(model: SavedModel, frame: pd.DataFrame, limits: dict) -> list[ReviewRow]:
    if frame.empty:
        return []
    ranked = predict_frame(model, frame)
    location = frame.set_index(frame["module_id"].astype(str), drop=False)
    rows: list[ReviewRow] = []
    for record in ranked.to_dict(orient="records"):
        module_id = str(record["module_id"])
        meta = location.loc[module_id] if module_id in location.index else None
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        file = str(meta["file"]) if meta is not None else ""
        function = str(meta["function"]) if meta is not None else ""
        lineno = int(meta["lineno"]) if meta is not None else 0
        values = {}
        if meta is not None:
            values = {key: meta[key] for key in limits if key in meta.index}
        rows.append(
            ReviewRow(
                rank=int(record["rank"]),
                module_id=module_id,
                file=file,
                function=function,
                lineno=lineno,
                call="DEFECTIVE" if record["predicted_defective"] else "clean",
                defect_probability=float(record["defect_probability"]),
                risk_score=float(record["risk_score"]),
                risk_tier=str(record["risk_tier"]),
                top_rule=str(record["top_rule"] or ""),
                top_rule_text=str(record["top_rule_text"] or ""),
                recommendation=str(record["recommendation"] or ""),
                outside=columns_outside(values, limits),
            )
        )
    return rows
