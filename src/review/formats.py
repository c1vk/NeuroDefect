"""Plain text, Markdown, and SARIF renderings of a review."""
from __future__ import annotations

import json

from src.review.report import ReviewReport

_TIER_LEVEL = {"CRITICAL": "warning", "HIGH": "warning", "MEDIUM": "note", "LOW": "note"}


def _scope_line(report: ReviewReport) -> str:
    if report.scope == "diff":
        return "Scope: functions changed since the base commit."
    return "Scope: whole tree."


def _summary(report: ReviewReport) -> str:
    counts = ", ".join(f"{name}={report.tier_counts.get(name, 0)}" for name in ("CRITICAL", "HIGH", "MEDIUM", "LOW"))
    return f"Showing {len(report.rows)} of {report.matched_count} functions. Tiers: {counts}."


def format_text(report: ReviewReport) -> str:
    lines = list(report.honesty)
    lines.append("")
    lines.append(_scope_line(report))
    lines.extend(report.warnings)
    lines.append(_summary(report))
    lines.append(f"Model: {report.model_dir}  threshold {report.threshold:.3f}")
    lines.append("")
    lines.append(f"{'#':>3}  {'call':10} {'tier':8} {'p(def)':>6}  {'rule':4}  location")
    for row in report.rows:
        lines.append(
            f"{row.rank:>3}  {row.call:10} {row.risk_tier:8} {row.defect_probability:6.3f}  "
            f"{row.top_rule or '-':4}  {row.location}  {row.function}"
        )
        if row.outside:
            lines.append(f"     outside training range: {', '.join(row.outside)}")
        if row.recommendation:
            lines.append(f"     {row.recommendation}")
    return "\n".join(lines)


def format_markdown(report: ReviewReport) -> str:
    lines = ["> " + line for line in report.honesty]
    lines.append("")
    lines.append(_scope_line(report))
    lines.extend(report.warnings)
    lines.append("")
    lines.append(_summary(report))
    lines.append("")
    lines.append("| # | Call | Tier | P(def) | Rule | Location |")
    lines.append("|---|---|---|---|---|---|")
    for row in report.rows:
        note = f"{row.location} {row.function}"
        if row.outside:
            note += f" (outside training range: {', '.join(row.outside)})"
        lines.append(
            f"| {row.rank} | {row.call} | {row.risk_tier} | {row.defect_probability:.3f} | "
            f"{row.top_rule or '-'} | {note} |"
        )
        if row.recommendation:
            lines.append("")
            lines.append(f"{row.location}: {row.recommendation}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_sarif(report: ReviewReport) -> str:
    rules = []
    seen: set[str] = set()
    results = []
    for row in report.rows:
        rule_id = row.top_rule or "priority"
        if rule_id not in seen:
            seen.add(rule_id)
            rules.append({
                "id": rule_id,
                "shortDescription": {"text": row.top_rule_text or "Derived testing priority"},
            })
        message = row.recommendation or row.top_rule_text or row.risk_tier
        if row.outside:
            message += " Outside training range: " + ", ".join(row.outside) + "."
        results.append({
            "ruleId": rule_id,
            "level": _TIER_LEVEL.get(row.risk_tier, "note"),
            "message": {"text": message},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": row.file.replace("\\", "/")},
                    "region": {"startLine": max(row.lineno, 1)},
                }
            }],
        })
    payload = {
        "version": "2.1.0",
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "runs": [{
            "tool": {"driver": {"name": "NeuroDefect", "rules": rules}},
            "invocations": [{
                "executionSuccessful": True,
                "toolExecutionNotifications": [{
                    "descriptor": {"id": "honesty"},
                    "message": {"text": " ".join(report.honesty)},
                    "level": "note",
                }],
            }],
            "results": results,
        }],
    }
    return json.dumps(payload, indent=2) + "\n"


def render(report: ReviewReport, kind: str) -> str:
    if kind == "md":
        return format_markdown(report)
    if kind == "sarif":
        return format_sarif(report)
    return format_text(report)
