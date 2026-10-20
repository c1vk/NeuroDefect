"""Extract NASA-PROMISE-style metrics from Python source with radon.

One row per function/method (plus one ``<module>`` row per file for top-level code
when it contains statements outside functions). Column names follow the NASA MDP
datasets where a reasonable mapping exists. See ``COLUMN_MAPPING`` / ``UNMAPPABLE``.
"""
from __future__ import annotations

import ast
import io
import keyword
import math
import tokenize
import textwrap
from pathlib import Path

import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import h_visit_ast
from radon.raw import analyze

# Shared NASA columns the scanner produces (used to train the shared-subset model).
SHARED_COLUMNS = [
    "loc", "v(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

# Columns the scanner emits but the shared model does NOT train on: in PROMISE CM1 the
# lOCode/lOComment/lOBlank values do not behave like line counts (median lOCode = 1 while
# median loc = 17), and Python docstrings are not '#' comments, so their distributions disagree.
EXCLUDED_FROM_MODEL = {
    "lOCode": "PROMISE lOCode median is 1 vs loc median 17 on CM1 — inconsistent semantics",
    "lOComment": "Python docstrings are not counted as comments; CM1 comment density differs strongly",
    "lOBlank": "blank-line conventions differ between the C sources and Python",
}
MODEL_COLUMNS = [c for c in SHARED_COLUMNS if c not in EXCLUDED_FROM_MODEL]

COLUMN_MAPPING = {
    "loc": "physical lines of the function (end line - start line + 1)",
    "v(g)": "radon cyclomatic complexity (McCabe)",
    "lOCode": "radon raw SLOC (source lines of code)",
    "lOComment": "radon raw comment lines (# comments)",
    "lOBlank": "radon raw blank lines",
    "uniq_Op": "Halstead h1 distinct operators (default: Python tokenize operators+keywords; --halstead radon for radon's AST counts)",
    "uniq_Opnd": "Halstead h2 distinct operands (identifiers, numbers, strings)",
    "total_Op": "Halstead N1 total operators",
    "total_Opnd": "Halstead N2 total operands",
    "n": "Halstead length N = N1 + N2",
    "v": "Halstead volume",
    "d": "Halstead difficulty",
    "l": "Halstead level = 1 / d (0 if d = 0)",
    "i": "Halstead intelligent content = v / d (0 if d = 0)",
    "e": "Halstead effort",
    "b": "Halstead delivered bugs v / 3000",
    "t": "Halstead time (e / 18 seconds)",
    "branchCount": "APPROXIMATION: 2 * (v(g) - 1) + 1 (NASA-style edge count of a decision graph)",
}

UNMAPPABLE = {
    "ev(g)": "essential complexity needs structured-reduction of the control-flow graph (not in radon)",
    "iv(g)": "design complexity needs call-graph/module-interaction analysis (not in radon)",
    "locCodeAndComment": "lines with both code and comment are not reported per function by radon",
}

_SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", "build", "dist", ".tox"}


def _halstead_row(report) -> dict[str, float]:
    h1, h2 = report.h1, report.h2
    N1, N2 = report.N1, report.N2
    d = float(report.difficulty or 0.0)
    v = float(report.volume or 0.0)
    return {
        "uniq_Op": h1, "uniq_Opnd": h2, "total_Op": N1, "total_Opnd": N2,
        "n": N1 + N2, "v": v, "d": d,
        "l": 1.0 / d if d > 0 else 0.0,
        "i": v / d if d > 0 else 0.0,
        "e": float(report.effort or 0.0),
        "b": float(report.bugs or 0.0),
        "t": float(report.time or 0.0),
    }


_NON_CODE = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
             tokenize.ENCODING, tokenize.ENDMARKER}


def token_halstead(segment: str) -> dict[str, float]:
    """Classical token-level Halstead counts (closer to the NASA MDP extractor than radon's AST view).

    Operators: punctuation/operator tokens and Python keywords. Operands: identifiers, numbers, strings.
    """
    ops: list[str] = []
    opnds: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(textwrap.dedent(segment)).readline):
            if tok.type in _NON_CODE:
                continue
            if tok.type == tokenize.OP or (tok.type == tokenize.NAME and keyword.iskeyword(tok.string)):
                if tok.string in {")", "]", "}"}:
                    continue  # count bracket pairs once
                ops.append(tok.string)
            elif tok.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING) or tok.type == getattr(tokenize, "FSTRING_START", -1):
                opnds.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return _empty_halstead()
    h1, h2, N1, N2 = len(set(ops)), len(set(opnds)), len(ops), len(opnds)
    vocab, length = h1 + h2, N1 + N2
    v = length * math.log2(vocab) if vocab > 1 else 0.0
    d = (h1 / 2.0) * (N2 / h2) if h2 else 0.0
    e = d * v
    return {"uniq_Op": h1, "uniq_Opnd": h2, "total_Op": N1, "total_Opnd": N2, "n": length, "v": v, "d": d,
            "l": 1.0 / d if d > 0 else 0.0, "i": v / d if d > 0 else 0.0, "e": e, "b": v / 3000.0, "t": e / 18.0}


def _empty_halstead() -> dict[str, float]:
    return {k: 0.0 for k in ("uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "n", "v", "d", "l", "i", "e", "b", "t")}


def _raw_counts(segment: str) -> tuple[int, int, int]:
    """(SLOC, comment lines, blank lines) via radon.raw; simple fallback if tokenizing fails."""
    try:
        raw = analyze(textwrap.dedent(segment))
        return int(raw.sloc), int(raw.comments), int(raw.blank)
    except Exception:
        lines = segment.splitlines()
        blank = sum(1 for ln in lines if not ln.strip())
        comments = sum(1 for ln in lines if ln.strip().startswith("#"))
        return len(lines) - blank - comments, comments, blank


def scan_file(path: Path, root: Path | None = None, halstead: str = "token") -> list[dict]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    rel = str(path.relative_to(root)) if root else str(path)
    rows: list[dict] = []
    func_nodes = {
        (n.lineno, n.name): n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for block in cc_visit(source):
        if block.letter == "C":  # classes: their methods are reported separately
            continue
        name = f"{block.classname}.{block.name}" if getattr(block, "classname", None) else block.name
        start, end = block.lineno, block.endline or block.lineno
        segment = "\n".join(lines[start - 1 : end])
        sloc, comments, blank = _raw_counts(segment)
        node = func_nodes.get((start, block.name))
        if halstead == "radon":
            hal = _halstead_row(h_visit_ast(node).total) if node is not None else _empty_halstead()
        else:
            hal = token_halstead(segment)
        vg = int(block.complexity)
        rows.append({
            "module_id": f"{rel}::{name}",
            "file": rel,
            "function": name,
            "lineno": start,
            "loc": end - start + 1,
            "v(g)": vg,
            "lOCode": sloc,
            "lOComment": comments,
            "lOBlank": blank,
            **hal,
            "branchCount": 2 * (vg - 1) + 1,
        })
    return rows


def scan_folder(folder: str | Path, halstead: str = "token") -> pd.DataFrame:
    root = Path(folder).resolve()
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*.py") if not any(part in _SKIP_DIRS for part in p.parts)
    )
    rows: list[dict] = []
    for f in files:
        rows.extend(scan_file(f, root.parent if root.is_file() else root, halstead=halstead))
    cols = ["module_id", "file", "function", "lineno"] + SHARED_COLUMNS
    return pd.DataFrame(rows, columns=cols)
