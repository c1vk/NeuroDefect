"""Limit a scan to Python functions touched since a git revision."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_PLUS = re.compile(r"^\+\+\+ b/(.+)$")


@dataclass
class ChangeSet:
    """Repo-relative POSIX paths. ``whole_files`` are included at every line."""

    lines: dict[str, set[int]] = field(default_factory=dict)
    whole_files: set[str] = field(default_factory=set)


def parse_unified_diff(text: str) -> dict[str, set[int]]:
    """Map each ``*.py`` path in a unified diff to the new-file line numbers it touches."""
    current: str | None = None
    found: dict[str, set[int]] = {}
    for line in text.splitlines():
        plus = _PLUS.match(line)
        if plus:
            path = plus.group(1).strip()
            current = path if path.endswith(".py") and path != "/dev/null" else None
            continue
        if current is None:
            continue
        hunk = _HUNK.match(line)
        if not hunk:
            continue
        start = int(hunk.group(1))
        count = int(hunk.group(2)) if hunk.group(2) is not None else 1
        if count <= 0:
            continue
        found.setdefault(current, set()).update(range(start, start + count))
    return found


def git_root(start: Path) -> Path | None:
    """Nearest repository that contains ``start``, or None when git cannot see one."""
    cwd = start if start.is_dir() else start.parent
    if not cwd.exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    root = proc.stdout.strip()
    return Path(root) if root else None


def collect_changes(start: Path, base: str = "HEAD") -> ChangeSet | None:
    """Changed and untracked Python lines. None when ``start`` is not in a git checkout."""
    root = git_root(start)
    if root is None:
        return None
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "-U0", base, "--"],
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:
        return None
    changes = ChangeSet(lines=parse_unified_diff(diff.stdout))
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        text=True,
    )
    if untracked.returncode != 0:
        return None
    for name in untracked.stdout.split("\0"):
        if name.endswith(".py"):
            changes.whole_files.add(name.replace("\\", "/"))
    return changes


def overlaps(scan_root: Path, git_root_path: Path, file: str, lineno: int, loc: int, changes: ChangeSet) -> bool:
    """True when the function's lines were edited or the file is untracked."""
    try:
        rel = (scan_root / file).resolve().relative_to(git_root_path.resolve()).as_posix()
    except ValueError:
        return False
    if rel in changes.whole_files:
        return True
    touched = changes.lines.get(rel, set())
    length = loc if loc and loc > 0 else 1
    return any(line in touched for line in range(int(lineno), int(lineno) + int(length)))
