from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.feature_map import TARGET_ALIASES, normalize
from src.utils.exceptions import DataError

SUPPORTED_SUFFIXES = {".csv", ".xls", ".xlsx"}


@dataclass
class LoadedDataset:
    name: str
    path: Path
    frame: pd.DataFrame
    target_column: str | None
    target_detection: str


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise DataError(f"Unsupported file type '{suffix}'. Use CSV or XLS/XLSX.")


def detect_target_column(columns: list[str], requested: str | None = None) -> tuple[str | None, str]:
    if requested:
        if requested not in columns:
            raise DataError(
                f"Target column '{requested}' was not found. "
                f"Available columns: {', '.join(columns)}"
            )
        return requested, "user"
    matches = [col for col in columns if normalize(col) in TARGET_ALIASES]
    if len(matches) == 1:
        return matches[0], "auto"
    if len(matches) > 1:
        raise DataError(
            "Automatic target detection is ambiguous. "
            f"Candidate columns: {', '.join(matches)}. Please select the target column manually."
        )
    return None, "missing"


def load_dataset(
    path: str | Path,
    target_column: str | None = None,
    name: str | None = None,
) -> LoadedDataset:
    file_path = Path(path)
    if not file_path.exists():
        raise DataError(f"Dataset was not found: {file_path}")
    try:
        frame = _read_table(file_path)
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Could not read '{file_path.name}'. The file may be invalid.") from exc
    if frame.empty:
        raise DataError("The dataset is empty.")
    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    detected, how = detect_target_column(list(frame.columns), target_column)
    return LoadedDataset(
        name=name or file_path.stem,
        path=file_path,
        frame=frame,
        target_column=detected,
        target_detection=how,
    )


def list_datasets(directories: list[Path]) -> list[Path]:
    found: list[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                found.append(path)
    return found
