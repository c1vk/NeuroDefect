from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


@dataclass
class AppConfig:
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "MVP")).upper()

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def training(self) -> dict[str, Any]:
        values = dict(self.raw.get("training", {}))
        if self.mode == "RESEARCH":
            values.update(self.raw.get("research", {}).get("training", {}))
        return values

    def ga(self) -> dict[str, Any]:
        values = dict(self.raw.get("ga", {}))
        if self.mode == "RESEARCH":
            values.update(self.raw.get("research", {}).get("ga", {}))
        return values

    def smote(self) -> dict[str, Any]:
        return dict(self.raw.get("smote", {}))

    def fuzzy(self) -> dict[str, Any]:
        return dict(self.raw.get("fuzzy", {}))

    def dataset(self) -> dict[str, Any]:
        return dict(self.raw.get("dataset", {}))

    def ui(self) -> dict[str, Any]:
        return dict(self.raw.get("ui", {}))

    def with_overrides(self, **overrides: Any) -> "AppConfig":
        merged = _deep_merge(self.raw, overrides)
        return AppConfig(raw=merged)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else ROOT / "config.yaml"
    if not config_path.exists():
        return AppConfig(raw={"mode": "MVP"})
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return AppConfig(raw=data)
