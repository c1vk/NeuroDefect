"""Static (PNG) plots for GA behaviour, used by the ablation script and the dashboard."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_ga_fitness(histories: dict[str, list[float]], path: Path, title: str) -> Path:
    """Best-so-far GA fitness per generation; one line per seed plus the mean."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    curves = [np.asarray(h, dtype=float) for h in histories.values() if len(h)]
    for label, h in histories.items():
        if len(h):
            ax.plot(range(len(h)), h, alpha=0.45, linewidth=1.2, label=f"seed {label}")
    if curves:
        n = min(len(c) for c in curves)
        mean = np.mean([c[:n] for c in curves], axis=0)
        ax.plot(range(n), mean, color="black", linewidth=2.2, label="mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_frequency(freq: dict[str, float], path: Path, title: str) -> Path:
    """Fraction of GA individuals (all generations) that included each feature."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.24 * len(names) + 1.0)), dpi=130)
    ax.barh(names[::-1], vals[::-1], color="#2563eb")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selection frequency")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
