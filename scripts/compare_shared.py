"""Full NASA feature set vs scanner-compatible shared subset (BPN softmax, held-out test).

Usage: python scripts/compare_shared.py [--datasets CM1 PC1 JM1] [--seeds 42 7 13]
Writes reports/shared_vs_full.csv and reports/shared_vs_full.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.columns import canonicalize_nasa_columns, shared_subset  # noqa: E402
from src.data.loader import LoadedDataset, load_dataset  # noqa: E402
from src.pipeline.trainer import train_pipeline  # noqa: E402
from src.scanner.radon_scanner import MODEL_COLUMNS  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DATASETS = {"CM1": "data/raw/cm1.csv", "PC1": "data/raw/pc1.csv", "JM1": "data/raw/jm1.csv"}
KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 13])
    ap.add_argument("--out", default=str(ROOT / "reports"))
    a = ap.parse_args()
    os.environ.setdefault("NEURODEFECT_MODELS_DIR", tempfile.mkdtemp(prefix="nd_shared_cmp_"))
    rows = []
    for name in a.datasets:
        ds = load_dataset(ROOT / DATASETS[name], name=name)
        full = canonicalize_nasa_columns(ds.frame)
        variants = {
            "full (21 NASA cols)": full,
            f"shared ({len(MODEL_COLUMNS)} scanner cols)": shared_subset(full, ds.target_column),
        }
        for seed in a.seeds:
            cfg = load_config().with_overrides(training={"random_state": seed}, ga={"random_state": seed})
            for label, frame in variants.items():
                d = LoadedDataset(name=name, path=ds.path, frame=frame, target_column=ds.target_column,
                                  target_detection=ds.target_detection)
                res = train_pipeline(d, cfg, with_baselines=False)
                m = res.metrics["Neuro-Fuzzy-GA"]
                rows.append({"dataset": name, "features": label, "seed": seed,
                             "n_selected": len(res.selected_features), **{k: float(getattr(m, k)) for k in KEYS}})
                print(f"{name} seed={seed} {label}: f1={m.f1:.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    df.to_csv(out / "shared_vs_full.csv", index=False)
    lines = ["# Full NASA features vs scanner-compatible shared subset", "",
             f"Softmax BPN on GA-selected features, mean ± std over seeds {a.seeds}; held-out 30% test split.",
             f"Shared subset: {', '.join(MODEL_COLUMNS)}.", "",
             "| Dataset | Features | " + " | ".join(KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
    for (d, f), g in df.groupby(["dataset", "features"], sort=False):
        lines.append(f"| {d} | {f} | " + " | ".join(f"{g[k].mean():.3f} ± {g[k].std(ddof=0):.3f}" for k in KEYS) + " |")
    (out / "shared_vs_full.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
