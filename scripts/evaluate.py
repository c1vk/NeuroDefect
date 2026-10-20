"""Headless evaluation: train the full pipeline on a dataset and print held-out metrics.

Usage: python scripts/evaluate.py data/raw/cm1.csv [--seeds 42 7 13] [--out results.json]
All preprocessing, SMOTE, GA, BPN early stopping and threshold tuning use the training
split only; metrics are computed once on the untouched stratified test split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset  # noqa: E402
from src.pipeline.trainer import train_pipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402

KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
MODELS = ["Logistic Regression", "Random Forest", "Support Vector Machine", "BPN", "Neuro-Fuzzy-GA"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out")
    args = ap.parse_args()
    os.environ.setdefault("NEURODEFECT_MODELS_DIR", tempfile.mkdtemp(prefix="nd_eval_"))
    ds = load_dataset(args.dataset)
    results = {}
    for seed in args.seeds:
        cfg = load_config().with_overrides(training={"random_state": seed}, ga={"random_state": seed})
        res = train_pipeline(ds, cfg)
        results[seed] = {m: {k: float(getattr(res.metrics[m], k)) for k in KEYS} for m in MODELS if m in res.metrics}
        results[seed]["_selected"] = res.selected_features
    agg = {}
    for m in MODELS:
        rows = [results[s][m] for s in args.seeds if m in results[s]]
        if rows:
            agg[m] = {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}
    print(f"{ds.name}  n={len(ds.frame)}  seeds={args.seeds}")
    print(f"{'model':24s}" + "".join(f"{k:>10s}" for k in KEYS))
    for m, v in agg.items():
        print(f"{m:24s}" + "".join(f"{v[k]:10.3f}" for k in KEYS))
    if args.out:
        Path(args.out).write_text(json.dumps({"per_seed": results, "mean": agg}, indent=2, default=str))


if __name__ == "__main__":
    main()
