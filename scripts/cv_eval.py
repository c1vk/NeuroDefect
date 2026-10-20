"""Run repeated stratified CV experiments.

  python scripts/cv_eval.py main --dataset CM1 [--repeats 5 --folds 5]
  python scripts/cv_eval.py selectors --dataset CM1     # none vs GA(logistic) vs GA(BPN) vs PSO
Per-fold results -> reports/cv/<experiment>_<dataset>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.imbalance import resolve_smote  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.evaluation.cv import FoldOptions, repeated_cv  # noqa: E402

DATASETS = {"synthetic_sample": "data/sample/NASA_CM1_sample.csv", "CM1": "data/raw/cm1.csv",
            "PC1": "data/raw/pc1.csv", "JM1": "data/raw/jm1.csv"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["main", "selectors"])
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selectors", nargs="+", default=["none", "ga_logistic", "ga_bpn", "pso"])
    ap.add_argument("--out", default=str(ROOT / "reports" / "cv"))
    a = ap.parse_args()
    ds = load_dataset(ROOT / DATASETS[a.dataset], name=a.dataset)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.experiment}_{a.dataset}.csv"
    frames = []
    use_smote = resolve_smote(a.dataset, True)
    runs = [("ga_logistic", FoldOptions(smote=use_smote))] if a.experiment == "main" else [
        (s, FoldOptions(selector=s, models=("Hybrid",), fuzzy=False, smote=use_smote)) for s in a.selectors]
    for sel, opts in runs:
        frames.append(repeated_cv(ds.frame, ds.target_column, opts, a.folds, a.repeats, a.seed,
                                  log=lambda m, s=sel: print(f"[{a.dataset}:{s}] {m}", flush=True)))
        pd.concat(frames).assign(dataset=a.dataset, folds=a.folds, repeats=a.repeats).to_csv(path, index=False)


if __name__ == "__main__":
    main()
