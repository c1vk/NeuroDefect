"""Train on one NASA dataset and score a different one.

Usage: python scripts/cross_project.py --train CM1 --test PC1
Writes reports/cross_project_CM1_to_PC1.md

The model is fit on a stratified split of the training set (the same pipeline as
the dashboard). The other dataset is scored in full with that fitted model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.cleaner import clean_frame  # noqa: E402
from src.data.columns import drop_derived_halstead  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.data.validator import validate_target  # noqa: E402
from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.evaluation.testing_budget import defect_recall_at_budget  # noqa: E402
from src.pipeline.trainer import train_pipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DATASETS = {
    "CM1": "data/raw/cm1.csv",
    "PC1": "data/raw/pc1.csv",
    "JM1": "data/raw/jm1.csv",
    "synthetic_sample": "data/sample/NASA_CM1_sample.csv",
}


def _xy(path: Path):
    ds = load_dataset(path)
    target = validate_target(ds.frame[ds.target_column])
    aligned = ds.frame.copy()
    aligned[ds.target_column] = target
    features, y, _, _ = clean_frame(aligned, ds.target_column)
    features, _ = drop_derived_halstead(features)
    return ds, features, validate_target(y).astype(int)


def score_transfer(result, features: pd.DataFrame, y: pd.Series) -> dict:
    missing = [col for col in result.all_features if col not in features.columns]
    if missing:
        raise SystemExit(f"Test set is missing model features: {', '.join(missing)}")
    scaled = result.preprocessor.transform(features[result.all_features])
    selected = scaled[result.selected_features].to_numpy(dtype=np.float32)
    scores = result.bpn.predict_proba(selected)[:, 1]
    thr = float(result.bpn.decision_threshold_)
    y_arr = np.asarray(y).astype(int)
    pred = (scores >= thr).astype(int)
    metrics = classification_metrics(y_arr, pred, scores)
    return {
        "n": int(len(y_arr)),
        "defective": int(y_arr.sum()),
        "f1": metrics.f1,
        "mcc": metrics.mcc,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "budget10": defect_recall_at_budget(y_arr, scores, 0.10),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="CM1", choices=list(DATASETS))
    parser.add_argument("--test", default="PC1", choices=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train == args.test:
        raise SystemExit("Train and test datasets must be different.")

    cfg = load_config().with_overrides(training={"random_state": args.seed}, ga={"random_state": args.seed})
    train_ds = load_dataset(ROOT / DATASETS[args.train])
    result = train_pipeline(train_ds, cfg, with_baselines=False)
    _, test_x, test_y = _xy(ROOT / DATASETS[args.test])
    held = score_transfer(result, result.X_test_raw, result.y_test)
    transferred = score_transfer(result, test_x, test_y)

    out = ROOT / "reports" / f"cross_project_{args.train}_to_{args.test}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Cross-project check: train {args.train}, test {args.test}",
        "",
        f"Seed {args.seed}. The model is the dashboard pipeline fit on {args.train}. "
        "Same-project numbers are that run's held-out split. "
        f"The {args.test} numbers score every row of the other dataset.",
        "",
        "| Set | Rows | Defective | Precision | Recall | F1 | MCC | Defects caught in top 10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {args.train} held-out | {held['n']} | {held['defective']} | {held['precision']:.3f} | "
            f"{held['recall']:.3f} | {held['f1']:.3f} | {held['mcc']:.3f} | {held['budget10'] * 100:.1f}% |"
        ),
        (
            f"| {args.test} (unseen project) | {transferred['n']} | {transferred['defective']} | "
            f"{transferred['precision']:.3f} | {transferred['recall']:.3f} | {transferred['f1']:.3f} | "
            f"{transferred['mcc']:.3f} | {transferred['budget10'] * 100:.1f}% |"
        ),
        "",
        "A drop from the first row to the second is the failure a tester hits when the training project is not the code under test.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
