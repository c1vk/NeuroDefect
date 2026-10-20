"""Ablation study for NeuroDefect.

Variants (each trained independently per seed, held-out test untouched):
  full        GA features + SMOTE + softmax BPN + GA-tuned fuzzy (fixed-quantile fuzzy also scored)
  no_ga       all features (no GA feature selection)
  no_smote    no SMOTE oversampling
  acc_floor   full, but BPN threshold chosen with an accuracy floor (threshold_mode=accuracy_floor)
"no-fuzzy" is the BPN decision alone (prob >= tuned threshold) from the full run, compared with
fuzzy risk tier >= HIGH decisions from fixed-quantile vs GA-tuned membership functions.

Usage: python scripts/ablation.py [--datasets ...] [--seeds 42 7 13] [--out reports]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.loader import load_dataset  # noqa: E402
from src.evaluation.plots import plot_feature_frequency, plot_ga_fitness  # noqa: E402
from src.pipeline.trainer import train_pipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_DATASETS = {
    "synthetic_sample": "data/sample/NASA_CM1_sample.csv",
    "CM1": "data/raw/cm1.csv",
    "PC1": "data/raw/pc1.csv",
    "JM1": "data/raw/jm1.csv",
}
CLS_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
VARIANTS = ["full", "no_ga", "no_smote", "acc_floor"]


def _cls_row(m) -> dict:
    return {k: float(getattr(m, k)) for k in CLS_KEYS}


def run(datasets: dict[str, str], seeds: list[int], out: Path, variants: list[str] = VARIANTS) -> pd.DataFrame:
    os.environ.setdefault("NEURODEFECT_MODELS_DIR", tempfile.mkdtemp(prefix="nd_ablation_"))
    rows: list[dict] = []
    fig_dir = out / "figures"
    for ds_name, path in datasets.items():
        ds = load_dataset(ROOT / path, name=ds_name)
        ga_hist, fz_hist, freqs = {}, {}, []
        for seed in seeds:
            for variant in variants:
                overrides = {
                    "training": {"random_state": seed},
                    "ga": {"random_state": seed},
                    "fuzzy": {"ga_tune": True, "ga": {"random_state": seed}},
                }
                if variant == "acc_floor":
                    overrides["training"]["threshold_mode"] = "accuracy_floor"
                cfg = load_config().with_overrides(**overrides)
                t0 = time.perf_counter()
                res = train_pipeline(
                    ds, cfg, variant="full" if variant == "acc_floor" else variant, with_baselines=False
                )
                secs = time.perf_counter() - t0
                base = {"dataset": ds_name, "seed": seed, "variant": variant, "seconds": round(secs, 1),
                        "n_features": len(res.selected_features)}
                rows.append({**base, "component": "BPN (softmax)", **_cls_row(res.metrics["Neuro-Fuzzy-GA"])})
                if variant == "full":
                    ga_hist[str(seed)] = res.ga_history
                    fz_hist[str(seed)] = res.fuzzy_tune_history
                    freqs.append(res.ga_feature_frequency)
                    for comp, fm in (("Fuzzy tier>=HIGH (fixed quantile)", res.fuzzy_metrics_fixed_test),
                                     ("Fuzzy tier>=HIGH (GA-tuned)", res.fuzzy_metrics_test)):
                        rows.append({**base, "component": comp,
                                     "precision": fm["high_precision"], "recall": fm["high_recall"],
                                     "f1": fm["high_f1"], "mcc": fm["high_mcc"], "risk_corr": fm["risk_corr"]})
                print(f"[{ds_name}] seed={seed} {variant:9s} {secs:6.1f}s  BPN F1={res.metrics['Neuro-Fuzzy-GA'].f1:.3f}",
                      flush=True)
        plot_ga_fitness(ga_hist, fig_dir / f"{ds_name}_ga_fitness.png", f"{ds_name}: GA feature-selection fitness")
        plot_ga_fitness(fz_hist, fig_dir / f"{ds_name}_fuzzy_ga_fitness.png", f"{ds_name}: fuzzy membership GA fitness (train-only val)")
        mean_freq = pd.DataFrame(freqs).mean().to_dict()
        plot_feature_frequency(mean_freq, fig_dir / f"{ds_name}_feature_frequency.png",
                               f"{ds_name}: GA feature selection frequency (mean of seeds)")
        pd.DataFrame(rows).to_csv(out / "ablation_raw.csv", index=False)  # checkpoint
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, seeds: list[int]) -> str:
    lines = [
        "# NeuroDefect ablation study",
        "",
        f"Mean ± std over seeds {', '.join(map(str, seeds))}. Stratified 70/30 split per seed; preprocessing, SMOTE, "
        "GA feature selection, BPN early stopping/threshold and fuzzy-membership GA use training data only "
        "(fuzzy-GA fitness uses 3-fold out-of-fold BPN probabilities on the training split). "
        "Metrics are on the untouched held-out test split.",
        "",
        "- **BPN (softmax)** rows: classifier decision (softmax class-1 probability ≥ validation-tuned threshold).",
        "- **no-fuzzy** = BPN decision alone (the `full` BPN row). **Fuzzy** rows: decision = derived risk tier ≥ HIGH; "
        "`risk_corr` = Pearson correlation of the 0–100 risk score with the defect label.",
        "- `acc_floor`: threshold maximises validation F1 subject to validation accuracy ≥ 0.65.",
        "",
    ]
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc", "risk_corr"]
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        lines += [f"## {ds}", "", "| Variant | Component | " + " | ".join(metric_cols) + " |",
                  "|---|---|" + "---|" * len(metric_cols)]
        for (variant, comp), g in sub.groupby(["variant", "component"], sort=False):
            cells = []
            for c in metric_cols:
                if c in g and g[c].notna().any():
                    cells.append(f"{g[c].mean():.3f} ± {g[c].std(ddof=0):.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {variant} | {comp} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = run({k: DEFAULT_DATASETS[k] for k in args.datasets}, args.seeds, out, args.variants)
    df.to_csv(out / "ablation_raw.csv", index=False)
    summary = df.groupby(["dataset", "variant", "component"], sort=False).agg(["mean", "std"]).drop(
        columns=["seed"], level=0, errors="ignore")
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out / "ablation_summary.csv", index=False)
    (out / "ablation.md").write_text(to_markdown(df, args.seeds), encoding="utf-8")
    print((out / "ablation.md").read_text())


if __name__ == "__main__":
    main()
