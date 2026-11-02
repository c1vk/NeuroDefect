"""NeuroDefect command line interface.

  neurodefect review <folder> [--top 10] [--changed] [--format text|md|sarif]
  neurodefect scan <folder> -o metrics.csv
  neurodefect train-shared --dataset data/raw/cm1.csv [--out models/shared]
  neurodefect predict metrics.csv [--model DIR] [--top 20] [--csv out.csv] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _cmd_scan(args) -> int:
    from src.scanner.radon_scanner import UNMAPPABLE, scan_folder

    df = scan_folder(args.folder, halstead=args.halstead)
    if df.empty:
        print(f"No Python functions found under {args.folder}", file=sys.stderr)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Scanned {df['file'].nunique()} files, {len(df)} functions -> {out}")
    print("Not computed (no radon equivalent): " + ", ".join(UNMAPPABLE))
    return 0


def _cmd_train_shared(args) -> int:
    import tempfile

    from src.data.columns import shared_subset
    from src.data.loader import LoadedDataset, load_dataset
    from src.pipeline.artifacts import save_run
    from src.pipeline.trainer import train_pipeline
    from src.utils.config import load_config

    ds = load_dataset(args.dataset)
    frame = shared_subset(ds.frame, ds.target_column)
    shared = LoadedDataset(name=f"{ds.name}_shared", path=ds.path, frame=frame,
                           target_column=ds.target_column, target_detection=ds.target_detection)
    cfg = load_config().with_overrides(training={"random_state": args.seed}, ga={"random_state": args.seed})
    os.environ["NEURODEFECT_MODELS_DIR"] = tempfile.mkdtemp(prefix="nd_shared_")
    res = train_pipeline(shared, cfg, with_baselines=not args.no_baselines)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"{ds.name}_seed{args.seed}"
    import shutil
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(res.run_dir, run_dir)
    (out_root / "latest.json").write_text(json.dumps({"run_dir": run_dir.name}) + "\n")
    m = res.metrics["Neuro-Fuzzy-GA"]
    print(f"Shared-subset model ({len(res.all_features)} columns, GA kept {len(res.selected_features)}) -> {run_dir}")
    print(f"Held-out: acc={m.accuracy:.3f} prec={m.precision:.3f} rec={m.recall:.3f} f1={m.f1:.3f} "
          f"auc={m.roc_auc:.3f} mcc={m.mcc:.3f}  threshold={res.bpn.decision_threshold_:.3f}")
    return 0


def _cmd_review(args) -> int:
    from src.review.formats import render
    from src.review.report import review_path

    report = review_path(
        args.path,
        model=args.model,
        top=args.top,
        changed_only=args.changed,
        base=args.base,
        halstead=args.halstead,
    )
    print(render(report, args.format), end="" if args.format == "sarif" else "\n")
    if report.scanned_count == 0:
        return 1
    if args.fail_on and report.tier_counts.get(args.fail_on, 0):
        return 1
    return 0


def _cmd_predict(args) -> int:
    import pandas as pd

    from src.pipeline.inference import load_model, predict_frame

    model = load_model(args.model)
    table = predict_frame(model, pd.read_csv(args.csv))
    if args.csv_out:
        table.to_csv(args.csv_out, index=False)
    if args.json_out:
        Path(args.json_out).write_text(table.to_json(orient="records", indent=2))
    shown = table.head(args.top)
    print(f"Model: {model.run_dir}  (threshold {model.threshold:.3f}, {len(table)} modules)")
    print(f"{'#':>3}  {'tier':8} {'risk':>5} {'p(def)':>6}  {'rule':4}  module")
    for _, r in shown.iterrows():
        print(f"{r['rank']:>3}  {r['risk_tier']:8} {r['risk_score']:5.1f} {r['defect_probability']:6.3f}  "
              f"{r['top_rule'] or '-':4}  {r['module_id']}")
    counts = table["risk_tier"].value_counts().to_dict()
    print("Tiers: " + ", ".join(f"{k}={counts.get(k, 0)}" for k in ("CRITICAL", "HIGH", "MEDIUM", "LOW")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurodefect", description="NeuroDefect defect-risk CLI")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="compute NASA-style metrics for Python code with radon")
    s.add_argument("folder")
    s.add_argument("-o", "--output", default="metrics.csv")
    s.add_argument("--halstead", choices=["token", "radon"], default="token",
                   help="token = classical token counts (default, closer to NASA); radon = radon AST operators only")
    s.set_defaults(func=_cmd_scan)
    t = sub.add_parser("train-shared", help="train on the scanner-compatible NASA column subset")
    t.add_argument("--dataset", default="data/raw/cm1.csv")
    t.add_argument("--out", default="models/shared")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-baselines", action="store_true")
    t.set_defaults(func=_cmd_train_shared)
    r = sub.add_parser("review", help="scan a Python tree and rank functions with the saved model")
    r.add_argument("path")
    r.add_argument("--top", type=int, default=10, help="rows to print; 0 prints every matched function")
    r.add_argument("--changed", action="store_true", help="only functions touched since the base commit")
    r.add_argument("--base", default="HEAD")
    r.add_argument("--format", choices=["text", "md", "sarif"], default="text")
    r.add_argument("--model", default=None, help="run dir or folder with latest.json (default models/shared)")
    r.add_argument("--fail-on", dest="fail_on", choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=None,
                   help="exit 1 when this derived tier is present; default is always 0 when functions were scanned")
    r.add_argument("--halstead", choices=["token", "radon"], default="token")
    r.set_defaults(func=_cmd_review)
    q = sub.add_parser("predict", help="rank modules in a metrics CSV by the network call, then defect probability")
    q.add_argument("csv")
    q.add_argument("--model", default=None, help="run dir or folder with latest.json (default models/shared)")
    q.add_argument("--top", type=int, default=20)
    q.add_argument("--csv-out", dest="csv_out")
    q.add_argument("--json-out", dest="json_out")
    q.set_defaults(func=_cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
