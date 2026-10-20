"""Generate a CM1-like NASA PROMISE sample used when no raw dataset is supplied.

The columns follow the standard NASA CM1 metric names. Defect labels are sampled
from a logistic function of complexity, size, and effort — they are synthetic,
not NASA ground truth. Drop a real PC5/JM1/CM1 CSV into data/raw/ for research use.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "NASA_CM1_sample.csv"

COLUMNS = [
    "module_id",
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
    "defects",
]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sample_row(rng: random.Random, idx: int) -> dict:
    loc = clamp(rng.lognormvariate(math.log(28), 0.7), 1, 284)
    vg = clamp(1 + loc / 12 + rng.gauss(0, 2.4), 1, 32)
    evg = clamp(vg * rng.uniform(0.35, 0.85), 1, 21)
    ivg = clamp(vg * rng.uniform(0.45, 0.95), 1, 28)
    uniq_op = clamp(rng.gauss(12.4, 6.1), 1, 41)
    uniq_opnd = clamp(uniq_op * rng.uniform(1.1, 2.4) + rng.gauss(0, 3), 1, 80)
    total_op = clamp(loc * rng.uniform(0.8, 1.6), 1, 400)
    total_opnd = clamp(total_op * rng.uniform(0.6, 1.1), 1, 350)
    n = total_op + total_opnd
    volume = clamp(rng.gauss(324, 210), 0, 1284)
    level = clamp(rng.gauss(0.08, 0.05), 0.001, 0.5)
    difficulty = clamp(1 / max(level, 0.01) * rng.uniform(0.3, 0.8), 0.5, 80)
    effort = clamp(volume * difficulty * rng.uniform(0.6, 1.4), 0, 45021)
    bugs = clamp(volume / 3000 + rng.gauss(0, 0.05), 0, 2)
    time_est = clamp(effort / 18, 0, 4000)
    locode = clamp(loc * rng.uniform(0.7, 0.95), 1, loc)
    comments = clamp(rng.gauss(4, 5), 0, 40)
    blanks = clamp(rng.gauss(6, 5), 0, 40)
    loc_cc = clamp(rng.gauss(0.4, 0.8), 0, 8)
    branch = clamp(vg * rng.uniform(1.2, 2.1), 1, 60)
    return {
        "module_id": f"CM1_module_{idx:03d}",
        "loc": round(loc, 2),
        "v(g)": round(vg, 2),
        "ev(g)": round(evg, 2),
        "iv(g)": round(ivg, 2),
        "n": round(n, 2),
        "v": round(volume, 2),
        "l": round(level, 4),
        "d": round(difficulty, 2),
        "i": round(volume * level, 2),
        "e": round(effort, 2),
        "b": round(bugs, 4),
        "t": round(time_est, 2),
        "lOCode": round(locode, 2),
        "lOComment": round(comments, 2),
        "lOBlank": round(blanks, 2),
        "locCodeAndComment": round(loc_cc, 2),
        "uniq_Op": round(uniq_op, 2),
        "uniq_Opnd": round(uniq_opnd, 2),
        "total_Op": round(total_op, 2),
        "total_Opnd": round(total_opnd, 2),
        "branchCount": round(branch, 2),
    }


def defect_logit(row: dict) -> float:
    z = (
        -2.35
        + 0.12 * (row["v(g)"] - 5.4)
        + 0.012 * (row["loc"] - 37)
        + 0.00004 * (row["e"] - 8000)
        + 0.04 * (row["branchCount"] - 8)
        + 0.001 * (row["v"] - 320)
    )
    return 1.0 / (1.0 + math.exp(-z))


def main() -> None:
    rng = random.Random(42)
    n_rows = 344
    n_defects = 49
    rows = [sample_row(rng, i) for i in range(n_rows)]
    ranked = sorted(range(n_rows), key=lambda i: defect_logit(rows[i]) + rng.gauss(0, 0.05), reverse=True)
    defective = set(ranked[:n_defects])
    for i, row in enumerate(rows):
        row["defects"] = 1 if i in defective else 0

    # A few missing values and one duplicate, matching a realistic quality report.
    rows[11]["v"] = ""
    rows[48]["e"] = ""
    rows[90]["e"] = ""
    rows.append(dict(rows[3]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
