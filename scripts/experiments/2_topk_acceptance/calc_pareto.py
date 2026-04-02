#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_rows(path: Path):
    data = json.loads(path.read_text())
    rows = []
    for r in data:
        if not isinstance(r, dict):
            continue
        if "score" not in r or "mean_topk" not in r:
            continue
        try:
            score = float(r["score"])
            mean_topk = float(r["mean_topk"])
            if not (math.isfinite(score) and math.isfinite(mean_topk)):
                continue
            rows.append(
                {
                    "formula": r.get("formula", ""),
                    "score": score,
                    "mean_topk": mean_topk,
                }
            )
        except Exception:
            continue
    return rows


def pareto_frontier_max_score_min_topk(scores: np.ndarray, topks: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of Pareto-optimal points for:
      maximize score, minimize mean_topk.
    """
    n = scores.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue

        dominates_i = (
            (scores >= scores[i])
            & (topks <= topks[i])
            & ((scores > scores[i]) | (topks < topks[i]))
        )
        dominates_i[i] = False
        if dominates_i.any():
            is_pareto[i] = False
            continue

        dominated_by_i = (
            (scores[i] >= scores)
            & (topks[i] <= topks)
            & ((scores[i] > scores) | (topks[i] < topks))
        )
        dominated_by_i[i] = False
        is_pareto[dominated_by_i] = False

    return is_pareto


def main():
    ap = argparse.ArgumentParser(
        description="Compute Pareto-frontier formulas from optimization result json."
    )
    ap.add_argument(
        "-i",
        "--infile",
        required=True,
        help="Path to results json (must contain score + mean_topk).",
    )
    args = ap.parse_args()

    path = Path(args.infile)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    rows = load_rows(path)
    if not rows:
        raise ValueError(f"No usable entries with both score and mean_topk in {path}")

    scores = np.array([r["score"] for r in rows], dtype=float)
    topks = np.array([r["mean_topk"] for r in rows], dtype=float)

    mask = pareto_frontier_max_score_min_topk(scores, topks)
    pareto_rows = [rows[i] for i in range(len(rows)) if mask[i]]
    pareto_rows = sorted(pareto_rows, key=lambda d: (d["mean_topk"], -d["score"]))

    emitted = 0
    seen = set()
    for r in pareto_rows:
        formula = str(r.get("formula", "")).strip()
        if not formula:
            continue
        if formula in seen:
            continue
        seen.add(formula)
        print(formula)
        emitted += 1

    if emitted == 0:
        print("No Pareto formula found (all empty or missing).", file=sys.stderr)


if __name__ == "__main__":
    main()
