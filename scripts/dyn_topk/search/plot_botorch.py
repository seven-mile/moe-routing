#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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
        # j dominates i if:
        #   score_j >= score_i AND topk_j <= topk_i AND at least one strict
        dominates_i = (
            (scores >= scores[i]) &
            (topks <= topks[i]) &
            ((scores > scores[i]) | (topks < topks[i]))
        )
        dominates_i[i] = False
        if dominates_i.any():
            is_pareto[i] = False
            continue

        # i dominates others -> mark them non-pareto
        dominated_by_i = (
            (scores[i] >= scores) &
            (topks[i] <= topks) &
            ((scores[i] > scores) | (topks[i] < topks))
        )
        dominated_by_i[i] = False
        is_pareto[dominated_by_i] = False

    return is_pareto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="optimization_results_lmeval.json",
                    help="Path to results json (must contain score + mean_topk).")
    ap.add_argument("--out_png", type=str, default="pareto_score_vs_meantopk.png")
    ap.add_argument("--out_pareto_json", type=str, default="pareto_frontier_points.json")
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

    # sort frontier for a clean connecting line (by mean_topk ascending, score descending)
    pareto_rows = sorted(pareto_rows, key=lambda d: (d["mean_topk"], -d["score"]))

    # Plot:
    # x = mean_topk (lower is better)
    # y = score (higher is better)
    plt.figure()
    plt.scatter(topks, scores, marker="o")
    plt.scatter([r["mean_topk"] for r in pareto_rows], [r["score"] for r in pareto_rows], marker="x")
    plt.plot([r["mean_topk"] for r in pareto_rows], [r["score"] for r in pareto_rows], marker="x")

    plt.xlabel("mean_topk (lower is better)")
    plt.ylabel("score (higher is better)")
    plt.title("Pareto frontier: maximize score, minimize mean_topk")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)

    Path(args.out_pareto_json).write_text(json.dumps(pareto_rows, indent=2))
    print(f"Saved plot: {args.out_png}")
    print(f"Saved pareto points: {args.out_pareto_json}")
    print(f"All points: {len(rows)} | Pareto points: {len(pareto_rows)}")


if __name__ == "__main__":
    main()
