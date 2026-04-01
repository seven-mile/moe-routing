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


def find_row(rows, target_score, target_topk, atol_score=1e-9, atol_topk=1e-9):
    for r in rows:
        if math.isclose(r["score"], target_score, abs_tol=atol_score) and math.isclose(
            r["mean_topk"], target_topk, abs_tol=atol_topk
        ):
            return r
    return None


def annotate_point(ax, x, y, label, xytext=(10, 10), color="#2b2d42",
                   marker_color=None, marker="o", s=90, fontsize=11,
                   ha="left", va="bottom", zorder=10):
    if marker_color is None:
        marker_color = color

    ax.scatter(
        [x], [y],
        s=s,
        marker=marker,
        color=marker_color,
        edgecolors="white",
        linewidths=1.4,
        zorder=zorder,
    )

    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=1.2,
            alpha=0.85,
            shrinkA=4,
            shrinkB=4,
        ),
        zorder=zorder + 1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="optimization_results_lmeval.json",
                    help="Path to results json (must contain score + mean_topk).")
    ap.add_argument("--out_png", type=str, default="pareto_score_vs_meantopk_ng.png")
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

    # Convert score to percentage for plotting
    scores_pct = scores * 100.0
    pareto_topks = np.array([r["mean_topk"] for r in pareto_rows], dtype=float)
    pareto_scores_pct = np.array([r["score"] * 100.0 for r in pareto_rows], dtype=float)

    # Named points
    lossless = find_row(
        rows,
        target_score=0.9112964366944655,
        target_topk=5.964637927818708,
        atol_score=1e-12,
        atol_topk=1e-12,
    )
    optimum = find_row(
        rows,
        target_score=0.9044730856709629,
        target_topk=5.45121673265149,
        atol_score=1e-12,
        atol_topk=1e-12,
    )

    baseline_topk = 8.0
    baseline_score_pct = 90.75

    # --- Style ---
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "--",
    })

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    # Palette: low-saturation, lively but not aggressive
    bg_point_color = "#9fb3c8"      # muted blue-gray
    frontier_color = "#5b8def"      # soft vivid blue
    frontier_fill = "#dce9ff"       # pale blue fill
    baseline_color = "#98a1ab"      # neutral gray
    lossless_color = "#2a9d8f"      # muted teal
    optimum_color = "#8b5fbf"       # muted violet

    # All points
    ax.scatter(
        topks,
        scores_pct,
        s=34,
        marker="o",
        color=bg_point_color,
        alpha=0.55,
        edgecolors="none",
        zorder=2,
        label="Search space",
    )

    # Baseline horizontal reference
    ax.axhline(
        baseline_score_pct,
        color=baseline_color,
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        alpha=0.65,
        zorder=1,
    )

    # Frontier: filled band + outlined line + markers
    ymin = min(scores_pct.min(), baseline_score_pct) - 0.35
    ax.fill_between(
        pareto_topks,
        pareto_scores_pct,
        ymin,
        color=frontier_fill,
        alpha=0.35,
        zorder=1.5,
    )

    # white under-stroke for cleaner separation
    ax.plot(
        pareto_topks,
        pareto_scores_pct,
        color="white",
        linewidth=5.5,
        alpha=0.95,
        zorder=4,
    )

    ax.plot(
        pareto_topks,
        pareto_scores_pct,
        color=frontier_color,
        linewidth=2.6,
        zorder=5,
        label="Pareto frontier",
    )

    ax.scatter(
        pareto_topks,
        pareto_scores_pct,
        s=54,
        color=frontier_color,
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
    )

    # Baseline point
    annotate_point(
        ax,
        baseline_topk,
        baseline_score_pct,
        "Baseline\n(k=8.0, score=90.75)",
        xytext=(-12, 12),
        color="#5f6770",
        marker_color="#b5bdc6",
        marker="D",
        s=82,
        fontsize=10.5,
        ha="right",
        va="bottom",
        zorder=9,
    )

    # Lossless point
    if lossless is not None:
        annotate_point(
            ax,
            lossless["mean_topk"],
            lossless["score"] * 100.0,
            f"Lossless\n(k={lossless['mean_topk']:.2f}, score={lossless['score']*100:.2f})",
            xytext=(12, 12),
            color=lossless_color,
            marker_color=lossless_color,
            marker="o",
            s=96,
            fontsize=10.5,
            ha="left",
            va="bottom",
            zorder=10,
        )

    # Optimum point
    if optimum is not None:
        annotate_point(
            ax,
            optimum["mean_topk"],
            optimum["score"] * 100.0,
            f"Optimum\n(k={optimum['mean_topk']:.2f}, score={optimum['score']*100:.2f})",
            xytext=(12, -16),
            color=optimum_color,
            marker_color=optimum_color,
            marker="o",
            s=96,
            fontsize=10.5,
            ha="left",
            va="top",
            zorder=10,
        )

    # Label baseline line near left side
    x_min, x_max = topks.min(), max(topks.max(), baseline_topk)
    x_span = x_max - x_min
    ax.text(
        x_min + 0.02 * x_span,
        baseline_score_pct + 0.06,
        "Baseline score",
        color=baseline_color,
        fontsize=10,
        ha="left",
        va="bottom",
    )

    # Axes / title
    ax.set_xlabel("Mean Top-k (lower is better)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Search Space and Pareto Frontier")

    # margins
    ax.set_xlim(x_min - 0.12 * x_span, x_max + 0.06 * x_span)
    y_min = min(scores_pct.min(), baseline_score_pct) - 0.45
    y_max = max(scores_pct.max(), baseline_score_pct) + 0.45
    ax.set_ylim(y_min, y_max)

    # legend
    leg = ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.92)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=240, bbox_inches="tight")

    print(f"Saved plot: {args.out_png}")
    print(f"All points: {len(rows)} | Pareto points: {len(pareto_rows)}")


if __name__ == "__main__":
    main()
