#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text())


def load_rows(path: Path):
    data = load_json(path)
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


def find_row(rows, target_score, target_topk, atol_score=1e-9, atol_topk=1e-9):
    for r in rows:
        if math.isclose(r["score"], target_score, abs_tol=atol_score) and math.isclose(
            r["mean_topk"], target_topk, abs_tol=atol_topk
        ):
            return r
    return None


def annotate_point(
    ax,
    x,
    y,
    label,
    xytext=(10, 10),
    color="#2b2d42",
    marker_color=None,
    marker="o",
    s=90,
    fontsize=11,
    ha="left",
    va="bottom",
    zorder=10,
    relpos=None,
):
    if relpos is None:
        ha_to_relx = {"left": 0.0, "center": 0.5, "right": 1.0}
        va_to_rely = {"bottom": 0.0, "center": 0.5, "top": 1.0}
        relpos = (ha_to_relx.get(ha, 0.5), va_to_rely.get(va, 0.5))

    if marker_color is None:
        marker_color = color

    ax.scatter(
        [x],
        [y],
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
            relpos=relpos,
            shrinkA=4,
            shrinkB=4,
        ),
        zorder=zorder + 1,
    )


def _data_delta_to_points(ax, x, y, dx_data, dy_data):
    p0 = ax.transData.transform((x, y))
    p1 = ax.transData.transform((x + dx_data, y + dy_data))
    # 1 point = 1/72 inch.
    return (p1[0] - p0[0]) * 72.0 / ax.figure.dpi, (p1[1] - p0[1]) * 72.0 / ax.figure.dpi


def auto_annotation_xytext(ax, kind, x, y, x_min, x_max, y_min, y_max):
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    # Default placement templates in data-space fractions of the current view.
    frac = {
        "baseline": (0, -0.24),
        "lossless": (0.08, -0.48),
        "optimum": (-0.05, -0.6),
    }
    dx_frac, dy_frac = frac.get(kind, (0.10, 0.10))

    # Flip directions near boundaries to keep text boxes on canvas.
    if x > x_min + 0.82 * x_span and dx_frac > 0:
        dx_frac = -abs(dx_frac)
    if x < x_min + 0.18 * x_span and dx_frac < 0:
        dx_frac = abs(dx_frac)
    if y < y_min + 0.22 * y_span and dy_frac < 0:
        dy_frac = abs(dy_frac)
    if y > y_min + 0.82 * y_span and dy_frac > 0:
        dy_frac = -abs(dy_frac)

    dx_pt, dy_pt = _data_delta_to_points(ax, x, y, dx_frac * x_span, dy_frac * y_span)

    # Clamp extremes so arrows remain readable even for unusual ranges.
    dx_pt = max(-150.0, min(150.0, dx_pt))
    dy_pt = max(-170.0, min(170.0, dy_pt))
    return (dx_pt, dy_pt)


def build_legend_figure():
    fig, ax = plt.subplots(figsize=(6.2, 1.35))
    ax.axis("off")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#9fb3c8",
            markeredgecolor="none",
            markersize=8,
            label="Search space",
        ),
        plt.Line2D([0], [0], color="#5b8def", linewidth=2.6, label="Pareto frontier"),
        plt.Line2D(
            [0],
            [0],
            color="#98a1ab",
            linestyle=(0, (4, 4)),
            linewidth=1.2,
            label="Baseline score",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#b5bdc6",
            markeredgecolor="white",
            markersize=8,
            label="Baseline point",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2a9d8f",
            markeredgecolor="white",
            markersize=8,
            label="Lossless",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#8b5fbf",
            markeredgecolor="white",
            markersize=8,
            label="Optimum",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        # handlelength=1.5,
        # handletextpad=0.45,
        # columnspacing=0.9,
        # labelspacing=0.25,
        # borderpad=0.25,
        # borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, legend


def write_outputs(
    path,
    rows,
    pareto_rows,
    baseline_topk,
    baseline_score_pct,
    show_baseline_label,
    fig_scale,
    out_json,
    out_png,
    out_pdf,
    legend_pdf=None,
    baseline_score_std=0.0,
):
    scores = np.array([r["score"] for r in rows], dtype=float)
    topks = np.array([r["mean_topk"] for r in rows], dtype=float)
    scores_pct = scores * 100.0
    pareto_topks = np.array([r["mean_topk"] for r in pareto_rows], dtype=float)
    pareto_scores_pct = np.array([r["score"] * 100.0 for r in pareto_rows], dtype=float)

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

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    frontier_json = [
        {
            "formula": r.get("formula", ""),
            "score": r["score"],
            "mean_topk": r["mean_topk"],
        }
        for r in pareto_rows
    ]
    out_json.write_text(json.dumps(frontier_json, indent=2, ensure_ascii=False))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "--",
        }
    )

    # Shrink source canvas so that at fixed LaTeX include width, text appears relatively larger.
    base_w, base_h = 8.8, 5.8
    fig_w = base_w / fig_scale
    fig_h = base_h / fig_scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bg_point_color = "#9fb3c8"
    frontier_color = "#5b8def"
    frontier_fill = "#dce9ff"
    baseline_color = "#98a1ab"
    lossless_color = "#2a9d8f"
    optimum_color = "#8b5fbf"

    ax.scatter(
        topks,
        scores_pct,
        s=34,
        marker="o",
        color=bg_point_color,
        alpha=0.55,
        edgecolors="none",
        zorder=2,
    )

    if baseline_score_std > 0:
        # Draw baseline uncertainty band first, then a standard I-shaped error bar.
        ax.axhspan(
            baseline_score_pct - baseline_score_std,
            baseline_score_pct + baseline_score_std,
            color=baseline_color,
            alpha=0.15,
            zorder=0.5,
        )

    ax.axhline(
        baseline_score_pct,
        color=baseline_color,
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        alpha=0.65,
        zorder=1,
    )

    ymin = min(scores_pct.min(), baseline_score_pct) - 0.35
    ax.fill_between(
        pareto_topks,
        pareto_scores_pct,
        ymin,
        color=frontier_fill,
        alpha=0.35,
        zorder=1.5,
    )

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

    x_min, x_max = topks.min(), max(topks.max(), baseline_topk)
    x_span = x_max - x_min
    baseline_y_low = baseline_score_pct - baseline_score_std
    baseline_y_high = baseline_score_pct + baseline_score_std
    y_min = min(scores_pct.min(), baseline_y_low) - 0.45
    y_max = max(scores_pct.max(), baseline_y_high) + 0.45
    ax.set_xlim(x_min - 0.12 * x_span, x_max + 0.06 * x_span)
    ax.set_ylim(y_min, y_max)

    baseline_xytext = auto_annotation_xytext(
        ax,
        kind="baseline",
        x=baseline_topk,
        y=baseline_score_pct,
        x_min=ax.get_xlim()[0],
        x_max=ax.get_xlim()[1],
        y_min=ax.get_ylim()[0],
        y_max=ax.get_ylim()[1],
    )

    annotate_point(
        ax,
        baseline_topk,
        baseline_score_pct,
        f"Baseline\n({baseline_topk:.2f}, {baseline_score_pct:.2f})",
        xytext=baseline_xytext,
        relpos=(0.9, 0),
        color="#5f6770",
        marker_color="#b5bdc6",
        marker="D",
        s=82,
        fontsize=10.5,
        ha="right",
        va="top",
        zorder=9,
    )

    if baseline_score_std > 0:
        ax.errorbar(
            baseline_topk,
            baseline_score_pct,
            yerr=baseline_score_std,
            fmt="none",
            ecolor="black",
            elinewidth=1.7,
            capsize=5.5,
            capthick=1.7,
            zorder=1.2,
        )

    if lossless is not None:
        lossless_xytext = auto_annotation_xytext(
            ax,
            kind="lossless",
            x=lossless["mean_topk"],
            y=lossless["score"] * 100.0,
            x_min=ax.get_xlim()[0],
            x_max=ax.get_xlim()[1],
            y_min=ax.get_ylim()[0],
            y_max=ax.get_ylim()[1],
        )
        annotate_point(
            ax,
            lossless["mean_topk"],
            lossless["score"] * 100.0,
            f"Lossless\n({lossless['mean_topk']:.2f}, {lossless['score']*100:.2f})",
            xytext=lossless_xytext,
            color=lossless_color,
            marker_color=lossless_color,
            marker="o",
            s=96,
            fontsize=10.5,
            ha="left",
            va="top",
            zorder=10,
        )

    if optimum is not None:
        optimum_xytext = auto_annotation_xytext(
            ax,
            kind="optimum",
            x=optimum["mean_topk"],
            y=optimum["score"] * 100.0,
            x_min=ax.get_xlim()[0],
            x_max=ax.get_xlim()[1],
            y_min=ax.get_ylim()[0],
            y_max=ax.get_ylim()[1],
        )
        annotate_point(
            ax,
            optimum["mean_topk"],
            optimum["score"] * 100.0,
            f"Optimum\n({optimum['mean_topk']:.2f}, {optimum['score']*100:.2f})",
            xytext=optimum_xytext,
            relpos=(0.5, 0),
            color=optimum_color,
            marker_color=optimum_color,
            marker="o",
            s=96,
            fontsize=10.5,
            ha="left",
            va="top",
            zorder=10,
        )

    if show_baseline_label:
        ax.text(
            x_min + 0.02 * x_span,
            baseline_score_pct + 0.06,
            "Baseline score",
            color=baseline_color,
            fontsize=10,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Mean Top-k (lower is better)")
    ax.set_ylabel("Score (%)")
    # ax.set_title("Search Space and Pareto Frontier")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    if legend_pdf is not None:
        legend_fig, legend_artist = build_legend_figure()
        legend_pdf.parent.mkdir(parents=True, exist_ok=True)

        # Export using the legend artist's own bbox to avoid PDF backend
        # including hidden/full-figure artists in "tight" calculations.
        legend_fig.canvas.draw()
        renderer = legend_fig.canvas.get_renderer()
        legend_bbox = legend_artist.get_window_extent(renderer=renderer)
        legend_bbox = legend_bbox.transformed(legend_fig.dpi_scale_trans.inverted())

        legend_fig.savefig(legend_pdf, bbox_inches=legend_bbox, pad_inches=0.0)
        plt.close(legend_fig)
        print(f"Saved legend pdf: {legend_pdf}")

    print(f"Saved frontier json: {out_json}")
    print(f"Saved plot png: {out_png}")
    print(f"Saved plot pdf: {out_pdf}")
    print(f"All points: {len(rows)} | Pareto points: {len(pareto_rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="optimization_results_lmeval.json", help="Path to results json.")
    ap.add_argument("--baseline_score", type=float, required=True, help="Baseline score percentage to draw as a horizontal line.")
    ap.add_argument("--baseline_topk", type=float, default=8.0, help="Baseline mean top-k to annotate and compare against.")
    ap.add_argument(
        "--fig_scale",
        type=float,
        default=1.8,
        help="Scale factor to shrink the original 8.8x5.8 canvas. Typical IEEE range: 1.7-2.0.",
    )
    ap.add_argument("--show_baseline_label", action="store_true", help="Show the baseline score label on the plot.")
    ap.add_argument("--baseline_score_std", type=float, default=0.0, help="Standard deviation or confidence interval width for baseline score (percentage points).")
    ap.add_argument("--out_json", type=str, default=None, help="Path to output pareto frontier json file.")
    ap.add_argument("--out_png", type=str, default=None, help="Path to output PNG file.")
    ap.add_argument("--out_pdf", type=str, default=None, help="Path to output PDF file.")
    ap.add_argument("--legend_pdf", type=str, default=None, help="Path to output legend PDF file.")
    ap.add_argument("--show_pareto", action="store_true", help="Print pareto points to console.")
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
    pareto_rows = sorted([rows[i] for i in range(len(rows)) if mask[i]], key=lambda d: (d["mean_topk"], -d["score"]))

    baseline_score_pct = float(args.baseline_score)
    baseline_topk = float(args.baseline_topk)
    baseline_score_std = float(args.baseline_score_std)
    fig_scale = float(args.fig_scale)
    if fig_scale <= 0:
        raise ValueError("--fig_scale must be > 0")

    out_json = Path(args.out_json) if args.out_json else path.parent / f"{path.stem}_pareto_frontier.json"
    out_png = Path(args.out_png) if args.out_png else path.parent / f"{path.stem}.png"
    out_pdf = Path(args.out_pdf) if args.out_pdf else path.parent / f"{path.stem}.pdf"
    legend_pdf = Path(args.legend_pdf) if args.legend_pdf else None

    write_outputs(
        path=path,
        rows=rows,
        pareto_rows=pareto_rows,
        baseline_topk=baseline_topk,
        baseline_score_pct=baseline_score_pct,
        show_baseline_label=args.show_baseline_label,
        fig_scale=fig_scale,
        out_json=out_json,
        out_png=out_png,
        out_pdf=out_pdf,
        legend_pdf=legend_pdf,
        baseline_score_std=baseline_score_std,
    )

    if args.show_pareto:
        print("\nPareto-optimal points:")
        for r in pareto_rows:
            print(f"  - score: {r['score']:.6f}, mean_topk: {r['mean_topk']:.6f}, formula: {r['formula']}")


if __name__ == "__main__":
    main()
