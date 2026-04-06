#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            filename = (r.get("filename") or "").strip()
            acc_text = r.get("Acceptance rate (%)")
            topk_text = r.get("Mean Token Top-K")
            if not filename or acc_text is None or topk_text is None:
                continue

            try:
                acc = float(acc_text)
                topk = float(topk_text)
            except Exception:
                continue

            if not (math.isfinite(acc) and math.isfinite(topk)):
                continue

            name = filename.lower()
            tag = "normal"
            if "baseline" in name:
                tag = "baseline"
            elif "lossless" in name:
                tag = "lossless"
            elif "optimum" in name:
                tag = "optimum"

            rows.append({"filename": filename, "acc": acc, "topk": topk, "tag": tag})
    return rows


def highlight_point(ax, x, y, marker, marker_color, s=90, zorder=10):
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


def pick_special_point(rows, tag):
    items = [r for r in rows if r["tag"] == tag]
    if not items:
        return None

    if tag == "baseline":
        return sorted(items, key=lambda r: (r["topk"], -r["acc"]))[0]
    if tag == "lossless":
        return sorted(items, key=lambda r: (-r["acc"], r["topk"]))[0]
    if tag == "optimum":
        return sorted(items, key=lambda r: (r["topk"], -r["acc"]))[0]
    return items[0]


def make_plot(rows, out_png: Path, out_pdf: Path):
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "--",
        }
    )

    bg_point_color = "#9fb3c8"
    baseline_color = "#98a1ab"
    lossless_color = "#2a9d8f"
    optimum_color = "#8b5fbf"

    topks = np.array([r["topk"] for r in rows], dtype=float)
    accs = np.array([r["acc"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(3.55, 2.70))

    ax.scatter(
        topks,
        accs,
        s=34,
        marker="o",
        color=bg_point_color,
        alpha=0.55,
        edgecolors="none",
        zorder=2,
    )

    x_min, x_max = topks.min(), topks.max()
    y_min, y_max = accs.min(), accs.max()
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    ax.set_xlim(x_min - 0.10 * x_span, x_max + 0.08 * x_span)
    ax.set_ylim(y_min - 0.10 * y_span, y_max + 0.10 * y_span)

    baseline = pick_special_point(rows, "baseline")
    if baseline is not None:
        highlight_point(
            ax,
            baseline["topk"],
            baseline["acc"],
            marker="D",
            marker_color="#b5bdc6",
            s=82,
            zorder=9,
        )

    lossless = pick_special_point(rows, "lossless")
    if lossless is not None:
        highlight_point(
            ax,
            lossless["topk"],
            lossless["acc"],
            marker="o",
            marker_color=lossless_color,
            s=96,
            zorder=10,
        )

    optimum = pick_special_point(rows, "optimum")
    if optimum is not None:
        highlight_point(
            ax,
            optimum["topk"],
            optimum["acc"],
            marker="o",
            marker_color=optimum_color,
            s=96,
            zorder=10,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=bg_point_color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=6.8,
            label="Policy",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#b5bdc6",
            markeredgecolor="white",
            markersize=5.8,
            label="Baseline",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=lossless_color,
            markeredgecolor="white",
            markersize=6.8,
            label="Lossless",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=optimum_color,
            markeredgecolor="white",
            markersize=6.8,
            label="Optimum",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.48,
        handletextpad=0.45,
        columnspacing=0.85,
    )

    ax.set_xlabel("Mean Token Top-K")
    ax.set_ylabel("Acceptance rate (%)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot Mean Token Top-K vs Acceptance rate from csv."
    )
    ap.add_argument("--infile", type=str, required=True, help="Path to input csv file.")
    ap.add_argument("--out_png", type=str, default=None, help="Path to output png file.")
    ap.add_argument("--out_pdf", type=str, default=None, help="Path to output pdf file.")
    args = ap.parse_args()

    in_path = Path(args.infile)
    if not in_path.exists():
        raise FileNotFoundError(f"Input csv not found: {in_path}")

    rows = load_rows(in_path)
    if not rows:
        raise ValueError(f"No valid rows found in {in_path}")

    out_png = Path(args.out_png) if args.out_png else in_path.with_name(f"{in_path.stem}_topk_vs_acc.png")
    out_pdf = Path(args.out_pdf) if args.out_pdf else in_path.with_name(f"{in_path.stem}_topk_vs_acc.pdf")

    make_plot(rows=rows, out_png=out_png, out_pdf=out_pdf)
    print(f"Saved plot png: {out_png}")
    print(f"Saved plot pdf: {out_pdf}")
    print(f"All points: {len(rows)}")


if __name__ == "__main__":
    main()
