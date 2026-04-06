#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"],
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "#F8FAFC",
    }
)


STYLE = {
    "panel_bg": "#F8FAFC",
    "grid": "#D8E0EA",
    "spine": "#B8C2CF",
    "text": "#1F2937",
    "muted": "#526076",
    "other": "#B8C1CC",
}


DEFAULT_COLORS = [
    # "#F28E2B",
    # "#59A14F",
    "#4E79A7",
    "#E15759",
    # "#9C755F",
    "#76B7B2",
    "#EDC948",
    "#AF7AA1",
]


def load_rows(path, case_col, total_col, component_cols):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        header = [name.strip() for name in reader.fieldnames if name is not None]
        if case_col not in header:
            raise ValueError(f"Missing case column {case_col!r} in {path}")
        if total_col not in header:
            raise ValueError(f"Missing total column {total_col!r} in {path}")

        if component_cols is None:
            component_cols = [name for name in header if name not in {case_col, total_col}]
        else:
            missing = [name for name in component_cols if name not in header]
            if missing:
                raise ValueError(f"Missing component columns {missing!r} in {path}")

        rows = []
        for row_idx, row in enumerate(reader, start=2):
            case = (row.get(case_col) or "").strip()
            if not case:
                continue

            try:
                total = float(row[total_col])
                components = [float(row[name]) for name in component_cols]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid numeric value in {path} at row {row_idx}") from exc

            component_sum = sum(components)
            else_value = total - component_sum
            if else_value < -1e-6:
                raise ValueError(
                    f"Row {row_idx} in {path} has component sum larger than total: "
                    f"total={total}, component_sum={component_sum}"
                )
            if else_value < 0:
                else_value = 0.0

            rows.append(
                {
                    "case": case,
                    "total": total,
                    "components": components,
                    "else": else_value,
                }
            )

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    return rows, component_cols


def build_component_colors(component_cols):
    colors = {}
    for idx, name in enumerate(component_cols):
        colors[name] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
    colors["Other"] = STYLE["other"]
    return colors


def make_legend_handles(component_cols, colors):
    return list(component_cols) + ["Other"]


def build_legend_box(labels, colors):
    def build_item(label):
        swatch = DrawingArea(16, 10, 0, 0)
        swatch.add_artist(
            Rectangle(
                (0, 0),
                16,
                10,
                facecolor=colors[label],
                edgecolor="none",
            )
        )
        text = TextArea(label, textprops={"color": STYLE["text"], "fontsize": 9.0})
        return HPacker(children=[swatch, text], align="center", pad=0, sep=3)

    rows = [labels[start : start + 3] for start in range(0, len(labels), 3)]
    row_boxes = []
    for row in rows:
        children = [build_item(label) for label in row]
        row_boxes.append(HPacker(children=children, align="center", pad=0, sep=10))

    return VPacker(children=row_boxes, align="center", pad=0, sep=4)


def save_legend(out_path, component_cols, colors, ncol=3):
    labels = make_legend_handles(component_cols, colors)

    probe_fig = plt.figure(figsize=(3.35, 0.58), facecolor="white")
    probe_fig.patch.set_facecolor("white")
    probe_ax = probe_fig.add_axes([0, 0, 1, 1])
    probe_ax.set_axis_off()
    legend_box = build_legend_box(labels, colors)
    anchored = AnchoredOffsetbox(
        loc="center",
        child=legend_box,
        frameon=False,
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=probe_ax.transAxes,
        borderpad=0.0,
    )
    probe_ax.add_artist(anchored)

    probe_fig.canvas.draw()
    renderer = probe_fig.canvas.get_renderer()
    bbox = anchored.get_window_extent(renderer=renderer).transformed(probe_fig.dpi_scale_trans.inverted())

    width = max(bbox.width, 1e-3)
    height = max(bbox.height, 1e-3)
    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    legend_box = build_legend_box(labels, colors)
    anchored = AnchoredOffsetbox(
        loc="center",
        child=legend_box,
        frameon=False,
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    ax.add_artist(anchored)

    fig.savefig(out_path, bbox_inches=None, pad_inches=0.0)
    plt.close(probe_fig)
    plt.close(fig)


def style_axes(ax, case_labels, x_max, x_label):
    ax.set_facecolor(STYLE["panel_bg"])
    ax.set_xlim(0, x_max)
    ax.set_xlabel(x_label, color=STYLE["text"])
    ax.set_yticks(list(range(len(case_labels))))
    ax.set_yticklabels(case_labels, color=STYLE["text"])
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, color=STYLE["grid"], alpha=0.9)
    ax.grid(False, axis="y")
    ax.tick_params(axis="x", colors=STYLE["muted"], length=4, width=0.8)
    ax.tick_params(axis="y", colors=STYLE["text"], length=0, width=0.0, pad=6)
    for spine_name, spine in ax.spines.items():
        if spine_name in {"top", "right"}:
            spine.set_visible(False)
        else:
            spine.set_color(STYLE["spine"])
            spine.set_linewidth(0.85)


def plot_breakdown(rows, component_cols, out_png, out_pdf, legend_pdf=None, x_label="Latency (ms)"):
    colors = build_component_colors(component_cols)
    case_labels = [row["case"] for row in rows]
    totals = [row["total"] for row in rows]
    max_total = max(totals)
    x_max = max_total * 1.18

    width = 3.45
    height = max(1.35, 0.42 * len(rows) + 0.42)
    fig, ax = plt.subplots(figsize=(width, height))

    y_positions = list(range(len(rows)))
    bar_height = 0.56

    left_offsets = [0.0] * len(rows)
    for comp_idx, comp_name in enumerate(component_cols):
        values = [row["components"][comp_idx] for row in rows]
        ax.barh(
            y_positions,
            values,
            left=left_offsets,
            height=bar_height,
            color=colors[comp_name],
            edgecolor="white",
            linewidth=0.8,
            label=comp_name,
            zorder=3,
        )
        left_offsets = [left + value for left, value in zip(left_offsets, values)]

    else_values = [row["else"] for row in rows]
    ax.barh(
        y_positions,
        else_values,
        left=left_offsets,
        height=bar_height,
        color=colors["Other"],
        edgecolor="white",
        linewidth=0.8,
        label="Other",
        zorder=3,
    )

    baseline_total = rows[0]["total"]
    pad = max_total * 0.018
    for y, row in zip(y_positions, rows):
        ratio = row["total"] / baseline_total * 100.0
        ax.text(
            row["total"] + pad,
            y,
            f"{ratio:.1f}%",
            va="center",
            ha="left",
            color=STYLE["text"],
            fontsize=9.0,
            fontweight="semibold",
        )

    ax.invert_yaxis()
    style_axes(ax, case_labels, x_max, x_label)

    fig.tight_layout(pad=0.45)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    if legend_pdf is not None:
        legend_pdf.parent.mkdir(parents=True, exist_ok=True)
        save_legend(legend_pdf, component_cols, colors)


def parse_args():
    ap = argparse.ArgumentParser(description="Plot latency breakdown stacked bars from a CSV file.")
    ap.add_argument("--infile", required=True, help="Input CSV file.")
    ap.add_argument("--out_png", default=None, help="Output PNG path.")
    ap.add_argument("--out_pdf", default=None, help="Output PDF path.")
    ap.add_argument("--legend_pdf", default=None, help="Optional standalone legend PDF path.")
    ap.add_argument("--case_col", default="Case", help="Case column name.")
    ap.add_argument("--total_col", default="Total", help="Total column name.")
    ap.add_argument(
        "--component_cols",
        nargs="*",
        default=None,
        help="Component columns to stack, in order. Defaults to all columns except case and total.",
    )
    ap.add_argument("--x_label", default="Latency (ms)", help="X-axis label.")
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.infile)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    rows, component_cols = load_rows(in_path, args.case_col, args.total_col, args.component_cols)

    out_png = Path(args.out_png) if args.out_png else in_path.with_suffix(".png")
    out_pdf = Path(args.out_pdf) if args.out_pdf else in_path.with_suffix(".pdf")
    legend_pdf = Path(args.legend_pdf) if args.legend_pdf else None

    plot_breakdown(
        rows=rows,
        component_cols=component_cols,
        out_png=out_png,
        out_pdf=out_pdf,
        legend_pdf=legend_pdf,
        x_label=args.x_label,
    )

    print(f"Saved plot png: {out_png}")
    print(f"Saved plot pdf: {out_pdf}")
    if legend_pdf is not None:
        print(f"Saved legend pdf: {legend_pdf}")


if __name__ == "__main__":
    main()