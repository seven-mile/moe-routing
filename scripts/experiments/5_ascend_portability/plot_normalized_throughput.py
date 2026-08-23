#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.facecolor": "#F7F8FA",
        "axes.linewidth": 0.8,
    }
)

CONCURRENCY = [128, 256, 384, 512]
SPECDECODE = [1.0000, 1.6822, 2.2938, 1.7774]
SPEC_K_OPTIMUM = [1.1423, 2.1643, 3.0308, 1.9573]


def main() -> None:
    x = np.arange(len(CONCURRENCY))
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.45, 2.05))

    ax.bar(
        x - width / 2,
        SPECDECODE,
        width,
        color="#000000",
        alpha=0.78,
        label="SpecDec Baseline",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        SPEC_K_OPTIMUM,
        width,
        color="#8b5fbf",
        label="Spec-K Optimum",
        zorder=3,
    )

    for xpos, baseline, optimum in zip(x, SPECDECODE, SPEC_K_OPTIMUM):
        ax.text(
            xpos + width / 2,
            optimum - 0.10,
            f"{optimum / baseline:.2f}x",
            ha="center",
            va="top",
            fontsize=7.5,
            color="white",
            fontweight="medium",
        )

    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Normalized throughput")
    ax.set_xticks(x, [str(value) for value in CONCURRENCY])
    ax.set_ylim(0, 3.65)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#D7DDE6", alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B7C0CC")
    ax.spines["bottom"].set_color("#B7C0CC")
    ax.tick_params(colors="#556070", length=3, width=0.8)
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=1.0, handlelength=1.2)

    output = Path(__file__).resolve().parents[3] / "paper" / "ascend" / "normalized_throughput.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
