import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.ticker import ScalarFormatter, NullLocator
from matplotlib.lines import Line2D

# 设置学术风格参数
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Noto Sans', 'Arial', 'sans-serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.titlepad': 10,
    'figure.facecolor': '#FCFCFD',
    'axes.facecolor': '#F7F8FA',
})


# 固定颜色映射，确保一致性
COLOR_MAP = {
    'vanilla': '#9E9E9E',      # 灰色 - 中立
    'baseline': '#000000',     # 黑色 - 参考
    'lossless': '#64B5F6',     # 浅蓝 - 主角1
    'optimum': '#E64B2E',      # 红色 - 主角2
}

# 主角的标记样式突出显示
MARKER_MAP = {
    'vanilla': 'o',
    'baseline': 's',
    'lossless': '^',           # 三角形强调
    'optimum': 'D',            # 菱形强调
}

# 主角的线宽加粗
LINEWIDTH_MAP = {
    'vanilla': 1.5,
    'baseline': 1.5,
    'lossless': 2.2,
    'optimum': 2.2,
}

# 统一视觉语义：提升现代感，同时维持论文可读性
STYLE = {
    'canvas_bg': '#FCFCFD',
    'panel_bg': '#F7F8FA',
    'grid_major': '#D7DDE6',
    'grid_minor': '#E9EDF3',
    'spine': '#B7C0CC',
    'text_main': '#1F2937',
    'text_muted': '#556070',
    'marker_border': (0.75, 0.81, 0.88, 0.55),
    'line_halo': (0.90, 0.93, 0.96, 0.95),
}


def load_and_extract_label(df):
    """从DataFrame提取关键指标，兼容不同字段名"""
    col_map = {
        "concurrency": [
            "Maximum request concurrency",
            "Peak concurrent requests"
        ],
        "throughput": [
            "Output token throughput (tok/s)",
            "Total token throughput (tok/s)"
        ],
        "tpot": [
            "Mean TPOT (ms)",
            "Median TPOT (ms)"
        ]
    }

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"None of columns {candidates} found")

    c_col = find_col(col_map["concurrency"])
    t_col = find_col(col_map["throughput"])
    l_col = find_col(col_map["tpot"])

    sub = df[[c_col, t_col, l_col]].copy()
    sub.columns = ["concurrency", "throughput", "tpot"]
    sub = sub.sort_values("concurrency")
    
    return sub


def load_and_extract(args):
    """从CSV加载数据并按照labels分组提取"""
    df = pd.read_csv(args.file)
    labels = args.labels
    
    data_list = []
    for label in labels:
        mask = df["filename"].str.contains(label, na=False)
        if not mask.any():
            raise ValueError(f"No data found for label: {label}")
        label_df = df[mask]
        data_list.append(load_and_extract_label(label_df))
    
    return data_list, labels


def add_performance_indicators(ax):
    """
    在图中添加性能指示符号
    - ↑ 表示 Throughput 越高越好
    - ↓ 表示 Latency 越低越好
    """
    # 在右上角添加指示符号（使用Unicode字符，简洁美观）
    indicator_box = dict(
        boxstyle='round,pad=0.25,rounding_size=0.15',
        facecolor='white',
        edgecolor='none',
        alpha=0.85
    )
    ax.text(0.98, 0.97, '↑  Higher Throughput',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            color='#2E7D32', fontweight='normal',
            bbox=indicator_box)
    ax.text(0.98, 0.90, '↓  Lower Latency',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            color='#E64B2E', fontweight='normal',
            bbox=indicator_box)


def style_axis(ax, batch_sizes, ylabel):
    """统一坐标轴样式，让画面更干净、层级更清晰。"""
    ax.set_facecolor(STYLE['panel_bg'])
    ax.set_xscale('log')
    ax.set_xlabel('Global Batch Size', color=STYLE['text_main'])
    ax.set_ylabel(ylabel, color=STYLE['text_main'])

    if len(batch_sizes) > 0:
        ax.set_xticks(batch_sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticklabels([str(int(x)) for x in batch_sizes])

    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='x', which='minor', bottom=False)

    ax.grid(True, axis='y', which='major', linestyle='--',
            linewidth=0.75, color=STYLE['grid_major'], alpha=0.9)
    ax.grid(True, axis='y', which='minor', linestyle='-',
            linewidth=0.45, color=STYLE['grid_minor'], alpha=0.5)
    ax.grid(False, axis='x')

    for spine_name, spine in ax.spines.items():
        if spine_name in ('top', 'right'):
            spine.set_visible(False)
        else:
            spine.set_color(STYLE['spine'])
            spine.set_linewidth(0.9)

    ax.tick_params(axis='both', colors=STYLE['text_muted'], length=4, width=0.8)


def save_standalone_legend(labels, out_file, fig_scale=1.35):
    """单独导出图例文件，便于拼接subplot时复用。"""
    base_w, base_h = 4.8, 0.75
    fig_legend = plt.figure(figsize=(base_w / fig_scale, base_h / fig_scale), facecolor='white')

    handles = []
    for label in labels:
        color = COLOR_MAP.get(label, '#000000')
        marker = MARKER_MAP.get(label, 'o')
        linewidth = LINEWIDTH_MAP.get(label, 1.5)
        is_focus = label in ['lossless', 'optimum']
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle='-',
                linewidth=linewidth,
                markersize=6.8 if is_focus else 5.2,
                markeredgewidth=0.6,
                markeredgecolor=STYLE['marker_border'],
                markerfacecolor=color,
                alpha=0.95 if is_focus else 0.82,
                label=label,
            )
        )

    legend = fig_legend.legend(
        handles=handles,
        labels=labels,
        ncol=len(labels),
        loc='center',
        bbox_to_anchor=(0.5, 0.5),
        frameon=True,
        fancybox=True,
        edgecolor='#D5DCE6',
        facecolor='white',
        framealpha=0.95,
        borderpad=0.5,
        handlelength=2.2,
        columnspacing=1.2,
    )
    for txt in legend.get_texts():
        txt.set_color(STYLE['text_main'])

    fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_legend.savefig(out_file + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_legend)


def plot_metric_advanced(data_list, labels, metric, ylabel, out_file,
                         show_legend=False, show=False, fig_scale=1.35):
    """
    高级绘图函数
    """
    # Shrink source canvas so that with fixed LaTeX include width, text appears relatively larger.
    base_w, base_h = 5.2, 4.2
    fig, ax = plt.subplots(figsize=(base_w / fig_scale, base_h / fig_scale), facecolor=STYLE['canvas_bg'])
    
    # 获取当前batch size值（用于显示具体数值）
    batch_sizes = data_list[0]["concurrency"].values if data_list else []
    
    # 记录每个标签的最后一个数据点，用于添加倍数标签
    last_point_data = {}
    
    for data, label in zip(data_list, labels):
        color = COLOR_MAP.get(label, '#000000')
        marker = MARKER_MAP.get(label, 'o')
        linewidth = LINEWIDTH_MAP.get(label, 1.5)
        is_focus = label in ['lossless', 'optimum']
        x = data["concurrency"]
        y = data[metric]
        
        # 记录最后一个数据点（用于标注倍数）
        last_x_val = x.iloc[-1] if hasattr(x, 'iloc') else x[-1]
        last_y_val = y.iloc[-1] if hasattr(y, 'iloc') else y[-1]
        # 确保都是标量
        if hasattr(last_x_val, 'item'):
            last_x_val = last_x_val.item()
        if hasattr(last_y_val, 'item'):
            last_y_val = last_y_val.item()
        last_point_data[label] = (float(last_x_val), float(last_y_val))
        
        # 为主角添加更高的zorder使其更突出
        zorder = 5 if is_focus else 3

        # 先画一层柔和halo线条（不带marker），避免marker出现双层边框。
        ax.plot(
            x,
            y,
            color=STYLE['line_halo'],
            linewidth=linewidth + 0.8,
            zorder=zorder - 1,
            linestyle='-',
            alpha=0.65,
            solid_capstyle='round',
            label='_nolegend_'
        )
        
        line, = ax.plot(
            x,
            y,
            marker=marker,
            markersize=6.8 if is_focus else 5.2,
            markeredgewidth=0.6,
            markeredgecolor=STYLE['marker_border'],
            markerfacecolor=color,
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
            linestyle='-',
            alpha=0.95 if is_focus else 0.82,
            solid_capstyle='round'
        )
    
    style_axis(ax, batch_sizes, ylabel)
    
    # 添加相对于baseline的吞吐倍数标签
    baseline_throughput = float(last_point_data.get('baseline', last_point_data[labels[0]][1])[1])
    for label in labels:
        if label not in last_point_data:
            continue
        
        last_x, last_y = last_point_data[label]
        color = COLOR_MAP.get(label, '#000000')
        ratio = last_y / baseline_throughput
        label_text = f"{ratio:.2f}×"
        
        # 在数据点右边添加标签
        ax.text(last_x, last_y, f"  {label_text}",
                color=color,
                fontsize=8.5,
                verticalalignment='center',
                horizontalalignment='left',
                fontweight='normal')
    
    # 添加性能指示符号
    # add_performance_indicators(ax)
    
    # 根据需要显示图例
    if show_legend:
        legend = ax.legend(
            frameon=True,
            fancybox=True,
            edgecolor='#D5DCE6',
            facecolor='white',
            framealpha=0.92,
            borderpad=0.55,
            handlelength=2.2,
            loc='best'
        )
        for txt in legend.get_texts():
            txt.set_color(STYLE['text_main'])
    
    # 调整边距
    ax.margins(x=0.05, y=0.05)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.savefig(out_file + '.pdf', bbox_inches='tight')
        plt.savefig(out_file + '.png', dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True,
                        help="CSV/log files for all method, per configuration")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each method (order matters)")
    parser.add_argument("--out_prefix", type=str, default="data/plot/main_res",
                        help="Output prefix for pdf files")
    parser.add_argument("--show", action="store_true",
                        help="Show plots")
    parser.add_argument("--fig_scale", type=float, default=1.35,
                        help="Scale factor to shrink source canvas. Typical range: 1.2-1.5.")
    parser.add_argument("--with_legend", action="store_true",
                        help="Enable legend in main plots (disabled by default)")
    parser.add_argument("--legend_file", action="store_true",
                        help="Enable standalone legend export")
    
    args = parser.parse_args()
    if args.fig_scale <= 0:
        raise ValueError("--fig_scale must be > 0")
    
    data_list, labels = load_and_extract(args)
    
    # 分别绘制吞吐量和延迟图
    plot_metric_advanced(
        data_list, 
        labels, 
        metric="throughput",
        ylabel="Throughput (tokens/s)",
        out_file=args.out_prefix + "_throughput",
        show_legend=args.with_legend,
        show=args.show,
        fig_scale=args.fig_scale,
    )

    if args.legend_file:
        save_standalone_legend(
            labels=labels,
            out_file=args.out_prefix + "_legend",
            fig_scale=args.fig_scale,
        )
    
    # plot_metric_advanced(
    #     data_list,
    #     labels,
    #     metric="tpot",
    #     ylabel="TPOT (ms)",
    #     out_file=args.out_prefix + "_latency",
    #     show_legend=args.with_legend,
    #     show=args.show
    # )


if __name__ == "__main__":
    main()
