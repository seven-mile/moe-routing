import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap  # 改为导入定色列表工具
import numpy as np

# --- 1. 数据准备：定义 8x8 色值矩阵 ---
# 这里填入从原图中提取的精确色值
# 直接使用十六进制色值的 8x8 矩阵，已按列对齐方便修改
data_hex = [
    #  0轴(列):    0          1          2          3          4          5          6          7
    ['#d9ed92', '#99d98c', '#d9ed92', '#d9ed92', '#d9ed92', '#52b69a', '#d9ed92', '#d9ed92'], # 行 0
    ['#99d98c', '#d9ed92', '#d9ed92', '#2f9ca5', '#d9ed92', '#d9ed92', '#d9ed92', '#d9ed92'], # 行 1
    ['#d9ed92', '#d9ed92', '#d9ed92', '#99d98c', '#d9ed92', '#d9ed92', '#1e90aa', '#329ea4'], # 行 2
    ['#d9ed92', '#2f9ca5', '#99d98c', '#d9ed92', '#99d98c', '#d9ed92', '#d9ed92', '#99d98c'], # 行 3
    ['#d9ed92', '#d9ed92', '#d9ed92', '#99d98c', '#d9ed92', '#1782a8', '#d9ed92', '#c2e78e'], # 行 4
    ['#52b69a', '#d9ed92', '#d9ed92', '#d9ed92', '#1782a8', '#d9ed92', '#d9ed92', '#d3eb91'], # 行 5
    ['#99d98c', '#99d98c', '#1e90aa', '#d9ed92', '#d9ed92', '#d9ed92', '#99d98c', '#d9ed92'], # 行 6
    ['#d9ed92', '#d9ed92', '#329ea4', '#99d98c', '#c2e78e', '#d3eb91', '#d9ed92', '#d9ed92']  # 行 7
]

# --- 2. 映射逻辑：将 Hex 转换为索引和自定义色板 ---
# 获取矩阵中出现的所有唯一颜色并排序，保证索引稳定
unique_colors = sorted(list(set(color for row in data_hex for color in row)))
color_to_idx = {color: i for i, color in enumerate(unique_colors)}

# 构建索引矩阵
data_indices = np.array([[color_to_idx[color] for color in row] for row in data_hex])

# 创建定色调色板
custom_cmap = ListedColormap(unique_colors)

# --- 3. 绘图辅助函数 ---
def draw_board(ax, data_idx, step_idx, selection, disabled_indices):
    rows, cols = data_idx.shape
    
    # 使用索引矩阵绘图
    # np.flipud 配合 invert_yaxis 确保矩阵[0,0]在左上角，且 pcolormesh 渲染正确
    ax.pcolormesh(data_idx, cmap=custom_cmap, edgecolors='white', 
                  linewidth=1.5, vmin=0, vmax=len(unique_colors)-1)
    
    ax.set_aspect('equal')
    ax.invert_yaxis() # 翻转Y轴，使 0 行在最上面
    ax.xaxis.tick_top() # X轴刻度移到上方
    
    # 刻度设置在方格中心 (0.5, 1.5 ...)
    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_xticklabels(range(cols), fontsize=9, fontfamily='sans-serif', fontweight='bold')
    ax.set_yticklabels(range(rows), fontsize=9, fontfamily='sans-serif', fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, top=False, bottom=False, right=False)

    # 绘制对称轴 (注意：翻转后的坐标轴，(0,0)是左上，(8,8)是右下)
    ax.plot([0, 8], [0, 8], color='#555555', linestyle='--', linewidth=0.8, alpha=0.5)

    # 绘制禁用区域
    for k in disabled_indices:
        # 矩形参数：(x, y), width, height
        ax.add_patch(patches.Rectangle((0, k), 8, 1, facecolor='gray', alpha=0.4, zorder=2))
        ax.add_patch(patches.Rectangle((k, 0), 1, 8, facecolor='gray', alpha=0.4, zorder=2))

    # 绘制当前选择 (红框)
    if selection:
        # selection 传入的是集合，转为列表
        nodes = list(selection)
        r, c = nodes[0], nodes[1]
        for row_idx, col_idx in [(r, c), (c, r)]:
            rect = patches.Rectangle((col_idx, row_idx), 1, 1, linewidth=2.5, 
                                     edgecolor='#e31a1c', facecolor='none', zorder=10)
            ax.add_patch(rect)
    
    sel_nodes = list(selection)
    sel_str = "{" + f"{sel_nodes[0]},{sel_nodes[1]}" + "}"
    ax.set_xlabel(f"({step_idx}) Selection {sel_str}", fontsize=10, fontweight='bold', labelpad=12)

# --- 4. 主程序：生成四个步骤的图 ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.45)

steps = [
    (1, {4, 5}, set()),
    (2, {2, 6}, {4, 5}),
    (3, {1, 3}, {4, 5, 2, 6}),
    (4, {0, 7}, {4, 5, 2, 6, 1, 3})
]

for ax, (idx, sel, disabled) in zip(axes, steps):
    # 注意：这里传入的是 data_indices
    draw_board(ax, data_indices, idx, sel, disabled)

# --- 5. 添加连接箭头 ---
fig.canvas.draw()
trans = fig.transFigure

for i in range(3):
    bbox1 = axes[i].get_position()
    bbox2 = axes[i+1].get_position()
    
    x_start = bbox1.x1 + 0.015 
    x_end = bbox2.x0 - 0.015
    y_mid = (bbox1.y0 + bbox1.y1) / 2
    
    arrow = patches.FancyArrowPatch(
        (x_start, y_mid), (x_end, y_mid),
        transform=trans, 
        arrowstyle='-|>', 
        mutation_scale=15, 
        color='#444444',
        linewidth=1.2
    )
    fig.add_artist(arrow)

plt.savefig('heatmap_process_hex_fixed.png', dpi=300, bbox_inches='tight')
plt.show()
