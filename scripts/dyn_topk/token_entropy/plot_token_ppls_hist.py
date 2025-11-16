import sys
import json
import matplotlib.pyplot as plt
import numpy as np # 新增：用于计算分位数
import argparse # 新增：用于处理更复杂的命令行参数

# --- 1. 定义和解析命令行参数 ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="从 JSON 文件中读取浮点数列表（LLM PPL值），并绘制其直方图，可选地在图上标记分位数。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 必需的参数：文件路径
    parser.add_argument(
        "file_path", 
        type=str, 
        help="包含浮点数列表的 JSON 文件路径。"
    )
    
    # 可选的参数：分位数列表
    parser.add_argument(
        "-q", "--quantiles", 
        type=float, 
        nargs='*', # 允许0个或多个浮点数
        default=[], 
        help=(
            "要在图上标记的分位数（0到100之间）。\n"
            "例如: -q 50 90 99"
        )
    )
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    file_path = args.file_path
    quantiles_to_mark = args.quantiles

    # --- A. 从 JSON 文件中读取浮点数列表 ---
    try:
        with open(file_path, 'r') as f:
            # 使用 json.load() 读取整个 JSON 结构
            ppl_list = json.load(f)
        
        if not isinstance(ppl_list, list) or not ppl_list:
            print(f"错误：JSON 文件中没有找到数据或列表为空。")
            return

        # 转换为 numpy 数组，方便后续处理
        ppl_data = np.array(ppl_list)
        print(f"成功读取 {len(ppl_data)} 个数据点。")

        # --- B. 绘制直方图 ---
        
        # 创建图形和坐标轴对象
        plt.figure(figsize=(12, 7))
        
        # 使用 plt.hist() 绘制直方图
        plt.hist(ppl_data, bins=100, edgecolor='black', alpha=0.7, color='skyblue')
        
        # --- C. 标记分位数 ---
        if quantiles_to_mark:
            print("\n--- 分位数标记 ---")
            
            # 过滤和排序分位数，确保它们在 [0, 100] 范围内
            valid_quantiles = sorted([q for q in quantiles_to_mark if 0 <= q <= 100])
            
            # 用于确定文本标签的垂直位置，避免重叠
            text_height_factor = 0.9 
            
            for i, q in enumerate(valid_quantiles):
                # 计算分位数的值 (例如：第 99 百分位点的值)
                q_value = np.percentile(ppl_data, q)
                
                # 打印信息
                print(f"Q{q:.1f} ({q:.1f}%) 值: {q_value:.4f}")

                # 1. 绘制垂直线
                plt.axvline(
                    q_value, 
                    color='red', 
                    linestyle='--', 
                    linewidth=1.5, 
                    alpha=0.8,
                    label=f'{q:.1f}% Quantile' # 可选地添加到图例
                )
                
                # 2. 标记分位数和值
                label_text = f'Q{q:.1f}: {q_value:.4f}'
                
                # 计算文本位置（y轴位置略有偏移，以防止多条文本重叠）
                y_max = plt.gca().get_ylim()[1]
                y_pos = y_max * (text_height_factor - 0.04 * (i % 5)) # 错开标记
                
                plt.text(
                    q_value, 
                    y_pos, # 垂直位置
                    label_text, 
                    color='red', 
                    fontsize=11, 
                    rotation=90, # 垂直显示，更节省空间
                    verticalalignment='top',
                    horizontalalignment='right'
                )

        # --- D. 图表美化和显示 ---
        
        # 添加标题和标签
        plt.title('LLM Token Perplexity (PPL) Distribution', fontsize=16)
        plt.xlabel('Perplexity (PPL) Value', fontsize=14)
        plt.ylabel('Frequency (Count)', fontsize=14)
        
        # 添加网格线，方便查看
        plt.grid(axis='y', alpha=0.5, linestyle='--')
        
        # 优化布局，确保所有元素都可见
        plt.tight_layout()
        
        # 显示图表
        plt.show()

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请确保文件路径正确。")
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的 JSON 格式。")
    except Exception as e:
        print(f"发生了一个错误: {e}")

if __name__ == "__main__":
    # 检查是否安装了必要的库
    try:
        import numpy as np
        import argparse
    except ImportError:
        print("错误：缺少必要的库。请运行 'pip install numpy matplotlib' 进行安装。")
        sys.exit(1)
        
    main()