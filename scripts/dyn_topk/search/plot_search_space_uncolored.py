import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_scatter(input_file, output_file, show_plot, constraint_threshold=1.40):
    """
    绘制散点图并标记最佳点
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出图像文件路径（可选）
        show_plot: 是否显示图表
        constraint_threshold: 约束阈值（ppl <= threshold）
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在")
        return
    
    # 读取JSON文件
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 文件 '{input_file}' 不是有效的JSON格式")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 提取数据
    avg_ppl = []
    benefit = []
    
    for item in data:
        avg_ppl.append(item['avg_ppl'])
        benefit.append(item['benefit'])
    
    # 转换为numpy数组便于计算
    avg_ppl = np.array(avg_ppl)
    benefit = np.array(benefit)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制所有散点
    plt.scatter(avg_ppl, benefit, alpha=0.5, s=50, color='#617fcc', 
                linewidth=0.5, label='Search points')
    
    # 标记灰色半平面 (ppl <= constraint_threshold)
    plt.axvspan(0, constraint_threshold, alpha=0.2, 
                color='gray', label=f'PPL ≤ {constraint_threshold} region')
    
    # 在约束区域内寻找benefit最小的点
    mask = avg_ppl <= constraint_threshold
    if np.any(mask):
        constrained_ppl = avg_ppl[mask]
        constrained_benefit = benefit[mask]
        
        # 找到benefit最小的点
        min_benefit_idx = np.argmin(constrained_benefit)
        best_ppl = constrained_ppl[min_benefit_idx]
        best_benefit = constrained_benefit[min_benefit_idx]
        
        # 用红色大号标记最佳点
        plt.scatter(best_ppl, best_benefit, color='red', s=200, marker='*', 
                    linewidth=2, zorder=5, 
                    label=f'Best point: PPL={best_ppl:.3f}, Benefit={best_benefit:.3f}')
    
    # 设置坐标轴标签
    plt.xlabel('Average PPL', fontsize=12)
    plt.ylabel('Relative Top-k', fontsize=12)
    
    # 设置标题
    plt.title('Automatic Search Space Distribution', fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(loc='best', frameon=True, framealpha=0.9)
    
    # 添加注释说明搜索目标
    plt.text(0.05, 0.05, 'Search Goal: Lower-left region is better\n(Both lower PPL and lower top-k are desirable)', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 调整坐标轴范围，留出一些边距
    plt.xlim(avg_ppl.min() - 0.001, avg_ppl.max() + 0.001)
    plt.ylim(benefit.min() - 0.001, benefit.max() + 0.001)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表到文件
    if output_file:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取文件扩展名以确定保存格式
        _, ext = os.path.splitext(output_file)
        if not ext:
            output_file += '.png'  # 默认保存为PNG格式
            ext = '.png'
        
        # 支持的图像格式
        supported_formats = ['.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps']
        if ext.lower() not in supported_formats:
            print(f"警告: 不支持的图像格式 '{ext}'，将使用PNG格式")
            output_file = output_file.replace(ext, '.png')
            ext = '.png'
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_file}")
    
    # 显示图表
    if show_plot:
        plt.show()
    
    # 如果没有保存也没有显示，关闭图表释放内存
    if not output_file and not show_plot:
        plt.close()
    
    # 打印最佳点的详细信息
    if np.any(mask):
        print(f"\n最佳点信息 (约束区域 PPL ≤ {constraint_threshold}):")
        print(f"  Average PPL: {best_ppl:.4f}")
        print(f"  Relative Top-k: {best_benefit:.4f}")
        print(f"  搜索空间总点数: {len(data)}")
        print(f"  满足约束的点数: {np.sum(mask)}")

def main():
    parser = argparse.ArgumentParser(
        description='绘制自动搜索空间分布散点图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s -i data.json -o plot.png --show
  %(prog)s -i data.json -o results/search_plot.pdf
  %(prog)s -i data.json --show
  %(prog)s -i data.json -o plot.svg --no-show
  %(prog)s -i data.json -c 1.35 -o plot.png --show
        '''
    )
    
    # 必需参数
    parser.add_argument('-i', '--input', required=True,
                       help='输入JSON文件路径')
    
    # 可选参数
    parser.add_argument('-o', '--output',
                       help='输出图像文件路径（支持格式: png, jpg, pdf, svg, eps）')
    
    parser.add_argument('--show', action='store_true', default=False,
                       help='显示图表（如果指定了输出文件，默认不显示）')
    
    parser.add_argument('-c', '--constraint', type=float, default=1.40,
                       help='约束阈值（默认: 1.40）')
    
    # 如果没有指定输出文件，默认显示图表
    args = parser.parse_args()
    
    # 如果没有指定输出文件且没有明确要求显示，则默认显示
    if not args.output and not args.show:
        args.show = True
    
    # 调用绘图函数
    plot_scatter(args.input, args.output, args.show, args.constraint)

if __name__ == "__main__":
    main()
