import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys

def plot_multidimensional_analysis(file_path, output_path=None, apply_jitter=False):
    """
    Generates scatter plots to analyze the effect of four formula dimensions on metrics.

    Each plot represents one dimension and uses a color gradient to show its values.
    
    Args:
        file_path (str): Path to the input JSON file.
        output_path (str, optional): Path to save the plot image. If not specified,
                                     the plot is displayed in a window.
        apply_jitter (bool): If True, applies a small x-axis offset to points
                             with different dimension values to reduce overlap.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        sys.exit(1)

    avg_ppl_values = np.array([item['avg_ppl'] for item in data])
    benefit_values = np.array([item['benefit'] for item in data])
    formula_values = np.array([
        tuple(map(float, item['formula'][1:-1].split(', '))) for item in data
    ])

    dimension_names = ["Dimension 1", "Dimension 2", "Dimension 3", "Dimension 4"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        dim_values = formula_values[:, i]

        # Get unique values and their inverse indices for vectorized coloring
        unique_vals, dim_indices = np.unique(dim_values, return_inverse=True)
        num_unique_vals = len(unique_vals)
        colors = cm.viridis(np.linspace(0, 1, num_unique_vals))
        point_colors = colors[dim_indices]
        
        # Determine x-axis values
        x_values = avg_ppl_values
        if apply_jitter:
            # Jitter based on the unique dimension value index
            jitter_offset = (dim_indices - np.median(np.unique(dim_indices))) * 0.0005
            x_values = avg_ppl_values + jitter_offset

        # Vectorized plotting with smaller points
        ax.scatter(x_values, benefit_values,
                   c=point_colors,
                   s=25, alpha=0.8)
        
        ax.set_xlabel('Average PPL (avg_ppl)')
        ax.set_ylabel('Benefit')
        ax.set_title(f'Effect of {dimension_names[i]} on Metrics')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Create a custom legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[j], markersize=10)
                   for j in range(num_unique_vals)]
        labels = [f'{val}' for val in unique_vals]
        
        ax.legend(handles, labels, title=dimension_names[i], loc='best')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to '{output_path}'")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multidimensional analysis plots.")
    parser.add_argument("input_json", type=str,
                        help="Path to the input JSON file containing the data.")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to save the output image file (e.g., plot.png).")
    parser.add_argument("-j", "--jitter", action="store_true",
                        help="Apply a small x-axis jitter to reduce point overlap.")
    
    args = parser.parse_args()
    
    plot_multidimensional_analysis(args.input_json, args.output, args.jitter)
