import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_multidimensional_analysis(file_path):
    """
    Reads data from a JSON file and generates four scatter plots
    to analyze the effect of each of the four formula dimensions
    on avg_ppl and benefit.

    Each plot represents one dimension and uses a color gradient
    to show the different values of that dimension.

    Args:
        file_path (str): The path to the JSON file containing the data.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    # Extracting and parsing data
    avg_ppl_values = [item['avg_ppl'] for item in data]
    benefit_values = [item['benefit'] for item in data]
    # Assuming formula is a string like "(4.0, 1.12, 1.04, 1.02)"
    formula_values = [
        tuple(map(float, item['formula'][1:-1].split(', '))) for item in data
    ]

    # There are four dimensions to analyze
    dimension_names = ["Dimension 1", "Dimension 2", "Dimension 3", "Dimension 4"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

    # Get a list of all unique values for each dimension to set up color mapping
    dimension_unique_values = [
        sorted(list(set(f[i] for f in formula_values))) for i in range(4)
    ]
    
    for i in range(4):
        ax = axes[i]
        dim_values = [f[i] for f in formula_values]
        
        # Create a color map for the unique values of the current dimension
        unique_vals = dimension_unique_values[i]
        num_unique_vals = len(unique_vals)
        colors = cm.viridis(np.linspace(0, 1, num_unique_vals))
        color_map = {val: colors[j] for j, val in enumerate(unique_vals)}

        # Plot each point with its corresponding color
        for j in range(len(data)):
            ax.scatter(avg_ppl_values[j], benefit_values[j],
                       c=[color_map[dim_values[j]]],
                       s=50, alpha=0.8,
                       label=f'{dimension_names[i]}={dim_values[j]}' if j==0 else "")
            
        # Add labels and title for the subplot
        ax.set_xlabel('Average PPL (avg_ppl)')
        ax.set_ylabel('Benefit')
        ax.set_title(f'Effect of {dimension_names[i]} on Metrics')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Create a custom legend using a proxy artist for each unique color
        handles, labels = [], []
        for val in unique_vals:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color_map[val],
                                      markersize=10))
            labels.append(f'{val}')
        
        ax.legend(handles, labels, title=dimension_names[i], loc='best')


    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming your JSON data is saved in a file named 'data.json'
plot_multidimensional_analysis('search.json')