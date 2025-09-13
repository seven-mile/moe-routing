import json
import matplotlib.pyplot as plt

def plot_scatter_from_json(file_path):
    """
    Reads data from a JSON file and plots a scatter graph of avg_ppl vs. benefit.

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

    # Extracting x and y coordinates
    avg_ppl_values = [item['avg_ppl'] for item in data]
    benefit_values = [item['benefit'] for item in data]

    # Creating the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_ppl_values, benefit_values, c='b', marker='o', alpha=0.7)

    # Setting plot labels and title
    plt.xlabel('Average PPL (avg_ppl)')
    plt.ylabel('Benefit')
    plt.title('Scatter Plot of Average PPL vs. Benefit')

    # Setting axis limits to be reasonable
    # You can adjust these values for better visualization
    plt.xlim(min(avg_ppl_values) * 0.95, max(avg_ppl_values) * 1.05)
    plt.ylim(min(benefit_values) * 0.95, max(benefit_values) * 1.05)

    # Adding a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Displaying the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# First, save your data into a JSON file, for example, 'data.json'
# [
#     {
#         "formula": "(4.0, 1.12, 1.04, 1.02)",
#         "avg_ppl": 1.390625,
#         "benefit": 0.8257640153169632
#     },
#     {
#         "formula": "(4.0, 1.12, 1.04, 1.04)",
#         "avg_ppl": 1.3984375,
#         "benefit": 0.8221638426184654
#     },
#     {
#         "formula": "(4.0, 1.12, 1.05, 1.02)",
#         "avg_ppl": 1.390625,
#         "benefit": 0.8244383111596107
#     }
# ]

# Uncomment the line below and change 'data.json' to your file name
import sys
assert len(sys.argv) == 2, "Usage: python plot_search.py <data.json>"
plot_scatter_from_json(sys.argv[1])
