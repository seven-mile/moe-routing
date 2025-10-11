import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

def get_token_counts(file_path: str, tokenizer) -> list[int]:
    """
    Reads a JSONL file, parses the 'resps' field, and returns a list of token counts for each response.
    
    Args:
        file_path (str): Path to the JSONL file.
        tokenizer: An initialized Hugging Face Tokenizer.
        
    Returns:
        list[int]: A list containing the token count for each response.
    """
    token_counts = []
    print(f"Reading data and counting tokens from file '{file_path}'...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing lines"):
            try:
                data = json.loads(line)
                if 'resps' in data and data['resps'] and data['resps'][0]:
                    response_text = data['resps'][0][0]
                    num_tokens = len(tokenizer.encode(response_text))
                    token_counts.append(num_tokens)
                else:
                    print(f"Warning: Skipping a line with an invalid or empty 'resps' field: {line.strip()}")
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Error parsing line, skipping. Error: {e}. Line content: {line.strip()}")
                
    return token_counts

def plot_histogram(token_counts: list[int], output_path: str, bins: int = 50):
    """
    Plots and saves a histogram based on a list of token counts.
    
    Args:
        token_counts (list[int]): A list of token counts.
        output_path (str): The file path to save the histogram image.
        bins (int): The number of bins for the histogram.
    """
    if not token_counts:
        print("Error: No data available for plotting.")
        return

    mean_val = np.mean(token_counts)
    median_val = np.median(token_counts)
    std_dev = np.std(token_counts)
    min_val = np.min(token_counts)
    max_val = np.max(token_counts)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    plt.hist(token_counts, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    
    plt.title('Distribution of Response Token Counts', fontsize=16)
    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    
    stats_text = (
        f"Total Samples: {len(token_counts)}\n"
        f"Mean: {mean_val:.2f}\n"
        f"Median: {median_val:.2f}\n"
        f"Std Dev: {std_dev:.2f}\n"
        f"Min: {min_val}\n"
        f"Max: {max_val}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    print(f"\nHistogram successfully saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Reads 'resps' field from a JSONL file and plots a histogram of its token counts.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output_path', type=str, help="Path to the output histogram image.")
    parser.add_argument('--bins', type=int, default=50, help="The number of bins for the histogram.")
    parser.add_argument('--tokenizer_path', type=str, default='Qwen/Qwen3-30B-A3B', help="Hugging Face Hub path for the tokenizer.")
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        print(f"Successfully loaded tokenizer: {args.tokenizer_path}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    token_counts = get_token_counts(args.file_path, tokenizer)
    if not args.output_path:
        args.output_path = args.file_path.rsplit('.', 1)[0] + '_token_count_histogram.png'
    plot_histogram(token_counts, args.output_path, args.bins)

if __name__ == '__main__':
    main()
