import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import os

score_files = glob.glob('data/lmsys/routing_scores/*.parquet')

num_layers = 48
num_experts = 128
top_k = 8

stats = {}

for f in tqdm(score_files):
    df = pd.read_parquet(f)
    scores = pd.pivot_table(
        df,
        values='routing_score',
        index=['layer_id', 'token_position_in_sequence'],
        columns='expert_id',
        aggfunc='mean'
    )
    # [layers, tokens, experts]
    scores = scores.to_numpy().reshape(num_layers, -1, num_experts)

    # Select Top-k
    top_k_experts = np.argsort(scores, axis=2)[:, :, -top_k:]

    # Count activations
    activation_counts = np.zeros((num_layers, num_experts, num_experts), dtype=int)
    np.add.at(activation_counts, (np.arange(num_layers)[:, None, None, None], top_k_experts[:, :, :, None], top_k_experts[:, :, None, :]), 1)
    stats[os.path.basename(f)] = activation_counts

np.savez(f'data/lmsys/coactivation_counts_top{top_k}.npz', **stats)
