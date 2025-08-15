import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import os

score_files = glob.glob('data/mixtral/routing_scores/*.parquet')

num_layers = 32
num_experts = 8
top_k = 2

top_ks = [top_k]
while top_ks[-1] > 1:
  top_ks.append(top_ks[-1] // 2)

stats = {k:{} for k in top_ks}

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

  for k in top_ks:
    # Select Top-k
    top_k_experts = np.argsort(scores, axis=2)[:, :, -k:]

    # Count activations
    activation_counts = np.zeros((num_layers, num_experts), dtype=int)
    np.add.at(activation_counts, (np.arange(num_layers)[:, None, None], top_k_experts), 1)
    stats[k][os.path.basename(f)] = activation_counts

for k in top_ks:
  np.savez(f'data/mixtral/activation_counts_top{k}.npz', **stats[k])
