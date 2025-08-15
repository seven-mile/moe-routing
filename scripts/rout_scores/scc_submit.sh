#!/bin/bash
#SBATCH -p g078t
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=0-03:20:00
#SBATCH --comment=bupthpc
#SBATCH --output=./logs/main.log

export HF_ENDPOINT=https://hf-mirror.com

module load gcc/9.3.0
module load nvidia/cuda/12.4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python profile_deepseek.py
