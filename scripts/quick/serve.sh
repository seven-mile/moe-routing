WORKSPACE_ROOT=/root/dev/repos/moe-routing

cd $WORKSPACE_ROOT

TRANSFORMERS_OFFLINE=1 \
VLLM_DYN_TOPKS_NO_DROP_TOKENS=1 \
vllm serve \
    Qwen/Qwen3-30B-A3B \
    --seed 42 \
    -dp 2 -ep \
    --no-enable-prefix-caching \
    --max-model-len 4096 \
    --gpu_memory_utilization 0.9 \
    --speculative-config '{
      "model": "Qwen/Qwen3-0.6B",
      "method": "draft_model",
      "num_speculative_tokens": 3,
      "disable_padded_drafter_batch": false
    }' \
    --enforce-eager \
    --all2all-backend=allgather_reducescatter \
    --no-async-scheduling
