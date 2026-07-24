#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: serve.sh [options] [-- <extra vllm serve arguments>]

Required for multi-node runs:
  --nnodes N                 Number of nodes.
  --node-rank N              Zero-based node rank.
  --gpus-per-node N          Number of local GPUs/ranks.
  --master-addr ADDR         Private address of node rank 0.

Model and server options:
  --model MODEL              Target model path or Hugging Face ID.
  --draft-model MODEL        Draft model path or Hugging Face ID.
  --model-revision REV       Target Hugging Face revision.
  --draft-revision REV       Draft Hugging Face revision.
  --rpc-port PORT            Data-parallel RPC port (default: 13345).
  --host HOST                API bind address on rank 0 (default: 127.0.0.1).
  --port PORT                API port on rank 0 (default: 8000).
  --gpu-memory-utilization F Fraction of GPU memory for vLLM (default: 0.9).
  --dry-run                  Print the command without starting vLLM.
  -h, --help                 Show this help.

Environment:
  TARGET_MODEL, DRAFT_MODEL, TARGET_REVISION, DRAFT_REVISION,
  NNODES, NODE_RANK, GPUS_PER_NODE,
  MASTER_ADDR, DP_RPC_PORT, VLLM_HOST, VLLM_PORT, PYTHON_BIN.
EOF
}

die() {
    echo "[serve.sh] ERROR: $*" >&2
    exit 1
}

print_command() {
    printf '[serve.sh] command:'
    printf ' %q' "$@"
    printf '\n'
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-30B-A3B}
DRAFT_MODEL=${DRAFT_MODEL:-Qwen/Qwen3-0.6B}
TARGET_REVISION=${TARGET_REVISION:-4c446470ba0aec43e22ac1128f9ffd915f338ba3}
DRAFT_REVISION=${DRAFT_REVISION:-e6de91484c29aa9480d55605af694f39b081c455}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-}
DP_RPC_PORT=${DP_RPC_PORT:-13345}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
PYTHON_BIN=${PYTHON_BIN:-python}
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) TARGET_MODEL=${2:?}; shift 2 ;;
        --draft-model) DRAFT_MODEL=${2:?}; shift 2 ;;
        --model-revision) TARGET_REVISION=${2:?}; shift 2 ;;
        --draft-revision) DRAFT_REVISION=${2:?}; shift 2 ;;
        --nnodes) NNODES=${2:?}; shift 2 ;;
        --node-rank) NODE_RANK=${2:?}; shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE=${2:?}; shift 2 ;;
        --master-addr) MASTER_ADDR=${2:?}; shift 2 ;;
        --rpc-port) DP_RPC_PORT=${2:?}; shift 2 ;;
        --host) VLLM_HOST=${2:?}; shift 2 ;;
        --port) VLLM_PORT=${2:?}; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION=${2:?}; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "unknown option: $1" ;;
    esac
done

for value_name in NNODES NODE_RANK GPUS_PER_NODE DP_RPC_PORT VLLM_PORT; do
    value=${!value_name}
    [[ $value =~ ^[0-9]+$ ]] || die "$value_name must be a non-negative integer"
done
(( NNODES > 0 )) || die "NNODES must be greater than zero"
(( GPUS_PER_NODE > 0 )) || die "GPUS_PER_NODE must be greater than zero"
(( NODE_RANK < NNODES )) || die "NODE_RANK must be smaller than NNODES"
if [[ -z $MASTER_ADDR ]]; then
    if (( NNODES == 1 )); then
        MASTER_ADDR=127.0.0.1
    else
        die "--master-addr is required when NNODES is greater than one"
    fi
fi

DP_SIZE=$((NNODES * GPUS_PER_NODE))
DP_START_RANK=$((NODE_RANK * GPUS_PER_NODE))

model_is_local() {
    [[ $1 == /* || $1 == ./* || $1 == ../* ]]
}

TARGET_REMOTE_REVISION=$TARGET_REVISION
DRAFT_REMOTE_REVISION=$DRAFT_REVISION
if model_is_local "$TARGET_MODEL"; then
    TARGET_REMOTE_REVISION=""
fi
if model_is_local "$DRAFT_MODEL"; then
    DRAFT_REMOTE_REVISION=""
fi

SPECULATIVE_CONFIG=$(
    "$PYTHON_BIN" - "$DRAFT_MODEL" "$DRAFT_REMOTE_REVISION" <<'PY'
import json
import sys

config = {
    "model": sys.argv[1],
    "method": "draft_model",
    "num_speculative_tokens": 3,
    "disable_padded_drafter_batch": False,
}
if sys.argv[2]:
    config["revision"] = sys.argv[2]
print(json.dumps(config, separators=(",", ":")))
PY
)

CMD=(
    vllm serve "$TARGET_MODEL"
    --seed 42
    --enable-expert-parallel
    --disable-log-requests
    --no-enable-prefix-caching
    --max-model-len 4096
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --all2all-backend deepep_low_latency
    --speculative-config "$SPECULATIVE_CONFIG"
    --enforce-eager
    --no-async-scheduling
    --data-parallel-size "$DP_SIZE"
    --data-parallel-size-local "$GPUS_PER_NODE"
    --data-parallel-start-rank "$DP_START_RANK"
    --data-parallel-address "$MASTER_ADDR"
    --data-parallel-rpc-port "$DP_RPC_PORT"
)

if [[ -n $TARGET_REMOTE_REVISION ]]; then
    CMD+=(
        --revision "$TARGET_REMOTE_REVISION"
        --tokenizer-revision "$TARGET_REMOTE_REVISION"
    )
fi

if (( NODE_RANK > 0 )); then
    CMD+=(--headless)
else
    CMD+=(--host "$VLLM_HOST" --port "$VLLM_PORT")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "[serve.sh] repo=$ROOT_DIR"
echo "[serve.sh] nnodes=$NNODES node_rank=$NODE_RANK gpus_per_node=$GPUS_PER_NODE"
echo "[serve.sh] ep_size=$DP_SIZE dp_start_rank=$DP_START_RANK master=$MASTER_ADDR:$DP_RPC_PORT"
echo "[serve.sh] target_model=$TARGET_MODEL"
echo "[serve.sh] draft_model=$DRAFT_MODEL"
echo "[serve.sh] target_revision=${TARGET_REMOTE_REVISION:-local-path}"
echo "[serve.sh] draft_revision=${DRAFT_REMOTE_REVISION:-local-path}"
print_command "${CMD[@]}"

if [[ $DRY_RUN == true ]]; then
    exit 0
fi

cd "$ROOT_DIR"
exec "${CMD[@]}"
