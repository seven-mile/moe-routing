#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [options] [-- <extra vllm bench serve args>]"
    cat <<'EOF'

Options:
    -H, --host HOST          Endpoint host to query models from and pass to vllm bench serve (default: localhost)
    -p, --port PORT          Endpoint port to query models from and pass to vllm bench serve (default: 8000)
    -c, --concurrency N      max_concurrency in vllm bench serve (default: 512)
    -n, --num-prompts N      num_prompts in vllm bench serve (default: c * 2)
    -l, --output-len N       sharegpt_output_len in vllm bench serve (default: 1024)
    -t, --tokenizer NAME     Tokenizer name (default: detected model id)
    -f, --formula JSON_ARRAY Formula list, e.g. '[2.0, 1.1]'
    --constant-k K           Real constant top-k baseline for all generated target tokens
    --naive-lower-layers N   Naive baseline: reduce first N MoE layers unconditionally
    --naive-k K              Naive baseline target k for those lower layers
    --policy-file PATH       Policy Python file path resolved by the server
                             (default: configs/ppl_to_ks.py)
    -D, --dataset-name NAME  Dataset name (default: sharegpt)
    -h, --help               Show this help

Behavior:
    - Auto query /v1/models and use the first model id.
    - If --tokenizer is not set, tokenizer defaults to the detected model id.
    - If --num-prompts is not set, it defaults to concurrency * 2.
    - If --formula is set, dyn-assisted-action-config uses spec_with_list_layer_range.
    - If --constant-k is set, dyn-assisted-action-config uses constant_k.
    - If --naive-lower-layers and --naive-k are set, use an equivalent
      spec_with_list_layer_range config that ignores preview perplexity.
    - If --formula is not set, dyn-assisted-action-config uses baseline.
    - dataset-name=sharegpt resolves dataset-path from local hf cache scan output.

Environment (defaults are applied if unset):
    TRANSFORMERS_OFFLINE=1
EOF
}

die() {
    echo "[bench_serve.sh] ERROR: $*" >&2
    exit 1
}

HOST="localhost"
PORT=8000
CONCURRENCY=512
NUM_PROMPTS=""
OUTPUT_LEN=1024
TOKENIZER=""
FORMULA=""
CONSTANT_K=""
NAIVE_LOWER_LAYERS=""
NAIVE_K=""
POLICY_FILE="${POLICY_FILE:-configs/ppl_to_ks.py}"
DATASET_NAME="sharegpt"
DATASET_PATH=""

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -H|--host)
            [[ $# -ge 2 ]] || die "--host requires a value"
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            [[ $# -ge 2 ]] || die "--port requires a value"
            PORT="$2"
            shift 2
            ;;
        -c|--concurrency)
            [[ $# -ge 2 ]] || die "--concurrency requires a value"
            CONCURRENCY="$2"
            shift 2
            ;;
        -n|--num-prompts)
            [[ $# -ge 2 ]] || die "--num-prompts requires a value"
            NUM_PROMPTS="$2"
            shift 2
            ;;
        -l|--output-len)
            [[ $# -ge 2 ]] || die "--output-len requires a value"
            OUTPUT_LEN="$2"
            shift 2
            ;;
        -t|--tokenizer)
            [[ $# -ge 2 ]] || die "--tokenizer requires a value"
            TOKENIZER="$2"
            shift 2
            ;;
        -f|--formula)
            [[ $# -ge 2 ]] || die "--formula requires a JSON array string"
            FORMULA="$2"
            shift 2
            ;;
        --constant-k)
            [[ $# -ge 2 ]] || die "--constant-k requires a value"
            CONSTANT_K="$2"
            shift 2
            ;;
        --naive-lower-layers)
            [[ $# -ge 2 ]] || die "--naive-lower-layers requires a value"
            NAIVE_LOWER_LAYERS="$2"
            shift 2
            ;;
        --naive-k)
            [[ $# -ge 2 ]] || die "--naive-k requires a value"
            NAIVE_K="$2"
            shift 2
            ;;
        --policy-file)
            [[ $# -ge 2 ]] || die "--policy-file requires a value"
            POLICY_FILE="$2"
            shift 2
            ;;
        -D|--dataset-name)
            [[ $# -ge 2 ]] || die "--dataset-name requires a value"
            DATASET_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ "$PORT" =~ ^[0-9]+$ ]] || die "--port must be an integer"
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || die "--concurrency must be an integer"
[[ "$OUTPUT_LEN" =~ ^[0-9]+$ ]] || die "--output-len must be an integer"
if [[ -n "$NUM_PROMPTS" ]]; then
    [[ "$NUM_PROMPTS" =~ ^[0-9]+$ ]] || die "--num-prompts must be an integer"
fi
if [[ -n "$CONSTANT_K" ]]; then
    [[ -z "$FORMULA" ]] || die "--formula cannot be combined with --constant-k"
    [[ -z "$NAIVE_LOWER_LAYERS" && -z "$NAIVE_K" ]] || die "--constant-k cannot be combined with naive baseline options"
    [[ "$CONSTANT_K" =~ ^[0-9]+$ ]] || die "--constant-k must be an integer"
    [[ "$CONSTANT_K" -le 8 ]] || die "--constant-k currently assumes base k=8 and must be <= 8"
fi
if [[ -n "$NAIVE_LOWER_LAYERS" || -n "$NAIVE_K" ]]; then
    [[ -z "$FORMULA" ]] || die "--formula cannot be combined with naive baseline options"
    [[ "$NAIVE_LOWER_LAYERS" =~ ^[0-9]+$ ]] || die "--naive-lower-layers must be an integer"
    [[ "$NAIVE_K" =~ ^[0-9]+$ ]] || die "--naive-k must be an integer"
    [[ "$NAIVE_K" -le 8 ]] || die "--naive-k currently assumes base k=8 and must be <= 8"
fi

if [[ -z "$NUM_PROMPTS" ]]; then
    NUM_PROMPTS=$((CONCURRENCY * 2))
fi

make_naive_cfg() {
    local k="$1"
    local base_k=8
    local vals=()
    local i
    for ((i = k; i < base_k; i++)); do
        vals+=("1.0e30")
    done
    for ((i = 0; i < k; i++)); do
        vals+=("0.0")
    done
    local joined
    joined="$(IFS=,; echo "${vals[*]}")"
    printf '[%s]' "$joined"
}

resolve_dataset_path() {
    local scan_output
    local line
    local cleaned
    local repo
    local repo_type
    local local_path
    local repo_cache_path=""

    if [[ "$DATASET_NAME" != "sharegpt" ]]; then
        die "Only --dataset-name sharegpt is currently supported"
    fi

    if command -v hf >/dev/null 2>&1; then
        scan_output="$(hf cache scan 2>/dev/null || true)"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        scan_output="$(huggingface-cli scan-cache 2>/dev/null || true)"
    else
        die "Neither hf nor huggingface-cli is available"
    fi

    [[ -n "$scan_output" ]] || die "HF cache scan returned no output"

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        [[ "$line" == REPO* ]] && continue
        [[ "$line" == ---* ]] && continue

        # hf cache scan is a human-readable table. Parse by columns:
        # repo=id (col1), type=dataset/model/space (col2), path=last col.
        cleaned="${line%${line##*[![:space:]]}}"
        repo="$(printf '%s\n' "$cleaned" | awk '{print $1}')"
        repo_type="$(printf '%s\n' "$cleaned" | awk '{print $2}')"
        local_path="$(printf '%s\n' "$cleaned" | awk '{print $NF}')"

        if [[ "$repo_type" == "dataset" && "$repo" == anon8231489123/ShareGPT_Vicuna_unfiltered && -n "$local_path" ]]; then
            repo_cache_path="$local_path"
            break
        fi
    done <<< "$scan_output"

    [[ -n "$repo_cache_path" ]] || die "Could not find ShareGPT_Vicuna_unfiltered in HF cache"
    [[ -d "$repo_cache_path" ]] || die "Resolved HF repo path does not exist: $repo_cache_path"

    SHAREGPT_SNAPSHOT="192ab2185289094fc556ec8ce5ce1e8e587154ca"
    SHAREGPT_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
    DATASET_PATH="$repo_cache_path/snapshots/$SHAREGPT_SNAPSHOT/$SHAREGPT_FILE"
    [[ -f "$DATASET_PATH" ]] || die "Resolved sharegpt file not found: $DATASET_PATH"
}

resolve_dataset_path

BASE_ENDPOINT="http://${HOST}:${PORT}/v1"
MODELS_URL="${BASE_ENDPOINT}/models"

echo "[bench_serve.sh] Querying models from: ${MODELS_URL}"
MODELS_JSON="$(curl --proxy "" -X GET -fsS "${MODELS_URL}")" || die "Failed to query ${MODELS_URL}"

MODEL_NAME="$({
    printf '%s' "$MODELS_JSON" \
        | jq '.data[0].id // empty' \
        | tr -d '"'
})"

if [[ -z "$MODEL_NAME" || "$MODEL_NAME" == "$MODELS_JSON" || "$MODEL_NAME" == " " ]]; then
    die "Could not parse first model id from /v1/models response"
fi

if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER="$MODEL_NAME"
fi

echo "[bench_serve.sh] Detected model: ${MODEL_NAME}"
echo "[bench_serve.sh] Using tokenizer: ${TOKENIZER}"

if [[ -n "$CONSTANT_K" ]]; then
    DYN_ASSISTED_ACTION_CONFIG=$(cat <<EOF
{
    "file": "${POLICY_FILE}",
    "function": "constant_k",
    "args": [
        ${CONSTANT_K}
    ]
}
EOF
)
elif [[ -n "$NAIVE_LOWER_LAYERS" ]]; then
    NAIVE_CFG="$(make_naive_cfg "$NAIVE_K")"
    DYN_ASSISTED_ACTION_CONFIG=$(cat <<EOF
{
    "file": "${POLICY_FILE}",
    "function": "spec_with_list_layer_range",
    "args": [
        ${NAIVE_CFG},
        [${NAIVE_LOWER_LAYERS}, 1000]
    ]
}
EOF
)
elif [[ -n "$FORMULA" ]]; then
    [[ "$FORMULA" =~ ^\[.*\]$ ]] || die "--formula must be a JSON array string, e.g. '[2.0, 1.1]'"
    DYN_ASSISTED_ACTION_CONFIG=$(cat <<EOF
{
    "file": "${POLICY_FILE}",
    "function": "spec_with_list_layer_range",
    "args": [
        ${FORMULA},
        [0, 0]
    ]
}
EOF
)
else
    DYN_ASSISTED_ACTION_CONFIG=$(cat <<EOF
{
    "file": "${POLICY_FILE}",
    "function": "baseline"
}
EOF
)
fi

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[bench_serve.sh] TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "[bench_serve.sh] host=${HOST}, port=${PORT}, concurrency=${CONCURRENCY}, num_prompts=${NUM_PROMPTS}, output_len=${OUTPUT_LEN}"
echo "[bench_serve.sh] dataset_name=${DATASET_NAME}"
echo "[bench_serve.sh] policy_file=${POLICY_FILE}"
if [[ -n "$CONSTANT_K" ]]; then
    echo "[bench_serve.sh] constant_k=${CONSTANT_K}"
fi
if [[ -n "$NAIVE_LOWER_LAYERS" ]]; then
    echo "[bench_serve.sh] naive_lower_layers=${NAIVE_LOWER_LAYERS}, naive_k=${NAIVE_K}, encoded_cfg=${NAIVE_CFG}"
fi
echo "[bench_serve.sh] dataset_path=${DATASET_PATH}"
echo "[bench_serve.sh] dyn_assisted_action_config:"
printf '%s' "$DYN_ASSISTED_ACTION_CONFIG" | jq .

VLLM_CMD=(
    vllm
    bench
    serve
    --backend vllm
    --model "$MODEL_NAME"
    --tokenizer "$TOKENIZER"
    --dataset-name "$DATASET_NAME"
    --dataset-path "$DATASET_PATH"
    --num-prompts "$NUM_PROMPTS"
    --sharegpt-output-len "$OUTPUT_LEN"
    --max-concurrency "$CONCURRENCY"
    --host "$HOST"
    --port "$PORT"
    --dyn-assisted-action-config "$DYN_ASSISTED_ACTION_CONFIG"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[bench_serve.sh] Running vllm bench serve ..."
"${VLLM_CMD[@]}"
