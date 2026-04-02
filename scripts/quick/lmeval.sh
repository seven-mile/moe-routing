#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [options] [-- <extra lm_eval args>]"
    cat <<'EOF'

Options:
    --port PORT              Endpoint port (default: 8000)
    --task TASK              Task name passed to lm_eval (default: gsm8k_cot)
    -c, --concurrency N      num_concurrent in model_args (default: 512)
    --formula JSON_ARRAY     Formula list, e.g. '[3.4, 2.1, 1.2]'
    --host HOST              Endpoint host (default: 127.0.0.1)
    --max-retries N          max_retries in model_args (default: 3)
    --show-config            Add --show_config to lm_eval (default: on)
    --no-show-config         Do not pass --show_config
    -h, --help               Show this help

Behavior:
    - Auto query /v1/models and use the first model id.
    - task=ifeval uses local-chat-completions + /chat/completions
    - other tasks use local-completions + /completions
    - If --formula is set: assisted_action.function=spec_with_list_layer_range
        assisted_action.args=[<formula>, [0, 0]]
    - If --formula is not set: assisted_action.function=baseline (no args)

Environment (defaults are applied if unset):
    HF_DATASETS_OFFLINE=1
    HF_ENDPOINT=https://hf-mirror.com
EOF
}

die() {
    echo "[bench.sh] ERROR: $*" >&2
    exit 1
}

PORT=8000
TASK="gsm8k_cot"
CONCURRENCY=512
HOST="127.0.0.1"
MAX_RETRIES=3
FORMULA=""
SHOW_CONFIG=1

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            [[ $# -ge 2 ]] || die "--port requires a value"
            PORT="$2"
            shift 2
            ;;
        --task)
            [[ $# -ge 2 ]] || die "--task requires a value"
            TASK="$2"
            shift 2
            ;;
        -c|--concurrency)
            [[ $# -ge 2 ]] || die "--concurrency requires a value"
            CONCURRENCY="$2"
            shift 2
            ;;
        --host)
            [[ $# -ge 2 ]] || die "--host requires a value"
            HOST="$2"
            shift 2
            ;;
        --max-retries)
            [[ $# -ge 2 ]] || die "--max-retries requires a value"
            MAX_RETRIES="$2"
            shift 2
            ;;
        --formula)
            [[ $# -ge 2 ]] || die "--formula requires a JSON array string"
            FORMULA="$2"
            shift 2
            ;;
        --show-config)
            SHOW_CONFIG=1
            shift
            ;;
        --no-show-config)
            SHOW_CONFIG=0
            shift
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
[[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || die "--max-retries must be an integer"

BASE_ENDPOINT="http://${HOST}:${PORT}/v1"
MODELS_URL="${BASE_ENDPOINT}/models"

echo "[bench.sh] Querying models from: ${MODELS_URL}"
MODELS_JSON="$(curl --proxy "" -X GET -fsS "${MODELS_URL}")" || die "Failed to query ${MODELS_URL}"

# Extract first model id from standard OpenAI schema: {"data": [{"id": "..."}, ...]}
MODEL_NAME="$({
    printf '%s' "$MODELS_JSON" \
        | jq '.data[0].id // empty' \
        | tr -d '"'
})"

if [[ -z "$MODEL_NAME" || "$MODEL_NAME" == "$MODELS_JSON" || "$MODEL_NAME" == " " ]]; then
    die "Could not parse first model id from /v1/models response"
fi

echo "[bench.sh] Detected model: ${MODEL_NAME}"

if [[ "$TASK" == "ifeval" ]]; then
    LM_MODEL="local-chat-completions"
    BASE_URL="${BASE_ENDPOINT}/chat/completions"
    EXTRA_MODEL_FIELDS='"enable_thinking": false,'
    EXTRA_ARGS+=(--apply_chat_template)
else
    LM_MODEL="local-completions"
    BASE_URL="${BASE_ENDPOINT}/completions"
    EXTRA_MODEL_FIELDS=''
fi

if [[ -n "$FORMULA" ]]; then
    [[ "$FORMULA" =~ ^\[.*\]$ ]] || die "--formula must be a JSON array string, e.g. '[3.4, 2.1, 1.0]'"
    ASSISTED_ACTION=$(cat <<EOF
"assisted_action": {
    "file": "configs/ppl_to_ks.py",
    "function": "spec_with_list_layer_range",
    "args": [
        ${FORMULA},
        [0, 0]
    ]
}
EOF
)
else
    ASSISTED_ACTION=$(cat <<'EOF'
"assisted_action": {
    "file": "configs/ppl_to_ks.py",
    "function": "baseline"
}
EOF
)
fi

MODEL_ARGS=$(cat <<EOF
{
    "model": "${MODEL_NAME}",
    "base_url": "${BASE_URL}",
    "num_concurrent": ${CONCURRENCY},
    "max_retries": ${MAX_RETRIES},
    "timeout": -1,
    "tokenized_requests": false,
    "tokenizer_backend": "none",
    ${EXTRA_MODEL_FIELDS}
    ${ASSISTED_ACTION}
}
EOF
)

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[bench.sh] HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "[bench.sh] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[bench.sh] task=${TASK}, lm_model=${LM_MODEL}, base_url=${BASE_URL}, concurrency=${CONCURRENCY}"
# print formatted JSON by JQ
echo "[bench.sh] model_args:"
printf '%s' "$MODEL_ARGS" | jq .

LM_CMD=(
    lm_eval
    --model "$LM_MODEL"
    --model_args "$MODEL_ARGS"
    --tasks "$TASK"
)

if [[ "$SHOW_CONFIG" -eq 1 ]]; then
    LM_CMD+=(--show_config)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    LM_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[bench.sh] Running lm_eval ..."
"${LM_CMD[@]}"
