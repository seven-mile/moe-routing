#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bench.sh --profile PROFILE --dataset-path PATH [options]

Profiles:
  smoke  Functional check: concurrency 8, 2x prompts, all three policies.
  ep4    Single-node extrapolation: concurrency 32..512, 8x prompts.
  ep8    Paper protocol: concurrency 64..1024, 4x prompts.
  ep16   Paper protocol: concurrency 128..4096, 2x prompts.
  ep32   Paper protocol: concurrency 256..4096, 1x prompts.

Options:
  --host HOST                 Server host (default: 127.0.0.1).
  --port PORT                 Server port (default: 8000).
  --model MODEL               Served model ID; otherwise query /v1/models.
  --tokenizer TOKENIZER       Tokenizer path/ID (default: served model).
  --dataset-path PATH         ShareGPT JSON file.
  --dataset-sha256 SHA256     Expected dataset digest (default: pinned ShareGPT).
  --skip-dataset-hash         Do not validate the dataset digest.
  --profile PROFILE           One of the profiles above.
  --modes CSV                 Override modes (default: profile-dependent).
  --concurrencies CSV         Override concurrency values.
  --prompt-multiplier N       Override num-prompts/concurrency ratio.
  --output-len N              Requested output length (default: 1024).
  --output-dir PATH           Log/result directory.
  --ready-timeout N           Endpoint wait timeout in seconds (default: 600).
  --cooldown-seconds N        Delay between runs (default: 2).
  --plot                       Generate throughput PDF and PNG.
  --strict-protocol           Fail on unequal counts or failed requests.
  --dry-run                   Print commands without contacting the server.
  -h, --help                  Show this help.
EOF
}

die() {
    echo "[bench.sh] ERROR: $*" >&2
    exit 1
}

print_command() {
    printf '[bench.sh] command:'
    printf ' %q' "$@"
    printf '\n'
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/experiments/1_main_results"
HOST=${VLLM_HOST:-127.0.0.1}
PORT=${VLLM_PORT:-8000}
MODEL_NAME=""
TOKENIZER=""
DATASET_PATH=${SHAREGPT_PATH:-}
DATASET_SHA256=${SHAREGPT_SHA256:-35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4}
CHECK_DATASET_HASH=true
PROFILE=""
MODES_CSV=""
CONCURRENCIES_CSV=""
PROMPT_MULTIPLIER=""
OUTPUT_LEN=1024
OUTPUT_DIR=""
READY_TIMEOUT=600
COOLDOWN_SECONDS=2
PLOT=false
STRICT_PROTOCOL=false
DRY_RUN=false
PYTHON_BIN=${PYTHON_BIN:-python}
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST=${2:?}; shift 2 ;;
        --port) PORT=${2:?}; shift 2 ;;
        --model) MODEL_NAME=${2:?}; shift 2 ;;
        --tokenizer) TOKENIZER=${2:?}; shift 2 ;;
        --dataset-path) DATASET_PATH=${2:?}; shift 2 ;;
        --dataset-sha256) DATASET_SHA256=${2:?}; shift 2 ;;
        --skip-dataset-hash) CHECK_DATASET_HASH=false; shift ;;
        --profile) PROFILE=${2:?}; shift 2 ;;
        --modes) MODES_CSV=${2:?}; shift 2 ;;
        --concurrencies) CONCURRENCIES_CSV=${2:?}; shift 2 ;;
        --prompt-multiplier) PROMPT_MULTIPLIER=${2:?}; shift 2 ;;
        --output-len) OUTPUT_LEN=${2:?}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
        --ready-timeout) READY_TIMEOUT=${2:?}; shift 2 ;;
        --cooldown-seconds) COOLDOWN_SECONDS=${2:?}; shift 2 ;;
        --plot) PLOT=true; shift ;;
        --strict-protocol) STRICT_PROTOCOL=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n $PROFILE ]] || die "--profile is required"
case "$PROFILE" in
    smoke)
        DEFAULT_MODES="baseline,lossless,optimum"
        DEFAULT_CONCURRENCIES="8"
        DEFAULT_MULTIPLIER=2
        ;;
    ep4)
        DEFAULT_MODES="baseline,lossless,optimum"
        DEFAULT_CONCURRENCIES="32,64,128,256,512"
        DEFAULT_MULTIPLIER=8
        ;;
    ep8)
        DEFAULT_MODES="baseline,lossless,optimum"
        DEFAULT_CONCURRENCIES="64,128,256,512,1024"
        DEFAULT_MULTIPLIER=4
        ;;
    ep16)
        DEFAULT_MODES="baseline,lossless,optimum"
        DEFAULT_CONCURRENCIES="128,256,512,1024,2048,4096"
        DEFAULT_MULTIPLIER=2
        ;;
    ep32)
        DEFAULT_MODES="baseline,lossless,optimum"
        DEFAULT_CONCURRENCIES="256,512,1024,2048,4096"
        DEFAULT_MULTIPLIER=1
        ;;
    *) die "unknown profile: $PROFILE" ;;
esac

MODES_CSV=${MODES_CSV:-$DEFAULT_MODES}
CONCURRENCIES_CSV=${CONCURRENCIES_CSV:-$DEFAULT_CONCURRENCIES}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-$DEFAULT_MULTIPLIER}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/logs/main_results/$(date -u +%Y%m%dT%H%M%SZ)_$PROFILE"}

for value_name in PORT PROMPT_MULTIPLIER OUTPUT_LEN READY_TIMEOUT COOLDOWN_SECONDS; do
    value=${!value_name}
    [[ $value =~ ^[0-9]+$ ]] || die "$value_name must be a non-negative integer"
done
(( PROMPT_MULTIPLIER > 0 )) || die "PROMPT_MULTIPLIER must be greater than zero"
[[ -n $DATASET_PATH ]] || die "--dataset-path or SHAREGPT_PATH is required"
if [[ $DRY_RUN == false ]]; then
    [[ -f $DATASET_PATH ]] || die "dataset not found: $DATASET_PATH"
    if [[ $CHECK_DATASET_HASH == true ]]; then
        [[ $DATASET_SHA256 =~ ^[0-9a-f]{64}$ ]] \
            || die "dataset SHA256 must be 64 lowercase hex characters"
        actual_dataset_sha256=$(sha256sum "$DATASET_PATH" | awk '{print $1}')
        [[ $actual_dataset_sha256 == "$DATASET_SHA256" ]] \
            || die "dataset SHA256 mismatch: expected $DATASET_SHA256, found $actual_dataset_sha256"
    fi
fi

IFS=',' read -r -a MODES <<< "$MODES_CSV"
IFS=',' read -r -a CONCURRENCIES <<< "$CONCURRENCIES_CSV"
for mode in "${MODES[@]}"; do
    case "$mode" in baseline|lossless|optimum) ;; *) die "unknown mode: $mode" ;; esac
done
for concurrency in "${CONCURRENCIES[@]}"; do
    [[ $concurrency =~ ^[0-9]+$ ]] || die "invalid concurrency: $concurrency"
    (( concurrency > 0 )) || die "concurrency must be greater than zero"
done

BASE_ENDPOINT="http://$HOST:$PORT/v1"
MODELS_URL="$BASE_ENDPOINT/models"

if [[ $DRY_RUN == false ]]; then
    deadline=$((SECONDS + READY_TIMEOUT))
    while ! MODELS_JSON=$(curl --noproxy '*' -fsS "$MODELS_URL" 2>/dev/null); do
        (( SECONDS < deadline )) || die "server did not become ready: $MODELS_URL"
        sleep 2
    done
    if [[ -z $MODEL_NAME ]]; then
        MODEL_NAME=$(
            "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' \
                <<< "$MODELS_JSON"
        )
    fi
else
    MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-30B-A3B}
fi
TOKENIZER=${TOKENIZER:-$MODEL_NAME}

echo "[bench.sh] profile=$PROFILE modes=$MODES_CSV"
echo "[bench.sh] concurrencies=$CONCURRENCIES_CSV prompt_multiplier=$PROMPT_MULTIPLIER"
echo "[bench.sh] model=$MODEL_NAME tokenizer=$TOKENIZER"
echo "[bench.sh] dataset=$DATASET_PATH"
if [[ $CHECK_DATASET_HASH == true ]]; then
    echo "[bench.sh] dataset_sha256=$DATASET_SHA256"
else
    echo "[bench.sh] dataset_sha256=unchecked"
fi
echo "[bench.sh] output_dir=$OUTPUT_DIR"

if [[ $DRY_RUN == false ]]; then
    mkdir -p "$OUTPUT_DIR"
    {
        printf 'repository_commit=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
        printf 'profile=%s\n' "$PROFILE"
        printf 'modes=%s\n' "$MODES_CSV"
        printf 'concurrencies=%s\n' "$CONCURRENCIES_CSV"
        printf 'prompt_multiplier=%s\n' "$PROMPT_MULTIPLIER"
        printf 'output_len=%s\n' "$OUTPUT_LEN"
        printf 'model=%s\n' "$MODEL_NAME"
        printf 'tokenizer=%s\n' "$TOKENIZER"
        printf 'dataset_path=%s\n' "$DATASET_PATH"
        printf 'dataset_sha256=%s\n' "$DATASET_SHA256"
    } > "$OUTPUT_DIR/run.env"
fi

LOG_FILES=()
for mode in "${MODES[@]}"; do
    DYN_CONFIG=$(
        "$PYTHON_BIN" "$SCRIPT_DIR/policy_config.py" \
            "$mode" "$ROOT_DIR/configs/ppl_to_ks.py"
    )
    for concurrency in "${CONCURRENCIES[@]}"; do
        num_prompts=$((concurrency * PROMPT_MULTIPLIER))
        log_file="$OUTPUT_DIR/${mode}_c${concurrency}.log"
        CMD=(
            vllm bench serve
            --backend vllm
            --model "$MODEL_NAME"
            --tokenizer "$TOKENIZER"
            --dataset-name sharegpt
            --dataset-path "$DATASET_PATH"
            --num-prompts "$num_prompts"
            --max-concurrency "$concurrency"
            --sharegpt-output-len "$OUTPUT_LEN"
            --host "$HOST"
            --port "$PORT"
            --seed 42
            --dyn-assisted-action-config "$DYN_CONFIG"
        )
        CMD+=("${EXTRA_ARGS[@]}")

        echo "[bench.sh] mode=$mode concurrency=$concurrency num_prompts=$num_prompts"
        print_command "${CMD[@]}"
        if [[ $DRY_RUN == true ]]; then
            continue
        fi

        "${CMD[@]}" 2>&1 | tee "$log_file"
        LOG_FILES+=("$log_file")
        if (( COOLDOWN_SECONDS > 0 )); then
            sleep "$COOLDOWN_SECONDS"
        fi
    done
done

if [[ $DRY_RUN == true ]]; then
    exit 0
fi

RESULT_CSV="$OUTPUT_DIR/results.csv"
"$PYTHON_BIN" "$SCRIPT_DIR/parse_results_bench.py" --fail-on-error \
    --output "$RESULT_CSV" "${LOG_FILES[@]}"

SUMMARY_ARGS=(
    "$RESULT_CSV"
    --output "$OUTPUT_DIR/summary.md"
    --required-modes "${MODES[@]}"
    --expected-prompt-multiplier "$PROMPT_MULTIPLIER"
)
if [[ $STRICT_PROTOCOL == true ]]; then
    SUMMARY_ARGS+=(--strict-protocol)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" "${SUMMARY_ARGS[@]}"

if [[ $PLOT == true ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/plot_main_res.py" \
        --file "$RESULT_CSV" \
        --labels "${MODES[@]}" \
        --out_prefix "$OUTPUT_DIR/$PROFILE" \
        --with_legend
fi

echo "[bench.sh] results=$RESULT_CSV"
echo "[bench.sh] summary=$OUTPUT_DIR/summary.md"
