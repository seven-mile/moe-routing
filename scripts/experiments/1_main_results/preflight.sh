#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: preflight.sh [--expected-commit REVISION]

Run source validation, command-generation checks, and parser regression tests
without downloading inputs, creating a Python environment, or using a GPU.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
EXPECTED_COMMIT=

while (( $# > 0 )); do
    case "$1" in
        --expected-commit) EXPECTED_COMMIT=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[preflight.sh] ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || { echo "[preflight.sh] ERROR: $PYTHON_BIN is unavailable" >&2; exit 1; }

TMP_ROOT=$(mktemp -d /tmp/spec-k-preflight.XXXXXX)
trap 'rm -rf -- "$TMP_ROOT"' EXIT

SOURCE_ARGS=(--source-only)
if [[ -n $EXPECTED_COMMIT ]]; then
    SOURCE_ARGS+=(--expected-commit "$EXPECTED_COMMIT")
fi
"$SCRIPT_DIR/check_env.sh" "${SOURCE_ARGS[@]}"

"$SCRIPT_DIR/prepare_nvshmem.sh" \
    --prefix "$TMP_ROOT/nvshmem-3.4.5" --dry-run
NVSHMEM_DIR="$TMP_ROOT/nvshmem-3.4.5" \
    "$SCRIPT_DIR/bootstrap_env.sh" \
    --venv "$TMP_ROOT/venv" --dry-run
"$SCRIPT_DIR/prepare_inputs.sh" \
    --root "$TMP_ROOT/inputs" --with-models --dry-run

PYTHON_BIN="$PYTHON_BIN" "$SCRIPT_DIR/serve.sh" \
    --model "$TMP_ROOT/inputs/models/Qwen3-30B-A3B" \
    --draft-model "$TMP_ROOT/inputs/models/Qwen3-0.6B" \
    --nnodes 1 --node-rank 0 --gpus-per-node 4 --dry-run

PYTHON_BIN="$PYTHON_BIN" "$SCRIPT_DIR/bench.sh" \
    --profile smoke \
    --dataset-path \
    "$TMP_ROOT/inputs/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json" \
    --output-dir "$TMP_ROOT/bench-output" \
    --output-len 64 --strict-protocol --plot --dry-run

"$PYTHON_BIN" "$SCRIPT_DIR/test_analysis.py"
echo "[preflight.sh] PASS"
