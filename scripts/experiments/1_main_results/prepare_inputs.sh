#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: prepare_inputs.sh [options]

Options:
  --root PATH       Destination root (default: .ae-inputs).
  --with-models     Also download the pinned target and draft model snapshots.
  --dry-run         Print commands without downloading or creating directories.
  -h, --help        Show this help.

The dataset is always downloaded and verified. Model downloads are optional
because they are large and may already be staged in the evaluation image.
EOF
}

die() {
    echo "[prepare_inputs.sh] ERROR: $*" >&2
    exit 1
}

print_command() {
    printf '[prepare_inputs.sh] command:'
    printf ' %q' "$@"
    printf '\n'
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUTPUT_ROOT="$ROOT_DIR/.ae-inputs"
WITH_MODELS=false
DRY_RUN=false

DATASET_REPO="anon8231489123/ShareGPT_Vicuna_unfiltered"
DATASET_REVISION="192ab2185289094fc556ec8ce5ce1e8e587154ca"
DATASET_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_SHA256="35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
TARGET_REPO="Qwen/Qwen3-30B-A3B"
TARGET_REVISION="4c446470ba0aec43e22ac1128f9ffd915f338ba3"
DRAFT_REPO="Qwen/Qwen3-0.6B"
DRAFT_REVISION="e6de91484c29aa9480d55605af694f39b081c455"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) OUTPUT_ROOT=${2:?}; shift 2 ;;
        --with-models) WITH_MODELS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

if [[ $DRY_RUN == false ]]; then
    command -v hf >/dev/null 2>&1 \
        || die "hf CLI is unavailable; activate the artifact environment"
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is unavailable"
fi

DATASET_DIR="$OUTPUT_ROOT/datasets/sharegpt"
DATASET_PATH="$DATASET_DIR/$DATASET_FILE"
TARGET_DIR="$OUTPUT_ROOT/models/Qwen3-30B-A3B"
DRAFT_DIR="$OUTPUT_ROOT/models/Qwen3-0.6B"

DATASET_CMD=(
    hf download "$DATASET_REPO" "$DATASET_FILE"
    --repo-type dataset
    --revision "$DATASET_REVISION"
    --local-dir "$DATASET_DIR"
)

print_command "${DATASET_CMD[@]}"
if [[ $DRY_RUN == false ]]; then
    mkdir -p "$DATASET_DIR"
    "${DATASET_CMD[@]}"
    [[ -f $DATASET_PATH ]] || die "dataset download did not create $DATASET_PATH"
    actual_sha256=$(sha256sum "$DATASET_PATH" | awk '{print $1}')
    [[ $actual_sha256 == "$DATASET_SHA256" ]] \
        || die "dataset SHA256 mismatch: expected $DATASET_SHA256, found $actual_sha256"
    echo "[prepare_inputs.sh] PASS: dataset_sha256=$actual_sha256"
fi

if [[ $WITH_MODELS == true ]]; then
    TARGET_CMD=(
        hf download "$TARGET_REPO"
        --revision "$TARGET_REVISION"
        --local-dir "$TARGET_DIR"
    )
    DRAFT_CMD=(
        hf download "$DRAFT_REPO"
        --revision "$DRAFT_REVISION"
        --local-dir "$DRAFT_DIR"
    )
    print_command "${TARGET_CMD[@]}"
    print_command "${DRAFT_CMD[@]}"
    if [[ $DRY_RUN == false ]]; then
        mkdir -p "$TARGET_DIR" "$DRAFT_DIR"
        "${TARGET_CMD[@]}"
        "${DRAFT_CMD[@]}"
    fi
fi

echo "[prepare_inputs.sh] dataset=$DATASET_PATH"
if [[ $WITH_MODELS == true ]]; then
    echo "[prepare_inputs.sh] target_model=$TARGET_DIR"
    echo "[prepare_inputs.sh] draft_model=$DRAFT_DIR"
fi
