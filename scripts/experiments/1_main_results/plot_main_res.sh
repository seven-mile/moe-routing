#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 RESULTS.csv OUTPUT_PREFIX [LABEL ...]" >&2
    exit 1
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/experiments/1_main_results"
RESULT_CSV=$1
OUTPUT_PREFIX=$2
shift 2

if (( $# > 0 )); then
    LABELS=("$@")
else
    LABELS=(baseline lossless optimum)
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"
python "$SCRIPT_DIR/plot_main_res.py" \
    --file "$RESULT_CSV" \
    --labels "${LABELS[@]}" \
    --out_prefix "$OUTPUT_PREFIX" \
    --legend_file
