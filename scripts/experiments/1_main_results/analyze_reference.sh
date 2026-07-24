#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: analyze_reference.sh [OUTPUT_DIR]

Validate, summarize, and plot the checked-in 30B EP8 reference CSV.
Outputs default to reference/generated/.
EOF
}

if (( $# > 1 )); then
    usage >&2
    exit 1
fi
if (( $# == 1 )) && [[ $1 == -h || $1 == --help ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REFERENCE_CSV="$SCRIPT_DIR/reference/30b_ep8.csv"
OUTPUT_DIR=${1:-"$SCRIPT_DIR/reference/generated"}
PYTHON_BIN=${PYTHON_BIN:-python}
MPLCONFIGDIR=${MPLCONFIGDIR:-"$OUTPUT_DIR/.matplotlib"}
export MPLCONFIGDIR

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" \
    --strict-protocol \
    "$REFERENCE_CSV" \
    --required-modes baseline lossless optimum \
    --expected-prompt-multiplier 4 \
    --output "$OUTPUT_DIR/30b_ep8_summary.md"

"$PYTHON_BIN" "$SCRIPT_DIR/plot_main_res.py" \
    --file "$REFERENCE_CSV" \
    --labels baseline lossless optimum \
    --out_prefix "$OUTPUT_DIR/30b_ep8" \
    --with_legend

echo "Reference summary: $OUTPUT_DIR/30b_ep8_summary.md"
echo "Reference plots: $OUTPUT_DIR/30b_ep8_throughput.{pdf,png}"
