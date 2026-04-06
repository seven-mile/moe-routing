#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/experiments/4_latency_breakdown"
RESULT_DIR="$ROOT_DIR/data/plot/4_latency_breakdown"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

cd "$SCRIPT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/plot_latency_breakdown.py" \
  --infile "$RESULT_DIR/30b_ep8.csv" \
  --out_png "$RESULT_DIR/30b_ep8.png" \
  --out_pdf "$RESULT_DIR/30b_ep8.pdf" \
  --legend_pdf "$RESULT_DIR/legend.pdf"

"$PYTHON_BIN" "$SCRIPT_DIR/plot_latency_breakdown.py" \
  --infile "$RESULT_DIR/235b_ep32.csv" \
  --out_png "$RESULT_DIR/235b_ep32.png" \
  --out_pdf "$RESULT_DIR/235b_ep32.pdf"