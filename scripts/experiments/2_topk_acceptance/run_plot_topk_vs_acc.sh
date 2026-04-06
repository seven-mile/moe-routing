#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

IN_CSV="$REPO_ROOT/data/plot/2_topk_acceptance/total.csv"
OUT_PNG="$REPO_ROOT/data/plot/2_topk_acceptance/total_topk_vs_acc.png"
OUT_PDF="$REPO_ROOT/data/plot/2_topk_acceptance/total_topk_vs_acc.pdf"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/plot_topk_vs_acc.py" \
    --infile "$IN_CSV" \
    --out_png "$OUT_PNG" \
    --out_pdf "$OUT_PDF"
