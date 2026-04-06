#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/dyn_topk/search"
RESULT_DIR="$ROOT_DIR/data/dyn_topk_search/0_results"

LEGEND_PDF="$RESULT_DIR/pareto_frontier_legend.pdf"

run_case() {
    local infile="$1"
    shift 1

    local stem
    stem=$(basename "$infile" .json)

    args=(
        --infile "$infile"
        "$@"
        --out_json "$RESULT_DIR/${stem}_pareto_frontier.json"
        --out_png "$RESULT_DIR/${stem}.png"
        --out_pdf "$RESULT_DIR/${stem}.pdf"
    )

    python3 "$SCRIPT_DIR/plot_botorch_ng.py" "${args[@]}"
}

run_case "$RESULT_DIR/optimization_results_lmeval_235b_0b6_gsm8k_cot.json" \
    --baseline_score 91.20 \
    --baseline_score_std 0.81 \
    --baseline_topk 8 \
    --legend_pdf "$LEGEND_PDF"

run_case "$RESULT_DIR/optimization_results_lmeval_30b_0b6_gsm8k_cot.json" \
    --baseline_score 90.17 \
    --baseline_score_std 0.81 \
    --baseline_topk 8 \

run_case "$RESULT_DIR/optimization_results_lmeval_30b_0b6_gsm8k_cot_neo.json" \
    --baseline_score 90.17 \
    --baseline_score_std 0.81 \
    --baseline_topk 8 \

run_case "$RESULT_DIR/optimization_results_lmeval_30b_0b6_ifeval.json" \
    --baseline_score 82.81 \
    --baseline_score_std 1.66 \
    --baseline_topk 8

run_case "$RESULT_DIR/optimization_results_lmeval_30b_2507_eagle3_gsm8k_cot.json" \
    --baseline_score 87.41 \
    --baseline_score_std 0.92 \
    --baseline_topk 8 \

run_case "$RESULT_DIR/optimization_results_lmeval_30b_2507_eagle3_ifeval.json" \
    --baseline_score 82.38 \
    --baseline_score_std 1.66 \
    --baseline_topk 8

run_case "$RESULT_DIR/optimization_results_lmeval_30b_eagle3_gsm8k_cot.json" \
    --baseline_score 90.17 \
    --baseline_score_std 0.81 \
    --baseline_topk 8

run_case "$RESULT_DIR/optimization_results_lmeval_30b_eagle3_ifeval.json" \
    --baseline_score 82.81 \
    --baseline_score_std 1.66 \
    --baseline_topk 8

run_case "$RESULT_DIR/optimization_results_lmeval_glm47flash_mtp_gsm8k_cot.json" \
    --baseline_score 84.53 \
    --baseline_score_std 1.00 \
    --baseline_topk 4

# run_case "$RESULT_DIR/optimization_results_lmeval_r1_mtp_gsm8k_cot_8d_apply_last.json" \
#     --baseline_score 91.74 \
#     --baseline_topk 8

run_case "$RESULT_DIR/optimization_results_lmeval_r1_mtp_gsm8k_cot_5d.json" \
    --baseline_score 92.34 \
    --baseline_score_std 0.57 \
    --baseline_topk 8
