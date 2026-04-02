#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: stat_pareto.sh -i RESULTS_JSON -o OUTPUT_DIR [-- EXTRA_BENCH_ARGS...]

Required:
  -i  Input optimization json for calc_pareto.py
  -o  Output directory for per-case logs

Behavior:
  - Run baseline once
  - Run one case for each formula emitted by calc_pareto.py
  - Each case runs:
      bash scripts/quick/bench_serve.sh -c 512 -n 512 -l 256 [ -f FORMULA ]
    - EXTRA_BENCH_ARGS are appended to every bench_serve.sh invocation
  - Save each case log to a separate file under OUTPUT_DIR

Examples:
    stat_pareto.sh -i results.json -o out
    stat_pareto.sh -i results.json -o out -- -D sharegpt --percentile-metrics ttft,tpot,itl
EOF
}

INFILE=""
OUTDIR=""

EXTRA_ARGS=()

while getopts ":i:o:h" opt; do
    case "$opt" in
        i) INFILE="$OPTARG" ;;
        o) OUTDIR="$OPTARG" ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "Missing value for -$OPTARG" >&2
            usage
            exit 2
            ;;
        \?)
            echo "Unknown option: -$OPTARG" >&2
            usage
            exit 2
            ;;
    esac
done

shift $((OPTIND - 1))
EXTRA_ARGS=("$@")

if [[ -z "$INFILE" || -z "$OUTDIR" ]]; then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CALC_PARETO="$SCRIPT_DIR/calc_pareto.py"
BENCH_SCRIPT="$REPO_ROOT/scripts/quick/bench_serve.sh"

if [[ ! -f "$CALC_PARETO" ]]; then
    echo "calc_pareto.py not found: $CALC_PARETO" >&2
    exit 1
fi
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "bench_serve.sh not found: $BENCH_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$INFILE" ]]; then
    echo "Input json not found: $INFILE" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

mapfile -t FORMULAS < <(python3 "$CALC_PARETO" -i "$INFILE")

run_case() {
    local case_name="$1"
    local log_file="$2"
    shift 2

    echo "[stat_pareto] running ${case_name}"
    (
        cd "$REPO_ROOT"
        echo "[case] ${case_name}"
        echo "[cmd] bash scripts/quick/bench_serve.sh $*"
        bash "$BENCH_SCRIPT" "$@"
    ) 2>&1 | tee "$log_file"
}

run_case "baseline" "$OUTDIR/baseline.log" -c 512 -n 512 -l 256 "${EXTRA_ARGS[@]}"

idx=1
for formula in "${FORMULAS[@]}"; do
    # remove by tr -d '()'
    formula="$(echo "$formula" | tr -d '()')"
    if [[ -z "${formula// }" ]]; then
        continue
    fi

    log_file="$OUTDIR/formula_$(printf "%03d" "$idx").log"
    run_case "formula_$idx" "$log_file" -c 512 -n 512 -l 256 -f "[$formula]" "${EXTRA_ARGS[@]}"
    idx=$((idx + 1))
done

echo "[stat_pareto] completed. output_dir=$OUTDIR total_formula_cases=$((idx - 1))"
