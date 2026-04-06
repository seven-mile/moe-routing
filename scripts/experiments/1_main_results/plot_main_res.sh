#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/experiments/1_main_results"
RESULT_DIR="$ROOT_DIR/data/plot/1_main_res"

cd $SCRIPT_DIR || exit -1

python $SCRIPT_DIR/plot_main_res.py --file $RESULT_DIR/30b_ep8.csv   --labels baseline lossless optimum --out_prefix $RESULT_DIR/30b_ep8  --legend_file
python $SCRIPT_DIR/plot_main_res.py --file $RESULT_DIR/30b_ep16.csv  --labels baseline lossless optimum --out_prefix $RESULT_DIR/30b_ep16
python $SCRIPT_DIR/plot_main_res.py --file $RESULT_DIR/30b_ep32.csv  --labels baseline lossless optimum --out_prefix $RESULT_DIR/30b_ep32
python $SCRIPT_DIR/plot_main_res.py --file $RESULT_DIR/235b_ep16.csv --labels baseline lossless optimum --out_prefix $RESULT_DIR/235b_ep16
python $SCRIPT_DIR/plot_main_res.py --file $RESULT_DIR/235b_ep32.csv --labels baseline lossless optimum --out_prefix $RESULT_DIR/235b_ep32
