# Scripts

The supported SC26 Artifacts Evaluated--Functional workflow is
[`experiments/1_main_results`](experiments/1_main_results/README.md). Evaluators
do not need to run any other directory under `scripts/`.

The remaining directories contain optional research workflows:

- `dyn_topk/search`: automatic Spec-K policy search.
- `experiments/2_topk_acceptance`: Top-k/acceptance analysis; its benchmark
  runner uses `quick/bench_serve.sh`.
- `quick/lmeval.sh` and `lm_eval/generate_report.py`: quality-evaluation
  helpers.
- Other directories: exploratory data preparation and analysis retained for
  research provenance. They are not part of the AE workflow.
