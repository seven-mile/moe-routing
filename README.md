# Spec-K

This repository contains the Spec-K implementation and its pinned source
dependencies. Spec-K applies draft-model signals to token-wise expert routing
for speculative mixture-of-experts serving.

The SC26 Artifacts Evaluated--Functional workflow is documented in
[`scripts/experiments/1_main_results`](scripts/experiments/1_main_results/README.md).
Its required path is a one-node, four-GPU functional validation of the
Qwen3-30B-A3B target, Qwen3-0.6B drafter, and baseline/lossless/optimum policy
modes. It does not impose a throughput threshold or request Results Reproduced.

## License

The Spec-K superproject is licensed under the Apache License 2.0. Git
submodules and downloaded dependencies remain subject to their own licenses.
