# Main-result artifact workflow

This directory is the AE entry point for the Qwen3-30B-A3B target and
Qwen3-0.6B draft model. The frozen artifact revision pins the superproject and
all source submodules used by this workflow.

## Scope

The scripts support EP4, EP8, EP16, and EP32 without cluster-specific
environment variables. The Functional AE scope is:

1. Run the required EP4 smoke test on one 4xH100 node. It validates the
   installation, serving path, benchmark protocol, and analysis pipeline.
2. Run the 30B EP8 profile only when two 4xH100 nodes provide the lossless
   RDMA, GPUDirect RDMA, and NVSHMEM IBGDA facilities required by DeepEP.
3. Use the EP16 and EP32 profiles when recreating the corresponding paper
   configurations on a suitable cluster; they are outside the Functional AE.

Record the GPU topology and complete logs for every run. The Functional
criteria do not include a throughput or speedup threshold.

## Source checkout

Clone the public artifact and use the frozen revision printed in the AD/AE
Appendix:

```bash
git clone --recurse-submodules https://github.com/ParCIS/Spec-K.git
cd Spec-K
REVISION="sc26-ae-v3"
git checkout "$REVISION"
git submodule update --init --recursive
scripts/experiments/1_main_results/preflight.sh \
  --expected-commit "$REVISION"
```

The public release uses credential-free HTTPS submodule URLs and supports an
anonymous recursive clone. The preflight requires an exact, clean checkout,
validates all five submodule gitlinks, expands the pinned setup/serve/benchmark
commands, and runs the parser tests without downloads or GPUs.

## Build the environment

`bootstrap_env.sh` encodes a staged installation without relying on uv to infer
native build order. It first syncs torch and ordinary dependencies while
excluding the four packages whose builds import torch. This sync also installs
the NVSHMEM package selected by the frozen dependency graph. The script then
installs DeepEP, flash-attn, and vLLM in that order, followed by read-only frozen
environment and dependency checks. PPLX remains in the pinned source and
dependency graph but is not used by the DeepEP main-result path; pass
`--with-pplx` only to build the optional PPLX backend as well.

The `--venv` option selects the environment path. The bootstrap passes that
path explicitly to every `uv sync` and package installation command.

Install the host prerequisites and a uv version in the supported
`>=0.11.21,<0.12` range:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl git python3-pip \
  libibverbs-dev librdmacm-dev rdma-core
python3 -m pip install --user 'uv>=0.11.21,<0.12'
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

The RDMA packages provide the verbs and connection-manager development files,
userspace providers, and core device setup required by NVSHMEM's IBGDA plugin.
They are required because DeepEP low-latency mode enables IBGDA even for the
required EP4 initialization. DeepEP uses NVSHMEM's unique-ID initialization,
so MPI, PMI, and PMIx bootstrap packages are not required by this workflow.

The Linux lock selects `nvidia-nvshmem-cu12==3.3.20`, which satisfies DeepEP's
NVSHMEM 3.3.9 minimum and supplies its headers, host/device libraries, and IBGDA
plugin. `bootstrap_env.sh` ignores an ambient `NVSHMEM_DIR` while building the
native packages so that DeepEP and torch use the same NVSHMEM installation. Do
not add another NVSHMEM library to `LD_LIBRARY_PATH` for this workflow.

The checked-in `uv.lock` uses public PyPI and pythonhosted.org URLs and records
the versions, environment markers, artifact hashes, and editable source paths
used by the workflow. The bootstrap verifies the lock before runtime checks.

Build the environment:

```bash
scripts/experiments/1_main_results/bootstrap_env.sh
```

By default, the bootstrap downloads native extensions from the pinned upstream
vLLM 0.16.0 wheel while retaining this artifact's Python changes through an
editable install. Pass `--build-vllm-from-source` to compile all vLLM native
extensions locally; this is a fallback and may take more than one hour. Use
`--dry-run` to inspect the complete sequence. The final
`uv sync --frozen --check` and `uv pip check` operations only validate the
resulting environment; they do not reinstall the editable native packages
installed in the preceding steps.

The required OpenAI-compatible HTTP endpoint does not use vLLM's separate gRPC
server. The optional `grpcio-tools` package is therefore not installed; a build
message that skips gRPC proto generation does not indicate a failed setup.

Before serving, activate and check the environment on every node:

```bash
source .venv/bin/activate
scripts/experiments/1_main_results/check_env.sh
```

The environment checker verifies the locked NVSHMEM package and confirms that
DeepEP resolves its host library from that package. A successful EP4 smoke run
remains the required compatibility evidence.

## Prepare inputs

After activating the environment, download the pinned ShareGPT file and verify
its SHA256. Add `--with-models` to stage local snapshots of both pinned model
revisions instead of downloading weights during server startup:

```bash
scripts/experiments/1_main_results/prepare_inputs.sh \
  --root "$PWD/.ae-inputs" \
  --with-models
```

The helper prints the dataset and model paths to use below. Without
`--with-models`, it downloads only the 673 MB dataset. `--dry-run` prints all
repository IDs, revisions, and destinations without contacting Hugging Face.
The two model snapshots require approximately 59 GB; reserve at least 70 GB for
models, the dataset, and download staging. If the site requires an egress proxy,
set its standard `HTTP_PROXY` and `HTTPS_PROXY` variables before cloning or
downloading.

The serving wrapper accepts any of the following equivalent model locations:

- Use the absolute paths printed by `prepare_inputs.sh --with-models`.
- For snapshots staged elsewhere, pass their absolute paths to `--model` and
  `--draft-model`.
- Omit both model options to use the pinned Hugging Face IDs and revisions. This
  reuses the standard Hugging Face cache when populated and otherwise downloads
  the missing files at server startup.

For a pre-warmed cache on a disconnected node, set `HF_HUB_OFFLINE=1`. Use the
standard `HF_HOME` or `HF_HUB_CACHE` variables when the cache is not in its
default location. These variables are unnecessary when absolute snapshot paths
are supplied.

## Start the server

Single-node EP4 smoke test:

```bash
scripts/experiments/1_main_results/serve.sh \
  --model "$PWD/.ae-inputs/models/Qwen3-30B-A3B" \
  --draft-model "$PWD/.ae-inputs/models/Qwen3-0.6B" \
  --nnodes 1 \
  --node-rank 0 \
  --gpus-per-node 4
```

For two-node EP8, use a private address for node rank 0. Run this on both nodes,
changing only `--node-rank`:

```bash
scripts/experiments/1_main_results/serve.sh \
  --model "$PWD/.ae-inputs/models/Qwen3-30B-A3B" \
  --draft-model "$PWD/.ae-inputs/models/Qwen3-0.6B" \
  --nnodes 2 \
  --node-rank 0 \
  --gpus-per-node 4 \
  --master-addr HEAD_PRIVATE_IP
```

Rank 0 starts the API server. Other ranks automatically use `--headless`. The
command preserves the paper configuration: expert parallelism,
`deepep_low_latency`, eager execution, no prefix cache, no async scheduling,
4096-token model length, and three speculative tokens.

For Hugging Face model IDs, `serve.sh` also pins the target and draft revisions.
Local model directories are assumed to have been downloaded from those
revisions and do not receive a Hub revision argument.

Use `serve.sh --dry-run ...` to inspect shell quoting and computed ranks without
loading a model.

## Run the client

Smoke test:

```bash
scripts/experiments/1_main_results/bench.sh \
  --profile smoke \
  --dataset-path "$PWD/.ae-inputs/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --output-len 64 \
  --strict-protocol \
  --plot
```

Two-node EP8 main-result curve:

```bash
scripts/experiments/1_main_results/bench.sh \
  --profile ep8 \
  --dataset-path "$PWD/.ae-inputs/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --strict-protocol \
  --plot
```

The supplied profiles use prompt multipliers of 4 for EP8, 2 for EP16, and 1
for EP32.
Within a new run, every mode receives the same number of prompts at a given
concurrency. Logs, parsed CSV, a Markdown summary, and optional plots are stored
under `logs/main_results/` by default.

The parser fails if any requested log lacks a complete benchmark-result block.
Strict protocol validation additionally requires every requested policy at
every concurrency, zero failed requests, equal successful-request counts, and
exactly `concurrency * prompt_multiplier` successful requests.

The analysis checks require no GPU and can be run independently:

```bash
python scripts/experiments/1_main_results/test_analysis.py
scripts/experiments/1_main_results/analyze_reference.sh
```

The runner verifies the pinned ShareGPT file by SHA256 before contacting the
server. Use `--dataset-sha256` for an explicitly substituted dataset or
`--skip-dataset-hash` only for a non-reference smoke test.

## Analyze the reference fixture

`reference/30b_ep8.csv` contains the protocol-relevant columns for the complete
30B EP8 concurrency curve used by the AE analysis check. It is a lossless
column selection from the original parsed CSV; its README records the
source-file digest and transformation.

`analyze_reference.sh` checks the fixture's policy set, zero-failure fields,
matched request counts, and prompt multiplier before producing a Markdown
summary and throughput plots under `reference/generated/`. An alternate output
directory may be supplied as its sole argument.

The reference fixture validates the analysis and request-matching protocol. It
is separate from the required live EP4 smoke test and carries no acceptance
threshold.
