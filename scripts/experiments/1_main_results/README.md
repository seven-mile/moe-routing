# Main-result artifact workflow

This directory is the AE entry point for the Qwen3-30B-A3B target and
Qwen3-0.6B draft model. It is based on superproject commit `d2016ca` and the
gitlinks recorded by that commit.

## Scope

The scripts support EP4, EP8, EP16, and EP32 without cluster-specific
environment variables. The Functional AE scope is:

1. Run an EP4 smoke test on one 4xH100 node. This checks installation and the
   serving workflow, but it does not reproduce a paper data point.
2. If two 4xH100 nodes and working IBGDA/RDMA are available, the 30B EP8
   profile may be run as an optional diagnostic. This corresponds to one of
   the paper's main-result curves, but it is not a Functional acceptance
   condition and no H100 speedup threshold is prescribed.
3. Keep EP16 and EP32 profiles available as documentation of the original
   experiment configuration, but outside the Functional badge workflow.

H100 throughput and policy ordering need not match the original H20
measurements. Record the complete topology, but do not apply a performance or
speedup threshold to the Functional run.

## Source checkout

Clone the public artifact and replace the placeholder below with the frozen
revision printed in the AD/AE Appendix:

```bash
git clone --recurse-submodules https://github.com/ParCIS/Spec-K.git
cd Spec-K
REVISION="sc26-ae-v1"
git checkout "$REVISION"
git submodule update --init --recursive
scripts/experiments/1_main_results/preflight.sh \
  --expected-commit "$REVISION"
```

The frozen AE release must use credential-free HTTPS submodule URLs and permit
an anonymous recursive clone. The preflight requires an exact, clean checkout,
validates all five submodule gitlinks, expands the pinned setup/serve/benchmark
commands, and runs the parser tests without downloads or GPUs.

Before publishing the frozen revision, run the stricter offline release gate:

```bash
scripts/experiments/1_main_results/release_check.sh \
  --expected-commit "$REVISION"
```

It additionally requires a tracked project license, rejects unresolved revision
placeholders and private paths, and parses the public TOML files. The command is
expected to fail in an unfrozen draft checkout.

## Environment status

`bootstrap_env.sh` encodes a staged installation without relying on uv to infer
native build order. It first syncs torch and ordinary dependencies while
excluding the four packages whose builds import torch. It then installs DeepEP,
flash-attn, and vLLM in that order, followed by read-only frozen environment and
dependency checks. PPLX remains in the pinned source and dependency graph but
is not used by the DeepEP main-result path; pass `--with-pplx` only to build the
optional PPLX backend as well.

Every `uv sync` sets `UV_PROJECT_ENVIRONMENT` to the path selected by
`--venv`. This is necessary because `uv sync --python /path/to/venv/bin/python`
selects an interpreter but does not, by itself, make that existing venv the
project environment. With an absolute `--venv` path, the bootstrap neither
discovers nor modifies a `.venv` belonging to another worktree.

Install the uv version used to validate the lock before running the bootstrap:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl git python3-pip xz-utils \
  libibverbs1 ibverbs-providers
python3 -m pip install --user 'uv==0.11.21'
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

The two RDMA runtime packages provide `libibverbs.so.1`, `libmlx5.so.1`, and
their dependencies. They are required because DeepEP low-latency mode enables
NVSHMEM's IBGDA transport even for the required EP4 initialization. DeepEP
uses NVSHMEM's unique-ID initialization, so MPI, PMI, and PMIx bootstrap
packages are not required by this workflow.

Always set `NVSHMEM_DIR` while building DeepEP. At the pinned DeepEP commit,
`setup.py` behaves as follows:

- An explicit `NVSHMEM_DIR` takes precedence over the `nvidia.nvshmem` Python
  package.
- It adds `$NVSHMEM_DIR/include` and `$NVSHMEM_DIR/lib`, links
  `libnvshmem_device.a` and `libnvshmem_host.so`, and embeds an rpath to the
  selected `lib` directory.
- If neither `NVSHMEM_DIR` nor the Python package is found, the build succeeds
  with internode and low-latency support disabled. That build is unsuitable for
  this experiment.

Prepare the official NVIDIA CUDA 12 archive with its pinned SHA256 before the
environment build:

```bash
scripts/experiments/1_main_results/prepare_nvshmem.sh \
  --prefix /opt/nvshmem-3.4.5
export NVSHMEM_DIR=/opt/nvshmem-3.4.5
```

The helper uses NVIDIA's `libnvshmem-linux-x86_64-3.4.5_cuda12` archive and
verifies SHA256
`058cbaddc4ff8646b8d1bd9322e93c90eae54c86e1ac8922f20d8a55a7fa8b7e`.
The URL and digest come from NVIDIA's
[`redistrib_3.4.5.json`](https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.4.5.json)
manifest. The helper refuses to overwrite an existing prefix and verifies that
the extracted layout includes the unversioned `lib/libnvshmem_host.so` required
by the pinned DeepEP link step.

The checked-in `uv.lock` now uses only public PyPI and pythonhosted.org URLs.
The migration preserves every locked version, environment marker, artifact
hash, and distribution path; `uv lock --check` passes. Treat that structural
check as a prerequisite, not a substitute for the complete frozen source build
and runtime checks below.

The intended source-build command is:

```bash
scripts/experiments/1_main_results/bootstrap_env.sh
```

Use `bootstrap_env.sh --dry-run` to inspect the complete sequence. The default
build compiles the pinned vLLM source. The final `uv sync --frozen --check` and
`uv pip check` operations only validate the resulting environment; they do not
reinstall the native editable packages built in the preceding steps.

The pinned vLLM build metadata lists `grpcio-tools` for its separate gRPC
server. That package is intentionally absent from the HTTP serving environment:
vLLM treats it as optional during its editable build, and this AE uses the
OpenAI-compatible HTTP endpoint. A build warning that gRPC proto generation was
skipped is therefore expected and does not affect the required path.

Before serving, activate the environment and select the same NVSHMEM build on
every node:

```bash
source .venv/bin/activate
export NVSHMEM_DIR=/opt/nvshmem-3.4.5
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
scripts/experiments/1_main_results/check_env.sh
```

The environment checker verifies that DeepEP resolves the pinned NVSHMEM host
library and rejects unresolved host dependencies in the archive's IBGDA
plugin. A successful clean source build and EP4 smoke run remain the required
compatibility evidence.

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

The historical prompt multipliers are 4 for EP8, 2 for EP16, and 1 for EP32.
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

## Reference protocol note

`reference/30b_ep8.csv` contains the protocol-relevant columns for the complete
30B EP8 concurrency curve used by the AE analysis check. It is a lossless
column selection from the original parsed CSV; its README records the
source-file digest and transformation.

`analyze_reference.sh` checks the fixture's policy set, zero-failure fields,
matched request counts, and prompt multiplier before producing a Markdown
summary and throughput plots under `reference/generated/`. An alternate output
directory may be supplied as its sole argument.

The 30B EP8 c1024 point is matched at 4096 successful requests per mode and
yields 1.437x in the original H20 data. It is useful as a reference-protocol
check and as an optional live diagnostic if Chameleon can reserve both H100
nodes and expose the required RDMA devices. It is not the required badge run.
