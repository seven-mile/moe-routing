#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bootstrap_env.sh [options]

Required tool:
  uv >=0.11.21,<0.12       Compatible lockfile tool range.

Options:
  --venv PATH               Virtual environment path (default: .venv).
  --python VERSION          Python version for uv venv (default: 3.12).
  --max-jobs N              Parallel native build jobs (default: 8).
  --with-pplx               Also build optional PPLX kernels.
  --build-vllm-from-source  Compile vLLM native extensions instead of using the pinned wheel.
  --reuse-venv              Allow installation into an existing venv.
  --dry-run                 Print installation commands without changing an env.
  -h, --help                Show this help.
EOF
}

die() {
    echo "[bootstrap_env.sh] ERROR: $*" >&2
    exit 1
}

warn() {
    echo "[bootstrap_env.sh] WARNING: $*" >&2
}

print_command() {
    printf '[bootstrap_env.sh] command:'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    print_command "$@"
    if [[ $DRY_RUN == false ]]; then
        "$@"
    fi
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/experiments/1_main_results"
UV_MIN_PATCH_VERSION=21
VLLM_PRECOMPILED_BASE_COMMIT=89a77b10846fd96273cce78d86d2556ea582d26e
VENV_DIR=.venv
PYTHON_VERSION=3.12
MAX_JOBS_VALUE=${MAX_JOBS:-8}
WITH_PPLX=false
PRECOMPILED_VLLM=true
REUSE_VENV=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) VENV_DIR=${2:?}; shift 2 ;;
        --python) PYTHON_VERSION=${2:?}; shift 2 ;;
        --max-jobs) MAX_JOBS_VALUE=${2:?}; shift 2 ;;
        --with-pplx) WITH_PPLX=true; shift ;;
        --build-vllm-from-source) PRECOMPILED_VLLM=false; shift ;;
        --reuse-venv) REUSE_VENV=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ $MAX_JOBS_VALUE =~ ^[1-9][0-9]*$ ]] || die "--max-jobs must be positive"

if [[ $VENV_DIR = /* ]]; then
    VENV_PATH=$VENV_DIR
else
    VENV_PATH="$ROOT_DIR/$VENV_DIR"
fi
PYTHON_BIN="$VENV_PATH/bin/python"

"$SCRIPT_DIR/check_env.sh" --source-only

if grep -qE 'mirrors\.sustech\.edu\.cn' "$ROOT_DIR/uv.lock"; then
    if [[ $DRY_RUN == true ]]; then
        warn "uv.lock still uses the SUSTech mirror; a real run would stop here"
    else
        die "uv.lock must be regenerated against public indexes before installation"
    fi
fi

if [[ $DRY_RUN == false ]]; then
    command -v uv >/dev/null 2>&1 || die "uv is not installed"
    UV_ACTUAL_VERSION=$(uv --version | awk '{print $2}')
    IFS=. read -r uv_major uv_minor uv_patch <<< "$UV_ACTUAL_VERSION"
    if [[ $uv_major == 0 && $uv_minor == 11 && $uv_patch =~ ^[0-9]+$ ]] \
            && (( uv_patch >= UV_MIN_PATCH_VERSION )); then
        :
    else
        die "uv >=0.11.21,<0.12 is required, found $UV_ACTUAL_VERSION"
    fi
    command -v nvcc >/dev/null 2>&1 || die "nvcc is not installed"
    nvcc --version | grep -qE 'release 12\.8' \
        || die "CUDA toolkit 12.8 is required"
    if [[ -e $VENV_PATH && $REUSE_VENV == false ]]; then
        die "$VENV_PATH already exists; use --reuse-venv or choose a new path"
    fi
fi

RUNTIME_PATH="$VENV_PATH/bin:$PATH"
BUILD_ENV=(
    env
    -u NVSHMEM_DIR
    "TORCH_CUDA_ARCH_LIST=9.0"
    "MAX_JOBS=$MAX_JOBS_VALUE"
    "CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS_VALUE"
)
VLLM_BUILD_ENV=("${BUILD_ENV[@]}")
if [[ $PRECOMPILED_VLLM == true ]]; then
    VLLM_BUILD_ENV+=(
        "VLLM_USE_PRECOMPILED=1"
        "VLLM_PRECOMPILED_WHEEL_COMMIT=$VLLM_PRECOMPILED_BASE_COMMIT"
        "VLLM_PRECOMPILED_WHEEL_VARIANT=cu129"
    )
fi

cd "$ROOT_DIR"
if [[ ! -e $VENV_PATH ]]; then
    run uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
fi

# Install torch and all ordinary dependencies first. The excluded packages
# import torch from their build backend and are installed explicitly below.
run env "UV_PROJECT_ENVIRONMENT=$VENV_PATH" \
    uv sync --frozen --python "$PYTHON_BIN" --no-install-project \
    --no-install-package deep-ep \
    --no-install-package pplx-kernels \
    --no-install-package flash-attn \
    --no-install-package vllm

run "$PYTHON_BIN" -c \
    'import jinja2, packaging, setuptools, setuptools_scm, torch'
run "$VENV_PATH/bin/cmake" --version
run "$VENV_PATH/bin/ninja" --version

run "${BUILD_ENV[@]}" uv pip install --python "$PYTHON_BIN" \
    --editable "$ROOT_DIR/deep-ep" --verbose --no-build-isolation --no-deps
if [[ $WITH_PPLX == true ]]; then
    run "${BUILD_ENV[@]}" uv pip install --python "$PYTHON_BIN" \
        --editable "$ROOT_DIR/pplx-kernels" --verbose --no-build-isolation --no-deps
fi
run "${BUILD_ENV[@]}" uv pip install --python "$PYTHON_BIN" \
    'flash-attn==2.8.3' --verbose --no-build-isolation --no-deps

run "${VLLM_BUILD_ENV[@]}" uv pip install --python "$PYTHON_BIN" \
    --editable "$ROOT_DIR/vllm" --verbose --no-build-isolation --no-deps

# Verify that the manually built packages satisfy their locked editable/source
# identities without giving uv another opportunity to rebuild native code.
FINAL_SYNC=(env "UV_PROJECT_ENVIRONMENT=$VENV_PATH" \
    uv sync --frozen --check --python "$PYTHON_BIN")
if [[ $WITH_PPLX == false ]]; then
    FINAL_SYNC+=(--no-install-package pplx-kernels)
fi
run "${FINAL_SYNC[@]}"
run uv pip check --python "$PYTHON_BIN"

CHECK_ENV=("$SCRIPT_DIR/check_env.sh")
if [[ $WITH_PPLX == true ]]; then
    CHECK_ENV+=(--with-pplx)
fi
run env \
    "PATH=$RUNTIME_PATH" \
    "${CHECK_ENV[@]}"
