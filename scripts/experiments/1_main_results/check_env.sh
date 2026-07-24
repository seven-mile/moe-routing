#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SOURCE_ONLY=false
WITH_PPLX=false
EXPECTED_COMMIT=
while (( $# > 0 )); do
    case "$1" in
        --source-only) SOURCE_ONLY=true; shift ;;
        --with-pplx) WITH_PPLX=true; shift ;;
        --expected-commit) EXPECTED_COMMIT=${2:?}; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--source-only] [--with-pplx] [--expected-commit REVISION]"
            exit 0
            ;;
        *)
            echo "Usage: $0 [--source-only] [--with-pplx] [--expected-commit REVISION]" >&2
            exit 1
            ;;
    esac
done

failures=0
fail() {
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
}

pass() {
    echo "PASS: $*"
}

BASE_COMMIT=d2016ca430617824b70196abbba2a1459d6a0025
HEAD_COMMIT=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)
if git -C "$ROOT_DIR" merge-base --is-ancestor "$BASE_COMMIT" HEAD; then
    pass "HEAD descends from $BASE_COMMIT"
else
    fail "HEAD does not descend from $BASE_COMMIT"
fi

if [[ -n $EXPECTED_COMMIT ]]; then
    RESOLVED_EXPECTED_COMMIT=$(
        git -C "$ROOT_DIR" rev-parse --verify "$EXPECTED_COMMIT^{commit}" 2>/dev/null \
            || true
    )
    if [[ -z $RESOLVED_EXPECTED_COMMIT ]]; then
        fail "expected revision cannot be resolved: $EXPECTED_COMMIT"
    elif [[ $HEAD_COMMIT != "$RESOLVED_EXPECTED_COMMIT" ]]; then
        fail "HEAD is $HEAD_COMMIT, expected $RESOLVED_EXPECTED_COMMIT"
    else
        pass "HEAD matches expected revision $RESOLVED_EXPECTED_COMMIT"
    fi
    if [[ -n $(git -C "$ROOT_DIR" status --porcelain --untracked-files=all --ignore-submodules=none) ]]; then
        fail "checkout has tracked, untracked, or submodule changes"
    else
        pass "checkout is clean"
    fi
fi

declare -A EXPECTED_SUBMODULES=(
    [deep-ep]=abba6adde6b7a2d5bf3651c58b9edd670d13c986
    [lm-evaluation-harness]=a77f086a7b7f114aecb40242a70e963fadc66477
    [pplx-kernels]=5f48247dc21e25502873b5583217421056a490ed
    [transformers]=d1f7a32a48c3430a4395b6de718d6ba1c1360adb
    [vllm]=e0fbd66125551b5e0e2b470e453be914ff57e0ef
)

for name in "${!EXPECTED_SUBMODULES[@]}"; do
    expected=${EXPECTED_SUBMODULES[$name]}
    actual=$(git -C "$ROOT_DIR/$name" rev-parse HEAD 2>/dev/null || true)
    if [[ $actual == "$expected" ]]; then
        pass "$name@$expected"
    else
        fail "$name expected $expected, found ${actual:-uninitialized}"
    fi
done

if grep -qE 'git@github\.com:' "$ROOT_DIR/.gitmodules"; then
    fail ".gitmodules still requires SSH credentials"
else
    pass ".gitmodules uses credential-free URL syntax"
fi

if grep -qE 'mirrors\.sustech\.edu\.cn' "$ROOT_DIR/uv.lock"; then
    echo "WARN: uv.lock still contains the SUSTech mirror; regenerate it before AE"
else
    pass "uv.lock contains no SUSTech URLs"
fi

if [[ $SOURCE_ONLY == true ]]; then
    (( failures == 0 )) || exit 1
    exit 0
fi

for command_name in uv python nvcc nvidia-smi; do
    if command -v "$command_name" >/dev/null 2>&1; then
        pass "$command_name is available"
    else
        fail "$command_name is not available"
    fi
done

uv_version=$(uv --version 2>/dev/null | awk '{print $2}' || true)
if [[ $uv_version == 0.11.21 ]]; then
    pass "uv version is 0.11.21"
else
    fail "uv version is ${uv_version:-unavailable}, expected 0.11.21"
fi

python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || true)
if [[ $python_version == 3.12 ]]; then
    pass "Python version is 3.12"
else
    fail "Python version is ${python_version:-unavailable}, expected 3.12"
fi

NVSHMEM_LIB_DIR=
if [[ -z ${NVSHMEM_DIR:-} ]]; then
    fail "NVSHMEM_DIR is not set"
else
    NVSHMEM_LIB_DIR="$NVSHMEM_DIR/lib"
    if [[ -f $NVSHMEM_DIR/include/nvshmem.h ]]; then
        pass "NVSHMEM headers found in $NVSHMEM_DIR"
    else
        fail "missing $NVSHMEM_DIR/include/nvshmem.h"
    fi
    if [[ -f $NVSHMEM_DIR/lib/libnvshmem_device.a ]]; then
        pass "NVSHMEM device library found"
    else
        fail "missing $NVSHMEM_DIR/lib/libnvshmem_device.a"
    fi
    if [[ -e $NVSHMEM_DIR/lib/libnvshmem_host.so ]]; then
        pass "unversioned NVSHMEM host library found"
    else
        fail "missing $NVSHMEM_DIR/lib/libnvshmem_host.so required at link time"
    fi

    ibgda_plugin=$(find -L "$NVSHMEM_DIR/lib" -maxdepth 1 -type f \
        -name 'nvshmem_transport_ibgda.so*' -print -quit 2>/dev/null || true)
    if [[ -z $ibgda_plugin ]]; then
        fail "NVSHMEM IBGDA transport plugin is missing"
    else
        ibgda_ldd=$(ldd "$ibgda_plugin" 2>&1 || true)
        if grep -q 'not found' <<< "$ibgda_ldd"; then
            fail "NVSHMEM IBGDA transport has unresolved host libraries"
            echo "$ibgda_ldd" >&2
        else
            pass "NVSHMEM IBGDA transport host libraries resolve"
        fi
    fi
fi

if python - "$WITH_PPLX" <<'PY' >/dev/null 2>&1
import sys

import deep_ep
import flash_attn
import torch
import transformers
import vllm

if sys.argv[1] == "true":
    import pplx_kernels
PY
then
    if [[ $WITH_PPLX == true ]]; then
        pass "torch, DeepEP, flash-attn, PPLX, Transformers, and vLLM import"
    else
        pass "torch, DeepEP, flash-attn, Transformers, and vLLM import"
    fi
else
    fail "one or more required Python runtime imports failed"
fi

package_check=$(
    python - "$ROOT_DIR" "$WITH_PPLX" <<'PY' 2>&1 || true
import importlib.metadata
import pathlib
import sys

import deep_ep
import transformers
import vllm

root = pathlib.Path(sys.argv[1]).resolve()
expected_paths = {
    "deep_ep": (pathlib.Path(deep_ep.__file__).resolve(), root / "deep-ep"),
    "transformers": (pathlib.Path(transformers.__file__).resolve(), root / "transformers"),
    "vllm": (pathlib.Path(vllm.__file__).resolve(), root / "vllm"),
}
for name, (actual, expected_root) in expected_paths.items():
    if not actual.is_relative_to(expected_root.resolve()):
        raise SystemExit(f"{name} resolves to {actual}, outside {expected_root}")

expected_versions = {
    "torch": "2.9.1",
    "flash-attn": "2.8.3",
    "transformers": "4.57.0.dev0",
}
if sys.argv[2] == "true":
    expected_versions["pplx-kernels"] = "0.0.1"
for name, expected in expected_versions.items():
    actual = importlib.metadata.version(name)
    if actual != expected:
        raise SystemExit(f"{name} version is {actual}, expected {expected}")

if "e0fbd6612" not in vllm.__version__:
    raise SystemExit(f"vLLM version {vllm.__version__} does not identify e0fbd6612")

print(f"python={sys.executable}")
for name, (actual, _) in expected_paths.items():
    print(f"{name}={actual}")
reported = ["torch", "flash-attn", "transformers", "deep-ep", "vllm"]
if sys.argv[2] == "true":
    reported.append("pplx-kernels")
for name in reported:
    print(f"{name}=={importlib.metadata.version(name)}")
PY
)
if grep -q '^python=' <<< "$package_check"; then
    pass "package versions and editable source paths match the frozen checkout"
    echo "$package_check"
else
    fail "package version or editable source-path validation failed"
    echo "$package_check" >&2
fi

torch_cuda=$(python -c 'import torch; print(torch.version.cuda or "none")' 2>/dev/null || true)
if [[ $torch_cuda == 12.8* ]]; then
    pass "PyTorch uses CUDA $torch_cuda"
else
    fail "PyTorch CUDA version is ${torch_cuda:-unavailable}, expected 12.8"
fi

deep_ep_extension=$(
    python -c 'import deep_ep_cpp; print(deep_ep_cpp.__file__)' 2>/dev/null || true
)
if [[ -n $deep_ep_extension && -f $deep_ep_extension ]]; then
    pass "DeepEP extension found at $deep_ep_extension"
    deep_ep_ldd=$(ldd "$deep_ep_extension" 2>&1 || true)
    if [[ -z $NVSHMEM_LIB_DIR ]]; then
        fail "cannot verify DeepEP NVSHMEM resolution without NVSHMEM_DIR"
    elif grep -Fq "$NVSHMEM_LIB_DIR" <<< "$deep_ep_ldd"; then
        pass "DeepEP resolves NVSHMEM from NVSHMEM_DIR"
    else
        fail "DeepEP does not resolve NVSHMEM from $NVSHMEM_LIB_DIR"
        echo "$deep_ep_ldd" >&2
    fi
else
    fail "deep_ep_cpp extension is unavailable"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_inventory=$(nvidia-smi --query-gpu=index,name,driver_version \
        --format=csv,noheader 2>&1 || true)
    if [[ -n $gpu_inventory && $gpu_inventory != *"NVIDIA-SMI has failed"* ]]; then
        echo "$gpu_inventory"
        gpu_count=$(wc -l <<< "$gpu_inventory")
        if (( gpu_count >= 4 )); then
            pass "at least four GPUs are visible"
        else
            fail "only $gpu_count GPU(s) are visible; EP4 requires four"
        fi
    else
        fail "nvidia-smi cannot query the GPU driver"
        echo "$gpu_inventory" >&2
    fi

    topology=$(nvidia-smi topo -m 2>&1 || true)
    if [[ -n $topology && $topology != *"NVIDIA-SMI has failed"* ]]; then
        echo "$topology"
    else
        fail "nvidia-smi cannot report GPU topology"
        echo "$topology" >&2
    fi
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc_output=$(nvcc --version)
    echo "$nvcc_output"
    if grep -qE 'release 12\.8' <<< "$nvcc_output"; then
        pass "CUDA toolkit is 12.8"
    else
        fail "CUDA toolkit is not 12.8"
    fi
fi
NVSHMEM_INFO_BIN=
if [[ -n ${NVSHMEM_DIR:-} && -x $NVSHMEM_DIR/bin/nvshmem-info ]]; then
    NVSHMEM_INFO_BIN=$NVSHMEM_DIR/bin/nvshmem-info
elif command -v nvshmem-info >/dev/null 2>&1; then
    NVSHMEM_INFO_BIN=$(command -v nvshmem-info)
fi
if [[ -n $NVSHMEM_INFO_BIN ]]; then
    nvshmem_output=$($NVSHMEM_INFO_BIN -a 2>&1)
    echo "$nvshmem_output"
    if grep -qE '3\.4\.5' <<< "$nvshmem_output"; then
        pass "NVSHMEM reports version 3.4.5"
    else
        fail "nvshmem-info does not report version 3.4.5"
    fi
else
    fail "nvshmem-info is unavailable under NVSHMEM_DIR or PATH"
fi

(( failures == 0 )) || exit 1
