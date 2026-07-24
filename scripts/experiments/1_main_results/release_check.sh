#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: release_check.sh --expected-commit REVISION

Run final, offline release gates for a frozen public artifact checkout.
This command does not download, create an environment, or use a GPU.
EOF
}

die() {
    echo "[release_check.sh] ERROR: $*" >&2
    exit 1
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/experiments/1_main_results"
PYTHON_BIN=${PYTHON_BIN:-python3}
EXPECTED_COMMIT=

while (( $# > 0 )); do
    case "$1" in
        --expected-commit) EXPECTED_COMMIT=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n $EXPECTED_COMMIT ]] || die "--expected-commit is required"
command -v git >/dev/null 2>&1 || die "git is unavailable"
command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || die "$PYTHON_BIN is unavailable"

license_file=
for candidate in LICENSE LICENSE.md COPYING; do
    if git -C "$ROOT_DIR" ls-files --error-unmatch "$candidate" \
            >/dev/null 2>&1; then
        license_file=$candidate
        break
    fi
done
[[ -n $license_file ]] || die "a tracked LICENSE, LICENSE.md, or COPYING is required"
echo "[release_check.sh] license=$license_file"

PUBLIC_PATHS=(
    .gitmodules
    README.md
    pyproject.toml
    uv.lock
    scripts/experiments/1_main_results
)
PUBLIC_SCAN_PATHS=(
    "${PUBLIC_PATHS[@]}"
    ':(exclude)scripts/experiments/1_main_results/release_check.sh'
)

if git -C "$ROOT_DIR" grep -n -E \
        'TO-BE-FROZEN|<frozen-revision>' -- "${PUBLIC_SCAN_PATHS[@]}"; then
    die "release placeholder remains in a public artifact file"
fi

if git -C "$ROOT_DIR" grep -n -E \
        '/root/|/mnt/|mirrors\.sustech|git@github\.com:|hf-mirror' \
        -- "${PUBLIC_SCAN_PATHS[@]}"; then
    die "private path, mirror, or credentialed Git URL remains"
fi

"$PYTHON_BIN" - "$ROOT_DIR" <<'PY'
import pathlib
import sys
import tomllib

root = pathlib.Path(sys.argv[1])
for name in ("pyproject.toml", "uv.lock"):
    with (root / name).open("rb") as handle:
        tomllib.load(handle)
    print(f"[release_check.sh] valid_toml={name}")
PY

PYTHON_BIN="$PYTHON_BIN" "$SCRIPT_DIR/preflight.sh" \
    --expected-commit "$EXPECTED_COMMIT"
echo "[release_check.sh] PASS"
