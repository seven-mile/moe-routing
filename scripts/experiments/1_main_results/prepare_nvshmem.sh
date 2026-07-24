#!/usr/bin/env bash
set -euo pipefail

VERSION=3.4.5
ARCHIVE_NAME=libnvshmem-linux-x86_64-3.4.5_cuda12-archive.tar.xz
ARCHIVE_URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/$ARCHIVE_NAME"
ARCHIVE_SHA256=058cbaddc4ff8646b8d1bd9322e93c90eae54c86e1ac8922f20d8a55a7fa8b7e

usage() {
    cat <<EOF
Usage: prepare_nvshmem.sh --prefix PATH [options]

Options:
  --prefix PATH             Final NVSHMEM installation prefix (required).
  --archive PATH            Use an existing archive instead of downloading.
  --cache-dir PATH          Download cache (default: /tmp/moe-routing-ae-cache).
  --dry-run                 Print the pinned source and destination only.
  -h, --help                Show this help.

Downloads the official NVIDIA NVSHMEM $VERSION CUDA 12 x86_64 archive,
verifies its SHA256, and extracts it without overwriting an existing prefix.
EOF
}

die() {
    echo "[prepare_nvshmem.sh] ERROR: $*" >&2
    exit 1
}

PREFIX=
ARCHIVE_PATH=
CACHE_DIR=/tmp/moe-routing-ae-cache
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX=${2:?}; shift 2 ;;
        --archive) ARCHIVE_PATH=${2:?}; shift 2 ;;
        --cache-dir) CACHE_DIR=${2:?}; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n $PREFIX ]] || die "--prefix is required"
if [[ $PREFIX != /* ]]; then
    PREFIX="$(pwd)/$PREFIX"
fi
[[ ! -e $PREFIX ]] || die "$PREFIX already exists"

echo "NVSHMEM archive: $ARCHIVE_URL"
echo "SHA256:          $ARCHIVE_SHA256"
echo "Install prefix:  $PREFIX"
if [[ $DRY_RUN == true ]]; then
    exit 0
fi

for command_name in curl sha256sum tar mktemp; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die "$command_name is required"
done

if [[ -z $ARCHIVE_PATH ]]; then
    mkdir -p "$CACHE_DIR"
    ARCHIVE_PATH="$CACHE_DIR/$ARCHIVE_NAME"
    if [[ ! -f $ARCHIVE_PATH ]]; then
        partial_path="$ARCHIVE_PATH.part"
        [[ ! -e $partial_path ]] || die "partial download exists: $partial_path"
        curl --fail --location --retry 3 --output "$partial_path" "$ARCHIVE_URL"
        mv "$partial_path" "$ARCHIVE_PATH"
    fi
elif [[ $ARCHIVE_PATH != /* ]]; then
    ARCHIVE_PATH="$(pwd)/$ARCHIVE_PATH"
fi
[[ -f $ARCHIVE_PATH ]] || die "archive not found: $ARCHIVE_PATH"

actual_sha256=$(sha256sum "$ARCHIVE_PATH")
actual_sha256=${actual_sha256%% *}
[[ $actual_sha256 == "$ARCHIVE_SHA256" ]] \
    || die "SHA256 mismatch for $ARCHIVE_PATH: $actual_sha256"

prefix_parent=$(dirname "$PREFIX")
mkdir -p "$prefix_parent"
stage_dir=$(mktemp -d "$prefix_parent/.nvshmem-$VERSION.XXXXXX")
cleanup() {
    if [[ -n ${stage_dir:-} && -d $stage_dir ]]; then
        rm -rf -- "$stage_dir"
    fi
}
trap cleanup EXIT

tar -xJf "$ARCHIVE_PATH" -C "$stage_dir"
shopt -s nullglob dotglob
entries=("$stage_dir"/*)
shopt -u nullglob dotglob
if (( ${#entries[@]} != 1 )) || [[ ! -d ${entries[0]} ]]; then
    die "archive must contain exactly one top-level directory"
fi

source_dir=${entries[0]}
for required_path in \
    include/nvshmem.h \
    lib/libnvshmem_device.a \
    lib/libnvshmem_host.so; do
    [[ -e $source_dir/$required_path ]] \
        || die "archive is missing $required_path"
done

mv "$source_dir" "$PREFIX"
echo "Installed NVSHMEM $VERSION at $PREFIX"
echo "export NVSHMEM_DIR=$(printf '%q' "$PREFIX")"
echo "export LD_LIBRARY_PATH=$(printf '%q' "$PREFIX/lib"):\${LD_LIBRARY_PATH:-}"
