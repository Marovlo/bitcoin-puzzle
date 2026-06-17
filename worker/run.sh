#!/usr/bin/env bash
#
# One-click build + run for the Bitcoin Puzzle Pool worker.
#
#   ./run.sh                  # auto-detect GPU/CPU, build, join the default pool
#   ./run.sh --url URL        # point at a different coordinator
#   ./run.sh --backend cpu    # force a backend (auto|metal|cpu|cuda|hip)
#   ./run.sh --build-only      # build, don't run
#   ./run.sh --test            # build + run correctness tests, don't join the pool
#
# Any extra flags are passed straight through to ./puzzle_worker
# (e.g. --cpu-threads N, --metal-batch N). See ./puzzle_worker --help.
#
# Platform / backend detection:
#   macOS            -> Metal (via Makefile)
#   Linux + nvcc     -> CUDA  (via CMake, sm auto)
#   Linux + hipcc    -> HIP   (via CMake)
#   Linux (CPU only) -> CPU   (via Makefile)
set -euo pipefail
cd "$(dirname "$0")"

BUILD_ONLY=0
RUN_TEST=0
PASSTHROUGH=()

# Pull out the flags we handle here; forward the rest to the worker binary.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only) BUILD_ONLY=1; shift ;;
        --test)       RUN_TEST=1;   shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

uname_s="$(uname -s)"
echo "==> Platform: $uname_s"

build_make() {
    echo "==> Building with make ($(nproc 2>/dev/null || sysctl -n hw.ncpu) cores)"
    make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
}

build_cmake() {
    local flag="$1"   # -DUSE_CUDA=ON or -DUSE_HIP=ON
    echo "==> Building with CMake ($flag)"
    mkdir -p build
    ( cd build && cmake .. "$flag" -DCMAKE_BUILD_TYPE=Release >/dev/null \
        && make -j"$(nproc)" )
    cp -f build/puzzle_worker ./puzzle_worker
}

BACKEND_NOTE="auto"
if [[ "$uname_s" == "Darwin" ]]; then
    BACKEND_NOTE="Metal"
    build_make
elif command -v nvcc >/dev/null 2>&1; then
    BACKEND_NOTE="CUDA"
    build_cmake -DUSE_CUDA=ON
elif command -v hipcc >/dev/null 2>&1; then
    BACKEND_NOTE="HIP"
    build_cmake -DUSE_HIP=ON
else
    BACKEND_NOTE="CPU"
    build_make
fi
echo "==> Built puzzle_worker (detected backend: $BACKEND_NOTE)"

if [[ "$RUN_TEST" == "1" ]]; then
    echo "==> Running correctness tests"
    make test
    exit 0
fi

if [[ "$BUILD_ONLY" == "1" ]]; then
    echo "==> --build-only: not joining the pool. Run: ./puzzle_worker"
    exit 0
fi

echo "==> Joining pool (Ctrl+C to stop)"
exec ./puzzle_worker "${PASSTHROUGH[@]}"
