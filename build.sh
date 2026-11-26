#!/bin/bash

################################################################################
# vLLM Custom Docker Image Build Script
################################################################################
#
# Builds vLLM from vllm/ source for the RTX 5090
#
# System Requirements:
# - NVIDIA Driver 620.70.10 (CUDA 13.5 UMD)
# - GPU: RTX 5090 (Blackwell architecture, compute 12.0)
# - CPU: 12-core Ryzen 9 9900X (24 threads)
# - RAM: 91GB available
# - Disk: ~40GB free for build cache
#
# Build Configuration:
# - Base + Runtime Image: nvcr.io/nvidia/vllm:26.11-py3 (see Dockerfile)
# - CUDA Arch: 12.0 only (RTX 5090 optimized)
# - Max Jobs: 16 (leaves 8 threads for system)
# - Expected Time: 45-75 minutes
# - Final Image Size: ~15GB
#
# Output Tags (per build):
# - vllm-custom:b<N>                          monotonic build number
# - vllm-custom:v0.22.1rc0-19-g50c80d79230    vLLM git-describe version
# - vllm-custom:latest                        most recent build
#
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/vllm"

# --- Version tagging --------------------------------------------------------
# Each build gets three tags:
#   vllm-custom:b<N>      monotonically increasing build number (build.number)
#   vllm-custom:<descr>   vLLM's own version, e.g. v0.22.1rc0-19-g50c80d79230
#   vllm-custom:latest    always points at the most recent build
# The build number only advances on a successful build, so failed builds
# don't burn a number.
BUILD_NUMBER_FILE="$SCRIPT_DIR/build.number"
LAST_BUILD_NUMBER=$(cat "$BUILD_NUMBER_FILE" 2>/dev/null || echo 0)
BUILD_NUMBER=$((LAST_BUILD_NUMBER + 1))

# vLLM-derived descriptive tag from setuptools_scm-style git describe.
VLLM_DESCRIBE=$(git -C "$SCRIPT_DIR/vllm" describe --tags --long 2>/dev/null \
  || git -C "$SCRIPT_DIR/vllm" rev-parse --short HEAD)

NUMBER_TAG="b${BUILD_NUMBER}"
VERSION_TAG="$VLLM_DESCRIBE"
# ---------------------------------------------------------------------------

echo "========================================"
echo "Building vLLM Custom Docker Image"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Build Number: $BUILD_NUMBER (previous: $LAST_BUILD_NUMBER)"
echo "  Tags:         vllm-custom:$NUMBER_TAG"
echo "                vllm-custom:$VERSION_TAG"
echo "                vllm-custom:latest"
echo "  Base Image:   nvcr.io/nvidia/vllm:26.11-py3 (CUDA 13.5, torch 2.15a0)"
echo "  Target Arch:  Blackwell (compute 12.0)"
echo "  Max Jobs:     16"
echo "  Build Time:   ~30-60 minutes"
echo ""
echo "Starting build at: $(date)"
echo ""

DOCKER_BUILDKIT=1 docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="12.0" \
  --build-arg MAX_JOBS=16 \
  --build-arg NVCC_THREADS=4 \
  -f "$SCRIPT_DIR/Dockerfile" \
  -t "vllm-custom:$NUMBER_TAG" \
  -t "vllm-custom:$VERSION_TAG" \
  -t vllm-custom:latest \
  "$SCRIPT_DIR"

BUILD_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $BUILD_EXIT_CODE -eq 0 ]; then
  # Persist the bumped build number only now that the build succeeded.
  echo "$BUILD_NUMBER" > "$BUILD_NUMBER_FILE"

  # Append a one-line entry to the build history (which version, when).
  # NOTE: this is build.history, not build.log (the latter is the full
  # build stdout/stderr written by build-async.sh).
  BUILD_HISTORY_FILE="$SCRIPT_DIR/build.history"
  if [ ! -f "$BUILD_HISTORY_FILE" ]; then
    printf '%-6s  %-25s  %s\n' "build" "date" "vllm-version" > "$BUILD_HISTORY_FILE"
  fi
  printf '%-6s  %-25s  %s\n' "$NUMBER_TAG" "$(date '+%Y-%m-%d %H:%M:%S %z')" "$VERSION_TAG" >> "$BUILD_HISTORY_FILE"

  echo "Build completed successfully at: $(date)"
  echo ""
  echo "Image tags created:"
  echo "  - vllm-custom:$NUMBER_TAG"
  echo "  - vllm-custom:$VERSION_TAG"
  echo "  - vllm-custom:latest"
  echo ""
  echo "Next steps:"
  echo "  1. Verify image: docker images | grep vllm-custom"
  echo "  2. vllm.sh is pinned to vllm-custom:latest — no tag edit needed"
  echo "  3. Test: cd ~/scripts && ./vllm.sh 1"
else
  echo "Build failed with exit code: $BUILD_EXIT_CODE"
  echo ""
  echo "Troubleshooting:"
  echo "  - Check disk space: df -h"
  echo "  - Check for OOM: dmesg | tail"
  echo "  - Reduce max_jobs: Edit this script, change max_jobs=16 to max_jobs=8"
fi
echo "========================================"
echo ""

exit $BUILD_EXIT_CODE
