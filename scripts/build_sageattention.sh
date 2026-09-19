#!/usr/bin/env bash
# Build and install SageAttention 2.2 for sm_120 from the pinned source.
#
#   bash scripts/build_sageattention.sh
#
# Requires the toolchain from setup_sageattention_env.sh.
#
# NOTES ON THE CROSS-COMPILE SETUP
#   * --no-build-isolation : we need torch visible during the build, and we do
#     not want pip to invent a fresh (wrong) environment.
#   * --no-deps             : torch is already installed; letting pip resolve
#     deps risks it "helpfully" replacing our CUDA-tuned torch.
#   * setsid + nohup + disown : a plain `nohup ... &` gets SIGINT'd (exit 130,
#     "ninja: build stopped: interrupted by user") when the parent `wsl -e bash`
#     session exits. Detaching the session is required.
#   * MAX_JOBS=4 : the default (nproc=20) OOM-kills the WSL VM.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./cuda_env.sh
source "$HERE/cuda_env.sh"

SAGE_DIR=${SAGE_DIR:-$HOME/ai/SageAttention}
LOG=${LOG:-$HOME/ai/sage_build.log}
VENV_PIP=${VENV_PIP:-$HOME/ai/lingbot-env/bin/pip}

cd "$SAGE_DIR"
echo "[build] dir=$SAGE_DIR commit=$(git rev-parse HEAD)"
echo "[build] nvcc=$($CUDA_HOME/bin/nvcc --version | tail -1)"
echo "[build] MAX_JOBS=$MAX_JOBS arch=$TORCH_CUDA_ARCH_LIST log=$LOG"

setsid nohup "$VENV_PIP" install -e . \
  --no-build-isolation --no-deps > "$LOG" 2>&1 < /dev/null &
echo "[build] started pid $! -> $LOG"
echo "[build] follow with: tail -f $LOG"
