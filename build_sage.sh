#!/usr/bin/env bash
# Build SageAttention 2.2 for sm_120 using the user-space CUDA 12.9 + gcc-14
# toolchain. No sudo. MAX_JOBS is capped because 32 parallel nvcc processes on a
# 12 GB WSL image OOM-killed the VM (observed: `uptime` reset to 0 min).
set -e
source "$HOME/ai/lingbot-world-v2/cuda_env.sh"

echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC"
echo "MAX_JOBS=$MAX_JOBS"

cd "$HOME/ai/SageAttention"
export MAX_JOBS=${MAX_JOBS:-3}
LOG=${LOG:-$HOME/ai/sage_build5.log}
nohup /home/zhang/ai/lingbot-env/bin/pip install -e . --no-build-isolation --no-deps > "$LOG" 2>&1 &
echo "build started, pid $!, log $LOG"
