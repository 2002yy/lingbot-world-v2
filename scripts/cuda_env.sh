#!/usr/bin/env bash
# Activate the user-space build environment for SageAttention on this machine.
#
#   source scripts/cuda_env.sh
#
# Puts nvcc 12.9 and gcc-14 on PATH and configures the include/library search
# paths so a stock `pip install` / cpp_extension build can find CUDA. See
# scripts/setup_sageattention_env.sh for how the pieces get there and why they
# are needed at all.
#
# Everything here is user-space ($HOME/ai). No system CUDA is required or used.

export LINGBOT_GCC14=$HOME/ai/gcc14
export LINGBOT_GCCBIN=$HOME/ai/gcc14bin
export LINGBOT_GLIBC_SHIM=$HOME/ai/glibc_shim

# --- CUDA 12.9 (12.9, not 12.8: see the glibc note below) -------------------
CUDA_HOME=$HOME/ai/cuda129/usr/local/cuda-12.9
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "cuda_env.sh: nvcc not found at $CUDA_HOME/bin/nvcc" >&2
  echo "  run: bash scripts/setup_sageattention_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$LINGBOT_GCCBIN:$PATH

# nvcc uses the host compiler for host code
export CC=$LINGBOT_GCCBIN/gcc
export CXX=$LINGBOT_GCCBIN/g++

# gcc-14 headers/libs (the distro default is gcc 15, which CUDA rejects)
export CPLUS_INCLUDE_PATH=$LINGBOT_GCC14/usr/include/c++/14:$LINGBOT_GCC14/usr/include/x86_64-linux-gnu/c++/14:$LINGBOT_GCC14/usr/include/c++/14/backward
export LIBRARY_PATH=$LINGBOT_GCC14/usr/lib/x86_64-linux-gnu:$LINGBOT_GCC14/usr/lib/gcc/x86_64-linux-gnu/14
export LD_LIBRARY_PATH=$LINGBOT_GCC14/usr/lib/x86_64-linux-gnu:$LINGBOT_GCC14/usr/lib/gcc/x86_64-linux-gnu/14:$LD_LIBRARY_PATH

# --- only build for the GPU we actually have --------------------------------
# Without this, setup.py builds for every arch in SUPPORTED_ARCHS
# (8.0/8.6/8.9/9.0/12.0), which multiplies build time and failure surface.
export TORCH_CUDA_ARCH_LIST="12.0"

# --- flags injected into every nvcc invocation ------------------------------
#   -std=c++17  : torch's headers assume at least C++17
#   -I$SHIM     : the glibc 2.43 shadow dir MUST come first so our
#                 bits/mathcalls.h wins over the system one
export NVCC_PREPEND_FLAGS="-std=c++17 -I$LINGBOT_GLIBC_SHIM"

# --- make CUDA headers/libs visible -----------------------------------------
# The apt .deb set does not cover everything: cuda_fp16.h needs nv/target from
# CCCL, and ATen's CUDAContextLight.h needs cusparse.h and cublas_v2.h. The
# torch wheel already ships all of them under site-packages/nvidia, so glob
# every nvidia/*/include and nvidia/*/lib rather than adding them one by one.
NVP=$HOME/ai/lingbot-env/lib/python3.12/site-packages/nvidia
NVINC=""
NVLIB=""
for d in "$NVP"/*/include; do [ -d "$d" ] && NVINC="$NVINC:$d"; done
for d in "$NVP"/*/lib; do [ -d "$d" ] && NVLIB="$NVLIB:$d"; done

# the shim must precede every real glibc header dir
export CPATH="$LINGBOT_GLIBC_SHIM:$NVINC:$CUDA_HOME/include"
export CPATH=${CPATH#:}

INC_FLAGS=""
for d in "$NVP"/*/include; do [ -d "$d" ] && INC_FLAGS="$INC_FLAGS -I$d"; done
export NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS $INC_FLAGS"

export LIBRARY_PATH="$NVLIB:$LIBRARY_PATH"
export LIBRARY_PATH=${LIBRARY_PATH#:}
export LD_LIBRARY_PATH="$NVLIB:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH#:}

# --- build parallelism ------------------------------------------------------
# The default is `nproc` (20 here). 20 concurrent nvcc processes inside a 13 GB
# WSL VM triggers the OOM killer, which takes the whole VM down (`uptime`
# resets to 0 min). 4 is comfortable; 3 also works.
export MAX_JOBS=${MAX_JOBS:-4}

if [ -n "${LINGBOT_ENV_VERBOSE:-}" ]; then
  echo "[env] CUDA_HOME=$CUDA_HOME"
  echo "[env] CC=$CC"
  echo "[env] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  echo "[env] MAX_JOBS=$MAX_JOBS"
  nvcc --version | tail -2
  "$CXX" --version | head -1
fi
