#!/usr/bin/env bash
# User-space CUDA 12.9 + gcc-14 toolchain for building SageAttention on
# Ubuntu 26.04 (glibc 2.43) / sm_120 / RTX 5060. No sudo, no system CUDA.
#
# Key findings baked in:
#   * nvidia-cuda-nvcc-cu12 pip package contains ONLY ptxas, not nvcc
#   * real nvcc comes from the ubuntu2404 .deb repo, extracted with dpkg-deb
#   * glibc 2.43 declares cospi/sinpi/rsqrt noexcept -> clashes with CUDA headers
#     -> fix is `-U_GNU_SOURCE` (verified: builds and runs sm_120a)
#   * gcc-15 is rejected (host_config.h #error) and genuinely incompatible;
#     gcc-14 extracted from resolute/universe via apt-get download
export LINGBOT_GCC14=$HOME/ai/gcc14
export LINGBOT_GCCBIN=$HOME/ai/gcc14bin
export CUDA_HOME=$HOME/ai/cuda129/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$LINGBOT_GCCBIN:$PATH

export CC=$LINGBOT_GCCBIN/gcc
export CXX=$LINGBOT_GCCBIN/g++

export CPLUS_INCLUDE_PATH=$LINGBOT_GCC14/usr/include/c++/14:$LINGBOT_GCC14/usr/include/x86_64-linux-gnu/c++/14:$LINGBOT_GCC14/usr/include/c++/14/backward
export LIBRARY_PATH=$LINGBOT_GCC14/usr/lib/x86_64-linux-gnu:$LINGBOT_GCC14/usr/lib/gcc/x86_64-linux-gnu/14
export LD_LIBRARY_PATH=$LINGBOT_GCC14/usr/lib/x86_64-linux-gnu:$LINGBOT_GCC14/usr/lib/gcc/x86_64-linux-gnu/14:$LD_LIBRARY_PATH

# only our arch
export TORCH_CUDA_ARCH_LIST="12.0"

# glibc 2.43 (Ubuntu 26.04) declares the C23 math functions cospi/sinpi/rsqrt
# under _GNU_SOURCE; CUDA's crt/math_functions.h declares the same names without
# noexcept -> "exception specification is incompatible with that of previous
# function". `-U_GNU_SOURCE` also cures it but strips POSIX/C23 features that
# libstdc++ needs (fwide/uselocale/pthread_* become undefined), so instead we
# shadow bits/mathcalls.h and force only the C23 gate off. Verified working:
# sm_120a binary builds and runs.
export LINGBOT_GLIBC_SHIM=$HOME/ai/glibc_shim
export NVCC_PREPEND_FLAGS="-std=c++17 -I$LINGBOT_GLIBC_SHIM"

# pip-shipped CUDA headers/libs that the extracted .deb set does not cover.
# Rather than adding them one at a time (cuda_fp16.h -> nv/target,
# ATen/CUDAContextLight.h -> cusparse.h + cublas_v2.h, ...) glob every
# nvidia/*/include and nvidia/*/lib the torch wheel already ships.
NVP=$HOME/ai/lingbot-env/lib/python3.12/site-packages/nvidia
NVINC=""
NVLIB=""
for d in "$NVP"/*/include; do [ -d "$d" ] && NVINC="$NVINC:$d"; done
for d in "$NVP"/*/lib; do [ -d "$d" ] && NVLIB="$NVLIB:$d"; done
# NOTE: the glibc shim must come FIRST so our bits/mathcalls.h wins.
export CPATH="$LINGBOT_GLIBC_SHIM:$NVINC:$CUDA_HOME/include"
export CPATH=${CPATH#:}
INC_FLAGS=""
for d in "$NVP"/*/include; do [ -d "$d" ] && INC_FLAGS="$INC_FLAGS -I$d"; done
export NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS $INC_FLAGS"
export LIBRARY_PATH="$NVLIB:$LIBRARY_PATH"
export LIBRARY_PATH=${LIBRARY_PATH#:}
export LD_LIBRARY_PATH="$NVLIB:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH#:}

echo "[env] CUDA_HOME=$CUDA_HOME"
echo "[env] CC=$CC"
echo "[env] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "[env] NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS"
nvcc --version | tail -2
$CXX --version | head -1
