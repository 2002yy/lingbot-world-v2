#!/usr/bin/env bash
# =============================================================================
#  setup_sageattention_env.sh
#
#  User-space build environment for SageAttention 2.2 on:
#      Ubuntu 26.04 (glibc 2.43) / WSL2 / RTX 5060 Laptop / sm_120 / torch cu128
#
#  NO SUDO. Everything lands under $HOME/ai. Nothing touches system CUDA.
#
#  WHY THIS SCRIPT EXISTS (read this before "fixing" it)
#  ----------------------------------------------------
#  1. There is no usable wheel.
#     - PyPI `sageattention` is stuck at 1.0.6; 2.x is not published there.
#     - GitHub releases only carry v2.0.1 and it has ZERO assets.
#     So SageAttention must be compiled from source for this exact
#     torch/CUDA/sm combination.
#
#  2. `pip install nvidia-cuda-nvcc-cu12` does NOT give you nvcc.
#     The package name is misleading: it ships only `ptxas` plus headers.
#     The real nvcc has to come from NVIDIA's apt repo, extracted with dpkg-deb.
#
#  3. CUDA 12.9 (not 12.8) is required *because of glibc 2.43*, not because of
#     the GPU. Details in section [C] below.
#
#  4. gcc 15 (the distro default) is rejected AND genuinely incompatible.
#
#  5. Building with the default parallelism OOM-kills the WSL VM.
#
#  Every one of those was hit and diagnosed on this machine; see
#  docs/SAGEATTENTION_BLACKWELL_WSL_BUILD.md for the full evidence.
#
#  USAGE
#      bash scripts/setup_sageattention_env.sh          # install toolchain
#      source scripts/cuda_env.sh                       # activate it
#      bash scripts/build_sageattention.sh              # build the extension
# =============================================================================
set -euo pipefail

AI=$HOME/ai
CUDA_VER=12.9
CUDA_PKG_VER_NVCC=12.9.86-1
CUDA_PKG_VER_CRT=12.9.86-1
CUDA_PKG_VER_CUDART=12.9.79-1
CUDA_PKG_VER_NVVM=12.9.86-1
SAGE_TAG=v2.2.0
SAGE_COMMIT=eb615cf6cf4d221338033340ee2de1c37fbdba4a
REPO_BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64

# the ubuntu2404 repo is used even though we run 26.04: the .debs are pure
# user-space toolchain payloads and the 2404 builds are the closest available.
# (26.04 has no matching NVIDIA apt repo yet.)

export https_proxy=${https_proxy:-http://127.0.0.1:7890}
export http_proxy=${http_proxy:-http://127.0.0.1:7890}

mkdir -p "$AI"
cd /tmp

# -----------------------------------------------------------------------------
# [A] CUDA 12.9 nvcc + headers + libraries, extracted into ~/ai/cuda129
# -----------------------------------------------------------------------------
echo "=== [A] fetching CUDA $CUDA_VER toolchain debs ==="
mkdir -p /tmp/cu_dl && cd /tmp/cu_dl
for spec in \
  "cuda-nvcc-12-9_${CUDA_PKG_VER_NVCC}_amd64.deb" \
  "cuda-crt-12-9_${CUDA_PKG_VER_CRT}_amd64.deb" \
  "cuda-cudart-dev-12-9_${CUDA_PKG_VER_CUDART}_amd64.deb" \
  "cuda-nvvm-12-9_${CUDA_PKG_VER_NVVM}_amd64.deb" \
  ; do
  if [ ! -s "$spec" ]; then
    echo "  downloading $spec"
    for attempt in 1 2 3; do
      curl -sL --max-time 600 -C - -o "$spec" "$REPO_BASE/$spec" && break
      echo "    retry $attempt"
    done
  fi
  dpkg-deb -x "$spec" "$AI/cuda129"
done
CUDA_HOME_129=$AI/cuda129/usr/local/cuda-12.9
test -x "$CUDA_HOME_129/bin/nvcc" || { echo "FATAL: nvcc missing"; exit 1; }
echo "  nvcc -> $CUDA_HOME_129/bin/nvcc"

# -----------------------------------------------------------------------------
# [B] gcc-14 toolchain, extracted into ~/ai/gcc14
#     CUDA 12.9 supports GCC 6-14. The distro default here is gcc 15.2, which
#     nvcc rejects via crt/host_config.h (#error __GNUC__ > 14) and which fails
#     for real if you force it with -allow-unsupported-compiler
#     (libstdc++ 15 headers blow up on __is_pointer etc).
# -----------------------------------------------------------------------------
echo "=== [B] fetching gcc-14 ==="
mkdir -p /tmp/gcc_dl && cd /tmp/gcc_dl
if [ ! -x "$AI/gcc14/usr/bin/gcc-14" ]; then
  PKGS=$(apt-cache depends --recurse --no-recommends --no-suggests \
           --no-conflicts --no-breaks --no-replaces --no-enhances g++-14 2>/dev/null \
         | grep -E '^\s+Depends:' | awk '{print $2}' | grep -v '<' | sort -u)
  # shellcheck disable=SC2086
  apt-get download $PKGS gcc-14 g++-14 libstdc++-14-dev libgcc-14-dev cpp-14 gcc-14-base 2>/dev/null || true
  for f in *.deb; do [ -s "$f" ] && dpkg-deb -x "$f" "$AI/gcc14"; done
fi
test -x "$AI/gcc14/usr/bin/g++-14" || { echo "FATAL: g++-14 missing"; exit 1; }

# nvcc's -ccbin wants a directory containing binaries literally named gcc/g++.
mkdir -p "$AI/gcc14bin"
ln -sf "$AI/gcc14/usr/bin/gcc-14"         "$AI/gcc14bin/gcc"
ln -sf "$AI/gcc14/usr/bin/g++-14"         "$AI/gcc14bin/g++"
ln -sf "$AI/gcc14/usr/bin/gcc-ar-14"      "$AI/gcc14bin/gcc-ar"      2>/dev/null || true
ln -sf "$AI/gcc14/usr/bin/gcc-nm-14"      "$AI/gcc14bin/gcc-nm"      2>/dev/null || true
ln -sf "$AI/gcc14/usr/bin/gcc-ranlib-14"  "$AI/gcc14bin/gcc-ranlib"  2>/dev/null || true
echo "  gcc-14 -> $AI/gcc14bin/gcc"

# -----------------------------------------------------------------------------
# [C] glibc 2.43 workaround  <-- THE SUBTLE ONE
#
#     Symptom:
#       bits/mathcalls.h(83): error: exception specification is incompatible
#       with that of previous function "cospi" (declared at .../crt/math_functions.h)
#       (also sinpi / tanpi / rsqrt / cospif / sinpif)
#
#     Cause: glibc >= 2.41, when _GNU_SOURCE is active, declares the C23 math
#     functions with `noexcept(true)`. CUDA's crt/math_functions.h declares the
#     same names without noexcept. The C++ front end refuses the mismatch.
#
#     Two obvious "fixes", both WRONG:
#       * -D__GLIBC_USE_IEC_60559_FUNCS_EXT_C23=0   -> no effect; the macro is
#         set by bits/libc-header-start.h with #undef + #define, and features.h
#         is include-guarded.
#       * -U_GNU_SOURCE      -> does silence cospi, but strips the POSIX/C23
#         feature set libstdc++ relies on:
#             cwchar: the global scope has no "fwide"
#             c++locale.h: "uselocale" is undefined
#             gthr-default.h: "pthread_mutex_timedlock" is undefined
#         i.e. it trades one error for a cascade.
#
#     What works: shadow bits/mathcalls.h. Ship our own file that forces just
#     the C23 gate off and then #include_next's the real header. Put that
#     directory FIRST on the include path. _GNU_SOURCE stays defined, so
#     libstdc++ is happy, and the conflicting declarations never appear.
# -----------------------------------------------------------------------------
echo "=== [C] installing glibc 2.43 shadow header ==="
mkdir -p "$AI/glibc_shim/bits"
cat > "$AI/glibc_shim/bits/mathcalls.h" <<'SHIM'
/* Shadow of glibc's bits/mathcalls.h.
 *
 * glibc >= 2.41 with _GNU_SOURCE declares the C23 math functions
 * cospi/sinpi/tanpi/rsqrt (and float variants) with noexcept(true); CUDA's
 * crt/math_functions.h declares the same names without noexcept, so C++
 * compilation fails with "exception specification is incompatible".
 *
 * bits/libc-header-start.h does #undef + #define on the gate macro and is
 * itself include-guarded, so it will not clobber us here. Force the C23 gate
 * off, then include the real file untouched.
 *
 * We never call the host versions of these C23 functions from CUDA code.
 */
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#include_next <bits/mathcalls.h>
SHIM
echo "  shim -> $AI/glibc_shim/bits/mathcalls.h"

# -----------------------------------------------------------------------------
# [D] fix dangling libcudart / CUDA libs
#
#     cuda-cudart-dev ships libcudart.so as a symlink to libcudart.so.12, but
#     the actual .so.12 comes from the runtime package, which we do not install
#     from the apt repo. Result: -lcudart fails with "cannot find -lcudart"
#     even though the symlink exists (it dangles).
#     The torch wheel already ships every CUDA lib under site-packages/nvidia,
#     so link those in.
# -----------------------------------------------------------------------------
echo "=== [D] linking CUDA libs from the torch wheel ==="
NVP=$HOME/ai/lingbot-env/lib/python3.12/site-packages/nvidia
mkdir -p "$CUDA_HOME_129/lib64"
for libdir in "$NVP"/*/lib; do
  [ -d "$libdir" ] || continue
  for f in "$libdir"/*.so*; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    [ -e "$CUDA_HOME_129/lib64/$b" ] || ln -sf "$f" "$CUDA_HOME_129/lib64/$b"
  done
done
for base in cudart cublas cublasLt cusparse curand cufft cusolver nvrtc nvjitlink; do
  src=$(ls "$CUDA_HOME_129/lib64/lib$base.so"* 2>/dev/null | head -1)
  if [ -n "$src" ] && [ ! -e "$CUDA_HOME_129/lib64/lib$base.so" ]; then
    ln -sf "$src" "$CUDA_HOME_129/lib64/lib$base.so"
  fi
done

# -----------------------------------------------------------------------------
# [E] SageAttention source, pinned
# -----------------------------------------------------------------------------
echo "=== [E] cloning SageAttention $SAGE_TAG ==="
if [ ! -d "$AI/SageAttention" ]; then
  git clone https://github.com/thu-ml/SageAttention.git "$AI/SageAttention"
fi
cd "$AI/SageAttention"
git fetch --tags --quiet || true
git checkout --quiet "$SAGE_COMMIT"
echo "  pinned at $(git rev-parse HEAD) ($(git describe --tags 2>/dev/null))"

echo
echo "=== toolchain ready ==="
echo "    now run:  source $(dirname "$0")/cuda_env.sh"
echo "              bash $(dirname "$0")/build_sageattention.sh"
