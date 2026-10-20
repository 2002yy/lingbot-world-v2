# Building SageAttention 2.2 for Blackwell (sm_120) on WSL2

A reproducible, **sudo-free** recipe for compiling SageAttention 2.2 on:

```
GPU        NVIDIA GeForce RTX 5060 Laptop (sm_120), driver 591.86, 8151 MiB
OS         Ubuntu 26.04 LTS "resolute" on WSL2, kernel 6.18.33.2-microsoft-standard-WSL2
glibc      2.43-2ubuntu2.4
Python     3.12.14
PyTorch    2.8.0+cu128  (CUDA runtime 12.8)
flash-attn 2.8.3
SageAttn   v2.2.0, commit eb615cf6cf4d221338033340ee2de1c37fbdba4a
nvcc       12.9.86      (user-space, ~/ai/cuda129)
gcc/g++    14.3.0       (user-space, ~/ai/gcc14)
MAX_JOBS   4
```

Scripts: `scripts/setup_sageattention_env.sh` → `scripts/cuda_env.sh` → `scripts/build_sageattention.sh`.
Everything installs under `$HOME/ai`. **No system CUDA, no apt install, no sudo.**

---

## 1. Why you cannot just `pip install`

Two independent reasons, both verified on this machine:

| attempt | result |
|---|---|
| `pip index versions sageattention` | `1.0.6, 1.0.5, … 0.1.0` — **PyPI has only 1.0.6** |
| GitHub releases for `thu-ml/SageAttention` | one release, `v2.0.1`, **0 assets** |

The project README says `pip install sageattention==2.2.0`, but public PyPI does
not carry 2.x. **Compile from source at a pinned commit.**

---

## 2. Why the toolchain must be user-space

The machine has **no nvcc** and **no `/usr/local/cuda*`**. Installing the full
CUDA Toolkit needs sudo and 3–4 GB.

There *is* a pip package that looks like the answer —
`nvidia-cuda-nvcc-cu12` (40 MB). **It is misnamed.**

```
$ pip show -f nvidia-cuda-nvcc-cu12
Files:
  nvidia/cuda_nvcc/bin/ptxas          <-- only ptxas
  nvidia/cuda_nvcc/include/crt/...    <-- headers
```

There is **no `nvcc` binary in it**. Real nvcc comes from NVIDIA's apt repo as a
`.deb`, which `dpkg-deb -x` can unpack into `$HOME` without root. That is what
`setup_sageattention_env.sh` [A] does.

We use the **ubuntu2404** repo because Ubuntu 26.04 has no matching NVIDIA apt
repo yet; the payloads are plain user-space toolchain files and work fine.

---

## 3. Why gcc 14, and why not `-allow-unsupported-compiler`

Distro default is **gcc 15.2.0**. CUDA 12.9 supports **GCC 6–14**.

Attempting nvcc with gcc 15:

```
crt/host_config.h:143:2: error: #error -- unsupported GNU version!
gcc versions later than 14 are not supported! The nvcc flag
'-allow-unsupported-compiler' can be used to override …
```

Forcing it with `-allow-unsupported-compiler` does **not** rescue the build —
libstdc++ 15 headers fail for real:

```
/usr/include/c++/15/type_traits(555): error: identifier "__is_pointer" is undefined
/usr/include/c++/15/type_traits(877): error: identifier "__is_volatile" is undefined
```

So gcc-14 is genuinely required. It comes from `resolute/universe` via
`apt-get download` (which needs no sudo) plus its full dependency closure,
extracted to `~/ai/gcc14`. A shim dir `~/ai/gcc14bin` provides binaries named
exactly `gcc` / `g++`, because nvcc's `-ccbin` expects that.

---

## 4. The glibc 2.43 problem (the subtle one)

### Symptom

```
bits/mathcalls.h(83): error: exception specification is incompatible with that
of previous function "cospi" (declared at line 2601 of
.../targets/x86_64-linux/include/crt/math_functions.h)
```

also for `sinpi`, `tanpi`, `rsqrt`, `cospif`, `sinpif`.

### Cause

glibc ≥ 2.41, when `_GNU_SOURCE` is active, declares the **C23 math functions**
with `noexcept(true)`. CUDA's `crt/math_functions.h` declares the same names
**without** `noexcept`. C++ rejects the mismatch.

Note this is **glibc-version-driven, not CUDA-version-driven**: CUDA **12.8 and
12.9 both fail identically**.

### Two obvious fixes that do NOT work

**✗ `-D__GLIBC_USE_IEC_60559_FUNCS_EXT_C23=0`**

No effect. The macro is defined in `bits/libc-header-start.h` with
`#undef` + `#define`, and `features.h` is include-guarded, so a command-line
`-D` gets overwritten.

**✗ `-U_GNU_SOURCE`**

This *does* silence `cospi` — and then destroys libstdc++:

```
gcc14/usr/include/c++/14/cwchar(148): error: the global scope has no "fwide"
x86_64-linux-gnu/c++/14/bits/c++locale.h(52): error: identifier "uselocale" is undefined
c++/14/bits/gthr-default.h(782): error: identifier "pthread_mutex_timedlock" is undefined
c++/14/mutex(205): error: identifier "clockid_t" is undefined
```

Trading one error for a cascade is not a fix.

### What works: shadow `bits/mathcalls.h`

Ship our own file that forces **only** the C23 gate off, then includes the real
header, and put that directory **first** on the include path:

```c
/* ~/ai/glibc_shim/bits/mathcalls.h */
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#include_next <bits/mathcalls.h>
```

`_GNU_SOURCE` stays defined (libstdc++ is happy) and the conflicting
declarations never appear. `include_next` guarantees we do not fork glibc's
implementation — only the gate.

Verified: an `sm_120a` binary compiles *and runs* with this shim.

---

## 5. Other traps that cost real time

### 5.1 `ninja -j 20` OOM-kills the WSL VM

`nproc` is 20. Twenty concurrent nvcc processes inside a 13 GB WSL image exceed
host memory and the OOM killer takes the whole VM down. The tell is that
`uptime` reports a fresh boot mid-build.

→ always `MAX_JOBS=4` (3 also works). This is exported by `cuda_env.sh`.

### 5.2 Background builds get SIGINT'd when the WSL session exits

```
ninja: build stopped: interrupted by user.        (exit status 130)
```

A plain `nohup ... &` is not enough — the process group still dies with the
parent `wsl -e bash` session. Detach the session:

```bash
setsid nohup <cmd> > log 2>&1 < /dev/null & disown
```

### 5.3 Missing headers: `cublas_v2.h`, `cusparse.h`, `nv/target`

Progressive failures, each one layer deeper:

```
torch/include/ATen/cuda/CUDAContextLight.h:8: fatal error: cusparse.h
torch/include/ATen/cuda/CUDAContextLight.h:9: fatal error: cublas_v2.h
cuda_fp16.h:4492: fatal error: nv/target
```

All three already exist inside the **torch wheel** under
`site-packages/nvidia/*/include`. Rather than chasing them one at a time,
`cuda_env.sh` globs **every** `nvidia/*/include` into both `CPATH` and
`NVCC_PREPEND_FLAGS`, and `nvidia/*/lib` into `LIBRARY_PATH` / `LD_LIBRARY_PATH`.

### 5.4 `cannot find -lcudart` even though `libcudart.so` exists

`cuda-cudart-dev` ships `libcudart.so` as a symlink pointing at
`libcudart.so.12` — but that target comes from the *runtime* package, which the
apt-repo extraction does not include. The symlink is **dangling**, so the linker
reports "cannot find".

→ link the runtime libs out of the torch wheel into `$CUDA_HOME/lib64`
(`setup_sageattention_env.sh` [D]).

---

## 6. Build

```bash
bash scripts/setup_sageattention_env.sh     # one-time
source scripts/cuda_env.sh                  # each shell
bash scripts/build_sageattention.sh         # ~15-25 min
```

Expected tail:

```
Target compute capabilities: {'12.0'}
Successfully built sageattention
Successfully installed sageattention-2.2.0
```

Resulting commit: **eb615cf** (`v2.2.0`), extensions
`_qattn_sm80`, `_qattn_sm89`, `_fused` all load.

`TORCH_CUDA_ARCH_LIST=12.0` matters: without it `setup.py`'s
`SUPPORTED_ARCHS = {"8.0","8.6","8.9","9.0","12.0"}` builds every architecture,
multiplying build time, disk and failure surface for GPUs we do not have.

---

## 7. Runtime configuration

There are TWO modes, and the distinction is not cosmetic — it is the main
finding of this investigation. See `wan/perf_mode.py` for the full evidence.

```
LINGBOT_MODE            repro | fast        (default: repro)
  repro   backend=fa2    compile=0           exact-trajectory reproduction
  fast    backend=hybrid compile=1           ~7% faster, different trajectory

LINGBOT_ATTN_BACKEND    fa2 | sdpa | sage | hybrid   per-field override
LINGBOT_COMPILE         0 | 1                        per-field override
LINGBOT_COMPILE_MODE    default                      Inductor only
LINGBOT_SAGE_MIN_KV     KV length above which hybrid uses Sage (default 2508)
```

### Why the split exists

Long-horizon testing (65 chunks ≈ 16 s of world time, scene 04, seed 42) showed
that the accelerated backends do **not** merely render the same world with
different texture — after roughly 32 chunks they produce a **different world**:

| | LPIPS @48-64 | SSIM | edge-SSIM | DINO cosine |
|---|---|---|---|---|
| B (hybrid, eager) vs FA2 | 0.4701 | 0.3646 | 0.2971 | 0.5176 |
| D (hybrid, compile) vs FA2 | 0.5207 | 0.3117 | 0.2560 | 0.6521 |

Crucially, **disabling compile does not fix this** — B and D are the same order.
The cause is the attention numerical path itself: FA2 is bit-for-bit
deterministic (two identical runs give PSNR 100.00 / LPIPS 0.0000), but the
recurrent rollout amplifies *any* perturbation to it. A zero-dependency SDPA
swap would be expected to behave the same way.

So "faster" and "seed-reproducible" are mutually exclusive here, and the honest
framing is: the fast backends are not wrong, they generate a different but
equally plausible trajectory.

### Observed boundary — NOT a guarantee

```
~20 chunks : structure preserved          (tested scene/seed only)
~32 chunks : structural divergence observed
```

This comes from **one seed in one scene**. It is *not* a universal safe rollout
length and must not be enforced as one without a multi-seed, multi-scene sweep.

### `compile` is Inductor, not CUDA Graph

`LINGBOT_COMPILE_MODE=default` means ordinary Inductor optimisation. It is not
`reduce-overhead` and not CUDA Graph. CUDA Graph is a separate, closed line:
CUDAGraph Trees skips capture because the KV cache is a mutated eager input
(`skipping cudagraphs due to mutated inputs` at `crossattn_cache["k"].copy_(k)`),
`cudagraph_support_input_mutation` already defaults to True and only covers
mutations "from prior cudagraph pool", and forcing capture dies in
`_cuda_setCheckpointPoolState` (`Expected curr_block->next == nullptr`).

### Always-on, no tradeoff: the host-sync refactor

Independent of the above, the KV-cache position bookkeeping
(`global_end_index` / `local_end_index` / `is_init`) is stored as plain Python
ints/bools rather than CUDA tensors. That removes a `.item()` CPU↔GPU sync per
layer per forward. Strict same-process paired measurement: **−3.1%, bit-exact**
(identical latent hash). This carries no reproducibility cost and is retained in
both modes.

### The hybrid dispatch rule

`hybrid` sends each path to its measured winner:

```
long-window self  (Lkv >= 2508) -> SageAttention
cross-attention                 -> SDPA
short self                      -> SDPA
```

Measured per-call median (RTX 5060, bf16; all call sites already hand over
contiguous NHD tensors, so **layout cost is exactly zero**):

| shape | FA2 | SDPA | Sage | best |
|---|---|---|---|---|
| 627x3762 self | 0.8008 | 0.6273 | **0.4729** | Sage |
| 627x3135 | 0.6335 | 0.5249 | **0.4118** | Sage |
| 627x2508 | 0.5615 | 0.4384 | **0.3351** | Sage |
| 627x1881 | 0.4564 | 0.3416 | 0.3740 | SDPA |
| 627x1254 | 0.3821 | **0.2394** | 0.2397 | SDPA |
| 627x627 | 0.2645 | **0.1393** | 0.1497 | SDPA |
| 627x512 cross | 0.2570 | **0.1216** | 0.1455 | SDPA |

Two things worth remembering: **flash-attn 2 is the slowest option at every
shape this model uses** (SDPA beats it by 22–53%), and Sage only wins where the
KV window is long — so running Sage at short KV or on cross-attention is
strictly worse (slower *and* more quantisation). Using Sage for only ~43% of
calls beats using it for all of them.

### Measured end-to-end

```
FA2 + eager           858.7 ms   0%        reproducible
Hybrid + eager        810.6 ms   -5.61%    different trajectory
Hybrid + compile      798.5 ms   -7.02%    different trajectory
```

VRAM delta: **zero** (peak 2978/3114 MiB identical across all arms), and
attention is only ~50 ms/chunk of a ~800 ms chunk, so this is a bounded win.

---

## 8. Upgrading / re-doing this later

Checklist if the environment is rebuilt:

1. `pip index versions sageattention` — is 2.x on PyPI yet? If so, prefer the wheel.
2. Check the GitHub release assets again.
3. Confirm the glibc version: `ldd --version`. **Below 2.41 the `cospi` conflict
   does not exist and the shadow header is unnecessary** (harmless, but check
   whether CUDA 12.8 then suffices).
4. Confirm `gcc --version`. If ≤ 14 the extracted gcc-14 is unnecessary.
5. Confirm torch's CUDA: this recipe targets **cu128 with nvcc 12.9**. A cu13
   torch would want matching nvcc.
6. Re-pin the SageAttention commit.
7. `nvidia-cuda-nvcc-cu12` may have been fixed to actually contain nvcc — check
   `pip show -f` before assuming it is still useless.
