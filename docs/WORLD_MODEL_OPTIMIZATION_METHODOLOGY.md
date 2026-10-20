# World-model inference optimisation: methodology and what we actually learned

This records the methodology transferred from
[`kaarelkaarelson/lingbot-world-v2-realtime`](https://github.com/kaarelkaarelson/lingbot-world-v2-realtime)
(a 1.3B world model at 16.1 FPS on one RTX 5090) into this machine
(RTX 5060 Laptop 8 GB, sm_120), and what our own measurements added or
contradicted.

The point is not the patches. It is the loop:

> reproducible baseline → profile → find the largest bucket → estimate the
> ceiling → run a bounded experiment → exactness gate → quality gate →
> long-rollout gate → keep it or close the line.

---

## 1. The reference ladder (verified from the upstream README)

Their optimisation table, in the order they applied it:

| step | before | after | s/chunk |
|---|---|---|---|
| Host syncs | CPU↔GPU sync on every layer | bookkeeping on the GPU | 2.68 → 2.57 |
| **Decoder** | Wan 2.1 VAE in fp32 | fp16 with sub-pixel upsampling | **2.57 → 1.95** |
| Compiler | PyTorch eager | one compiled graph | 1.95 → 1.68 |
| Matmuls | bf16 linears | FP8 rowwise via torchao | 1.68 → 1.47 |
| **Attention** | FlashAttention-2 | SageAttention 2.2 | **1.47 → 1.04** |
| Kernel fusion | one kernel per operation | fused norm / RoPE / residual / FP8 quant | 1.04 → 0.98 |

Totals: 6.0 → 16.1 FPS, 2.68 → 0.98 s/chunk; DiT 1.62 → 0.64 s; decoder
1.06 → 0.34 s; GPU busy 90 % → 98 %; kernel launches ≈20 000 → ≈4 800; host
syncs 110 → 2. Presets: `stock` 2.68/6.0, `exact` 1.07/14.8 (DiT latents
bit-identical), `fast` (default) 0.98/16.1.

### Correction worth keeping

**The single largest step in their ladder is the decoder (−0.62 s/chunk), not
attention (−0.43 s/chunk).** Attention is the largest *remaining* headroom, not
the largest *realised* win. This is easy to misremember and it matters: on our
machine attention is only ~6 % of chunk time, which is consistent with the
decoder being the bigger lever, not with attention being the headline.

### Their roofline, and their decision to stop

| kernel | reached | peak | % |
|---|---|---|---|
| FP8 matmuls | 390 TFLOP/s | 419 | 90 % |
| Decoder convolutions | 173 TFLOP/s | 210 | 83 % |
| Fused elementwise | ~1.3 TB/s | 1.8 | ~70 % |
| SageAttention | 543 TOPS | 838 | 65 % |

Their own conclusion: *"Attention has the most room. A hand written kernel at
90 % of peak would gain about one frame per second, so there is none."*

That is the most transferable lesson in the whole repository — they **quantified
the remaining prize and then declined it**. Knowing when to close a line is part
of the method, not a failure of it.

Numbers I could not verify from the README and therefore do not repeat as fact:
the attention/GEMM/elementwise/memcpy split (51/29/16/3), the "13 graphs /
12 breaks → 1 graph / 0 breaks" figure, and the "1.68 → 1.47" being described as
the largest single win. Those live in `OPTIMIZATIONS.md` sections 13/17 which I
did not read line by line. Treat them as plausible but unverified here.

---

## 2. What we reproduced, and at what scale

| lever | their result | ours (RTX 5060) | comparable? |
|---|---|---|---|
| host syncs | 2.68 → 2.57 (**−4.1 %**) | **−3.1 %, bit-exact** | yes, same order |
| decoder | 2.57 → 1.95 (−24 %) | earlier work, own route (bf16 VAE −37 % time, −860 MB) | yes, same direction |
| compiler | 1.95 → 1.68 (−14 %) | **−8.3 % short horizon, −1.5 % at 65 chunks** | partially |
| FP8 | 1.68 → 1.47 (−13 %) | selective FP8 already in use | yes |
| attention | 1.47 → 1.04 (−29 %) | **−3.98 % (hybrid) / −7.02 % (+compile)** | **no** |
| fusion | 1.04 → 0.98 (−6 %) | not yet attempted | — |

The attention row is where our hardware diverges hardest, and the reason is
structural rather than surprising: their KV window is 18 frames and attention is
a much larger share of their chunk; ours is a window of 6 with attention at
~50 ms of a ~800 ms chunk. **Copying their ordering would have been wrong.**

---

## 3. What our own work added beyond the reference

### 3.1 The attention line produced a mode split, not a patch

Full chain: FA2 → SDPA → Sage → hybrid dispatch → +compile → 65-chunk gate.

```
FA2 + eager           858.7 ms   0%        reproducible
Hybrid + eager        810.6 ms   -5.61%    different trajectory
Hybrid + compile      798.5 ms   -7.02%    different trajectory
```

Zero VRAM delta; attention is only ~49.8 ms/chunk steady state, so this is a
bounded win by construction.

Three findings inside that line are worth keeping independently:

1. **flash-attn 2 was the slowest option at every shape this model uses** —
   SDPA beats it by 22–53 %. A "backwards compatible" default was leaving
   performance on the table for free.
2. **Dispatch beats blanket replacement.** Sage wins only where the KV window is
   long; `hybrid` uses it for ~43 % of calls and beats using it for all of them.
   The right framing is *workload-aware attention backend dispatch*, not
   *install SageAttention*.
3. **Layout cost was zero**, because every call site already passes contiguous
   NHD tensors. This is exactly where a kernel microbench usually dies during
   integration, so it had to be measured rather than assumed.

### 3.2 The result that exceeds the reference

The reference validates at *kernel level / same latent*: four of six steps are
bit-identical and FP8 / attention were checked "on identical inputs", with
quality reported as PSNR 43.6 dB / SSIM 0.981 / LPIPS 0.004 on decoded clips.
That is the strongest claim available at single-inference granularity.

We went further and measured the *rollout*:

```
65 chunks (~16 s of world time), scene 04, seed 42

                    LPIPS    SSIM    edge-SSIM  DINO cos
B (hybrid eager)    0.4701   0.3646   0.2971     0.5176
D (hybrid compile)  0.5207   0.3117   0.2560     0.6521

FA2 vs FA2 (control) PSNR 100.00, LPIPS 0.0000 over all 21 measured chunks
```

**"Approximately lossless in a single inference" is not "lossless across a
causal rollout."** FA2 is bit-for-bit deterministic, yet *any* change to the
attention numerical path is chaotically amplified by the recurrent rollout into
a different — equally plausible — world after roughly 32 chunks. Falling back
from compile to eager does not help; the two are the same order. A
zero-dependency SDPA swap would be expected to behave identically.

This is a world-model-specific principle, and it generalises past attention:

> For a recurrent generator, the correctness unit is the **trajectory**, not the
> **call**. Every optimisation that perturbs the numerical path must be gated on
> rollout length, not on a per-operation similarity metric.

That is why this line ended in an explicit mode split rather than a patch:

```
LINGBOT_MODE=repro  (default)  fa2    + eager     exact trajectory
LINGBOT_MODE=fast              hybrid + compile   -7.02 %, different trajectory
```

Boundary wording is deliberately weak because the evidence is: **~20 chunks
structure preserved / ~32 chunks divergence observed, one seed in one scene.**
Not a universal threshold.

---

## 4. Levers still open on this machine

Ordered by expected value, to be confirmed by a fresh profile rather than by
importing the reference's ordering.

**A. A fresh DiT profile.** Attention is now ~6 %. We do not currently know the
split of the other ~94 % (GEMM / norm / modulation / residual / RoPE / cast /
cache write). Everything below depends on that measurement.

**B. Eliminate repeated work (no numerical change, so it can stay in `repro`).**
The reference identified three instances; these are attractive precisely because
they do not touch the numerical path and can therefore be bit-exact:
- camera-injection MLP recomputed on every denoise step within a chunk
- cross-attention K/V recomputed although the conditioning is constant within a
  generation
- RoPE tables recomputed per step

**C. Finish the sync-free denoising loop.** Our host-sync work (−3.1 %,
bit-exact) covered KV-cache position bookkeeping. The same treatment extends to
GPU-resident timesteps/sigmas, and to removing `.to()` / `nonzero()` from the
inner loop. Grep discipline: `.item()`, `.cpu()`, `.tolist()`, `nonzero()`,
tensor→Python→tensor.

**D. Kernel fusion / Inductor tuning.** Highest ceiling of the remaining items
(their elementwise sits at ~70 % of memory peak), and it does not change the
numerical path if done as fusion rather than reordering. Requires knowing (A).

**E. FP8 rowwise / accumulate mode.** Already partially in use here via
`LINGBOT_FP8`. The open frontier is fp16-accumulate FP8 GEMMs, which the
reference lists as unfinished.

**F. Interaction layer, and the real headline metric.** `input → pixel` latency,
not FPS. That chain is: input sampling → control update → conditioning → DiT →
VAE decode → streaming → present. Chunk length and remaining-generation-time at
the moment of input both matter, so `798.5 ms` is *not* "input to screen".

---

## 5. Lines already closed on this machine (do not reopen without new evidence)

| line | why it is closed |
|---|---|
| CUDA Graph / `reduce-overhead` | CUDAGraph Trees skips capture: the KV cache is a mutated eager input. `cudagraph_support_input_mutation` already defaults to True and only covers mutations **from a prior cudagraph pool**. Forcing capture dies in `_cuda_setCheckpointPoolState` (`Expected curr_block->next == nullptr`). |
| attention threshold sweeping | `hybrid` beats `all-Sage` by 0.3 pp; further tuning is < 1 % of chunk time and risks overfitting one shape/window. |
| Sage KV quantisation, KV ring buffer | reference measured these as neutral or slower; not re-tested here. |
| tiny decoder (TAEHV-class) as a drop-in | known softness / foliage loss; we already have a 3-step streaming decode route. |
| TensorRT | complexity not justified at the measured roofline. |

---

## 6. Reproducibility: what we can and cannot promise

- **`repro` mode** — FA2 + eager, plus the host-sync refactor. Bit-exact and
  deterministic; a seed reproduces the same trajectory.
- **`fast` mode** — faster, but produces a *different* trajectory beyond roughly
  32 chunks. Not a defect of SageAttention or of `torch.compile`; a property of
  the recurrent rollout.
- **Open, and not yet addressed**: length-independent conditioning. The
  reference's TODO flags that clip-level noise / camera-path normalisation can
  make different total rollout lengths fail to reproduce the same opening even
  with an identical seed. That is a world-model *semantics* problem, not a kernel
  problem, and it sits upstream of everything in this document.
