# Resolution authority audit: production is 512x768, not 304x528

## Why this gate existed

M1-2 would wire the streamed encode into the production prepare phase, and
resolution affects latent HxW, token count, the condition `y` shape, KV and
attention workload, encode peak, DiT latency and TAE latency -- and therefore the
comparability of every earlier number (1208 ms, 1566 ms, 3.1 FPS-equiv). So the
authority had to be established from real execution evidence, not from formula
inference.

## Evidence, captured from a real pipe.generate()

    native image        5472x3648 (WxH)   aspect h/w = 0.666667
    intrinsics          fx=415.53 fy=415.69 cx=415.78 cy=239.78
                        (documented for 480x832 = 832x480 WxH)

    vae.encoder  in=[1, 3, 1, 512, 768]  out=[1, 32, 1, 64, 96]
                                        ^^^^^^^^^^^^^^^^^^ pixel 512x768
                                        ^^^^^^^^^^ latent 64x96

The run then OOM'd at peak_alloc 6272 MiB, which by itself says the production
geometry is large.

## Three geometries exist in the tree

    image2video.py (the real production path)
        lat 64x96  pixel 512x768  frame_seqlen 1536  M(cs=3) 4608
        aspect = native h/w, max_area defaults to 480*832 = 399360

    production_loop.py
        lat 40x60  pixel 320x480  frame_seqlen  600  M(cs=3) 1800
        same formula but max_area = --area = 512*320 = 163840
        NOTE: it calls itself "production pipeline" while NOT using the
        production geometry

    pb3_run / M0 / M1 / P2b / S3 harnesses
        lat 38x66  pixel 304x528  frame_seqlen  627  M(cs=3) 1881
        forces the 480/832 training aspect with area 512*320

## Verdict: case 2, and worse than case 3

Production model input is **512x768 (lat 64x96, frame_seqlen 1536, M(cs=3)
4608)**.

Every performance number so far -- 1208 ms, 1566 ms, 3.1 FPS-equiv, the M=1881
GEMM analysis, the M0 5426 MiB memory gate, the M1 6508 MiB encode gate -- was
measured on the **304x528 benchmark profile**, not on the production geometry.

Those numbers remain internally valid and reproducible, but they must be
**relabelled as a benchmark profile** and must not be extrapolated to 512x768.

This also completes the "1800 tokens" explanation: 1800 is exactly
production_loop.py's geometry (lat 40x60 -> 3*20*30 = 1800). It was not a random
number but a third real geometry. Resolution drift is therefore not a theoretical
risk; it has already happened twice.

## What survives and what must be re-measured

Still valid, because the properties are geometry-independent:

- FP8 weight-only is a capacity-for-speed trade, not a speed optimisation. The
  dequant is a per-call cost and the direction of the conclusion does not depend
  on M.
- addcmul fusion is repro-safe but performance-neutral (a per-op property).
- The RoPE microbenchmark's device lesson (methodology).
- bf16 vs bf16 determinism establishes the new reference mode.
- The condition encode is bit-exact when chunked (encoder internals are
  independent of resolution).

Must be re-measured, and may be substantially worse:

- **M0 memory gate.** 512x768 has 2.45x the tokens of 304x528 (1536 vs 627), so
  KV, attention and activations all scale up. bf16 defaulting is NOT yet
  validated at production geometry and may even OOM.
- P2b's absolute FFN-down savings (GEMM efficiency differs at M=4608 vs 1881).
- All latency and throughput figures.

## Actions

1. Lock production authority to image2video.py's geometry (native aspect,
   max_area = 480*832).
2. Fix production_loop.py's misleading naming/geometry: it calls itself the
   production pipeline while using a different geometry, and it will keep
   producing mislabelled "production" numbers until that is addressed.
3. Re-run at 512x768: (a) the M0 memory gate with bf16 and the worst-case KV
   window, to decide whether bf16 can actually be the default; (b) the P2b
   FP8-vs-bf16 comparison, to confirm the direction still holds at the new M.
4. Only then M1-2, one of whose gates is "bf16 full production peak stays safe"
   -- and "production" has only now been defined correctly.

## Method

`m1_1_5_resolution_audit.py` drives the real `pipe.generate()` and captures
tensors with forward hooks rather than inferring from formulas, and lists the
three candidate geometries side by side for contrast. Three fixes were needed
along the way: `conv1` lives on `vae.model`; the `generate` signature differs
from what was assumed; and `action_path` is a directory, not a file.
