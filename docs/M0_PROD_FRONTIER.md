# M0-prod: 512x768 is infeasible on 8 GB -- the resolution frontier

## The question

Can 512x768 + bf16 + worst-case KV + real TAE + preview/display/control survive on
an 8 GB card?

## The answer: no, and it fails far earlier than expected

Not "the runtime does not fit" -- the **condition encode OOMs**, with fp8 as well
as bf16, and even at F=5 frames:

    512x768  fsl1536  F5    fp8   FAIL  OOM @ 6680 MiB
    512x768  fsl1536  F249  fp8   FAIL  OOM @ 6680 MiB
    512x768  fsl1536  F249  bf16  FAIL  OOM @ 6538 MiB

The bottleneck is the encoder's **per-block activation** (512x768 has 2.45x the
pixels of 304x528), not the accumulated length and not the weight mode.

## The frontier (streamed condition encode, block=4, F=249, model resident)

    geometry   fsl   weight   result   peak_reserved   driver free
    304x528    627   fp8      OK        5470 MiB       1541 MiB
    304x528    627   bf16     OK        6448 MiB        563 MiB
    320x480    600   fp8      OK        5250 MiB       1761 MiB
    320x480    600   bf16     OK        6346 MiB        665 MiB
    384x576    864   fp8      OK        6494 MiB        517 MiB
    384x576    864   bf16     FAIL      OOM @ 7382
    480x832   1560   fp8      FAIL      OOM @ 6740
    512x768   1536   fp8      FAIL      OOM @ 6680
    512x768   1536   bf16     FAIL      OOM @ 6538

    bf16 feasible upper bound  ~ fsl 627   (304x528 passes, 384x576 fails)
    fp8  feasible upper bound  ~ fsl 864   (384x576 passes, 480x832 fails)

## Final ruling on "resolution authority"

The M1-1.5 conclusion that production is 512x768 still holds -- but it is
**image2video.py's default geometry, i.e. a default aimed at large GPUs.** On this
card:

    512x768 (the production default)  infeasible (condition encode OOMs)
    480x832                           infeasible
    384x576                           fp8 only
    320x480                           both work (665-1761 MiB free)
    304x528 (our benchmark profile)   both work (563-1541 MiB free)

So there are **two authorities**, not one:

1. **Model/upstream authority** = image2video.py's default (native aspect,
   max_area 480*832) -> 512x768. The model's nominal configuration, which cannot
   run on 8 GB.
2. **Deployment authority** = must be explicitly chosen inside the feasible
   frontier. On 8 GB with bf16 the practical bound is fsl ~600-627, i.e. 320x480
   or 304x528.

This **vindicates the existing 304x528 work**: it is not image2video.py's default,
but it *is* a feasible 8 GB deployment geometry. All prior 304x528 numbers remain
valid as an **8 GB deployment profile** -- they simply must not be called
"production numbers at the model's default geometry".

## Code evidence for why the production path must OOM

    wan/image2video.py:831 and :1084
        y = self.vae.encode([
            torch.concat([
                interpolate(img[None].cpu(), size=(h,w), mode='bicubic').transpose(0,1),
                torch.zeros(3, F-1, h, w)
            ], dim=1)
        ])

The production path builds the full zero-padded input in one shot and encodes it
-- exactly the pattern already shown to OOM. `offload_model=True` does not help:
it only calls `empty_cache()` between generation steps and moves the model to CPU
after generation; the DiT is not offloaded during the condition encode, so the
encode peak is unaffected.

## Impact on the planned route

    planned: M0-prod(512x768) -> P2b-prod -> M1-2 -> final gate -> bf16 default
    actual:  M0-prod FAILS at the first step, and not because of bf16 -- the
             geometry itself is infeasible.

So a **deployment-geometry decision must come first**:

    A  304x528 (fsl 627)  all historical data exists; bf16 works but only 563 MiB free
    B  320x480 (fsl 600)  better margin (665 MiB); production_loop's geometry
    C  384x576 (fsl 864)  fp8 only; bf16 infeasible

Note that A and B both have narrow margins (563 / 665 MiB), and that is the
**encode** peak; the runtime peak stacks on top. The two must be evaluated
together -- M0 already measured the runtime peak at 304x528 as 5426 MiB.

## Method

`m0prod_frontier.py` probes one point per process, so allocator state cannot leak
between points, and captures the peak on OOM because a failed point's peak is
exactly what defines the frontier. `m0prod_build_y.py` uses the M1 streamed
encoder to build the condition latent at production geometry, since the
whole-clip form necessarily OOMs at 512x768.

One mistake along the way: `--max_area 409600` was labelled "304x528" but actually
computes 480x832, so the first frontier's "304x528 FAIL" was bogus. `max_area`
must equal the intended profile's W*H.

## Next

1. Decide the deployment geometry (A/B/C) and put it in configuration rather than
   recomputing it per script.
2. Rebuild the production baseline (latency, VRAM, FPS) at that geometry.
3. Then the bf16-vs-fp8 trade-off (P2b-prod).
4. M1-2 becomes about **lowering the encode peak further to buy bf16 margin**,
   not merely about fixing an OOM.
