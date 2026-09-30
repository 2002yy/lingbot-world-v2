# M1-2: streamed condition encode wired into the real generate()

## Scope, deliberately narrow

Not done: a session VAE manager, a vLLM-Omni runtime, background encode, per-tick
encode, multi-session eviction, or any new encoder state abstraction.

Two edits only:

1. `Wan2_1_VAE.encode_streamed(img, F, h, w, block=4)` in
   `wan/modules/vae2_1.py`. Mirrors `encode()`'s own block boundaries (frame 0
   alone, then `block` frames at a time) but builds each block on the fly, so the
   full "real frame plus hundreds of zero frames" input never exists. It calls
   `clear_cache()` on entry, so its state is independent of the caller and can be
   poisoned neither by a previous whole-clip encode nor by prewarm.

2. `WanI2VCausal._condition_latent(img, F, h, w)` in `wan/image2video.py`,
   dispatched by `LINGBOT_STREAM_ENCODE` (default 0, so updating the tree cannot
   change generated worlds). It replaces the two identical
   `y = self.vae.encode([...])` sites in `_generate_causal_fast` and
   `_generate_causal_pretrain`.

## GATE 1, bit-exact: PASS

    81 frames:
      condition y hash  7741be09201d5d6d   (whole == streamed)
      final output hash 4a718b0e1f0a3e5f   (whole == streamed)
      y shape [16, 21, 66, 38]

    249 frames:
      condition y hash  84f02569e1937a1e   (whole == streamed)
      y shape [16, 63, 66, 38]

"Visually similar" was explicitly not accepted. Both the condition latent and the
final decoded output are hash-identical.

## GATE 2/3, and an important correction: M1's 1392 MiB was a 777-frame figure

At the production length, 81 frames (generate's own default):

    whole:    peak max 7162 MiB   min free 1061   warm p50 26.92 s
    streamed: peak max 7044 MiB   min free  941   warm p50 27.58 s

So the peak saves only ~118 MiB, and the latency difference is inside the noise
band (min 26.32-26.39, max 37.37-41.65). Note that min free is marginally *worse*
with streaming (941 vs 1061), plausibly because many small allocations nudge
`reserved` up.

The reason: M1's 1392 MiB saving was measured at 777 frames (65 chunks), which is
my stress length, not the production default. At 81 frames the whole-clip input is
only 78 MB, so there is little to save. **M1-2's benefit scales with request
length; it is not a constant.**

## Where M1-2 actually pays: long horizons

    frames=249 (21 chunks):
      whole 1/2 ok   | streamed 2/2 ok   | y hash identical
    frames=501 (~41 chunks):
      whole    [0] FAIL OOM @7220 MiB (free 0)
      streamed [0] ok @6678 MiB, [1] FAIL
    frames=777 (65 chunks):
      whole    [0] FAIL OOM @7280 MiB
      streamed [0] ok @6680 MiB, [1] FAIL

At 501 and 777 frames the whole-clip path cannot complete even the FIRST request
while the streamed path can. That is the difference between unusable and usable,
not a percentage. The second request fails for both at those lengths because the
generation itself exceeds 8 GB at 304x528 -- unrelated to the condition encode.

## GATE 4, control-to-real: not yet measured

The encode sits in request prepare, not on the camera/control hot path, so it
should not affect control-to-real. But per the discipline it must be measured, and
`m0prodA_prodchain.py` is not the interactive loop. Outstanding: an interactive
warm control-to-real p50/p95 comparison before and after.

## Verdict

    GATE 1 bit-exact            PASS (y hash and output hash identical)
    GATE 2 VRAM at 81 frames    +118 MiB peak (modest); min free marginally lower
    GATE 3 latency at 81 frames neutral (within noise)
    GATE 4 control-to-real      not measured

M1-2 is correctness-neutral, very low cost, and its benefit scales with length. At
the production default it is modest; at long horizons it is the boundary between
OOM and runnable. Recommendation: keep it opt-in until GATE 4 is done. The case
against flipping the default now is that the 81-frame gain is modest and min free
dips slightly; the case for it is that it removes a length-dependent cliff at no
bit-exactness risk.

## The larger lesson of this round: three denominators, permanently separated

    micro / kernel benchmark
            |
    DiT + TAE benchmark profile
            |
    full deployment request

For P2b specifically:

    ~22.8%      benchmark / harness layer
    ~29.8%      production DiT layer
    ~5.6-6.2%   full-request, user-visible layer

All three are true; they simply have different denominators. Optimisation results
now settle against what the user actually waits for, rather than against a local
kernel's flattering percentage.
