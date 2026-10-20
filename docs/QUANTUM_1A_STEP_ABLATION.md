# Quantum-1A: DiT step-count ablation -- 2-step FAILS, stop rule applied

## The boundary this respected

"step0 preview already resembles the final frame" is evidence about **image-level**
progressive refinement. It is not evidence that a coarser latent can serve as
**authoritative state**, because the authoritative path also writes that latent into
the KV cache, and errors there can accumulate across chunks into long-horizon
trajectory drift. S3-C is the precedent: short-horizon visual similarity did not imply
long-horizon safety.

So the headline question was never single-frame quality. It was: after dropping a step,
does the KV/state update carry the error into the future?

Only the DiT step count changed. Preview was OFF so its 26.7 ms could not contaminate
the comparison; scheduler, KV implementation and decode were untouched.

Arms, as index subsets of `scheduler.timesteps` (values 999 -> 0):

    A3   3-step, current baseline   [0, 250, 750]  ->  t = 999, 749, 249
    B2   2-step                     [0, 750]       ->  t = 999, 249
    C1   1-step                     [0]            ->  t = 999

## Latency: the reason this was worth testing

    A3   724.1 ms   (warm p50)
    B2   548.8 ms   (-175.3 ms, -24.2%)
    C1   373.6 ms   (-350.5 ms, -48.4%)

## Layer 1: per-chunk visual quality against the 3-step baseline

    B2   ssim p50 0.5842  min 0.4881  |  edgeSSIM p50 0.5365  min 0.4401
    C1   ssim p50 0.3927  min 0.3187  |  edgeSSIM p50 0.3548  min 0.2818

This is not a mild degradation. For scale, the variant D preview reaches roughly 0.86
against the authoritative frame, so 0.58 is well outside "slightly different".

## Layer 2: KV/state direction divergence -- the decisive measurement

```
chunk   dim         B2 cos     B2 relL2    C1 cos     C1 relL2
    0   28892160   0.997905   0.06472    0.995511   0.09476
    1   57784320   0.991220   0.13250    0.988108   0.15422
    2   86676480   0.987644   0.15720    0.979864   0.20070
    3  115568640   0.984891   0.17384    0.971282   0.23970
    4  144460800   0.980840   0.19576    0.959599   0.28431
    5  173352960   0.977369   0.21276    0.951706   0.31085
    6  173352960   0.973928   0.22837    0.939440   0.34811
    7  173352960   0.970551   0.24271    0.927106   0.38191
    8  173352960   0.965459   0.26285    0.911367   0.42109
    9  173352960   0.962479   0.27395    0.903941   0.43837
```

    B2   cos 0.997905 -> 0.962479  (trend -0.0354)   relL2 0.065 -> 0.274
    C1   cos 0.995511 -> 0.903941  (trend -0.0916)   relL2 0.095 -> 0.438

**Both are monotonically divergent and neither plateaus.** Over ten chunks B2's relative
L2 grows 4.2x and C1's 4.6x, and the per-chunk increments do not shrink
(0.016, 0.015, 0.020, 0.012 across the last four for B2), so there is no sign of a
bound being reached.

Note the shape of the earlier norm-only measurement: the KV L2 **norms** were nearly
identical across arms (22368.7 / 22370.9 / 22374.1). That is exactly why norm alone
could not answer the question -- two states can share a magnitude and point in
different directions -- and it is why this measurement exists.

## Verdict: FAIL, stop rule applied

    stop rule: if 2-step shows clear state divergence by chunk 10, do not run 65

B2 shows monotonic, unplateaued KV divergence by chunk 10, so **the 65-chunk run was
not performed**. C1 is strictly worse on every axis and is closed with it.

The failure is precisely the one the boundary anticipated: the coarser latent does not
merely render differently, it poisons the recurrent state, and the error grows with each
chunk rather than settling. Against S3-C's precedent, a monotonic divergence in a
recurrent state at 10 chunks is not something a longer run fixes.

**So the 175 ms was not free, and the authoritative quantum cannot be reduced by
dropping DiT steps.** 2-step would have taken the product timeline from
231 / 790 / 840 ms to roughly 231 / 600 / 650 ms, which is why it was worth testing --
and why the measurement had to be about the state, not the picture.

## What this does NOT close

Step count is one way to reduce the DiT's 549 ms, and the most obvious. It is now
closed. What remains on the same 71.8%:

- **early-exit** per chunk rather than a fixed global reduction, which would let the
  step count depend on how much the chunk is actually changing;
- **a smaller interaction quantum**, which reduces per-chunk work without changing the
  numerical path -- the more honest lever, since it does not touch the schedule;
- **the KV update's 22.8%**, which this stage deliberately did not touch;
- **scheduling and pipelining**, to hide the quantum rather than shrink it.

The one thing this stage rules out is the cheapest-looking option.

## Method note

The first Layer 2 attempt concatenated the whole KV into one vector (173M elements,
347 MB in fp32) and kept several alive at once, which OOM'd. Cosine and relative L2 need
three scalars, so they are now accumulated layer by layer with no large temporary, and
the three arms run in lockstep so each chunk's comparison happens before the next.
