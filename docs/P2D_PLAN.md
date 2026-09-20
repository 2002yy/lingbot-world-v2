# P2d-1a2-r2 — next session's execution plan

Status: **NOT RUN**. Recorded 2026-09-21 so the two problems found in the first
run are not mis-attributed or forgotten.

---

## Why r2 is needed (two independent problems)

### Problem 1 — the `cam` candidate was algebraically wrong

`torch.addcmul(input, t1, t2)` computes `input + t1 * t2`. The reference is

```python
(1.0 + cam_scale) * x + cam_shift
```

so the correct mapping is

```python
scale1 = 1.0 + cam_scale
torch.addcmul(cam_shift, x, scale1)        # cam_shift + x * scale1   == reference
```

but what was actually tested was

```python
torch.addcmul(x, 1.0 + cam_scale, cam_shift)   # = x + (1+cs) * cam_shift
```

which is **not the same expression**. So:

* the `cam` result (27/40, max|diff| 1.953e-03) is **not** evidence about dtype
  promotion or broadcast paths;
* and the fact that `compile` and `addcmul` produced an *identical* error is
  explained by both candidates implementing the same wrong rewrite, not by a
  shared numerical property.

**Do not attribute the cam failure to dtype promotion.** Re-measure first.

### Problem 2 — sampling coverage collapsed onto one forward

```
coverage: chunks=[0] forwards=[0] blocks=30
```

`per_chain=40` was filled by the first forward (30 blocks) plus 10 more, so the
remaining forward kinds and chunk 1 never contributed. The four chains that
reported `40/40 exact` are therefore only validated on **chunk 0 / denoise step
0**, i.e. early denoise on the first chunk.

Until this is fixed those results are NOT eligible to be called repro-safe.

---

## Requirements for r2

### A. Fix the `cam` candidate

```python
scale1 = 1.0 + cam_scale
ref  = scale1 * x
ref  = ref + cam_shift
cand = torch.addcmul(cam_shift, x, scale1)
```

### B. Keep the reference's evaluation boundary intact

Never let the candidate recompute `1 + scale` itself, or the probe silently
acquires a second variable:

```python
scale1 = 1.0 + scale            # computed ONCE, exactly as the model does
ref  = n.float() * scale1 ; ref = ref + shift
cand = torch.addcmul(shift, n.float(), scale1)
```

### C. Stratified sampling

Quota per chain = 40, distributed over chunk x forward x block depth rather than
first-come-first-served:

```
chunk 0 : fwd 0/1/2/3  x  8 blocks  = 32
chunk 1 : fwd 0/1/2/3  x  2 blocks  =  8
```

and the blocks must be spread across network depth, not taken from the front.
Example sets (exact indices are not important, coverage is):

```
8-block set : 0, 4, 8, 12, 17, 21, 25, 29
2-block set : 7, 23
```

Goal coverage: chunk dimension ✓, denoise/KV dimension ✓, depth dimension ✓.

### D. Print every dtype involved

```
x.dtype, scale.dtype, scale1.dtype, shift.dtype, ref.dtype, cand.dtype
```

Only if `cam` still fails after the algebraic fix are we entitled to discuss
dtype promotion / broadcast differences.

### E. Do not spend a full corpus on `torch.compile`

Four correctly-written chains already went 0/40, so compiler fusion is
established as unusable for repro on this path. In r2, `compile` is at most a
small stratified negative control, not a co-equal candidate.

---

## r2 verdict table (target format)

```
chain   exact/total   maxdiff   chunks   fwds       blocks
mod1    40/40         0         0,1      0,1,2,3    distributed
mod2    40/40         0         0,1      0,1,2,3    distributed
resA    40/40         0         0,1      0,1,2,3    distributed
resB    40/40         0         0,1      0,1,2,3    distributed
cam     ?             ?         0,1      0,1,2,3    distributed
```

Only at this point does `40/40 exact` upgrade to **repro-safe candidate**. It is
still only a candidate: a full-network rollout latent-hash check is required
afterwards.

---

## If r2 is clean, do NOT lay down all five chains at once

Layer them so each family's contribution is attributable:

```
P2d-1b1  modulation family only (mod1 + mod2)
           -> local exactness
           -> rollout latent hash exactness
           -> bare whole-chunk latency  (need >= 1%, i.e. ~15.4 ms)
P2d-1b2  + residual family (resA + resB)      -> measure the increment
P2d-1b3  + cam                                -> measure the increment
```

Reason: if the total lands at e.g. -2.4%, we must be able to say whether
modulation contributed 2.1% and cam 0.1%, rather than not knowing. This project's
methodology already depends on "do not optimise the wrong thing", so one more
ablation is worth keeping.

---

## Performance expectation, stated honestly

A rough traffic argument says the modulation chain currently moves ~86 MB per
invocation against ~52 MB with `addcmul`, times 240 invocations per chunk, i.e.
on the order of tens of ms. Treat that only as a **direction**, not a prediction:
cache hits reduce real DRAM traffic, the original kernels are not necessarily
memory-bound, some of the win is launch overhead, `addcmul`'s own
TensorIterator/kernel efficiency is unknown, and neighbouring kernels may
already have implicit dependencies.

The durable statement is: **theoretically it removes one intermediate
materialisation and one kernel launch; the direction is clear, the whole-chunk
gain is decided by measurement.**

---

## Current P2d-1a2 results (first run, for reference only)

```
chain   cand       equal   max|diff|   dtype      a-shape
cam     addcmul    27/40   1.953e-03   float32    (1,1800,1536)   <- WRONG candidate
cam     compile    27/40   1.953e-03   float32    (1,1800,1536)   <- same wrong rewrite
mod1    addcmul    40/40   0.000e+00   bfloat16   (1,1800,1536)
mod1    compile     0/40   4.768e-07   bfloat16   (1,1800,1536)
mod2    addcmul    40/40   0.000e+00   float32    (1,1800,1536)
mod2    compile     0/40   4.768e-07   float32    (1,1800,1536)
resA    addcmul    40/40   0.000e+00   bfloat16   (1,1800,1536)
resA    compile     0/40   3.815e-06   bfloat16   (1,1800,1536)
resB    addcmul    40/40   0.000e+00   float32    (1,1800,1536)
resB    compile     0/40   7.629e-06   float32    (1,1800,1536)

coverage for ALL chains: chunks=[0] forwards=[0] blocks=30   <- insufficient
```

Also note the shapes are `(1, 1800, 1536)` rather than the expected 1881: the
captured `x` in the block is a view after the patchify/rope split, and 1800 came
from the prewarm-shaped dummy pass in the very first capture. Confirm the real
in-block token count in r2 as well, since an off-by-81 shape would mean the
corpus was not the production shape.
