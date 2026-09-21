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


---

# ENVIRONMENT BLOCKER (2026-09-21) — root cause confirmed, waiting on the host fix

## Symptom

Every attempt to build `WanI2VCausal` kills the WSL VM. `uptime` resets to
"up 0 min", the log is left at 0 bytes, and even the first `print` never lands.
Trivial python, all imports and CUDA allocations up to 2 GiB all work; only the
model build dies. It dies mid-build, after "step2: building pipe ...".

## Root cause — Windows Resource Exhaustion Detector, event 2004

```
System log, Microsoft-Windows-Resource-Exhaustion-Detector, Event ID 2004
"Windows successfully diagnosed a low virtual memory condition", largest
consumer listed as vmmemWSL:

  21:14   vmmemWSL  8,107,376,640 B  ~  7.6 GB
  21:16   vmmemWSL  9,220,898,816 B  ~  8.6 GB
  21:37   vmmemWSL  8,261,619,712 B  ~  7.7 GB
  21:41   vmmemWSL  7,324,598,272 B  ~  6.8 GB

32 such events in the last 3 hours.
```

The causal chain is complete:

```
model load produces a transient commit peak in vmmemWSL (7.5-9.2 GB)
        -> host CommitFree falls below what the peak needs
        -> Windows Resource Exhaustion Detector fires (event 2004)
        -> WSL is terminated
```

## Baseline captured BEFORE the fix (use this to verify the fix)

```
Physical total  15.5 GB
Physical free    5.2 GB
CommitLimit     31.5 GB          (= 15.5 physical + 16 pagefile)
CommitFree       6.1 GB
AutoManagedPF    False
PageFile         C:\pagefile.sys   allocated 16 GB, peak 7.7 GB
Event 2004 in last 3h: 32
```

## The fix (host-side, needs admin)

**A — primary, long-term.** Raise the page file. Either turn on "Automatically
manage paging file size for all drives", or set it manually to 32 GB:

```
RAM             ~15.5 GB
pagefile         32 GB
CommitLimit    ~47.5 GB     (+16 GB headroom vs today)
```

Then **reboot Windows**, even though some resizes apply dynamically: this is the
substrate of the performance experiments and it is not worth leaving any doubt
about whether the new page file is fully in effect.

Verify after reboot (all three):

```
PageFile allocated  ~ 32 GB
CommitLimit         ~ 47 GB
CommitFree          clearly above the old 8 GB
```

**B — temporary only.** Closing host apps frees roughly 5-6 GB of commit:
msedge 1.24 GB, OpenCode 0.82, node 0.70, MsMpDef 0.62, FlClash 0.53,
steamwebhelper 0.44, QQ x2 0.65. That would put CommitFree at ~13-14 GB against a
worst observed peak of 9.2 GB, so it is probably enough to finish the current
experiment without rebooting — but it is not stable, because browser/Steam/AV
commit fluctuates. Whether the model can start must not depend on how many Edge
tabs are open today.

**C — do NOT use `.wslconfig memory=10GB` for this.** It caps what WSL may use;
it does not add Windows commit budget. Since the startup phase itself needs close
to 9 GB, a 10 GB cap would simply move the failure from
"Windows kills WSL" to "Linux/Python OOMs". The failure location changes, not the
problem. Demoted from the fix list.

**D — T5 CPU load peak: record only, do not act.**
`t5_cpu=True` keeps umt5-xxl resident on the CPU, and `pipe.text_encoder = None`
immediately after the prompt embedding, so the 8-9 GB is very likely a transient
startup peak. But changing the load dtype now would introduce a fresh question
("are the prompt embeddings bit-identical, does the conditioning change, does the
rollout change") and would contaminate the current P2d bit-exact line. Filed as

```
Startup-memory optimization / T5 CPU load peak
  -> future investigation
  NOT part of P2d
  NOT needed to unblock the current experiment
```

## Methodology lesson to keep

> A failed measurement must not be trusted merely because its result looks
> plausible. A diagnostic probe itself needs execution evidence.

The specific trap here: the memory probe used `Start-Process ... -ArgumentList`
with a command string whose quoting did not survive, so **the target program
never started**. The probe then reported a beautifully clean, very plausible
1.1 GB vmmemWSL figure -- and that figure was used to *refute* the correct
hypothesis.

Resource monitors of this kind should therefore always record:

```
target PID
process start time
command line
exit code / alive state
number of samples taken
```

If there is no evidence the target process existed, the resource data is **not
eligible for attribution**. This is the same discipline as the `repro vs repro`
determinism control: measure that the thing you think you are measuring is
actually running.

## Recovery point

Nothing needs reverting. Last commit before the blocker is `89160de`
(`p2d1a2_r3.py`, 15804 bytes, tracked tree clean). Once the host has commit
headroom, run:

```
wsl -e bash -c "cd ~/ai/lingbot-world-v2 && LINGBOT_FP8=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/ai/lingbot-env/bin/python -u p2d1a2_r3.py --scene 04 --chunks 2"
```

No performance conclusion from before the blocker is invalidated.
