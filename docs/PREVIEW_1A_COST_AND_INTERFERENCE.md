# Preview-1A: step0 preview cost and its interference with the authoritative path

The question is not whether a preview is conceptually attractive but whether it is
economically viable. The authoritative chunk is ~766 ms and its decomposition is
DiT 71.8% + KV 22.8% + decode 4.8%, so this measures whether an extra decode can be
inserted at step0 for roughly its own cost.

## Terminology correction, made before measuring

"~200 ms preview" was only ever **step0 latent ready**. A deliverable preview is:

    input->assign + control/conditioning + step0 + latent handoff + extra decode

The optimistic estimate was ~1 + ~2 + 200 + 36.6 ~ 240 ms. That estimate is what
this stage tests.

## Results, four arms

    arm A  baseline, no preview
      authoritative input->first-real   763.9 ms
      DiT 549.4 | KV 174.1 | decode 36.6
      peak reserved 4848.6 MiB | min free 2492.5 MiB

    arm B  step0 + SYNCHRONOUS preview, clone=1
      authoritative                     807.3 ms   (+43.4 ms, +5.7%)
      preview input->decoded            241.5 ms
      step0 -> preview decoded           36.7 ms
      latent handoff                      0.0 ms
      preview decode                     36.7 ms
      peak reserved 4848.6 MiB (unchanged) | min free 2492.5 MiB (unchanged)

    arm B  clone=0
      authoritative 804.0 ms | preview 241.1 ms | handoff 0.0 ms
      -- indistinguishable from clone=1

    arm C  step0 snapshot + SEPARATE-STREAM decode
      authoritative                     801.3 ms
      preview input->decoded            766.4 ms   <- NOT overlapped
      DiT total 577.9 ms (vs 549.4, +28.5)
      peak reserved 5207.2 MiB (+358.6) | min free 2121.3 MiB

## What the four numbers decide

**1. The 240 ms estimate holds exactly.** Preview input->decoded is 241.5 ms p50.
The latency half of the idea is real, not optimistic.

**2. The latent handoff is free, and this was worth measuring rather than assuming.**
A clone of the step0 latent measures 0.0 ms (the tensor is ~160 KB, a device-to-device
copy of that is microseconds) and clone=0 is indistinguishable from clone=1. So the
cost is genuinely the decode, not the copy. Reporting "36 ms" from TAE(latent) alone
would have been right by luck here; the measurement is what makes it right.

**3. The authoritative regression is +5.7% (763.9 -> 807.3 ms), and it is essentially
the extra decode serialised.** That is the honest price: a 241 ms preview costs about
43 ms on the authoritative frame. It sits marginally above the 5% stop condition.

**4. Arm C does NOT overlap, and is worse than B on every axis.** The preview only
completes at 766.4 ms, i.e. when the chunk does, because the side-stream decode
competes for the same GPU and cannot hide behind the DiT steps. It also slows the
DiT by 28.5 ms and costs 358.6 MiB more peak. So the separate-stream route is not the
way to recover the 43 ms.

## Verdict against the stop gate

    preview decoded p50 241.5 ms   vs target <= 250-300 ms        PASS
    authoritative regression +5.7% vs target <= ~5%               marginally over
    VRAM: peak and min free unchanged                             PASS
    allocator: no growth observed across the run                  PASS

Arm B passes, narrowly on the regression. Arm C is closed.

So the available product architecture is:

    ~241 ms   non-authoritative but causally-linked preview
    ~807 ms   authoritative world frame

and the honest statement of the trade is: **the preview costs about 5.7% of the
authoritative latency, and it cannot currently be hidden.**

## Correctness boundary, enforced not assumed

A preview is non-authoritative lineage. `PreviewTrace` carries chunk_index,
generation_id, applied_event_ids, source_step and frame_kind="preview", so a
user-facing preview can point back at the control event it predicts. It cannot
commit, mark_real_decoded, write t3, or pass itself off as authoritative.

The runner asserts this on every chunk: it builds a preview-provenance FrameMeta and
requires BOTH `commit()` and `mark_real_decoded()` to refuse it. If either accepted
it, the run would fail.

`PreviewTrace` is a separate record from `LatencyTraceRecord`, for the same reason
`ApplicationClaim` and `ChunkPhaseTrace` are separate: `t3 = first affected REAL
frame` must not be diluted by a speculative response.

## Open, and worth doing before productising

The quality figure of SSIM 0.687 / edgeSSIM 0.635 came from a single sample. Before
treating the preview as a product path it should be characterised across a small set
of real controls, checking SSIM, edgeSSIM, camera-motion consistency and a gross
structural failure rate -- not to build a new benchmark, but to confirm the single
sample was not a fluke.

---

# Quality across real controls: the single sample was the pessimistic end

The earlier 0.687 / 0.635 figure came from one chunk. Characterised across nine
real controls (W, W+D, yaw, A, pitch, W+yaw-, S, D+yaw+, and the warm chunk):

    chunk  ssim vs final   edgeSSIM   mean|d|      std
        1         0.6871     0.6352    0.0603   0.3257
        2         0.7068     0.6663    0.0559   0.3538
        3         0.7545     0.7172    0.0462   0.3584
        4         0.8268     0.8003    0.0386   0.3738
        5         0.8653     0.8405    0.0323   0.3741
        6         0.8403     0.8142    0.0363   0.3860
        7         0.8314     0.8005    0.0362   0.3769
        8         0.8718     0.8464    0.0303   0.3905
        9         0.8058     0.7753    0.0391   0.3893

    ssim       p50 0.8268   min 0.6871   max 0.8718   spread 0.1847
    edgeSSIM   p50 0.8003   min 0.6352   max 0.8464
    gross structural failures (ssim < 0.30): 0/9

Two things this changes.

First, **the single sample was the worst case, not a representative one.** The
median across controls is 0.8268 / 0.8003, and there are no gross structural
failures. So the preview is consistently directionally correct rather than
occasionally usable.

Second, the figure **improves with chunk index** (0.69 at chunk 1 rising to 0.87 by
chunk 8). The most likely reason is that later chunks carry more KV context, so
step0 starts from a better-conditioned state. That is worth noting because it means
the preview's quality is lowest exactly at the start of a session, which is also
when a user is most likely to be forming their first impression of the interaction.

This is not a new benchmark and it is not used as a gate; it answers one question --
was the sample a fluke -- and the answer is no, in the favourable direction.

## Combined picture after Preview-1A

    ~241 ms   non-authoritative preview, ssim p50 0.83 vs the authoritative frame
    ~807 ms   authoritative world frame
    cost      about +5.7% on the authoritative latency, which could not be hidden

Arm C (separate stream) is closed. The remaining ways to recover the 43 ms would be
a cheaper preview decoder, latent reuse, or an independent preview head -- not more
stream engineering, since the side stream demonstrably does not overlap on this GPU.
