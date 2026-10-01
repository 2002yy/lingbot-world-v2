# Latency-2A: authoritative chunk critical-path decomposition

No optimisation in this stage. The output is a causal account that decides which
route to take next.

## First: an ordering bug, found before any profiling

`t2 ≈ t3` in the measured traces had two possible explanations, and they are
architecturally different:

    DiT -> decode -> commit -> mark_real   (input->commit already contains decode)
    DiT -> commit -> decode -> mark_real   (t3-t2 ~ 0 means decode is very short)

Inspecting `play.py` showed the first, and worse: the order was

    DiT -> KV update -> decode -> mark_real (t3) -> commit (t2)

so **t3 was set BEFORE t2**, violating the correctness rule that an authoritative
real frame must not be marked before its chunk commits. If a commit were refused,
the frame would already have claimed to be the first real frame, and the measured
input->first-real latency would describe a frame whose generation was never
accepted. `test_control_exactly_once` did not cover this, so it was a real gap.

Fixed in both places:

- the runtime now **enforces** the order: `mark_real_decoded` refuses unless the
  frame's chunk has been committed, in the same generation;
- `play.py` now commits **before** decoding, which is also the correct semantic:
  the generation state becomes authoritative when the last state-mutating step
  (the KV update) completes, and the decode is a read-only projection of it.

Four new gates (`test_commit_before_mark.py`): t3 requires a committed chunk; a
refused commit cannot leave t3 set; t2 is not after t3 in the correct order; and a
prewarm-generation frame cannot mark t3.

Full regression after the change: 14 + 11 + 11 + 14 + 4 = **54/54**.

## The account

    chunk wall (p50)               765.6 ms
    ----------------------------------------------------------
    control / reduction             1.5 ms    0.2%
    conditioning / state prep       0.9 ms    0.1%
    DiT step 0                    200.2 ms   26.1%
    DiT step 1                    175.1 ms   22.9%
    DiT step 2                    174.1 ms   22.7%
    KV / state update             174.4 ms   22.8%
    TAE decode                     36.6 ms    4.8%
    post-decode / bookkeeping       0.0 ms    0.0%
    ----------------------------------------------------------
    accounted                     762.8 ms   99.6%
    unattributed                    2.9 ms    0.4%

    n=9, accounted p50 99.9%   (stop condition was >= 95%)

Aggregated: DiT three steps 549.4 ms (71.8%) + KV update 174.4 ms (22.8%) =
**94.6% on the critical path**; TAE decode is 4.8%; the control plane is 0.3%.

## The instrumentation-overhead gate

    profiling OFF:  chunk wall p50 765.2 ms
    profiling ON:   chunk wall p50 765.6 ms
    delta           +0.4 ms (+0.05%)

The instrument is not measuring itself.

## Early-step usability

The decomposition says a cheaper decoder would address 4.8%, so the remaining
question is whether something could be shown sooner. One chunk's step0 / step1 /
final latents were saved and decoded with the same decoder:

    step      vs FINAL ssim   edgeSSIM    mean|d|      std    edgeE
    step0            0.6871     0.6352     0.0603   0.3257   0.1268
    step1            0.7474     0.6922     0.0513   0.3270   0.1308
    step2            1.0000     1.0000     0.0000   0.3269   0.1290

At step0, roughly 200 ms into the chunk, the decoded frame is already SSIM 0.69 and
edgeSSIM 0.64 against the final, with comparable edge energy. That is a
directionally correct, low-fidelity picture rather than noise.

So the available interaction architecture is roughly:

    ~200 ms   preview (after step 0), structurally ~69% aligned with the final
    ~375 ms   preview (after step 1), ~75% aligned
    ~766 ms   authoritative frame

## What the account decides

| where the time goes | measured | what it implies |
|---|---|---|
| DiT steps | 71.8% | fewer steps, early-exit, an interaction-specific coarse step, speculative |
| KV / state update | 22.8% | incremental state, a smaller interaction quantum, async or pipelined |
| decode | 4.8% | warp and preview-decoder work addresses almost nothing |
| chunk boundary / scheduling gap | 0.4% unattributed | no gap to recover |

A warp or low-resolution preview decoder would optimise the 4.8%. The 94.6% is DiT
steps plus the KV update, and the early-step result shows a usable picture is
available at ~200 ms without any of that being made faster.

## Three measurement disciplines, frozen

1. Both exclusive and wall-clock are reported, and `accounting()` reconciles against
   wall time. If overlap ever appears, summing exclusive durations can exceed the
   wall, so route decisions must rest on the critical path rather than on a sum of
   percentages.
2. CUDA time and host time are never mixed. GPU phases use CUDA events; host seams
   use `perf_counter_ns`. They live in different fields and `accounting()` never
   combines them.
3. Instrumentation overhead passes its own gate.

`ChunkPhaseTrace` is deliberately a separate record from `LatencyTraceRecord`:
`LatencyTrace` answers how long the user's event took, `ChunkPhaseTrace` answers
where the chunk spent its time. Two authorities, two schemas -- the same reasoning
that kept `ApplicationClaim` out of the trace record.

## Next

The next stage is therefore not §Warp-1 or §Preview-Decoder-1. It is a fast visual
response that does not wait for the authoritative chunk: a cheap response
immediately on input, with the authoritative world catching up later. The
early-step measurement supplies the evidence that such a response can be
structurally correct rather than a placeholder.
