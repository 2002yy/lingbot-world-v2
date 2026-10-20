# §Latency-3B-A: preemption capability and correctness audit

    STATUS   AUDIT DONE. Contract drafted below. NO cancellation code was written.
    GATE     PASSES WITH CONDITIONS -- see section 9. It is not an unconditional yes.
    NEXT     §Latency-3B-B, a minimal prototype, only if section 9 is accepted as the rule

## 0. Why preemption is the right target, and why the gain is real

`keypress -> renderer submit` is now lineage-bound and measured at p50 815 ms, while
`real decode -> submit` is p50 19 ms. The display side is no longer the bottleneck, so the
remaining experience is dominated by waiting for a chunk that is already running.

Model it. An input arrives at `t0` during chunk N, which started at 0 and ends at C.

    no preemption   N finishes at C, then N+1 (which consumes the input) runs C -> 2C
                    frame at 2C                latency = 2C - t0
    preemption      N is cancelled at t0, N' restarts at t0 consuming the input
                    frame at t0 + C            latency = C

The gain is exactly `C - t0 - (C - C)` = **the work already done on chunk N**, i.e. the
elapsed part `e`. With a uniform phase that is `C/2`. On this machine `C` is 550-900 ms,
so the expected saving is **275-450 ms**, or roughly a third to a half of the current
authoritative figure.

It is a trade, not a free win: the `e` of GPU work already spent is discarded. That is the
right trade for interactivity and the wrong one for throughput, which is why section 9
imposes a bound.

## 1. Q1 -- at which stages may a chunk be cancelled?

The chunk has five stages, and the answer is not uniform.

    S0  before begin_chunk()          nothing bound, nothing mutated         cancellable, trivially
    S1  after begin_chunk(), before the first forward
                                      runtime state bound; NO model state written
                                                                              cancellable, free
    S2  during the denoise steps      the KV cache IS being written           cancellable, see Q2
    S3  after the KV write, before commit()
                                      KV holds this chunk's final values      cancellable, see Q2
    S4  after commit()                authoritative                           NOT cancellable

**S4 is not cancellable by definition**, and this is the hard boundary the whole stage rests
on. Everything before it is recoverable; nothing after it is.

### The correction that makes S2/S3 tractable

The recorded design note says "4 forwards per chunk: 3 denoise + 1 writing the clean x0",
which reads as though only the fourth mutates the cache. **The code says otherwise**, and
the comment in `wan/modules/model_fast.py` is explicit about it:

    reading them used to be `.item()` -- a CPU<->GPU sync on EVERY layer, EVERY one of
    the 4 forwards per chunk

`WanSelfAttention` writes `kv_cache["k"]` and `["v"]` in place, and advances
`global_end_index` / `local_end_index`, on **every** forward that is handed the cache
(`model_fast.py:290-312`). So a chunk's KV slot is dirty from its first forward onward, not
only after the last one.

That sounds like it makes cancellation expensive. It does not, for a reason in section 3.

## 2. Q2 -- what must be discarded, and what does it cost?

Sizes, computed from the frozen deployment geometry and the real config
(`dim 1536`, `num_layers 30`, `latent 38x66`, `fsl 627`, `local_window 6`, bf16):

    kv_size (tokens)        3762
    full self-KV cache     661 MiB      k and v, all 30 layers
    ONE chunk's slot       110 MiB      the unit a naive rollback would restore
    the rolling memmove    ~924 MB of TRAFFIC (462 read + 462 write), not 661 -- the
                           earlier note's "924.5 MB" is traffic, and it reconciles exactly

Four things would have to be discarded:

1. **Runtime**: the `_inflight` snapshot. Its `candidate_camera` was never adopted, so
   dropping it is free. Nothing else in `InteractiveRuntime` changes during generation.
2. **The bound events**: `begin_chunk()` popped them from the queue and set them
   `in_flight`. Cancellation must **return them for re-assignment**, or abort them. This is
   a product decision with a correctness consequence -- see Q4.
3. **Model state**: the chunk's KV slot, plus `global_end_index` and `local_end_index`.
4. **Session locals**: `cur`, `x0`, the preview frame. Garbage-collected. The preview the
   user already saw cannot be unshown, which is allowed (previews are non-authoritative by
   construction) but is a product effect, not a correctness one.

## 3. The finding that decides the stage: cancellation is KV-neutral, so no rollback buffer is needed

A naive reading is that cancelling at S2/S3 requires restoring 110 MiB of KV, and that in
the steady state the rolling eviction has already shifted the cache, so restoring would need
a ~578 MiB pre-image -- unaffordable, since a bf16 run has only about 821 MiB free at peak
on an 8 GB card.

**That is avoidable, and for a structural reason.** Cancelling a chunk means restarting a
chunk with the **same `chunk_index`**, and every KV write is a pure function of
`current_start`:

    current_start = chunk_index * frame_seqlen         identical on the restart
    current_end   = current_start + frame_seqlen       identical

After the cancelled attempt's first forward, `global_end_index == current_end`. On the
restart the eviction guard is

    (current_end > global_end_index) and (num_new_tokens + local_end_index > kv_cache_size)
     ---------- false on the restart ----------

so the roll does **not** fire a second time, and the `else` branch computes

    local_end_index = local_end_index_prev + current_end - global_end_index
                    = local_end_index_prev + 0

which is the same slot the cancelled attempt wrote. The restart therefore **overwrites the
same slot in place** and leaves the eviction-shifted body exactly as the uninterrupted run
would have left it. The destroyed oldest tokens were going to be destroyed by this chunk
anyway.

**Consequence: preemption needs no rollback buffer, and no per-chunk KV snapshot.** The one
thing it must not do is fire a second eviction, and the existing guard already prevents that.

**Status of this finding: analytic, from the code path above.** It is the central thing
§Latency-3B-B must confirm empirically, and until it is confirmed the prototype must assume
it is wrong. Note that `local_end_index_prev` is only equal to the expected value if
cancellation happens at a point where the bookkeeping is consistent, which is why Q3
requires cancellation to be a synchronised point.

## 4. Q3 -- how do late results fail closed?

**The runtime is already fail-closed and needs no new rule.** `commit()` requires an
in-flight chunk and validates `chunk_index`, `generation_id` and `applied_event_ids` against
it. A cancelled chunk has no in-flight snapshot, so any late commit raises. A preview cannot
commit or mark t3 by construction, and `_chunk_log` is appended only inside `commit()`, so a
cancelled chunk leaves no trace in the authoritative record. All of this is already tested.

What is **not** already safe is the GPU. An asynchronous KV write that lands *after*
cancellation would corrupt the slot with no corresponding committed chunk. Therefore:

    cancellation must be a CUDA synchronised point. No cancellation may be requested while
    a forward for that chunk is still in flight on the stream.

That is one `torch.cuda.synchronize()` at the cancellation boundary, which the chunk loop
already performs between phases, so it is a placement constraint rather than a new cost.

## 5. Q4 -- does a preempted input count as settled or as processed?

It depends on what cancellation does with the bound events, and the two options have very
different product meanings.

    REBASE   the cancelled chunk's events are returned and re-assigned to the restarted
             chunk. `reduce_controls` folds a LIST, so the restarted chunk can legitimately
             carry both the old batch and the new input. The event commits normally, so
             `processed` and `settled` both advance, and the user's earlier keypress is
             not lost.

    ABORT    the events go terminal as `aborted`. `settled` advances past them,
             `processed` stops there **permanently**, and the user's keypress does nothing.

**Rebase is the right default**, because aborting makes WASD feel broken under fast input,
which is exactly the case preemption exists to serve. The §Latency-1C closeout already
froze the consequence for the abort path: a permanently stalled `processed_input_index` is
correct behaviour, and a finer gap/range summary is the future addition if operational
reporting wants one. Preemption is the first feature that would actually create the gap at
scale, so that summary moves from "someday" to "if 3B-B rebases, not needed; if it aborts,
required".

One more consequence of rebase: re-assignment runs `begin_chunk()`'s
`r.t1_assign_ns = t1` again, which would **overwrite the original t1** with a later time and
silently lengthen `accept_to_assign`. The contract must decide whether t1 is first-assignment
or last-assignment. First-assignment is the honest one, and preserving it requires
`begin_chunk()` to not clobber a t1 that is already set.

## 6. Q5 -- can a new input claim the current chunk?

**Not under the current claim rule, and this is the sharpest structural constraint found.**

`_next_free_chunk()` returns `committed.chunk_index + 1`, plus one more if a chunk is in
flight. So an input accepted during chunk N claims **N+1**. Cancel chunk N and
`committed.chunk_index` is unchanged, so the frontier is N, the restarted chunk is N, and
the input's claim N+1 does not match. The input is still consumed one chunk later:

    accept during N (claim N+1) -> cancel N -> restart N -> N commits
                               -> frontier N+1 -> input consumed by N+1
    frame at t0 + 2C ···· the same as no preemption at all

So **claiming the current chunk is impossible** if the accept happens while the chunk is in
flight, and this is not a fixable detail -- the claim is immutable by design, and stale
detection depends on that.

**The resolution is an ordering constraint, not a change to the claim mechanism:**

    preemption must CLEAR the in-flight state BEFORE the input is accepted.

Then `_next_free_chunk()` sees no in-flight chunk, returns `committed + 1 = N`, and the
restarted chunk N consumes the input. The claim stays immutable, bound exactly once, and
stale detection is untouched.

This is why the prototype's order is fixed:

    detect a pending input  ->  cancel (synchronise, clear _inflight, return the batch)
                            ->  accept  ->  begin_chunk  ->  generate

and it is also why the detection has to happen *inside* the generation loop rather than only
at the loop top: the worker reaches the top of the loop only when the chunk it is trying to
shorten has already finished.

## 7. Q6 -- does preemption break exactly-once or lineage?

**Not exactly-once, and not lineage, provided section 6's ordering is respected.** The
argument:

    exactly-once   an event appears in at most one COMMITTED chunk. A cancelled chunk
                   produces no commit, so a rebased event still appears in exactly one
                   committed chunk, and an aborted one in none. The existing invariant and
                   its eleven tests are untouched.
    lineage        a cancelled chunk produces no CommittedChunk (the log is appended only in
                   commit) and no committed FrameMeta, so no frame can cite it. The frame
                   that eventually carries the events cites the restarted chunk, correctly.
    generation     cancellation is not a reset and must NOT bump generation_id. The restarted
                   chunk reuses the same index under the same generation, and since only one
                   of the two attempts can commit, CommittedChunk(N, gen) stays unique.

Two real hazards are not correctness bugs but must be recorded:

**Hazard A -- the RNG stream is perturbed.** The per-step noise comes from a *running*
`torch.Generator` (`add_noise(..., generator=self._gen, ...)`), so a cancelled attempt
consumes draws that an uninterrupted run would not. A preempted run is therefore **not
reproducible against a non-preempted reference**, which touches gate 8's reference-sequence
reproducibility. If preemption ships, the per-step noise must become a deterministic
function of `(chunk_index, step)` rather than a draw from a shared generator.

**Hazard B -- session state that is advanced too early.** `self._prev_pose` is advanced
*before* the forwards, and it is what the plucker's relative pose is computed from. A
restart would then compute `inv(chunk_pose) @ chunk_pose = I` and silently lose the camera
motion of that chunk. `_prev_pose` must be advanced only after commit, or be restored on
cancellation. This is a concrete, must-fix item for 3B-B, and it is invisible unless looked
for.

## 8. The cost, and the one hazard that is not about correctness

    gain     ~C/2, i.e. 275-450 ms      (section 0)
    cost     the wasted `e` of GPU work, plus one synchronise, plus (per section 3) no
             KV restore at all
    risk     LIVENESS. If inputs arrive faster than a chunk can complete, every chunk is
             cancelled and no chunk ever commits. The user then gets no frames at all,
             which is strictly worse than waiting. Unbounded preemption is therefore not
             an option.

Any prototype needs an explicit bound: at most one preemption per chunk, or a minimum
completed-fraction before a chunk becomes preemptible, or a rate limit. The audit does not
pick one; it records that one is mandatory.

## 9. The gate, stated as the rule it is

> **Preemption is worth building only if a chunk can be cancelled without breaking the
> committed/in-flight boundary, exactly-once application, the processed/settled semantics,
> or generation lineage. If it cannot, it is not built.**

Verdict, against the evidence above:

    committed/in-flight boundary   PASS  -- commit() already fail-closes; S4 is excluded
    exactly-once                   PASS  -- no commit from a cancelled chunk (section 7)
    processed / settled            PASS with a default: REBASE, not ABORT (section 5)
    generation lineage             PASS  -- no bump, and CommittedChunk(N, gen) stays unique
    claim immutability             PASS, but only with cancel-BEFORE-accept (section 6)
    KV reversibility               PASS, analytically, and only because of the index
                                         argument in section 3 -- needs empirical proof
    synchronisation                REQUIRED: cancellation is a synchronised point (section 4)
    RNG reproducibility            CONDITIONAL: breaks gate 8 unless noise is derived from
                                         (chunk, step) (section 7, hazard A)
    `_prev_pose`                   CONDITIONAL: must move after commit or be restored
                                         (section 7, hazard B)
    liveness                      CONDITIONAL: a bound is mandatory (section 8)

Three of these are conditions that must be met **before** any prototype, because they change
code that already exists and are not part of the cancellation machinery:

    C1  move `_prev_pose` advancement to after commit (or restore it on cancel)
    C2  derive per-step noise from (chunk_index, step) instead of a running generator
    C3  make `begin_chunk()` not clobber an already-set t1, so rebase preserves t1

C2 and C3 touch generation and commit paths that the RC has frozen, which makes them a real
decision rather than bookkeeping.

## 10. What this stage did NOT do

- No cancellation code, no prototype, no timer, no A/B.
- No change to `interactive_runtime.py`, `wan/`, or any test.
- `§Latency-4` remains PARKED and its reopening condition is now written down: a backend or
  capture mechanism that provides **verifiable display-present completion** and can be tied
  to the existing `frame_id` / `generation_id` lineage. Without that, `t5 = None` is the
  correct answer, not a gap to be filled with a proxy.
- `§Runtime-GPU-1` remains a side track, opened only if the ghost occupancy reproduces or
  starts affecting one-click startup. `preflight_gpu()`'s fail-closed refusal already
  protects correctness, and the runtime must never try to kill an unknown GPU context.

## 11. Roadmap as frozen by this audit

    §Latency-1A..1D   CLOSED
    §Latency-3B-A     CLOSED  -- this document: capability + correctness contract
    §Latency-3B-B     NEXT    -- minimal preemption prototype, only after C1..C3
    §Latency-3B-C     then    -- real WASD latency A/B
    §Runtime-GPU-1    side track, only if reproduced
    §Latency-4        PARKED until a real present-complete capability exists
