# Latency-3B-B2: single-rebase preemption prototype

    STATUS   landed. P0 oracle PASS on the GPU at three cut points.
    SCOPE    one preemption per chunk, forward-boundary only, logical cancel (no kernel
             kill), fresh generation_id, cancel-before-accept, RNG restore, same-chunk
             replay, oracle equivalence, raw observed timestamp retained.
    NOT      repeated preemption, arbitrary kernel interruption, warp, t5/present,
             Runtime-GPU-1.

## What "cancel" means here, precisely

**Logical cancellation at a safe boundary.** Python and PyTorch cannot interrupt a CUDA
kernel that has already been launched. What this does instead is stop launching further
forwards, synchronise the work already issued, and then reuse the same KV slot for the
new attempt -- which §Latency-3B-B0 proved is safe with no rollback buffer.

    chunk N attempt A running
      forward boundary  ->  detection (a CPU lock read, no synchronise)
      decision taken    ->  CUDA synchronise        <- the only sync, and only here
      rebase_chunk()    ->  new generation, batch re-claimed, _inflight cleared
      accept(trigger)   ->  claim is N, because nothing is in flight
      restore_chunk_rng()
      begin_chunk()     ->  attempt B, same chunk index, new generation
      replay from forward 1
      commit attempt B

## The one place the spec as written collided with a frozen invariant

The contract says `generation_id: A -> B`. `begin_chunk` partitions the queue by
`claim.key() == (generation_id, chunk_index)`, and a claim's key IS
`(generation_id, target_chunk_index)`. So bumping the generation first would make the
batch's existing claims compare **less** than the new frontier, classify them **stale**,
and drop them -- which is exactly what a rebase must not do.

The generation therefore cannot move without moving the claims, and the claim was
deliberately designed to be bound once and never mutated.

**Resolution: an explicit, separate, recorded operation.** The invariant is restated
rather than relaxed:

    accept()   binds a claim exactly once
    rebase()   rebinds it exactly once, explicitly, and is recorded in rebases()
    nothing else ever rebinds

`rebase_count(event_id)` exposes the number of times an input was carried across a rebase,
and `t1` is **not** rewritten (C3): t1 is when the input was first assigned, which does not
change because the chunk was attempted twice.

## The correctness oracle

**P0 -- Rebase Oracle.** The strongest available gate, and not the obvious one.

    REFERENCE   both inputs known BEFORE chunk N starts -> N generated once, normally
    PREEMPT     N starts with E_old only, E_new arrives at a forward boundary, the
                attempt is cancelled and rebased, N is replayed with both

Comparing against the ordinary no-preemption path would be the wrong oracle, because that
path pushes E_new into chunk N+1 -- a different world by construction, so equality there
would prove nothing.

Result at cut points after forward 1, 2 and 3, in every case:

    x0                      bit-identical
    KV cache                identical, k, v and every index
    cache bookkeeping       identical
    _prev_pose              identical
    generator state         identical at the end
    applied_event_ids       identical, E_old AND E_new carried by the same chunk
    chunk_index             identical
    CommittedChunk lineage  identical
    generation_id           DIFFERS, and must: that is what makes a discarded attempt's
                            artefacts fail a generation check instead of matching

## Latency is not allowed to decide correctness

A preemption's own detection and synchronisation cost is real, and the frozen `t0` would
hide it, because `t0` is deliberately the physical input time. Redefining `t0` to start at
acceptance would corrupt the metric that all of §Latency-1 rests on.

So `t0` is **not** redefined, and the preemption cost is recorded separately, as timing
only: `t_observed_ns`, `t_admitted_ns`, `observed_to_admitted_ms` and `rebase_cost_ms` in
`Worker.preemption_trace`. None of them participates in claims, processed or settled.

    accepted -> real / submit      the existing authority metric
    observed -> real / submit      the same, because t0 is the press
    observed -> admitted           preemption's own admission cost

## The two performance disciplines

**No synchronise on the normal path.** Detection is a short lock read of a CPU flag, and
the branch is not even entered when `preempt_budget == 0`. An uninterrupted run never
synchronises for preemption, so its throughput is untouched. The single synchronise happens
only after the decision to rebase has been taken.

**A hard liveness bound.** `max_preemptions_per_chunk = 1`. Without it, sustained input
could cancel every attempt and the run would produce **no frames at all**, which is worse
than waiting. This is asserted behaviourally: a session that would preempt forever is
driven through the real worker and must still produce a frame, with no chunk restarted
twice.

## Gates

    P0   preempt-replay == all-inputs-known-before-N oracle     PASS  (GPU, 3 cut points)
    P1   a cancelled generation can never commit                PASS
    P2   a cancelled generation can never emit an authoritative frame  PASS
    P3   the triggering input claims the CURRENT chunk          PASS
    P3n  accepting before cancelling would claim N+1            PASS  (negative control)
    P4   exactly-once survives a rebase                         PASS
    P5   processed/settled unchanged, no gap                    PASS
    P5n  the abort path still DOES open a gap                   PASS  (control)
    P6   chunk-start RNG restored; final state == oracle        PASS  (in P0)
    P7   _prev_pose advances once, only at the final commit     PASS
    P8   KV K1-K5 remain true                                   PASS  (B0 re-run)
    P9   one preemption per chunk guarantees progress           PASS
    P10  no-preemption path remains bit-identical               PASS  (B1 reference)

## What this does NOT do

- **Repeated preemption.** One per chunk, by design, until single preemption is proven
  end to end in real use.
- **Kernel interruption.** Not possible, not attempted.
- **After the clean-x0 write.** That is not a boundary; an input arriving then waits for
  the next chunk. After forward 3 the remaining work is one write, so the gain there is
  about a quarter of a chunk rather than most of one -- kept as a boundary anyway, because
  a silent hole in the trigger set is worse than a small gain, and the per-boundary gain is
  measured rather than assumed.
- **Warp frames, t5/present, Runtime-GPU-1.** Unchanged.
- **A latency A/B.** §Latency-3B-C. Correctness first, and B2 deliberately does not let
  latency numbers decide whether the mechanism is right.

## Reproduce

    python test_single_rebase_preemption.py     # P1-P9, no GPU
    python b2_oracle.py                         # P0, needs GPU
