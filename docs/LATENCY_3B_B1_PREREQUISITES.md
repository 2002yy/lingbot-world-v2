# Latency-3B-B1: preemption prerequisites

    STATUS   C1, C3 and C2 landed. NO cancellation, NO preemption, NO state machine.
    SCOPE    exactly the three prerequisites §Latency-3B-A listed, so that if generation
             semantics ever go wrong it cannot be confused with a cancellation bug.
    NEXT     §Latency-3B-B2, the minimal single-preemption prototype.

## C1 -- the reference pose advances only on a successful commit

**It was a real correctness bug.** `WanSession.denoise` set `_prev_pose = chunk_pose` at
the start of generation, and `_prev_pose` is what the next chunk's relative pose is
computed from:

    rel0 = inv(prev_pose) @ chunk_pose

So a chunk that was cancelled or retried had already moved the reference. A replay would
then compute `inv(chunk_pose) @ chunk_pose = I` and **silently lose that chunk's camera
motion** -- not an error, a wrong world.

Frozen semantics:

    begin / generate / fail / rebase   ->  _prev_pose unchanged
    successful authoritative commit    ->  _prev_pose := committed pose

Implemented as `_candidate_pose` set in `denoise` and promoted by `on_commit()`, which the
worker calls immediately after `rt.commit()` returns.

### Output neutrality was measured, not argued

C1 changes when a variable is written. It must not change what is generated, and "it
should not" is not a gate. `b1_reference.py` drives the **real `WanSession.denoise`**
through six chunks with a deterministic control per chunk and digests each chunk's `x0`,
each chunk's KV slot, the final KV and the final pose. Run before the change and after:

    chunk 0  x0_sum  45455075      ==  45455075
    chunk 1  x0_sum   3228633      ==   3228633
    chunk 2  x0_sum 205737102      == 205737102
    chunk 3  x0_sum  12824753      ==  12824753
    chunk 4  x0_sum 204053961      == 204053961
    chunk 5  x0_sum  45145741      ==  45145741
    slot sums, gei/lei, pose sums, final_k, final_v, prev_pose_sum   all identical

Bit-identical, field for field. The audit's rule was: **if C1 changed an uninterrupted
run's output, stop and re-explain, because that would mean the current path depended on the
wrong advancement order.** It did not, so the change is confirmed instrumentation-order
only.

## C3 -- t1 is a write-once assignment authority

`begin_chunk` used to write `r.t1_assign_ns = t1` unconditionally. A rebase re-begins the
same batch, so the second begin would have overwritten the original assignment time, and
`accept_to_assign` would then measure **how many times the chunk was attempted** rather than
when the input was assigned.

    if t1_assigned_ns is None: set it       # first assignment wins
    else:                      preserve it

`assigned_chunk` and `generation_id` still move with the re-bind, because they describe
which chunk actually committed the input, which is a different question from when it was
first assigned.

**This cannot change generation output**: t1 is observation. The gate is the 85/85 unit
suite plus the reference equality above.

## C2 -- reproducible per-step noise, without redefining the schedule

The requirement is that a replayed chunk sees the same per-step noise. The tempting design
is to key the noise on `(seed, epoch, chunk_index, step)`, and the objection to it is real:
a key needs a namespace, or the same chunk under different seeds collides.

**This is not what was done**, and the reason is that the simpler design is also the safer
one:

> **A `torch.Generator`'s state IS the position in the noise stream.** Capturing it at
> chunk start and restoring it reproduces that chunk's draws exactly, and two different
> seeds produce different states, so there is nothing to namespaced -- the state cannot
> collide with itself.

Implemented as `_chunk_rng_state` captured in `denoise` plus `restore_chunk_rng()`. The
uninterrupted path only **reads** the generator, so its sequence is untouched.

That gives the two graded results the audit asked for, and the second one for free:

    C2-A  replay deterministic                      PASS  (state round-trip tests)
    C2-B  no-preemption reference unchanged          PASS  (bit-identical, above)

**Had keyed noise been necessary, the reference would have changed and that would have had
to be declared as a generation-semantics change** -- not disguised as an instrumentation
fix, and not covered by the old RC's numerical reproduction. It was not necessary.

## Gates

    C1  pose commit semantics              PASS
    C3  t1 write-once                      PASS
    C2  replay deterministic               PASS
    no-preemption reference output         PASS, bit-identical
    existing 1A-1D suites                  PASS   85/85
    KV replay proof invariant              re-run, still PASS
    release smoke                          PASS

`test_preemption_prerequisites.py` adds 11 cases: four for C3 (first assignment sets t1; a
rebind preserves it; a new claim gets its own; `accept_to_assign` does not inflate), three
for C1 (advances exactly once; a non-committing chunk leaves it alone; the worker calls
`on_commit` only after a successful commit), and four for C2 (state round-trip; capture
does not perturb the sequence; seeds produce distinct states; the replayed-boundary shape).

## What this does NOT do

- **No cancellation, no preemption, no detection loop, no liveness policy.** Those are
  §Latency-3B-B2, and its first version is limited to **a single preemption per chunk**
  because §Latency-3B-B0 did not cover a chunk being interrupted repeatedly within one
  generation.
- **No change to claims, `_inflight`, or commit.** The rebase in the C3 tests is simulated
  inside the test by un-inlining the flight, because no production path does it yet.
- **The KV proof was not re-derived**, only re-run; §Latency-3B-B0 already established it.
- One geometry and one chunk position, exactly as B0 recorded for its own limits.

## Reproduce

    python test_preemption_prerequisites.py       # C1/C3/C2, no GPU
    python b1_reference.py OUT.json               # the reference digests, needs GPU
    python kv_replay_proof.py                     # B0's proof, needs GPU
