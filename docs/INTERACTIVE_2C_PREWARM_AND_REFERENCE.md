# Interactive-2C: prewarm isolation and reference reproducibility

The last cut of the §Interactive-2 correctness line. No performance work.

## Part 1 — prewarm isolation (gates 1-7)

Prewarm may build caches, compile kernels, allocate, and run a dummy forward. It
may not advance the committed camera, consume a real input event, increment the
authoritative chunk counter, emit authoritative real-frame lineage, or consume an
authoritative ID.

### The mechanism: a verified provenance scope

    with rt.prewarm_scope() as pw:
        pw.prewarm_frame_meta("real")

`prewarm_scope()` snapshots an **authoritative fingerprint** on entry and compares
it on exit, normal or exceptional. The fingerprint covers everything prewarm is
forbidden to touch:

    committed chunk_index, generation_id, camera pose bytes, camera velocity bytes,
    camera gate, committed applied_event_ids,
    the queue as (event_id, claim) pairs,
    every record's status and t1/t2/t3/assigned_chunk/first_real_frame_id,
    AND the ID counters _next_event_id and _next_frame_id

The counters are in the fingerprint deliberately. A prewarm that leaves the camera
alone but advances `_next_frame_id` would still poison lineage, and that is exactly
the kind of hidden pollution the user flagged.

`prewarm_frame_meta()` draws from a **separate ID space** (`_next_prewarm_frame_id`)
and stamps `provenance="prewarm"`, `chunk_index=-1`, `generation_id=-1`,
`applied_event_ids=()`. Such a frame can neither `commit()` nor `mark_real_decoded()`
— the runtime refuses non-authoritative provenance outright, from Interactive-2A.

### Gates

    1  prewarm does not advance committed.chunk_index
    2  prewarm does not change committed CameraState (structural bytes, not a
       numeric approximation)
    3  prewarm does not consume the queue: pending ids and claims are unchanged
       and the event is still pending
    4  prewarm produces no authoritative lineage: provenance is "prewarm", no
       applied events, and it can neither commit nor set t3
    5  prewarm writes no t1/t2/t3 and no first_real_frame_id
    6  a FAILED prewarm pollutes nothing -- the partial pass raises, and the
       fingerprint is still identical
    7  prewarm does not consume the authoritative ID space; five prewarm frames
       take ids 1..5 while the next authoritative frame keeps the id it would
       otherwise have had
    7b the verifier is not vacuous: a deliberate mutation inside the scope IS
       caught

## Part 2 — reference reproducibility (gates 8-10)

Reproducibility is compared per authoritative chunk, not on the final pose. Each
step is a full projection:

    ReferenceStep { chunk_index, generation_id, applied_event_ids,
                    camera_pose, camera_v }

A right-then-left pair returns to where it started, so a final-state-only
comparison would pass a broken sequence; a test asserts the intermediate states
actually differ, so the comparison cannot be vacuous.

The script is W, W+D, yaw, A, pitch, and the comparison is over the whole
C0..C5 sequence.

This validates **control and runtime determinism**, not DiT output determinism. GPU
image bit-identity is explicitly not required here.

### Gates, including three perturbations

    8   identical sequence -> identical per-chunk projections
    8b  intermediate states are compared, and are shown to differ
    9   retry injection (two fail_chunk calls mid-sequence) does not change the
        sequence
    10  prewarm injection does not change the sequence
    10b a future event injected during the LAST flight does not alter any
        committed step, and ends still pending, never stale, never consumed
    10c the complement: the same kind of event IS consumed once its frontier
        arrives, so 10b cannot pass merely because the event was dropped

### A test bug worth recording

10b first failed, twice, for reasons that were the test's fault rather than the
runtime's:

- injecting the future event while **no chunk was in flight** gives it a claim on
  the very next chunk, so it is not future at all — it simply joins that chunk;
- after fixing that, it failed again because the event then legitimately joined
  chunk 2 at its frontier, which is correct behaviour.

Both are the runtime behaving correctly against a wrong expectation. 10c was added
specifically so 10b cannot pass by dropping the event.

## Full regression

    test_interactive_runtime         14/14
    test_control_exactly_once        11/11
    test_stale_frontier              11/11
    test_prewarm_and_reference       14/14
    ----------------------------------------
    TOTAL                            50/50

## GPU sanity

`play.py --prewarm_scope 1` on the frozen performance preset:

    ev    t0(s)   ->assign   ->commit     ->real  input->real  chunk  frame
     1  308.783        1.4      715.7      715.7        715.7      2      3  committed
     2  310.989        1.0      766.1      766.1        766.1      5      6  committed
     3  312.525        0.8      770.8      770.8        770.8      7      8  committed

    accept -> first affected real frame: n=3, p50 766 ms
    committed chunk_index 9, generation_id 0 -- identical to the no-prewarm run

    PREWARM SCOPE: authoritative fingerprint unchanged at exit (verified by
    prewarm_scope itself); every committed event still has a first_real_frame_id
    -> no prewarm provenance reached the authoritative lineage

## What is now established

    input has identity
    the claim has a frontier
    the camera effect is exactly-once
    retry does not replay
    staleness fails closed and is not event age
    reset cancellation is not staleness
    prewarm cannot pollute, even partially and even by ID consumption
    per-chunk committed state is reproducible
    frame lineage points all the way back to a real input

§Interactive-2's correctness line can be closed.

## What comes next is a different question

With correctness closed, the remaining interaction problem is the one the
measurements have been pointing at all along: the control plane is milliseconds
while the model chunk is ~750 ms. Making the feel right is about preview, warp,
speculative visual feedback or a smaller interaction quantum -- not about further
correctness work.
