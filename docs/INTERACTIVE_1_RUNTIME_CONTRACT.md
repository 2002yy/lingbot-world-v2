# Interactive-1: minimal runtime contract (event identity, assignment, commit)

## What this is, and what it deliberately is not

The Latency-1A audit found four missing structures behind the t0-t5 GAPs:

    A. no event identity      -> t0/t1 cannot have lineage
    B. no commit seam         -> t2 is unrepresentable
    C. no renderer / present  -> t4/t5 ARCHITECTURALLY unrepresentable
    D. no frame lineage       -> t3 cannot be shown to be affected

This stage supplies A, B and D. It does not supply C, so t4 and t5 remain
unrepresentable and are declared not measurable in the current architecture
rather than approximated.

Nothing here is a session manager, a background worker, an eviction policy or a
new state abstraction beyond the three structures above.

## The three structures

### InputEvent (A)

    event_id    monotonic for the whole session, including across reset(), so an
                id is never reused and lineage can never alias
    t0_ns       the moment the authoritative queue accepted it (t0)
    kind        control | reset | pause | quit
    controls    the payload

### FrameMeta (D)

    frame_id            unique per produced frame
    frame_kind          real | preview | warp
    chunk_index         which chunk produced it
    generation_id       which generation it belongs to
    applied_event_ids   exactly the events that chunk carried

This is what makes t3 meaningful: without lineage, a decode timestamp only proves
that *some* frame decoded, which is the decode-complete proxy the G4-0 audit
demoted.

### LatencyTrace

One record per event with t0..t5, assigned_chunk, generation_id and
first_real_frame_id. Every field is `Optional` and missing fields stay None.
`derived()` returns None for any figure whose endpoint is missing, so a value can
never be inferred from a different clock base -- the failure mode that made
hotswap_loop's "relative elapsed minus script timestamp" meaningless.

Only `control_to_real_display = t5 - t0` may be called measured input-to-display,
and in this architecture it is necessarily None.

## The state split, and why it is structural rather than procedural

    committed state   the authoritative state; only commit() may write it
    in-flight state   a speculative copy for the chunk being generated

`begin_chunk()` copies committed state into a snapshot and hands that out.
`commit()` is the only writer of committed state.

Correctness therefore does not depend on remembering to bump anything. This is the
general form of the `_CAM_EPOCH` poisoning bug, which was hit twice: prewarm's
dummy forward poisoned chunk 0 because state ownership was implicit. Here the
separation is enforced by construction.

## Fail-closed commit

`commit(meta)` refuses unless the returned metadata exactly equals what was
submitted: same chunk_index, same generation_id, same applied_event_ids.

A mismatch means the output cannot be attributed to the events that were
submitted, so committing would attach lineage to the wrong state. On refusal the
committed state is untouched and the chunk remains in flight, so a correct retry
is still possible. This is the rule taken from the vLLM-Omni audit, where the
session commits only when returned metadata equals the submitted tick snapshot.

## Reset semantics

A reset re-anchors `chunk_index` to zero and increments `generation_id`, but event
ids keep increasing. So a post-reset event can never be confused with a pre-reset
one, which is the identity property the vLLM-Omni design also enforces by keeping
event ids monotonic across reset.

## Invariants asserted by the test suite (16 cases, no GPU required)

    event ids monotonic; ids survive reset; unknown kinds rejected
    single assignment: an event belongs to exactly one chunk
    no double begin_chunk
    commit requires an exact match
    a failed commit leaves committed state untouched, and the chunk stays retryable
    abort leaves committed state untouched and the chunk retryable
    committed state is isolated from in-flight mutation
    commit adopts the in-flight snapshot
    t3 requires a real frame, and only the FIRST real frame sets it
    missing timestamps stay None; derived figures are None when either end is None
    note() refuses to be used for a measurable field
    the lineage chain event -> chunk -> frame is mechanically walkable
    an unknown event id in frame metadata is rejected

## What this unblocks

    t0  queue accept      -> now representable (accept() with t0_ns)
    t1  tick assignment   -> now representable (begin_chunk() with t1_ns)
    t2  state commit      -> now representable (commit() with t2_ns, fail-closed)
    t3  affected real     -> now representable (mark_real_decoded() with lineage)
    t4  renderer submit   -> still not measurable: no renderer exists
    t5  physical present  -> still not measurable: no present signal exists

So the next stage can produce a real input -> commit -> first-real-frame chain, and
must honestly report input-to-submit and input-to-present as unavailable rather
than substituting a proxy.
