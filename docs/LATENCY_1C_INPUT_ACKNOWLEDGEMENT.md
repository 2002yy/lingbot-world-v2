# Latency-1C: input acknowledgement and the processing watermark

## Why this is 1C and not a renumbering of 1A/1B

A proposal arrived to write this stage up as `Latency-1A` through `Latency-1E`
(identity → ack → lineage → instrumentation → benchmark). `Latency-1A` and
`Latency-1B` already exist, are published, and contain different things:

    Latency-1A   the t0-t5 contract, plus a reachability audit that found 6 of 6
                 GAPs, two of them architecturally unrepresentable
    Latency-1B   the trace record, its terminal statuses, and four frozen semantics

Re-using those numbers would overwrite two public documents. The proposed sequence
also collides in substance, not just in name: most of it is already built.

## Reconciliation: what already existed before this stage

| Proposed | Already in the tree | Where |
|---|---|---|
| `InputEvent(event_id, kind, payload, t0)` | **exists** | `InputEvent` |
| t0–t5 chain | **frozen** | `docs/LATENCY_1A_CONTRACT_AND_AUDIT.md` |
| assignment `event_id -> target_chunk` | **exists** | `ApplicationClaim`, `record.assigned_chunk` |
| `CommittedChunk(chunk_index, generation_id, applied_event_ids, t2)` | **existed as fields, not as an object** | now `CommittedChunk` |
| `FrameMeta(…, applied_event_ids, frame_kind)` | **exists** | `FrameMeta` |
| state machine | **exists**, plus `rejected`/`stale`/`reset_invalidated` | `LatencyTraceRecord` statuses |
| exactly-once: one event in at most one committed application | **enforced and tested** | `test_control_exactly_once.py`, 11 cases |
| processed ack emitted at t2, never at assignment | **already the design** | `test_commit_before_mark.py`, 4 cases |
| prepare → validate → commit | **exists** | `commit()` validates chunk_index, generation_id, applied_event_ids |

So the genuinely new material is small, and this document is mostly about the two
places where the proposal was **wrong** and the one place where it was **right**.

## The two acknowledgements

    InputAcceptedAck(event_id, input_index, t0_accepted_ns)
        The runtime has taken responsibility for this input. It is queued, it has an
        identity, and it will not be silently dropped. It does NOT say the model has
        consumed it.

    InputProcessedAck(event_id, input_index, chunk_index, generation_id, t2_committed_ns)
        THIS INPUT IS IN A COMMITTED MODEL STATE.

`processed_ack()` **raises** `RuntimeStateError` unless the record's status is
`committed` and `t2` is set. It does not return a placeholder and it does not return
`None`. A caller must be structurally incapable of mistaking "queued" or "assigned"
for "processed" — at assignment the GPU may not have consumed the input at all, and
the chunk carrying it can still be aborted.

Both are **views over `LatencyTraceRecord`**, not new stored state. The record stays
the single authority, which is the same rule that keeps derived latency figures out
of the serialised form.

## The watermark, and why there are two of them

    processed_input_index   every input_index up to it reached `committed`
    settled_input_index     no input_index up to it is still pending or in_flight

The proposal defined a single `processed_input_index` as "the contiguous prefix
[0..N] is fully processed". That is the right idea and the wrong single number,
because the prefix property breaks **permanently** on any input that terminates
without being processed:

    aborted          the chunk was abandoned          -> never consumed
    stale            the frontier had already passed  -> never consumed
    rejected         refused before assignment        -> never consumed
    reset_invalidated  cleared by a reset             -> never consumed

After such an input, `processed` can never advance past it again, while `settled`
walks on. Reporting only `processed` would let a clean-looking deadline tag hide a
dropped input. Reporting only `settled` would let a client believe an aborted input
was consumed. Both are therefore exposed, both are monotone, and `processed <=
settled` is asserted.

The prefix property itself holds, and for a specific reason worth recording: a
claim is taken from `_next_free_chunk()` at accept time, which is monotone in accept
order, and `begin_chunk()` assigns every event whose claim equals the frontier. A
lower-index input can therefore never still be queued while a higher-index one has
committed.

The watermark is **session-wide, not per-generation**. A reset does not rewind it:
the inputs it invalidated are terminal, so `settled` keeps moving and `processed`
stops at the last real commit. Rewinding would destroy the one thing the number is
for.

## `input_index` is a property, not a stored counter

The proposal wanted `event_id` (identity, possibly a UUID later) and `input_index`
(strictly monotone ordering) as two stored fields.

Today **they are the same number**, and verifiably so: `_next_event_id` is assigned
once and only ever incremented, and `reset` does not touch it. Adding a second stored
field that must always equal the first is a second authority for one fact — the same
hazard as `delta_applied` (rejected in favour of a materialised candidate) and
`_CAM_EPOCH` (the canonical "bookkeeping bit someone forgets to set").

So `input_index` is a read-only property returning `event_id`, with the migration
condition written down where it matters: it becomes a stored counter, **with a test
that it strictly increases across a reset**, on the day `event_id` stops being
monotone. Until then the API surface exists without the drift, and two tests assert
the monotonicity across a reset that the migration would depend on.

## `CommittedChunk`, and the `FrameMeta` trap

    CommittedChunk(chunk_index, generation_id, applied_event_ids,
                   processed_input_index, settled_input_index, t2_committed_ns)

Appended inside `commit()` from the state it summarises, so it cannot disagree with
the records. `committed_chunk(k)` / `committed_chunks()` read it, and
`frame_lineage(meta)` uses it to answer, for a frame a client is holding:

    strict        event_id in meta.applied_event_ids            exact, always available
    coarse_upto   the chunk's historical processed watermark    cheap, None for prewarm

**A watermark field was put on `FrameMeta` first, and it was wrong.** `FrameMeta` is
created *before* commit — `commit()` validates against it — so any watermark
snapshotted at construction carries the *previous* chunk's value and would answer
"does this frame include my input?" with a confident no. That is why the field is
gone and the per-chunk log exists instead. The trap is recorded in the code, because
it looks correct at a glance.

## The three invariants

    I1  Every accepted input has a stable identity.
        event_id + monotonically increasing input_index, across resets.

    I2  Every committed generation declares exactly what input it consumed.
        chunk_index + generation_id + applied_event_ids + processed_input_index.

    I3  Every displayed frame preserves generation lineage.
        frame -> generation -> chunk -> input.

Any runtime that cannot satisfy these three cannot be an Interactive Deployment
Authority, because without lineage a precise `t3`/`t4`/`t5` cannot say **which
input** it measured. Build causal attribution first, timing second — which is the
ordering the proposal argued for and which the actual work already followed.

## What this does NOT claim

- **`t4` and `t5` remain unavailable.** No renderer-submit seam and no
  present-completion signal exist in the runtime, and a missing time is still missing:
  no proxy is written into either field. `accept_to_renderer_ms` and
  `accept_to_present_ms` stay `None`.
- **`warp` frames do not exist.** There is no warp path in this tree. `frame_kind` is
  a free-form string and accepts one, but nothing produces it, and it is not declared
  here.
- **`SUPERSEDED` does not exist.** An input arriving while a chunk is in flight is
  claimed to the *next* chunk; nothing supersedes or preempts anything. A supersede
  transition only becomes meaningful with §Latency-3B, which is not started.
- **The demo's `flip()` is not a present.** §Latency-1A declared t4/t5 architecturally
  unrepresentable *for the runtime*, and that stands. `demo_wasd.py` does now own a
  real window, so a submit-shaped timestamp is arguably available there — but
  `pygame.display.flip()` returns after queuing or swapping a buffer, it is not a
  present-completion acknowledgement, and the recorded take runs on SDL's dummy
  driver where it presents nothing at all. Calling it t5 would be exactly the proxy
  the earlier stages spent effort undoing.

## Verification

    test_input_acknowledgement.py                15/15
    existing suites, unchanged                   54/54
      test_interactive_runtime 14, test_control_exactly_once 11,
      test_stale_frontier 11, test_prewarm_and_reference 14, test_commit_before_mark 4

The new suite is **mutation-tested**, because a suite that passes on its first run
has not been shown to have teeth. Three deliberate defects were injected into
`interactive_runtime.py` and each one was caught:

    guard removed from processed_ack()          10/15   (5 failures)
    watermark not refreshed in commit()          7/15   (8 failures)
    watermark not refreshed on stale             14/15  (1 failure)

and the file was then restored byte-identical and re-ran 15/15.
