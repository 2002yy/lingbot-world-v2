# Interactive-2B: staleness as an expired application claim

## The definition, tightened

> **stale means the application claim has expired, not that the event is old.**

Nothing may be inferred from event_id ordering, queue residency time or wall-clock
age. An old event whose frontier has not yet arrived is perfectly valid; a
brand-new event whose frontier has already passed is stale.

## The mechanism

At `accept()`, a control event is bound **once** to an immutable claim:

    ApplicationClaim = { generation_id, target_chunk_index }

The target is `_next_free_chunk()`: the committed chunk index plus one, plus one
more if a chunk is currently in flight. That single rule is what stops a retry from
swallowing input that arrived during the retry window, and it does so by
construction rather than by a check somewhere in the generation loop.

At `begin_chunk()`, the queue is partitioned by claim against the frontier the call
is about to realise:

    current < target   -> still pending (a future frontier is not due yet)
    current == target  -> assign, exactly once
    current > target   -> STALE, fail closed

**The partition runs before the camera reduction**, so a stale event cannot
contribute to a candidate even transiently.

## reset_invalidated is not stale

| situation | status |
|---|---|
| the event was legitimate but a reset actively cancelled it | `reset_invalidated` |
| the event's legitimate chunk has passed and it tries a later chunk | `stale` |
| a terminal event is pushed back in | API invariant violation; the canonical record is never reclassified |
| a stale claim surfaces from an external or corrupted queue without having gone through a normal reset | fail-closed `stale`, treated as a defensive path |

The reset is now applied **at accept time**, not deferred to `begin_chunk()`.
Deferring it let events accepted after the reset carry claims in the old generation
and be wrongly invalidated. A reset while a chunk is in flight is refused outright,
because the in-flight claim would otherwise be silently orphaned.

## Claim is runtime authority, not metrics authority

The claim is deliberately **not** added to `LatencyTraceRecord`. The Latency-1B
schema was just frozen, and stale detection is no reason to expand it. The queue
holds `QueuedInput { event, claim }`; the trace continues to answer only "this
event ended up stale". If stale forensics ever needs expected/observed frontier,
that is a separate schema migration rather than a field smuggled in now.

Likewise `note` carries human diagnostics only ("application frontier passed").
The correctness tests assert against `ApplicationClaim` and runtime state directly,
so `note` cannot quietly become a second authority.

## Seven gates, all passing

    1  the claim is created once at accept and is immutable; it carries no
       event_id and no timestamp, so age cannot leak into staleness
    2  an exact frontier match assigns
    3  a future event is not consumed early -- and an event accepted during a
       flight targets the NEXT free chunk
    4  a same-generation past frontier is stale, with no t1 and no membership
    4b the stale check precedes the camera reduction: the candidate equals the
       reduction of the good events alone, and the committed camera is untouched
    5  staleness has zero state impact: committed camera unchanged, candidate
       excludes the event, applied_event_ids excludes it, t1/t2/t3 are None, and
       t0 is kept because the event genuinely was accepted
    6  reset keeps `reset_invalidated` and is never reclassified later; events
       accepted after a reset carry new-generation claims; a reset during a flight
       is refused
    7  retry does not advance the frontier: after five simulated failures the
       claim is unchanged and the event still lands in the chunk it claimed

Plus: a terminal record is never reclassified.

    14/14 Latency-1B and 11/11 Interactive-2A tests still pass
    11/11 Interactive-2B tests pass

## GPU sanity, including a real fault injection

`play.py --inject_stale 3` accepts an event and then forces the frontier past its
claim, exercising staleness on the real GPU path:

      ev    t0(s)   ->assign   ->commit     ->real  input->real  chunk  frame
       1  209.009        1.4      714.9      714.9        714.9      2      3  committed
       2  209.725          -          -          -            -   None   None  stale
       3  211.205        0.8      787.9      787.9        787.9      7      6  committed
       4  213.535        1.8      769.5      769.5        769.5     10      9  committed

    STALE INJECTION: event 2 status=stale assigned=None t1=None
                     in_any_frame_lineage=False
      stale detection on the real GPU path: PASS
      the next real frame's lineage is unaffected: PASS

    accept -> first affected real frame: n=3, p50 769 ms

## A note of statistical restraint on 0.9-1.1 ms

The mechanism changed from a 60 Hz integrator to event-driven input, so a drop in
`input->assign` is plausible. But three GPU events is not a sample. Do not freeze
"faster than the previous 1.6-6.0 ms" as a performance conclusion. What both
datasets already establish is the ORDER OF MAGNITUDE: the control plane is
milliseconds and the model chunk is ~750 ms. That is stable enough to act on.

## Next

Interactive-2C, prewarm isolation and reference reproducibility, is simpler now
that there is a formal definition of which things are eligible for which
authoritative frontier.
