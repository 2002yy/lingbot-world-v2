# Latency-1B: the formal trace record and its terminal semantics

## Why this was done before Interactive-2, not after

Interactive-1 proved the chain is real. This stage freezes what "one interaction,
one attribution, one latency record" means **while the semantics are still
simple**, so that adding WASD, stale detection and prewarm isolation afterwards
does not force a redefinition of the observation contract at the same time as an
expansion of the product semantics.

Without this, §Interactive-2 would immediately raise questions that are really
observation questions: is a control event one camera delta or several, is a
simultaneous keypress one event or many, is stale a trace field or a runtime
error, do rejected events get records, do prewarm frames enter the latency table,
and how does a trace terminate when an event was assigned but the chunk aborted.

Deliberately not a metrics framework. One record type, one status set, one
`derived()` function.

## The record

    identity    trace_id, event_id, event_kind
    lineage     assigned_chunk, generation_id, applied_event_ids,
                first_real_frame_id
    timestamps  t0_accept_ns, t1_assign_ns, t2_commit_ns, t3_first_real_ns,
                t4_renderer_submit_ns (Optional), t5_present_ns (Optional)
    terminal    terminal_status, note
    derived     accept_to_assign_ms, accept_to_commit_ms,
                accept_to_first_real_ms, accept_to_renderer_ms,
                accept_to_present_ms, assign_to_commit_ms,
                commit_to_first_real_ms

Terminal statuses: `pending`, `in_flight`, `committed`, `aborted`, `rejected`,
`stale`, `reset_invalidated`.

## Four semantics frozen

### 1. A missing time is missing. No proxy, ever.

t4 and t5 have no seam in this architecture, so `accept_to_renderer_ms` and
`accept_to_present_ms` are None and must not be filled from any other clock base.
The runtime has no API that writes them at all, so they cannot be set by accident.

### 2. A failed event still gets a terminal record.

An event accepted, assigned, then aborted produces a complete record with
`terminal_status="aborted"` and None for the stages it never reached. Otherwise a
latency distribution silently contains only successful samples and the worst
stalls, retries and rejections vanish from the statistics.

    event=1 kind=control status=aborted
      t0=100 t1=200 t2=None t3=None t4=None t5=None
      chunk=0 gen=0 frame=None
      accept_to_assign=0.0001 accept_to_commit=None
      accept_to_first_real=None accept_to_renderer=None accept_to_present=None

### 3. Raw timestamps are the authority; derived values are functions.

Only raw fields are serialized. `export()` writes no field ending in `_ms`, and a
test asserts this. Persisting `accept_to_first_real_ms = 762` as an independent
fact would create a second authority that can drift when a timestamp is corrected.

### 4. event_kind is reserved for Interactive-2.

Only `"control"` is used today and the payload stays in `controls`. No hierarchy
was designed now for W/A/S/D, mouse look, joystick, continuous hold or key repeat;
that must be driven by Interactive-2's real requirements.

## Canonicality

One event_id maps to at most one record. The runtime stores records in a dict keyed
by event_id, so a duplicate is structurally impossible rather than prevented by
discipline.

## Acceptance tests: 14 cases, no GPU required, all passing

    1  the four real GPU traces map losslessly into the formal record
       (711/751/793/772 ms reproduced from raw timestamps)
    2  the printed table is a pure function of the records
    3  t4/t5 None never yields a display latency; and no API can write them
    4  abort, reject and stale each produce a complete terminal record, and
       failed samples stay in the distribution
    5  one event_id maps to exactly one record
    6  reset does not confuse generation lineage; queued pre-reset events are
       invalidated while post-reset events survive
    7  serialization round-trips with identical derived() and no derived field
       persisted
    +  prewarm-provenance frames cannot commit and cannot set t3

## One real runtime bug found by these tests

The first implementation cleared the whole queue in `begin_chunk()` when a reset
was pending, which also discarded events accepted **after** the reset. Those had a
legitimate claim on the new generation and were wrongly marked
`reset_invalidated`. Fixed: the reset now invalidates only what was queued before
it, at the moment it is accepted, and post-reset events survive.

## Verification against the real chain

`play.py` re-run on the frozen performance preset through the new record API:

      ev    t0(s)   ->assign   ->commit     ->real  input->real  chunk  frame
       1  206.295        6.0      725.0      725.0        725.0      1      2  committed
       2  208.487        3.0      763.1      763.1        763.1      4      5  committed
       3  210.807        2.5      775.0      775.0        775.0      7      8  committed
       4  212.360        1.9      777.8      777.8        777.8      9     10  committed

      accept -> first affected real frame: n=4, p50 769 ms, min 725, max 778

p50 769 ms against 762 ms in the previous run, i.e. inside run-to-run noise. The
chain, the lineage and the terminal statuses all survive the formalisation.

## Standing architectural observation

`input->assign` is 1.6-6.0 ms while `input->real` is 725-778 ms. The interaction
latency is therefore almost entirely the **next model chunk boundary**, not the
input queue or the runtime control plane. Even once WASD is wired, if one control
still waits for a full ~770 ms chunk before any visible result, the feel will
remain laggy. The eventual fix is preview / warp / speculative visual feedback or a
smaller interaction quantum, not shaving 2-3 ms off input polling. That work comes
after Interactive-2's correctness gates, not before.

## Next

    Interactive-2A  WASD / control wiring through the same InputEvent authority
    Interactive-2B  exactly-once, stale detection, retry double-apply
    Interactive-2C  prewarm isolation, reference reproducibility
