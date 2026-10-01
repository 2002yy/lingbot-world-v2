# Interactive-2A: control wiring, immutable in-flight camera, exactly-once

## Scope

Real control wiring + camera candidate state + exactly-once. Not stale detection,
not prewarm isolation, not preview/warp, not performance.

## Two design decisions, both structural rather than procedural

### 1. No `delta_applied` flag

The exactly-once authority is a **materialised, immutable `candidate_camera`**,
built once in `begin_chunk()` from (committed camera + this chunk's unique event
batch). A retry reads the stored candidate and never re-runs the reduction.

Recording `delta + applied=true/false` instead would be the same class of hazard as
`_CAM_EPOCH`: correctness resting on a bookkeeping bit that some future code path
can forget to set. Here a retry cannot double-apply because it never re-runs the
reduction at all.

    chunk 37 receives events [104, 105], base camera C36
      flight = { chunk_index: 37, generation_id: 4,
                 applied_event_ids: [104, 105],
                 base_camera: C36,
                 candidate_camera: reduce(C36, [104, 105]) }
    candidate_camera is immutable from that moment on

### 2. Retryable failure is not terminal abort

A generation attempt that fails keeps the flight, keeps the candidate, and leaves
the trace `in_flight`; the same flight is retried. `abort()` is the only path to
the terminal `aborted` status and is not retryable.

    retryable failure != abort

This resolved a genuine contract tension left by Latency-1B: `aborted` was
terminal while retry was still permitted, so one canonical record could revive
from `aborted` to `committed`, contradicting the freeze. The new split:

    generate attempt failure -> fail_chunk()  -> trace stays in_flight, retryable
    genuinely abandoning     -> abort_chunk() -> terminal aborted, not retryable

## Control semantics, kept minimal

`event_kind = "control"` with payload `forward / right / yaw / pitch`. One
InputEvent is one **discrete control intent**, so W, D and W+D all express through
the same payload without deciding now how OS key repeat, keydown/keyup, held keys,
mouse look or gamepad axes should be sampled. Those belong to the input-sampling
stage and must not contaminate 2A's state correctness.

The reduction is **time-independent**: each event contributes a fixed integration
window (75 steps at 60 Hz * 1.25, i.e. one chunk period), so the same
(base, events) always yields the same candidate regardless of arrival time. That is
what makes gate 8 meaningful.

## The eight gates, all passing

    1  single-event camera effect exactly-once: C0 + W == apply(C0, W) and is
       explicitly NOT equal to a double application
    2  combined control deterministic: same base + same controls -> same candidate
    3  retry does not re-apply: after two simulated failures the candidate is
       unchanged and the committed camera equals it, not a double apply
    3b a retryable failure leaves the trace in_flight and still reachable by commit
    3c abort is terminal and not retryable; commit afterwards raises
    4  a new input arriving during a retry cannot enter the retrying chunk; its
       lineage is unchanged and it belongs to the next chunk
    5  a refused commit leaves both the committed camera and the in-flight
       candidate untouched
    6  commit adopts the candidate VERBATIM -- perturbing the candidate after
       materialisation shows up in the committed state, proving commit does not
       recompute
    7  frame lineage and camera lineage share one source; a frame not carrying the
       event cannot set t3
    8  the reference sequence matches state by state, C0 -> C1 -> C2 -> C3 -> C4,
       with a right-then-left pair explicitly checked so a final-pose-only
       comparison could not pass a broken sequence

Plus: the reduction is pure (the input state is not mutated) and non-control
events contribute nothing.

    14/14 Latency-1B tests still pass
    11/11 Interactive-2A tests pass

## Two bugs found while building this

1. `control_reduce` imported `CameraState` from `interactive_runtime` while
   `interactive_runtime` imported `reduce_controls` back, a circular import. Now
   the module is dependency-free and the state is constructed lazily.
2. `reduce_controls(base, events)` iterated a bare controls dict as a sequence,
   feeding its KEYS (strings) into the reduction. A single dict is now wrapped in
   a list, and a non-event non-dict raises rather than being silently skipped.

## GPU end-to-end

    ev    t0(s)   ->assign   ->commit     ->real  input->real  chunk  frame
     1  225.764        0.9      718.0      718.0        718.0      2      3  committed
     2  227.216        1.1      751.6      751.6        751.6      4      5  committed
     3  229.509        1.0      769.5      769.5        769.5      7      8  committed

    accept -> first affected real frame: n=3, p50 752 ms, min 718, max 769

Note `input->assign` is now 0.9-1.1 ms against 1.6-6.0 ms previously, because the
camera is event-driven and no longer waits for a 60 Hz integrator to be advanced to
the current wall-clock time. The chain, the lineage and the terminal statuses are
unchanged.

## What can now be said

Pressing W no longer merely enters a trace: its causal effect on CameraState has
been demonstrated all the way to a real model frame, with the camera lineage and
the frame lineage sharing a single source.
