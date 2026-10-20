"""§Interactive-2A: deterministic control reduction and immutable in-flight camera.

TWO DESIGN DECISIONS, both aimed at structural rather than procedural correctness.

1. NO `delta_applied` FLAG. The exactly-once authority is a MATERIALISED,
   immutable `candidate_camera`, built once in `begin_chunk()` from
   (committed camera + this chunk's unique event batch). A retry reads the same
   candidate. Recording `delta + applied=true/false` instead would be the same
   class of hazard as `_CAM_EPOCH`: correctness resting on a bookkeeping bit that
   some future code path can forget to set. Here a retry cannot double-apply
   because it never re-runs the reduction at all.

2. RETRYABLE FAILURE IS NOT TERMINAL ABORT. A generation attempt that fails keeps
   the flight, keeps the candidate, and leaves the trace `in_flight`; the same
   flight is retried. `abort()` is reserved for genuinely abandoning the batch and
   is the only path that sets the terminal `aborted` status. Without this split,
   `aborted` would be terminal while retry was still permitted, and a single
   canonical record could revive from `aborted` to `committed` -- a direct
   contradiction of the Latency-1B freeze.

CONTROL SEMANTICS, kept minimal. `event_kind = "control"` with a payload of
`forward / right / yaw / pitch`. One InputEvent is one DISCRETE CONTROL INTENT, so
W, D and W+D all express through the same payload without deciding now how OS key
repeat, keydown/keyup, held keys, mouse look or gamepad axes should be sampled.
Those belong to Interactive-2's input-sampling stage and must not contaminate 2A's
state correctness.

The reduction is TIME-INDEPENDENT: each event contributes a fixed integration
window, so the same (base, events) always yields the same candidate regardless of
when the events happened to arrive. That is what makes gate 8 (reference sequence
reproducibility) meaningful.
"""
from __future__ import annotations

import numpy as np

# NOTE: this module deliberately does NOT import from interactive_runtime, to
# avoid a circular import. Events are duck-typed: anything with `.kind` and
# `.controls` is treated as an event, anything else as a raw controls dict.

CTRL_HZ = 60.0
CHUNK_PERIOD_S = 1.25
STEPS_PER_EVENT = int(round(CHUNK_PERIOD_S * CTRL_HZ))   # 75 steps == one chunk
YAW_RATE_MAX = 6.0
PITCH_RATE_MAX = 2.0
V_MAX = 1.0


def _as_pose(p):
    """Accept a 4x4 numpy pose. Anything else means the caller is testing the
    state machine with an opaque sentinel; then the reduction is a no-op copy."""
    if isinstance(p, np.ndarray) and p.shape == (4, 4):
        return p
    return None


def _make_state(pose, v, gate):
    """Build a CameraState without importing it, to keep this module free of a
    circular dependency on interactive_runtime."""
    from interactive_runtime import CameraState
    return CameraState(pose=pose, v=v, gate=gate)


def _apply_one(state, controls):
    """One discrete control intent -> one deterministic camera step."""
    pose = _as_pose(state.pose)
    if pose is None:
        # opaque sentinel: preserve it, so state-machine tests stay valid
        c = state.copy()
        c.gate = state.gate
        return c

    from cam_controller import CameraController
    ctl = CameraController(pose[:3, :3].copy(), pose[:3, 3].copy())
    ctl.cfg.yaw_rate_max = YAW_RATE_MAX
    ctl.cfg.pitch_rate_max = PITCH_RATE_MAX
    ctl.cfg.v_max = V_MAX
    ctl.set_input(fwd=float(controls.get("forward", controls.get("fwd", 0.0))),
                  strafe=float(controls.get("right", controls.get("strafe", 0.0))),
                  yaw=float(controls.get("yaw", 0.0)),
                  pitch=float(controls.get("pitch", 0.0)),
                  boost=bool(controls.get("boost", False)))
    for _ in range(STEPS_PER_EVENT):
        ctl.step(dt=1.0 / (CTRL_HZ * CHUNK_PERIOD_S))
    return _make_state(ctl.pose.copy(), ctl.v.copy(), state.gate)


def reduce_controls(base, events):
    """Deterministic, pure: the same (base, events) always yields the same state.

    Called exactly ONCE per chunk, inside begin_chunk(). Nothing else may call it,
    which is why a retry cannot double-apply.

    `events` may be a list of events, a list of raw controls dicts, or a SINGLE
    controls dict. That last case matters: iterating a bare dict yields its keys,
    which is a silent way to feed strings into the reduction.
    """
    if isinstance(events, dict):
        events = [events]
    c = base.copy()
    for ev in events:
        if hasattr(ev, "kind"):
            if ev.kind != "control":
                continue
            c = _apply_one(c, ev.controls)
        elif isinstance(ev, dict):              # a raw controls dict
            c = _apply_one(c, ev)
        else:
            raise TypeError(f"cannot reduce {type(ev).__name__}: expected an "
                            f"event or a controls dict")
    return c


def reduce_sequence(base, batches):
    """Fold a sequence of event batches, returning EVERY intermediate state.

    Gate 8 compares the whole committed sequence, not just the final pose: a
    right-then-left pair can end where it started while every intermediate state,
    and therefore the generated trajectory, was wrong.
    """
    out = []
    cur = base
    for b in batches:
        cur = reduce_controls(cur, b)
        out.append(cur)
    return out
