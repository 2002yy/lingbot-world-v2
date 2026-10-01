#!/usr/bin/env python
"""§Interactive-2A acceptance tests: exactly-once camera, retry, atomic commit.

The eight gates, plus the retry-vs-abort contract check that motivated the
retryable/terminal split. No GPU required.
"""
import sys
import traceback

import numpy as np

from control_reduce import reduce_controls, reduce_sequence
from interactive_runtime import (
    ABORTED, COMMITTED, IN_FLIGHT, CameraState, InteractiveRuntime,
    RuntimeStateError,
)

RESULTS = []


def case(fn):
    RESULTS.append(fn)
    return fn


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception as e:
        raise AssertionError(f"expected {exc.__name__}, got "
                             f"{type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    """A real 4x4 pose so the reduction actually runs."""
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.eye(4)
    p[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p[:3, 3] = [x, y, z]
    return p


def cam(rt):
    return rt.committed.camera


def same(a, b, tol=0.0):
    return np.allclose(np.asarray(a.pose), np.asarray(b.pose), atol=tol,
                       rtol=0.0) and np.allclose(np.asarray(a.v),
                                                 np.asarray(b.v), atol=tol,
                                                 rtol=0.0)


# ------------------------------------------------------ gate 1: exactly-once
@case
def gate1_single_event_exactly_once():
    """C0 + event(W) must equal apply(C0, W), not 2x W."""
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    c0 = cam(rt).copy()
    rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()
    cand = snap["candidate_camera"]

    ref = reduce_controls(c0, [{"forward": 1.0}])
    assert same(cand, ref), "candidate != apply(C0, W)"

    twice = reduce_controls(c0, [{"forward": 1.0}, {"forward": 1.0}])
    assert not same(cand, twice), \
        "candidate equals a DOUBLE application -- exactly-once is broken"

    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.mark_real_decoded(m)
    rt.commit(m)
    assert same(cam(rt), ref), "committed != C0 + W"


# ------------------------------------------------- gate 2: determinism
@case
def gate2_combined_control_deterministic():
    rt1 = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    rt2 = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    for rt in (rt1, rt2):
        rt.accept({"forward": 1.0, "right": 1.0})
    a = rt1.begin_chunk()["candidate_camera"]
    b = rt2.begin_chunk()["candidate_camera"]
    assert same(a, b), "same base + same controls gave different candidates"

    # order within a batch is part of the contract; both orders are deterministic
    o1 = reduce_controls(rt1.committed.camera,
                         [{"forward": 1.0}, {"right": 1.0}])
    o2 = reduce_controls(rt2.committed.camera,
                         [{"forward": 1.0}, {"right": 1.0}])
    assert same(o1, o2)


# ------------------------------------------------------ gate 3: retry
@case
def gate3_retry_does_not_reapply():
    """The headline test: a retry must read the SAME candidate, not re-run the
    reduction."""
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    rt.accept({"forward": 1.0})
    snap1 = rt.begin_chunk()
    c1 = snap1["candidate_camera"].copy()

    rt.fail_chunk("simulated generation failure")
    # the flight is retained; a retry reads the stored candidate
    assert rt._inflight is not None
    c_again = rt._inflight["candidate_camera"]
    assert same(c1, c_again), "retry produced a different candidate"

    rt.fail_chunk("second failure")
    assert same(c1, rt._inflight["candidate_camera"])

    m = rt.new_frame_meta("real", rt._inflight["chunk_index"],
                          rt._inflight["generation_id"],
                          rt._inflight["applied_event_ids"])
    rt.mark_real_decoded(m)
    rt.commit(m)
    assert same(cam(rt), c1), "committed camera != the candidate"

    # and explicitly: NOT a double application
    twice = reduce_controls(CameraState(pose=pose(), v=np.zeros(3)),
                            [{"forward": 1.0}, {"forward": 1.0}])
    assert not same(cam(rt), twice), "committed camera shows a double apply"


@case
def gate3b_retryable_failure_is_not_terminal():
    """retryable failure != abort. The trace stays in_flight throughout."""
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    ev = rt.accept({"forward": 1.0})
    rt.begin_chunk()
    rt.fail_chunk("boom")
    r = rt.record(ev.event_id)
    assert r.terminal_status == IN_FLIGHT, r.terminal_status
    assert not r.is_terminal(), "a retryable failure must not be terminal"
    # and it can still reach committed, which a terminal status would forbid
    snap = rt._inflight
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.mark_real_decoded(m)
    rt.commit(m)
    assert rt.record(ev.event_id).terminal_status == COMMITTED


@case
def gate3c_abort_is_terminal_and_not_retryable():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    ev = rt.accept({"forward": 1.0})
    rt.begin_chunk()
    rt.abort_chunk("giving up")
    r = rt.record(ev.event_id)
    assert r.terminal_status == ABORTED and r.is_terminal()
    assert rt._inflight is None
    raises(RuntimeStateError, rt.commit,
           rt.new_frame_meta("real", 0, 0, ()))


# --------------------------------------------- gate 4: no smuggling on retry
@case
def gate4_new_input_cannot_enter_a_retrying_chunk():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    e1 = rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()
    assert snap["applied_event_ids"] == (e1.event_id,)
    rt.fail_chunk("failure")

    e2 = rt.accept({"right": 1.0})           # arrives during the retry window
    assert rt._inflight["applied_event_ids"] == (e1.event_id,), \
        "a new event leaked into the retrying chunk"
    assert rt.pending()[0].event_id == e2.event_id, "e2 should still be queued"

    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.mark_real_decoded(m)
    rt.commit(m)
    # e2 belongs to the NEXT chunk
    snap2 = rt.begin_chunk()
    assert snap2["applied_event_ids"] == (e2.event_id,)
    assert snap2["chunk_index"] == snap["chunk_index"] + 1


# ------------------------------------------- gate 5: failed commit
@case
def gate5_failed_commit_does_not_change_camera_authority():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()
    c0 = rt.committed.camera.copy()
    c1 = snap["candidate_camera"].copy()

    bad = rt.new_frame_meta("real", 999, snap["generation_id"],
                            snap["applied_event_ids"])
    raises(RuntimeStateError, rt.commit, bad)
    assert same(cam(rt), c0), "committed camera changed after a refused commit"
    assert same(rt._inflight["candidate_camera"], c1), \
        "the in-flight candidate changed after a refused commit"


# ------------------------------------------- gate 6: atomic adoption
@case
def gate6_commit_adopts_candidate_verbatim():
    """Commit must adopt the candidate, never recompute camera state."""
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()
    # perturb the candidate AFTER materialisation: commit must adopt the perturbed
    # object verbatim, which proves it is not recomputing
    snap["candidate_camera"].pose = pose(x=42.0)
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    assert np.allclose(cam(rt).pose[:3, 3], [42.0, 0, 0]), cam(rt).pose[:3, 3]


# ------------------------------------------- gate 7: same-source lineage
@case
def gate7_frame_and_camera_lineage_share_a_source():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    e = rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.mark_real_decoded(m)
    rt.commit(m)
    assert e.event_id in m.applied_event_ids
    assert rt.committed.applied_event_ids == snap["applied_event_ids"]
    assert rt.record(e.event_id).first_real_frame_id == m.frame_id
    # a frame that does not carry the event must not set t3
    other = rt.new_frame_meta("real", 123, 99, ())
    rt.mark_real_decoded(other)
    assert rt.record(e.event_id).first_real_frame_id == m.frame_id


# ------------------------------------------- gate 8: full sequence
@case
def gate8_reference_sequence_matches_state_by_state():
    script = [{"forward": 1.0}, {"right": 1.0},
              {"forward": 1.0, "right": 1.0}, {"forward": -1.0}]
    base = CameraState(pose=pose(), v=np.zeros(3))

    # reference: fold the whole sequence
    ref = reduce_sequence(base, script)
    assert len(ref) == 4

    # runtime: one event per chunk, committed in order
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    got = []
    for ctrl in script:
        rt.accept(ctrl)
        snap = rt.begin_chunk()
        m = rt.new_frame_meta("real", snap["chunk_index"],
                              snap["generation_id"], snap["applied_event_ids"])
        rt.mark_real_decoded(m)
        rt.commit(m)
        got.append(rt.committed.camera.copy())

    for i, (a, b) in enumerate(zip(ref, got)):
        assert same(a, b), f"committed state {i} differs from the reference"

    # and explicitly: the intermediate states matter. right-then-left returns to
    # the start, so comparing only the final pose would pass a broken sequence.
    rt2 = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    for ctrl in [{"right": 1.0}, {"right": -1.0}]:
        rt2.accept(ctrl)
        s = rt2.begin_chunk()
        m = rt2.new_frame_meta("real", s["chunk_index"], s["generation_id"],
                               s["applied_event_ids"])
        rt2.mark_real_decoded(m)
        rt2.commit(m)
    ref2 = reduce_sequence(base, [{"right": 1.0}, {"right": -1.0}])
    assert same(rt2.committed.camera, ref2[-1])
    assert not same(ref2[0], ref2[1]), "the two steps must differ, or the test " \
                                       "would not detect a missing intermediate"


# ------------------------------------------- reduction purity
@case
def reduction_is_pure_and_time_independent():
    base = CameraState(pose=pose(), v=np.zeros(3))
    a = reduce_controls(base, [{"forward": 1.0}])
    b = reduce_controls(base, [{"forward": 1.0}])
    assert same(a, b)
    assert np.allclose(np.asarray(base.pose), pose()), "input state was mutated"
    # non-control events contribute nothing
    class E:
        kind = "pause"
        controls = {"forward": 1.0}
    c = reduce_controls(base, [E()])
    assert same(c, base), "a non-control event changed the camera"


def main():
    ok, fail = 0, []
    for fn in RESULTS:
        try:
            fn()
            ok += 1
            print(f"  PASS  {fn.__name__}")
        except Exception:
            fail.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}")
            print("        " +
                  traceback.format_exc().strip().replace("\n", "\n        "))
    print()
    print(f"  {ok}/{len(RESULTS)} passed")
    if fail:
        print(f"  FAILED: {fail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
