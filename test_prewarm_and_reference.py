#!/usr/bin/env python
"""§Interactive-2C acceptance tests: prewarm isolation and reference reproducibility.

Ten gates. No GPU required.

Reference reproducibility compares each authoritative chunk's full projection
rather than only the final pose: a right-then-left pair returns to where it
started, so a final-state-only comparison would pass a broken sequence.
"""
import sys
import traceback

import numpy as np

from interactive_runtime import (
    CameraState, InteractiveRuntime, RuntimeStateError,
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


def pose(z=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    p = np.eye(4)
    p[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p[:3, 3] = [0.0, 0.0, z]
    return p


def fresh():
    return InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))


def do_chunk(rt, ctrl):
    """Accept one control and commit one authoritative chunk."""
    rt.accept(ctrl)
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.mark_real_decoded(m)
    rt.commit(m)
    return rt.committed_projection()


def run_sequence(script, prewarm=False, retry_at=None, future_at=None):
    """Run a scripted event sequence, optionally perturbed.

    NOTE on `future_at`: an event accepted while no chunk is in flight targets the
    very next chunk, so it is not "future" at all -- it simply joins that chunk.
    To create a genuinely future claim the event must be accepted DURING a flight,
    which is why it is injected after begin_chunk() below. An earlier version
    injected it before, and the test failed for the right reason: the runtime had
    done nothing wrong.
    """
    rt = fresh()
    if prewarm:
        with rt.prewarm_scope() as pw:
            pw.prewarm_frame_meta("real")          # a prewarm pass happened
    steps = []
    for i, ctrl in enumerate(script):
        rt.accept(ctrl)
        snap = rt.begin_chunk()
        if future_at is not None and i == future_at:
            # now a chunk IS in flight, so this claim targets the chunk after it
            rt.accept({"right": 0.5})
        if retry_at is not None and i == retry_at:
            rt.fail_chunk("injected retryable failure")
            rt.fail_chunk("injected retryable failure")
            snap = rt._inflight
        m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                              snap["applied_event_ids"])
        rt.mark_real_decoded(m)
        rt.commit(m)
        steps.append(rt.committed_projection())
    return rt, steps


SCRIPT = [{"forward": 1.0}, {"forward": 1.0, "right": 1.0}, {"yaw": 1.0},
          {"right": -1.0}, {"pitch": -1.0}]


# =================================================== PREWARM ISOLATION (1-7)
@case
def p1_prewarm_does_not_advance_committed_chunk():
    rt = fresh()
    do_chunk(rt, {"forward": 1.0})
    before = rt.committed.chunk_index
    with rt.prewarm_scope() as pw:
        pw.prewarm_frame_meta("real")
    assert rt.committed.chunk_index == before


@case
def p2_prewarm_does_not_change_committed_camera():
    rt = fresh()
    do_chunk(rt, {"forward": 1.0})
    fp = rt.authoritative_fingerprint()
    with rt.prewarm_scope() as pw:
        pw.prewarm_frame_meta("real")
    after = rt.authoritative_fingerprint()
    assert fp["camera_pose"] == after["camera_pose"]
    assert fp["camera_v"] == after["camera_v"]
    assert fp["camera_gate"] == after["camera_gate"]


@case
def p3_prewarm_does_not_consume_the_queue():
    rt = fresh()
    e = rt.accept({"forward": 1.0})
    q_before = [(q.event.event_id, q.claim.key()) for q in rt.queued()]
    with rt.prewarm_scope() as pw:
        pw.prewarm_frame_meta("real")
    q_after = [(q.event.event_id, q.claim.key()) for q in rt.queued()]
    assert q_before == q_after, (q_before, q_after)
    assert rt.record(e.event_id).terminal_status == "pending"


@case
def p4_prewarm_produces_no_authoritative_lineage():
    rt = fresh()
    with rt.prewarm_scope() as pw:
        m = pw.prewarm_frame_meta("real")
        assert m.provenance == "prewarm"
        assert m.applied_event_ids == (), \
            "a prewarm frame must carry no applied events"
        # and it can neither commit nor set t3
        raises(RuntimeStateError, rt.mark_real_decoded, m)
        raises(RuntimeStateError, rt.commit, m)


@case
def p5_prewarm_does_not_write_t1_t2_t3():
    rt = fresh()
    e = rt.accept({"forward": 1.0})
    with rt.prewarm_scope() as pw:
        pw.prewarm_frame_meta("real")
    r = rt.record(e.event_id)
    assert (r.t1_assign_ns, r.t2_commit_ns, r.t3_first_real_ns) == \
        (None, None, None)
    assert r.first_real_frame_id is None


@case
def p6_prewarm_failure_pollutes_nothing():
    """The case most likely to be missed: a PARTIAL prewarm that raised."""
    rt = fresh()
    do_chunk(rt, {"forward": 1.0})
    fp = rt.authoritative_fingerprint()
    try:
        with rt.prewarm_scope() as pw:
            pw.prewarm_frame_meta("real")
            raise RuntimeError("simulated prewarm failure")
    except RuntimeError:
        pass
    assert rt.authoritative_fingerprint() == fp, \
        "a failed prewarm changed authoritative state"


@case
def p7_prewarm_does_not_consume_authoritative_ids():
    rt = fresh()
    do_chunk(rt, {"forward": 1.0})
    fp = rt.authoritative_fingerprint()
    with rt.prewarm_scope() as pw:
        ids = [pw.prewarm_frame_meta("real").frame_id for _ in range(5)]
    assert ids == [1, 2, 3, 4, 5], ids
    after = rt.authoritative_fingerprint()
    assert fp["next_frame_id"] == after["next_frame_id"], \
        "prewarm consumed the authoritative frame id space"
    assert fp["next_event_id"] == after["next_event_id"]
    # a subsequent authoritative frame keeps the id it would have had
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    assert m.frame_id == fp["next_frame_id"]


@case
def p7b_prewarm_scope_detects_an_actual_leak():
    """The verifier must not be vacuous: a real mutation has to be caught."""
    rt = fresh()
    raises(RuntimeStateError, lambda: _leaking_prewarm(rt))


def _leaking_prewarm(rt):
    with rt.prewarm_scope() as pw:
        rt.committed.chunk_index += 1          # deliberate pollution


# ================================================= REFERENCE REPRODUCIBILITY (8-10)
@case
def r8_identical_sequence_yields_identical_projections():
    _, a = run_sequence(SCRIPT)
    _, b = run_sequence(SCRIPT)
    assert len(a) == len(b) == len(SCRIPT)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.chunk_index == y.chunk_index, i
        assert x.generation_id == y.generation_id, i
        assert x.applied_event_ids == y.applied_event_ids, i
        assert x.camera_pose == y.camera_pose, f"step {i} pose differs"
        assert x.camera_v == y.camera_v, f"step {i} velocity differs"


@case
def r8b_intermediate_states_are_compared_not_just_the_final_one():
    """A right-then-left pair returns to the start, so a final-only comparison
    would pass a broken sequence. The step list must actually differ mid-way."""
    script = [{"right": 1.0}, {"right": -1.0}]
    _, steps = run_sequence(script)
    assert steps[0].camera_pose != steps[1].camera_pose, \
        "the two steps are identical, so this test could not detect a missing " \
        "intermediate state"
    _, steps2 = run_sequence(script)
    assert [s.camera_pose for s in steps] == [s.camera_pose for s in steps2]


@case
def r9_retry_injection_does_not_change_the_sequence():
    _, ref = run_sequence(SCRIPT)
    _, got = run_sequence(SCRIPT, retry_at=2)
    assert len(ref) == len(got)
    for i, (x, y) in enumerate(zip(ref, got)):
        assert x.applied_event_ids == y.applied_event_ids, f"step {i} lineage"
        assert x.camera_pose == y.camera_pose, f"step {i} pose"
        assert x.chunk_index == y.chunk_index, f"step {i} chunk"


@case
def r10_prewarm_injection_does_not_change_the_sequence():
    _, ref = run_sequence(SCRIPT)
    _, got = run_sequence(SCRIPT, prewarm=True)
    assert len(ref) == len(got)
    for i, (x, y) in enumerate(zip(ref, got)):
        assert x.applied_event_ids == y.applied_event_ids, f"step {i} lineage"
        assert x.camera_pose == y.camera_pose, f"step {i} pose"


@case
def r10b_future_pending_event_does_not_change_the_sequence():
    """Perturbation C: an event whose frontier is never reached must not alter any
    committed step, and must end still pending -- not stale, not consumed.

    Injected during the LAST chunk's flight, so its claim targets a chunk beyond
    the run. Injecting it earlier would be wrong: a future claim is supposed to be
    consumed exactly when its frontier arrives, and an earlier version of this test
    asserted otherwise.
    """
    _, ref = run_sequence(SCRIPT)
    rt, got = run_sequence(SCRIPT, future_at=len(SCRIPT) - 1)
    assert len(ref) == len(got)
    for i, (x, y) in enumerate(zip(ref, got)):
        assert x.applied_event_ids == y.applied_event_ids, \
            f"step {i}: the future event leaked into a chunk before its frontier"
        assert x.camera_pose == y.camera_pose, f"step {i} pose"
        assert x.chunk_index == y.chunk_index, f"step {i} chunk"
    # and it is still pending at the end: never stale, never consumed
    pend = rt.pending()
    assert len(pend) == 1, [e.event_id for e in pend]
    r = rt.record(pend[0].event_id)
    assert r.terminal_status == "pending", r.terminal_status
    assert r.t1_assign_ns is None
    assert rt.claim_of(pend[0].event_id).target_chunk_index == len(SCRIPT), \
        "the future claim should target the chunk after the last one run"


@case
def r10c_future_event_is_consumed_at_its_frontier():
    """The complement: the same event IS consumed once its frontier arrives, so
    the previous test is not passing merely because the event was dropped."""
    rt = fresh()
    rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()                       # chunk 0 in flight
    future = rt.accept({"right": 1.0})            # claims chunk 1
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.mark_real_decoded(m)
    rt.commit(m)
    snap2 = rt.begin_chunk()                      # frontier is now chunk 1
    assert snap2["applied_event_ids"] == (future.event_id,), \
        "the future event was not consumed at its own frontier"
    assert rt.record(future.event_id).terminal_status == "in_flight"


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
