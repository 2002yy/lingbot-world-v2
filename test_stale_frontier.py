#!/usr/bin/env python
"""§Interactive-2B acceptance tests: application-claim staleness.

Seven gates. Correctness is asserted against ApplicationClaim and runtime state
directly; `note` is human diagnostics and is never used as an authority here.

No GPU required.
"""
import sys
import traceback

import numpy as np

from control_reduce import reduce_controls
from interactive_runtime import (
    ABORTED, COMMITTED, IN_FLIGHT, PENDING, RESET_INVALIDATED, STALE,
    ApplicationClaim, CameraState, InteractiveRuntime, RuntimeStateError,
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


def pose(z=0.0):
    p = np.eye(4)
    p[:3, 3] = [0.0, 0.0, z]
    return p


def cam(rt):
    return rt.committed.camera


def same(a, b):
    return (np.allclose(np.asarray(a.pose), np.asarray(b.pose), atol=0, rtol=0)
            and np.allclose(np.asarray(a.v), np.asarray(b.v), atol=0, rtol=0))


def commit_next(rt):
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    rt.mark_real_decoded(m)
    return snap, m


# ---------------------------------------------------- gate 1: claim immutability
@case
def gate1_claim_created_once_and_immutable():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    ev = rt.accept({"forward": 1.0})
    c1 = rt.claim_of(ev.event_id)
    assert c1 is not None, "no claim was bound at accept time"
    assert isinstance(c1, ApplicationClaim)
    assert c1.generation_id == 0 and c1.target_chunk_index == 0, c1
    # re-reading yields the same object, and it is frozen
    assert rt.claim_of(ev.event_id) is c1
    raises(Exception, setattr, c1, "target_chunk_index", 99)
    # nothing about the claim is derived from event_id or age
    assert not hasattr(c1, "event_id")
    assert not hasattr(c1, "t0_ns")


# ---------------------------------------------------- gate 2: exact frontier
@case
def gate2_exact_frontier_assigns():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    ev = rt.accept({"forward": 1.0})
    assert rt.frontier() == (0, 0)
    snap = rt.begin_chunk()
    assert snap["applied_event_ids"] == (ev.event_id,)
    assert rt.record(ev.event_id).terminal_status == IN_FLIGHT


# ---------------------------------------------------- gate 3: future not early
@case
def gate3_future_event_is_not_consumed_early():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    commit_next(rt)                                   # chunk 0 done
    snap = rt.begin_chunk()                           # chunk 1 in flight
    e = rt.accept({"right": 1.0})                     # accepted mid-flight
    assert rt.claim_of(e.event_id).target_chunk_index == 2, \
        "an event accepted during a flight must target the NEXT free chunk"
    assert snap["applied_event_ids"] == (), "it leaked into the in-flight chunk"
    assert rt.record(e.event_id).terminal_status == PENDING


# ---------------------------------------------------- gate 4: past frontier
@case
def gate4_same_generation_past_frontier_is_stale():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    e = rt.accept({"forward": 1.0})                   # claim (0, 0)
    assert rt.claim_of(e.event_id).key() == (0, 0)
    # force the frontier past the claim without ever assigning it: a synthetic
    # regression, which is exactly the defensive path this gate covers
    rt.committed.chunk_index = 5                      # generation unchanged
    snap = rt.begin_chunk()                           # frontier (0, 6)
    r = rt.record(e.event_id)
    assert r.terminal_status == STALE, r.terminal_status
    assert r.t1_assign_ns is None, "a stale event must not be assigned"
    assert e.event_id not in snap["applied_event_ids"]


@case
def gate4b_stale_check_precedes_camera_reduction():
    """A stale event must not contribute to the candidate even transiently."""
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    c_before = cam(rt).copy()
    e_stale = rt.accept({"forward": 1.0})
    e_ok = rt.accept({"right": 1.0})
    rt.committed.chunk_index = 5
    # give the good event a matching claim by hand, so only the stale one is out
    for q in rt._queue:
        if q.event.event_id == e_ok.event_id:
            q.claim = ApplicationClaim(0, 6)
    snap = rt.begin_chunk()
    assert rt.record(e_stale.event_id).terminal_status == STALE
    assert snap["applied_event_ids"] == (e_ok.event_id,)
    ref = reduce_controls(c_before, [{"right": 1.0}])
    assert same(snap["candidate_camera"], ref), \
        "the stale event influenced the candidate camera"
    assert same(cam(rt), c_before), "committed camera changed"


# ---------------------------------------------------- gate 5: zero impact
@case
def gate5_stale_has_zero_state_impact():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    commit_next(rt)
    c_before = cam(rt).copy()
    e = rt.accept({"forward": 1.0})
    rt.committed.chunk_index = 9                      # frontier moves past
    snap = rt.begin_chunk()
    r = rt.record(e.event_id)
    assert r.terminal_status == STALE
    assert r.t0_accept_ns is not None, "t0 is kept: it was genuinely accepted"
    assert r.t1_assign_ns is None and r.t2_commit_ns is None
    assert r.t3_first_real_ns is None
    assert r.assigned_chunk is None
    assert e.event_id not in snap["applied_event_ids"]
    assert same(cam(rt), c_before)


# ---------------------------------------------------- gate 6: reset vs stale
@case
def gate6_reset_keeps_reset_invalidated_semantics():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    e = rt.accept({"forward": 1.0})                   # claim (0, 0)
    assert rt.claim_of(e.event_id).key() == (0, 0)
    rt.accept(kind="reset")
    r = rt.record(e.event_id)
    assert r.terminal_status == RESET_INVALIDATED, r.terminal_status
    assert r.terminal_status != STALE, "reset cancellation must not be stale"
    # and it must NOT be reclassified by a later begin_chunk
    commit_next(rt)
    commit_next(rt)
    assert rt.record(e.event_id).terminal_status == RESET_INVALIDATED


@case
def gate6b_post_reset_events_get_new_generation_claims():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    commit_next(rt)
    rt.accept(kind="reset")
    e = rt.accept({"forward": 1.0})
    c = rt.claim_of(e.event_id)
    assert c.generation_id == 1, c
    assert c.target_chunk_index == 0, c
    snap = rt.begin_chunk()
    assert snap["applied_event_ids"] == (e.event_id,)
    assert rt.record(e.event_id).generation_id == 1


@case
def gate6c_reset_while_in_flight_is_refused():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    rt.accept({"forward": 1.0})
    rt.begin_chunk()
    raises(RuntimeStateError, rt.accept, None, "reset")


# ---------------------------------------------------- gate 7: retry frontier
@case
def gate7_retry_does_not_advance_the_frontier():
    """The subtle one: failed attempts must not change input semantics."""
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    snap10 = rt.begin_chunk()
    f0 = rt.frontier()
    e = rt.accept({"forward": 1.0})
    assert rt.claim_of(e.event_id).target_chunk_index == snap10["chunk_index"] + 1

    for _ in range(5):
        rt.fail_chunk("simulated failure")
        # the frontier must not have moved: chunk_index is still uncommitted
        assert rt.frontier() == f0, (rt.frontier(), f0)
        assert rt.claim_of(e.event_id).target_chunk_index == \
            snap10["chunk_index"] + 1
        assert rt.record(e.event_id).terminal_status == PENDING

    # and it is still assignable once the retried chunk finally commits
    m = rt.new_frame_meta("real", snap10["chunk_index"], snap10["generation_id"],
                          snap10["applied_event_ids"])
    rt.commit(m)
    rt.mark_real_decoded(m)
    assert rt.frontier() == (0, snap10["chunk_index"] + 1)
    snap = rt.begin_chunk()
    assert snap["applied_event_ids"] == (e.event_id,), \
        "five failed attempts changed which chunk the event landed in"


# ---------------------------------------------------- terminal is terminal
@case
def terminal_records_are_never_reclassified():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    e = rt.accept({"forward": 1.0})
    rt.reject(e, STALE)
    assert rt.record(e.event_id).terminal_status == STALE
    raises(RuntimeStateError, rt.reject, e, STALE)


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
