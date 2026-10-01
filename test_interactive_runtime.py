#!/usr/bin/env python
"""Unit tests for the §Interactive-1 runtime contract. No GPU required.

Every invariant that the Latency-1A audit said was missing is asserted here:
event identity, single assignment, fail-closed commit, committed/in-flight
separation, reset semantics, and the rule that a missing timestamp is never
inferred.
"""
import sys
import traceback

from interactive_runtime import (
    CameraState, FrameMeta, InteractiveRuntime, LatencyTrace, RuntimeStateError,
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


# ------------------------------------------------------------------ identity
@case
def test_event_ids_monotonic():
    rt = InteractiveRuntime()
    ids = [rt.accept({"fwd": 0.5}).event_id for _ in range(5)]
    assert ids == [1, 2, 3, 4, 5], ids


@case
def test_event_ids_survive_reset():
    """A reset must re-anchor chunk_index but NEVER reuse an event id."""
    rt = InteractiveRuntime()
    a = rt.accept({"fwd": 1.0})
    snap = rt.begin_chunk()
    rt.commit(rt.new_frame_meta("real", snap["chunk_index"],
                                snap["generation_id"],
                                snap["applied_event_ids"]))
    rt.accept(kind="reset")
    b = rt.accept({"fwd": -1.0})
    assert b.event_id > a.event_id, (a.event_id, b.event_id)
    snap2 = rt.begin_chunk()
    assert snap2["chunk_index"] == 0, snap2["chunk_index"]
    assert snap2["generation_id"] == 1, snap2["generation_id"]


@case
def test_unknown_kind_rejected():
    rt = InteractiveRuntime()
    raises(RuntimeStateError, rt.accept, {}, "teleport")


# ---------------------------------------------------------------- assignment
@case
def test_single_assignment():
    """An event is assigned to exactly one chunk."""
    rt = InteractiveRuntime()
    e1 = rt.accept({"fwd": 1.0})
    e2 = rt.accept({"yaw": 0.5})
    s1 = rt.begin_chunk()
    assert s1["applied_event_ids"] == (e1.event_id, e2.event_id)
    rt.commit(rt.new_frame_meta("real", s1["chunk_index"], s1["generation_id"],
                                s1["applied_event_ids"]))
    s2 = rt.begin_chunk()
    assert s2["applied_event_ids"] == (), s2["applied_event_ids"]
    assert rt.trace(e1.event_id).assigned_chunk == 0
    assert rt.trace(e2.event_id).assigned_chunk == 0


@case
def test_no_double_begin():
    rt = InteractiveRuntime()
    rt.accept({"fwd": 1.0})
    rt.begin_chunk()
    raises(RuntimeStateError, rt.begin_chunk)


# -------------------------------------------------------------------- commit
@case
def test_commit_requires_exact_match():
    rt = InteractiveRuntime()
    rt.accept({"fwd": 1.0})
    s = rt.begin_chunk()
    bad = FrameMeta(frame_id=99, frame_kind="real", chunk_index=s["chunk_index"],
                    generation_id=s["generation_id"], applied_event_ids=())
    raises(RuntimeStateError, rt.commit, bad)


@case
def test_failed_commit_leaves_committed_untouched():
    """Fail-closed: a mismatched commit must not advance committed state."""
    rt = InteractiveRuntime()
    rt.accept({"fwd": 1.0})
    s = rt.begin_chunk()
    before = (rt.committed.chunk_index, rt.committed.generation_id,
              rt.committed.applied_event_ids)
    bad = FrameMeta(frame_id=1, frame_kind="real", chunk_index=12345,
                    generation_id=s["generation_id"],
                    applied_event_ids=s["applied_event_ids"])
    raises(RuntimeStateError, rt.commit, bad)
    after = (rt.committed.chunk_index, rt.committed.generation_id,
             rt.committed.applied_event_ids)
    assert before == after, (before, after)
    # the chunk is still in flight, so a correct retry is still possible
    good = rt.new_frame_meta("real", s["chunk_index"], s["generation_id"],
                             s["applied_event_ids"])
    rt.commit(good)
    assert rt.committed.chunk_index == s["chunk_index"]


@case
def test_abort_leaves_committed_untouched():
    rt = InteractiveRuntime()
    rt.accept({"fwd": 1.0})
    s = rt.begin_chunk()
    before = rt.committed.chunk_index
    rt.abort_chunk()
    assert rt.committed.chunk_index == before
    s2 = rt.begin_chunk()
    assert s2["chunk_index"] == s["chunk_index"], "aborted chunk must be retried"


# ------------------------------------------------------- committed/in-flight
@case
def test_committed_isolated_from_inflight():
    """Mutating the in-flight camera snapshot must not touch committed state."""
    rt = InteractiveRuntime(CameraState(pose="P0", v="V0", gate=1.0))
    rt.accept({"fwd": 1.0})
    s = rt.begin_chunk()
    s["camera"].pose = "MUTATED"
    s["camera"].gate = 0.0
    assert rt.committed.camera.pose == "P0", rt.committed.camera.pose
    assert rt.committed.camera.gate == 1.0


@case
def test_commit_adopts_snapshot():
    rt = InteractiveRuntime(CameraState(pose="P0"))
    rt.accept({"fwd": 1.0})
    s = rt.begin_chunk()
    s["camera"].pose = "P1"
    rt.commit(rt.new_frame_meta("real", s["chunk_index"], s["generation_id"],
                                s["applied_event_ids"]))
    assert rt.committed.camera.pose == "P1"


# ---------------------------------------------------------------- t3 rules
@case
def test_t3_requires_real_frame():
    rt = InteractiveRuntime()
    e = rt.accept({"fwd": 1.0})
    s = rt.begin_chunk()
    prev = rt.new_frame_meta("preview", s["chunk_index"], s["generation_id"],
                             s["applied_event_ids"])
    raises(RuntimeStateError, rt.mark_real_decoded, prev)
    assert rt.trace(e.event_id).t3_real_decoded is None


@case
def test_t3_first_real_only():
    """t3 is the FIRST affected real frame; later frames must not overwrite it."""
    rt = InteractiveRuntime()
    e = rt.accept({"fwd": 1.0})
    s = rt.begin_chunk()
    m1 = rt.new_frame_meta("real", s["chunk_index"], s["generation_id"],
                           s["applied_event_ids"])
    rt.mark_real_decoded(m1, _now_ns=1000)
    m2 = rt.new_frame_meta("real", s["chunk_index"], s["generation_id"],
                           s["applied_event_ids"])
    rt.mark_real_decoded(m2, _now_ns=2000)
    tr = rt.trace(e.event_id)
    assert tr.t3_real_decoded == 1000, tr.t3_real_decoded
    assert tr.first_real_frame_id == m1.frame_id


# ------------------------------------------------------------ no inference
@case
def test_missing_timestamps_stay_none():
    """t4/t5 have no seam; derived values must be None, never inferred."""
    rt = InteractiveRuntime()
    e = rt.accept({"fwd": 1.0}, _now_ns=100)
    s = rt.begin_chunk(_now_ns=200)
    m = rt.new_frame_meta("real", s["chunk_index"], s["generation_id"],
                          s["applied_event_ids"])
    rt.mark_real_decoded(m, _now_ns=300)
    rt.commit(m, _now_ns=250)
    d = rt.trace(e.event_id).derived()
    assert d["input_to_assign"] == 100, d
    assert d["input_to_commit"] == 150, d
    assert d["commit_to_real"] == 50, d
    assert d["input_to_real"] == 200, d
    assert d["real_to_submit"] is None, d
    assert d["submit_to_present"] is None, d
    assert d["control_to_real_display"] is None, d


@case
def test_note_rejects_measurable_fields():
    rt = InteractiveRuntime()
    raises(RuntimeStateError, rt.note, "t3")
    rt.note("t4")   # allowed: explicitly recording an unmeasurable field


# ------------------------------------------------------------------- lineage
@case
def test_lineage_chain_is_walkable():
    """event -> assigned chunk -> committed chunk -> real frame, mechanically."""
    rt = InteractiveRuntime()
    e = rt.accept({"yaw": 0.7}, _now_ns=10)
    s = rt.begin_chunk(_now_ns=20)
    m = rt.new_frame_meta("real", s["chunk_index"], s["generation_id"],
                          s["applied_event_ids"])
    rt.mark_real_decoded(m, _now_ns=30)
    rt.commit(m, _now_ns=25)
    tr = rt.trace(e.event_id)
    assert tr.assigned_chunk == s["chunk_index"]
    assert tr.generation_id == s["generation_id"]
    assert tr.first_real_frame_id == m.frame_id
    assert e.event_id in rt.committed.applied_event_ids


@case
def test_unknown_event_in_frame_meta_rejected():
    rt = InteractiveRuntime()
    rt.accept({"fwd": 1.0})
    s = rt.begin_chunk()
    m = FrameMeta(frame_id=1, frame_kind="real", chunk_index=s["chunk_index"],
                  generation_id=s["generation_id"], applied_event_ids=(999,))
    raises(RuntimeStateError, rt.mark_real_decoded, m)


def main():
    ok = 0
    fail = []
    for fn in RESULTS:
        try:
            fn()
            ok += 1
            print(f"  PASS  {fn.__name__}")
        except Exception:
            fail.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}")
            print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    print()
    print(f"  {ok}/{len(RESULTS)} passed")
    if fail:
        print(f"  FAILED: {fail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
