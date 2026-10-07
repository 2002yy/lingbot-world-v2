#!/usr/bin/env python
"""§Latency-1C gate: input acknowledgement and the processing watermark.

WHAT THIS PROTECTS, AND WHY IT IS NOT THE SAME AS A TIMER

A latency figure is only meaningful once it is clear WHICH input was measured. The
earlier stages built the identity, the commit seam and the frame lineage; this one
freezes the two acknowledgements and the watermark that make the attribution
mechanically checkable instead of inferred from elapsed time.

    accepted_ack    the runtime has taken responsibility for the input
    processed_ack   THIS INPUT IS IN A COMMITTED MODEL STATE   <- emitted at t2 only

The single most important property here is NEGATIVE: `processed_ack` must be
unobtainable for an input that is merely queued or merely assigned. At assignment the
GPU may not have consumed the input at all, and the chunk carrying it can still be
aborted, so an ack issued there would be a lie that also silently corrupts every
latency distribution derived from it.

THE WATERMARK HAS TWO VALUES ON PURPOSE

    processed_input_index   every input up to it is COMMITTED
    settled_input_index     no input up to it is still pending or in flight

They differ exactly when an input terminates without being processed -- aborted,
stale, rejected, or invalidated by a reset -- and the gap is permanent, because
`processed` is a contiguous-prefix property. Reporting only `processed` would let a
clean-looking deadline hide a dropped input; reporting only `settled` would let a
client believe an aborted input was consumed. Both are asserted below.

Run:  python test_input_acknowledgement.py
"""
import sys
import traceback

import numpy as np

from interactive_runtime import (
    COMMITTED, CameraState, InteractiveRuntime, RuntimeStateError, STALE,
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


def rt_new():
    return InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))


def commit_one(rt, event_ids=None):
    """Drive one full chunk and return (snapshot, meta)."""
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    rt.mark_real_decoded(m)
    return snap, m


# ---------------------------------------------------------------- I1 identity
@case
def i1_every_accepted_input_has_a_stable_identity():
    rt = rt_new()
    evs = [rt.accept({"forward": 0.6}) for _ in range(5)]
    ids = [e.event_id for e in evs]
    assert ids == sorted(set(ids)), "event_id is not unique and increasing"
    for e in evs:
        assert e.input_index == e.event_id
    # the ack is available as soon as the input is accepted, and carries the same id
    a = rt.accepted_ack(evs[2].event_id)
    assert a.event_id == evs[2].event_id and a.input_index == evs[2].event_id
    assert a.t0_accepted_ns == evs[2].t0_ns


@case
def i1_identity_survives_a_reset():
    """The input stream is session-wide: a reset must not recycle ids, or 'input
    1042' would mean two different things across a session."""
    rt = rt_new()
    before = [rt.accept({"forward": 0.6}).event_id for _ in range(3)]
    rt.accept({"reset": 1}, kind="reset")
    after = [rt.accept({"forward": 0.6}).event_id for _ in range(3)]
    assert min(after) > max(before), \
        f"event_id was reused across a reset: {before} then {after}"


@case
def i1_input_index_is_strictly_increasing_across_a_reset():
    rt = rt_new()
    idx = []
    for _ in range(3):
        idx.append(rt.accept({"forward": 0.6}).input_index)
    rt.accept({"reset": 1}, kind="reset")
    for _ in range(3):
        idx.append(rt.accept({"forward": 0.6}).input_index)
    assert idx == sorted(set(idx)) and len(set(idx)) == 6, idx


# ------------------------------------------------- I2 commit declares its input
@case
def i2_processed_ack_is_unobtainable_before_commit():
    """The negative property this whole stage exists for."""
    rt = rt_new()
    ev = rt.accept({"forward": 1.0})

    # queued
    state = rt.record(ev.event_id).terminal_status
    assert state == "pending", state
    raises(RuntimeStateError, rt.processed_ack, ev.event_id)

    # assigned, but the chunk is in flight and can still be aborted
    snap = rt.begin_chunk()
    assert snap["applied_event_ids"] == (ev.event_id,)
    assert rt.record(ev.event_id).terminal_status == "in_flight"
    raises(RuntimeStateError, rt.processed_ack, ev.event_id)

    # only commit unlocks it
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    p = rt.processed_ack(ev.event_id)
    assert p.chunk_index == snap["chunk_index"]
    assert p.generation_id == snap["generation_id"]
    assert p.t2_committed_ns == rt.record(ev.event_id).t2_commit_ns


@case
def i2_an_aborted_chunk_never_yields_a_processed_ack():
    rt = rt_new()
    ev = rt.accept({"forward": 1.0})
    rt.begin_chunk()
    rt.abort_chunk("test")
    assert rt.record(ev.event_id).terminal_status == "aborted"
    raises(RuntimeStateError, rt.processed_ack, ev.event_id)


@case
def i2_an_aborted_input_opens_a_permanent_gap_in_processed():
    """settled runs ahead of processed, and stays ahead."""
    rt = rt_new()
    e1 = rt.accept({"forward": 1.0})
    commit_one(rt)                              # chunk 0 consumes e1
    assert rt.committed.processed_input_index == e1.event_id

    e2 = rt.accept({"forward": 1.0})
    rt.begin_chunk()
    rt.abort_chunk("test")                      # e2 will never be processed
    assert rt.record(e2.event_id).terminal_status == "aborted"

    e3 = rt.accept({"yaw": 0.5})
    commit_one(rt)                              # chunk 1 consumes e3

    assert rt.committed.settled_input_index == e3.event_id, \
        "settled must advance past a terminal input"
    assert rt.committed.processed_input_index == e1.event_id, \
        ("processed must NOT jump over an aborted input: the prefix is broken at "
         "e2 forever, and a watermark that hid that would be lying")


@case
def i2_a_stale_input_opens_the_same_gap():
    rt = rt_new()
    e1 = rt.accept({"forward": 1.0})
    commit_one(rt)
    before = rt.committed.processed_input_index

    stale = rt.accept({"forward": 1.0})
    rt.committed.chunk_index += 2               # synthetic frontier regression
    rt.begin_chunk()
    assert rt.record(stale.event_id).terminal_status == STALE
    assert rt.committed.settled_input_index == stale.event_id
    assert rt.committed.processed_input_index == before, \
        "a stale input was counted as processed"
    raises(RuntimeStateError, rt.processed_ack, stale.event_id)


@case
def i2_a_reset_advances_settled_and_not_processed():
    rt = rt_new()
    e1 = rt.accept({"forward": 1.0})
    commit_one(rt)
    queued = [rt.accept({"forward": 1.0}) for _ in range(2)]

    rt.accept({"reset": 1}, kind="reset")
    for q in queued:
        assert rt.record(q.event_id).terminal_status == "reset_invalidated"
    assert rt.committed.processed_input_index == e1.event_id, \
        "a reset must not rewind or advance processed"
    assert rt.committed.settled_input_index == queued[-1].event_id, \
        "the invalidated inputs are settled"
    for q in queued:
        raises(RuntimeStateError, rt.processed_ack, q.event_id)


@case
def i2_committed_chunk_records_what_it_consumed():
    rt = rt_new()
    a = rt.accept({"forward": 0.6})
    b = rt.accept({"yaw": 0.8})
    snap, _ = commit_one(rt)
    c = rt.committed_chunk(snap["chunk_index"])
    assert c is not None
    assert c.applied_event_ids == (a.event_id, b.event_id)
    assert c.generation_id == snap["generation_id"]
    assert c.includes(a.event_id) and not c.includes(999)
    assert c.consumed_up_to(b.event_id)
    assert not c.consumed_up_to(b.event_id + 1), \
        "consumed_up_to must not claim an input that was never accepted"
    assert c.t2_committed_ns is not None


@case
def i2_a_input_free_chunk_still_commits_and_holds_the_watermark():
    rt = rt_new()
    e1 = rt.accept({"forward": 0.6})
    snap0, _ = commit_one(rt)
    snap1, _ = commit_one(rt)                   # no input in this chunk
    assert snap1["applied_event_ids"] == ()
    c1 = rt.committed_chunk(snap1["chunk_index"])
    assert c1.applied_event_ids == ()
    assert c1.processed_input_index == e1.event_id, \
        "an input-free chunk must carry the watermark forward, not reset it"
    assert c1.consumed_up_to(e1.event_id)


# ------------------------------------------------------- I3 frame lineage
@case
def i3_frame_lineage_is_available_and_exact():
    rt = rt_new()
    e1 = rt.accept({"forward": 0.6})
    e2 = rt.accept({"yaw": 0.8})
    snap, m = commit_one(rt)
    lin = rt.frame_lineage(m)
    assert lin["strict"] == (e1.event_id, e2.event_id)
    assert lin["coarse_upto"] == e2.event_id
    assert lin["generation_id"] == snap["generation_id"]
    assert lin["chunk_index"] == snap["chunk_index"]


@case
def i3_a_prewarm_frame_has_no_authoritative_watermark():
    rt = rt_new()
    with rt.prewarm_scope() as pw:
        pm = pw.prewarm_frame_meta("real")
        lin = rt.frame_lineage(pm)
        assert lin["provenance"] == "prewarm"
        assert lin["coarse_upto"] is None, \
            "a prewarm frame must not be able to borrow the authoritative watermark"
        assert lin["strict"] == ()


@case
def i3_the_question_a_client_actually_asks():
    """'Is my W already in this frame?' answered mechanically, not by elapsed time."""
    rt = rt_new()
    w = rt.accept({"forward": 0.6})             # input 1
    _, m1 = commit_one(rt)                      # chunk 0 carries it
    later = rt.accept({"yaw": 0.8})             # input 2, not yet processed

    assert rt.committed_chunk(0).includes(w.event_id) is True
    assert rt.frame_lineage(m1)["coarse_upto"] >= w.input_index
    raise_ok = False
    try:
        rt.processed_ack(later.event_id)
    except RuntimeStateError:
        raise_ok = True
    assert raise_ok, "an unprocessed input must not be claimable as processed"


# --------------------------------------------------- watermarks at the boundary
@case
def watermarks_start_at_zero_and_are_monotone():
    rt = rt_new()
    assert rt.committed.processed_input_index == 0
    assert rt.committed.settled_input_index == 0
    last_p = last_s = 0
    for _ in range(4):
        rt.accept({"forward": 0.6})
        commit_one(rt)
        p, s = rt.committed.processed_input_index, rt.committed.settled_input_index
        assert p >= last_p and s >= last_s, "a watermark went backwards"
        assert p <= s, "processed can never run ahead of settled"
        last_p, last_s = p, s
    assert rt.committed.processed_input_index == 4
    assert rt.committed.settled_input_index == 4


@case
def a_prewarm_pass_cannot_move_the_watermarks():
    rt = rt_new()
    for _ in range(2):
        rt.accept({"forward": 0.6})
        commit_one(rt)
    before = rt.authoritative_fingerprint()
    with rt.prewarm_scope() as pw:
        for _ in range(3):
            pw.prewarm_frame_meta("real")
    after = rt.authoritative_fingerprint()
    assert before == after
    assert before["processed_input_index"] == 2


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
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
