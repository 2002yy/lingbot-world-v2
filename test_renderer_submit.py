#!/usr/bin/env python
"""§Latency-1D gate: renderer submit authority.

§Latency-1B froze `t4_renderer_submit_ns` to None and noted that the runtime had no API
writing it, so it could not be set by accident. This stage adds the first writer, so the
whole risk of that change is concentrated in one question: can a t4 be produced that
describes something other than "this authoritative real frame was handed to a real
display backend, for the first time"?

Every case below is a way that could go wrong.

    T4  lineage: the submit record resolves to the same generation / chunk / inputs
    write-once: a redraw cannot overwrite the first submit
    fail-closed: prewarm, uncommitted, non-real, unknown-event and t4 < t3 are refused
    T5  no path anywhere yields a t5_presented_ns
    T6  t3 <= t4 wherever both exist

The viewer-level half of T1 (first submit rather than blend completion) and T3 (headless
produces no t4 at all) live in demo_wasd.py --mock / --mock_windowed, because they are
properties of a display that the runtime does not own.

Run:  python test_renderer_submit.py
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


def rt_new():
    return InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))


def committed_frame(rt, controls=None):
    """Drive one full chunk and return (event, snapshot, meta)."""
    ev = rt.accept(controls or {"forward": 0.6})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    rt.mark_real_decoded(m, _now_ns=1_000_000)
    return ev, snap, m


# ------------------------------------------------------------------ happy path
@case
def t4_is_recorded_and_visible_as_a_raw_timestamp():
    rt = rt_new()
    ev, snap, m = committed_frame(rt)
    assert rt.record(ev.event_id).t4_renderer_submit_ns is None
    rt.mark_renderer_submit(m, _now_ns=2_000_000)
    r = rt.record(ev.event_id)
    assert r.t4_renderer_submit_ns == 2_000_000
    # raw-only serialisation still holds: no derived field is persisted
    d = r.to_dict()
    assert d["t4_renderer_submit_ns"] == 2_000_000
    assert not any(k.endswith("_ms") for k in d), \
        "a derived value leaked into the exported record"
    g = r.derived()
    assert g["accept_to_renderer_ms"] == (2_000_000 - ev.t0_ns) / 1e6
    # t5 is still None and accept_to_present_ms must therefore stay None
    assert r.t5_present_ns is None
    assert g["accept_to_present_ms"] is None


@case
def t4_awards_a_submit_time_to_every_event_the_frame_carries():
    rt = rt_new()
    a = rt.accept({"forward": 0.6})
    b = rt.accept({"yaw": 0.8})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    rt.mark_real_decoded(m, _now_ns=5_000)
    rt.mark_renderer_submit(m, _now_ns=9_000)
    for e in (a, b):
        assert rt.record(e.event_id).t4_renderer_submit_ns == 9_000


# ------------------------------------------------------------------- T4 lineage
@case
def t4_lineage_resolves_to_the_same_generation_and_inputs():
    rt = rt_new()
    ev, snap, m = committed_frame(rt)
    rt.mark_renderer_submit(m, _now_ns=3_000_000)
    rec = rt.renderer_submit_record(m)
    assert rec["frame_id"] == m.frame_id
    assert rec["generation_id"] == snap["generation_id"]
    assert rec["source_chunk_index"] == snap["chunk_index"]
    assert rec["applied_event_ids"] == tuple(snap["applied_event_ids"])
    assert rec["t4_renderer_submit_ns"] == 3_000_000
    # and it agrees with the authoritative chunk record those inputs came from
    cc = rt.committed_chunk(snap["chunk_index"])
    assert tuple(cc.applied_event_ids) == rec["applied_event_ids"]
    assert cc.generation_id == rec["generation_id"]


@case
def a_frame_with_no_inputs_has_no_t4_to_report():
    rt = rt_new()
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"], ())
    rt.commit(m)
    rt.mark_real_decoded(m, _now_ns=1_000)
    rt.mark_renderer_submit(m, _now_ns=2_000)
    assert rt.renderer_submit_record(m)["t4_renderer_submit_ns"] is None, \
        "there is no event to carry the submit time, so none may be invented"


# ----------------------------------------------------------------- write-once
@case
def t4_is_write_once():
    rt = rt_new()
    ev, snap, m = committed_frame(rt)
    rt.mark_renderer_submit(m, _now_ns=2_000_000)
    raises(RuntimeStateError, rt.mark_renderer_submit, m, _now_ns=3_000_000)
    assert rt.record(ev.event_id).t4_renderer_submit_ns == 2_000_000, \
        "a redraw overwrote the first renderer submit"


@case
def a_second_frame_can_still_get_its_own_t4():
    """write-once is per frame, not per runtime."""
    rt = rt_new()
    e1, _, m1 = committed_frame(rt)
    rt.mark_renderer_submit(m1, _now_ns=2_000_000)
    e2, _, m2 = committed_frame(rt)
    rt.mark_renderer_submit(m2, _now_ns=4_000_000)
    assert rt.record(e1.event_id).t4_renderer_submit_ns == 2_000_000
    assert rt.record(e2.event_id).t4_renderer_submit_ns == 4_000_000


# ---------------------------------------------------------------- fail-closed
@case
def t4_is_refused_before_commit():
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    raises(RuntimeStateError, rt.mark_renderer_submit, m, _now_ns=2_000)
    assert rt.record(ev.event_id).t4_renderer_submit_ns is None


@case
def t4_is_refused_for_a_prewarm_frame():
    rt = rt_new()
    with rt.prewarm_scope() as pw:
        m = pw.prewarm_frame_meta("real")
        raises(RuntimeStateError, rt.mark_renderer_submit, m, _now_ns=1)


@case
def t4_is_refused_for_a_non_real_frame_kind():
    rt = rt_new()
    rt.accept({"forward": 0.6})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("preview", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    raises(RuntimeStateError, rt.mark_renderer_submit, m, _now_ns=2_000)


@case
def t4_is_refused_when_the_chunk_index_does_not_match():
    rt = rt_new()
    rt.accept({"forward": 0.6})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    bad = rt.new_frame_meta("real", 999, snap["generation_id"],
                            snap["applied_event_ids"])
    raises(RuntimeStateError, rt.mark_renderer_submit, bad, _now_ns=2_000)


@case
def t4_is_refused_when_it_would_precede_the_decode():
    rt = rt_new()
    ev, snap, m = committed_frame(rt)          # t3 = 1_000_000
    raises(RuntimeStateError, rt.mark_renderer_submit, m, _now_ns=500_000)
    assert rt.record(ev.event_id).t4_renderer_submit_ns is None, \
        "a submit earlier than the decode of the same frame is impossible"


@case
def lineage_that_does_not_match_the_submitted_chunk_is_refused_at_commit():
    """The reachable protection against an invented lineage.

    `mark_renderer_submit` also has an unknown-event-id guard, mirroring
    mark_real_decoded, but that branch is unreachable through a commit-validated meta:
    commit refuses any applied_event_ids that differ from the in-flight chunk, and the
    runtime never removes a record. So the guard is a defensive net, and the property
    that actually protects the t4 lineage is the one asserted here.
    """
    rt = rt_new()
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"], (999,))
    raises(RuntimeStateError, rt.commit, m)


@case
def t4_is_refused_across_a_generation_change():
    rt = rt_new()
    _, _, m = committed_frame(rt)
    rt.accept({"reset": 1}, kind="reset")
    raises(RuntimeStateError, rt.mark_renderer_submit, m, _now_ns=9_000)


# ------------------------------------------------------------------- T5, T6
@case
def t5_is_never_written_by_anything():
    rt = rt_new()
    ev, snap, m = committed_frame(rt)
    rt.mark_renderer_submit(m, _now_ns=2_000_000)
    for r in rt.records():
        assert r.t5_present_ns is None
        assert r.derived()["accept_to_present_ms"] is None
    assert "t5_present_ns" in rt.record(ev.event_id).to_dict()


@case
def t0_to_t4_is_monotone_where_the_fields_exist():
    # every stamp is supplied so the ordering is asserted against a known sequence
    # rather than against whatever wall-clock order the test happened to run in
    rt = rt_new()
    ev = rt.accept({"forward": 0.6}, _now_ns=10)
    snap = rt.begin_chunk(_now_ns=20)
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m, _now_ns=50)
    rt.mark_real_decoded(m, _now_ns=80)
    rt.mark_renderer_submit(m, _now_ns=95)
    r = rt.record(ev.event_id)
    assert (r.t0_accept_ns, r.t1_assign_ns, r.t2_commit_ns, r.t3_first_real_ns,
            r.t4_renderer_submit_ns) == (10, 20, 50, 80, 95)
    assert r.t5_present_ns is None            # never assert ordering against t5


@case
def a_prewarm_pass_cannot_move_t4():
    rt = rt_new()
    ev, _, m = committed_frame(rt)
    rt.mark_renderer_submit(m, _now_ns=2_000_000)
    before = rt.authoritative_fingerprint()
    with rt.prewarm_scope() as pw:
        for _ in range(3):
            pw.prewarm_frame_meta("real")
    assert before == rt.authoritative_fingerprint()
    assert rt.record(ev.event_id).t4_renderer_submit_ns == 2_000_000


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
