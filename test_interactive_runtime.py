#!/usr/bin/env python
"""§Latency-1B acceptance tests. No GPU required.

The seven acceptance criteria plus the abort-record test that motivated the
terminal-status design.
"""
import json
import sys
import traceback

from interactive_runtime import (
    ABORTED, COMMITTED, IN_FLIGHT, PENDING, REJECTED, RESET_INVALIDATED, STALE,
    CameraState, FrameMeta, InteractiveRuntime, LatencyTraceRecord,
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


def run_chunk(rt, t1, t2, t3, gen=None):
    snap = rt.begin_chunk(_now_ns=t1)
    m = rt.new_frame_meta("real", snap["chunk_index"],
                          snap["generation_id"] if gen is None
                          else gen, snap["applied_event_ids"])
    rt.mark_real_decoded(m, _now_ns=t3)
    rt.commit(m, _now_ns=t2)
    return snap, m


# ------------------------------------------------- 1. lossless mapping
@case
def test_1_real_traces_map_losslessly():
    """The four real GPU traces must land in the formal record without loss."""
    # exactly the shape play.py produced: p50 762 ms, chunks 1/4/7/9
    rt = InteractiveRuntime()
    plan = [(2.010e9, 2.0148e9, 2.7211e9, 2.7211e9),
            (4.189e9, 4.1913e9, 4.9403e9, 4.9403e9),
            (6.507e9, 6.5099e9, 7.3000e9, 7.3000e9),
            (8.079e9, 8.0806e9, 8.8513e9, 8.8513e9)]
    base = 127.944e9
    for i, (t0, t1, t2, t3) in enumerate(plan):
        rt.accept({"k": i}, _now_ns=int(base + t0))
        run_chunk(rt, int(base + t1), int(base + t2), int(base + t3))
    recs = rt.records()
    assert len(recs) == 4, len(recs)
    got = [round(r.derived()["accept_to_first_real_ms"]) for r in recs]
    assert got == [711, 751, 793, 772], got
    for r in recs:
        assert r.terminal_status == COMMITTED, r.terminal_status
        assert r.assigned_chunk is not None
        assert r.first_real_frame_id is not None
        assert r.generation_id == 0


# ------------------------------------------------- 2. table from record
@case
def test_2_table_is_derived_from_record():
    """The printed table must be a pure function of the records."""
    rt = InteractiveRuntime()
    # 1 ms = 1e6 ns, so use real nanosecond magnitudes
    rt.accept({"fwd": 1.0}, _now_ns=1_000_000)
    run_chunk(rt, 2_000_000, 3_000_000, 4_000_000)
    line = rt.records()[0].line()
    assert "accept_to_assign=1.0" in line, line
    assert "accept_to_commit=2.0" in line, line
    assert "accept_to_first_real=3.0" in line, line
    # nothing was measured for t4/t5, and the line says so explicitly
    assert "accept_to_renderer=None" in line, line
    assert "accept_to_present=None" in line, line


# ------------------------------------------------- 3. no display proxy
@case
def test_3_no_display_latency_without_t4_t5():
    rt = InteractiveRuntime()
    rt.accept({"fwd": 1.0}, _now_ns=1)
    run_chunk(rt, 2, 3, 4)
    d = rt.records()[0].derived()
    assert d["accept_to_renderer_ms"] is None
    assert d["accept_to_present_ms"] is None


@case
def test_3b_t4_t5_are_never_written_by_the_runtime():
    """There is no API that sets t4/t5, so they cannot be filled by accident."""
    rt = InteractiveRuntime()
    rt.accept({"fwd": 1.0}, _now_ns=1)
    run_chunk(rt, 2, 3, 4)
    r = rt.records()[0]
    assert r.t4_renderer_submit_ns is None
    assert r.t5_present_ns is None
    assert not hasattr(rt, "mark_submitted")
    assert not hasattr(rt, "mark_presented")


# ------------------------------------------------- 4. terminal records
@case
def test_4a_abort_produces_complete_terminal_record():
    """accepted -> assigned -> abort must still yield a full record."""
    rt = InteractiveRuntime()
    ev = rt.accept({"fwd": 1.0}, _now_ns=100)
    rt.begin_chunk(_now_ns=200)
    recs = rt.abort_chunk("generation failed")
    assert len(recs) == 1
    r = rt.record(ev.event_id)
    assert r.terminal_status == ABORTED, r.terminal_status
    assert r.t0_accept_ns == 100
    assert r.t1_assign_ns == 200
    assert r.t2_commit_ns is None
    assert r.t3_first_real_ns is None
    g = r.derived()
    assert g["accept_to_assign_ms"] == 0.0001
    assert g["accept_to_commit_ms"] is None
    assert g["accept_to_first_real_ms"] is None


@case
def test_4b_reject_produces_terminal_record():
    rt = InteractiveRuntime()
    ev = rt.accept({"fwd": 1.0}, _now_ns=10)
    rt.reject(ev, REJECTED)
    r = rt.record(ev.event_id)
    assert r.terminal_status == REJECTED
    assert r.t1_assign_ns is None
    assert r.derived()["accept_to_assign_ms"] is None


@case
def test_4c_stale_produces_terminal_record():
    rt = InteractiveRuntime()
    ev = rt.accept({"fwd": 1.0}, _now_ns=10)
    rt.reject(ev, STALE)
    assert rt.record(ev.event_id).terminal_status == STALE
    raises(RuntimeStateError, rt.reject, ev, "teleported")


@case
def test_4d_failed_samples_stay_in_the_distribution():
    """The point of terminal records: failures must not vanish from statistics."""
    rt = InteractiveRuntime()
    e1 = rt.accept({}, _now_ns=1)
    run_chunk(rt, 2, 3, 4)
    e2 = rt.accept({}, _now_ns=5)
    rt.begin_chunk(_now_ns=6)
    rt.abort_chunk("boom")
    e3 = rt.accept({}, _now_ns=7)
    rt.reject(e3, STALE)
    recs = rt.records()
    assert len(recs) == 3, len(recs)
    statuses = sorted(r.terminal_status for r in recs)
    assert statuses == [ABORTED, COMMITTED, STALE], statuses
    # and every one of them is terminal
    assert all(r.is_terminal() for r in recs)


# ------------------------------------------------- 5. canonicality
@case
def test_5_one_record_per_event():
    rt = InteractiveRuntime()
    ids = [rt.accept({}, _now_ns=i).event_id for i in range(1, 6)]
    recs = rt.records()
    assert [r.event_id for r in recs] == ids
    assert len({r.trace_id for r in recs}) == len(ids)
    # the store is keyed by event_id, so a duplicate is structurally impossible
    assert isinstance(rt._records, dict)


# ------------------------------------------------- 6. reset lineage
@case
def test_6_reset_does_not_confuse_generation():
    rt = InteractiveRuntime()
    e1 = rt.accept({}, _now_ns=1)
    run_chunk(rt, 2, 3, 4)
    assert rt.record(e1.event_id).generation_id == 0
    rt.accept(kind="reset", _now_ns=5)
    e2 = rt.accept({}, _now_ns=6)
    run_chunk(rt, 7, 8, 9)
    r2 = rt.record(e2.event_id)
    assert r2.generation_id == 1, r2.generation_id
    assert r2.assigned_chunk == 0, r2.assigned_chunk
    assert r2.event_id > e1.event_id


@case
def test_6b_reset_invalidates_queued_events():
    rt = InteractiveRuntime()
    e1 = rt.accept({}, _now_ns=1)          # queued, never assigned
    rt.accept(kind="reset", _now_ns=2)
    rt.begin_chunk(_now_ns=3)
    r = rt.record(e1.event_id)
    assert r.terminal_status == RESET_INVALIDATED, r.terminal_status
    assert r.is_terminal()


# ------------------------------------------------- 7. serialization
@case
def test_7_roundtrip_preserves_derived():
    rt = InteractiveRuntime()
    rt.accept({"fwd": 1.0}, _now_ns=100)
    run_chunk(rt, 200, 350, 400)
    rt.accept({}, _now_ns=500)
    rt.begin_chunk(_now_ns=600)
    rt.abort_chunk()
    rt.accept({}, _now_ns=700)
    rt.reject(rt.pending()[0], STALE)

    before = [r.derived() for r in rt.records()]
    blobs = rt.export()
    # no derived value may be persisted as an independent fact
    for b in blobs:
        for k in b:
            assert not k.endswith("_ms"), f"derived field persisted: {k}"
    back = InteractiveRuntime.import_records(json.loads(json.dumps(blobs)))
    after = [r.derived() for r in back]
    assert before == after, (before, after)
    assert [r.terminal_status for r in back] == \
           [r.terminal_status for r in rt.records()]


@case
def test_7b_export_is_json_safe():
    rt = InteractiveRuntime()
    rt.accept({}, _now_ns=10)
    run_chunk(rt, 20, 30, 40)
    s = json.dumps(rt.export())
    assert "accept_to" not in s, "derived values leaked into the export"


# ------------------------------------------------- prewarm provenance
@case
def test_prewarm_frames_cannot_commit_or_set_t3():
    rt = InteractiveRuntime()
    ev = rt.accept({}, _now_ns=1)
    snap = rt.begin_chunk(_now_ns=2)
    warm = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                             snap["applied_event_ids"], provenance="prewarm")
    raises(RuntimeStateError, rt.mark_real_decoded, warm)
    raises(RuntimeStateError, rt.commit, warm)
    assert rt.record(ev.event_id).t3_first_real_ns is None
    assert rt.committed.chunk_index == -1


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
