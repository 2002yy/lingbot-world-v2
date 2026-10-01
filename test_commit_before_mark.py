#!/usr/bin/env python
"""Gate for the ordering fix: a refused commit must not leave t3 set.

This was a real gap. play.py marked the real frame BEFORE committing, so a
refused commit would have left t3 pointing at a frame whose generation was never
accepted -- and the measured input->first-real latency would have described that
frame. The runtime now enforces commit-before-mark; this test asserts it.

Run:  python -m ... not needed; it is appended to the 2A suite as a case.
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


def pose():
    p = np.eye(4)
    return p


@case
def t3_requires_a_committed_chunk():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    ev = rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    # the chunk is still in flight -> marking is refused
    raises(RuntimeStateError, rt.mark_real_decoded, m)
    assert rt.record(ev.event_id).t3_first_real_ns is None
    rt.commit(m)
    rt.mark_real_decoded(m)
    assert rt.record(ev.event_id).t3_first_real_ns is not None


@case
def refused_commit_cannot_leave_t3_set():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    ev = rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()
    bad = rt.new_frame_meta("real", 999, snap["generation_id"],
                            snap["applied_event_ids"])
    raises(RuntimeStateError, rt.commit, bad)
    raises(RuntimeStateError, rt.mark_real_decoded, bad)
    r = rt.record(ev.event_id)
    assert r.t3_first_real_ns is None, \
        "a refused commit left t3 set, so a rejected frame claimed to be the " \
        "first real frame"
    assert r.first_real_frame_id is None
    assert r.t2_commit_ns is None


@case
def t2_is_not_after_t3_in_the_correct_order():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    ev = rt.accept({"forward": 1.0})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m, _now_ns=100)
    rt.mark_real_decoded(m, _now_ns=250)
    r = rt.record(ev.event_id)
    assert r.t2_commit_ns == 100 and r.t3_first_real_ns == 250
    assert r.t2_commit_ns <= r.t3_first_real_ns, \
        "commit must not come after the first real frame is marked"


@case
def a_prewarm_generation_cannot_mark_t3():
    rt = InteractiveRuntime(CameraState(pose=pose(), v=np.zeros(3)))
    with rt.prewarm_scope() as pw:
        m = pw.prewarm_frame_meta("real")
        raises(RuntimeStateError, rt.mark_real_decoded, m)


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
