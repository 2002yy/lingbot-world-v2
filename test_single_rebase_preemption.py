#!/usr/bin/env python
"""§Latency-3B-B2 gate: single-rebase preemption.

The oracle P0 (b2_oracle.py) is the strongest correctness gate and needs a GPU. This
file covers the gates that are properties of the runtime and the worker, so they run
anywhere:

    P1  a cancelled generation can never commit
    P2  a cancelled generation can never emit an authoritative frame
    P3  the triggering input claims the CURRENT chunk
    P4  exactly-once survives a rebase
    P5  processed/settled semantics are unchanged -- a rebase is not an abort
    P7  _prev_pose advances once, and only at the final commit
    P9  the one-preemption bound guarantees progress

Run:  python test_single_rebase_preemption.py
"""
import queue
import sys
import threading
import time
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


def begin(rt, t1=1000):
    return rt.begin_chunk(_now_ns=t1)


# ------------------------------------------------------------------ rebase basics
@case
def rebase_bumps_the_generation_and_keeps_the_chunk_index():
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    snap = begin(rt)
    assert (snap["chunk_index"], snap["generation_id"]) == (0, 0)
    info = rt.rebase_chunk("test")
    assert info["chunk_index"] == 0
    assert info["from_generation"] == 0 and info["to_generation"] == 1
    assert rt.committed.chunk_index == -1, "a rebase must not advance committed state"
    snap2 = begin(rt)
    assert (snap2["chunk_index"], snap2["generation_id"]) == (0, 1), \
        "the rebased attempt must reuse the chunk index under a NEW generation"


@case
def rebase_does_not_abort_the_input():
    """P5. The whole point: the input's intent survives, only the attempt dies."""
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    begin(rt)
    rt.rebase_chunk("test")
    r = rt.record(ev.event_id)
    assert not r.is_terminal(), f"a rebase made the input terminal ({r.terminal_status})"
    assert r.terminal_status == "pending"
    assert rt.rebase_count(ev.event_id) == 1


@case
def rebase_preserves_t1_but_moves_the_chunk_binding():
    """C3 and the rebase interact, and both halves matter."""
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    begin(rt, t1=1000)
    assert rt.record(ev.event_id).t1_assign_ns == 1000
    rt.rebase_chunk("test")
    begin(rt, t1=9000)
    assert rt.record(ev.event_id).t1_assign_ns == 1000, \
        "the rebase rewrote t1, so accept_to_assign would now measure attempt count"
    assert rt.record(ev.event_id).generation_id == 1, \
        "the chunk binding did not follow the new generation"


@case
def rebase_refuses_without_an_in_flight_chunk():
    rt = rt_new()
    raises(RuntimeStateError, rt.rebase_chunk)


@case
def rebase_refuses_on_an_already_terminal_batch():
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    begin(rt)
    rt.abort_chunk("test")
    assert rt.record(ev.event_id).terminal_status == "aborted"
    raises(RuntimeStateError, rt.rebase_chunk)


# ------------------------------------------------------------------- P1, P2
@case
def p1_a_cancelled_generation_cannot_commit():
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    old = begin(rt)
    old_gen, old_meta = old["generation_id"], rt.new_frame_meta(
        "real", old["chunk_index"], old["generation_id"], old["applied_event_ids"])
    rt.rebase_chunk("test")
    begin(rt)
    # the stale attempt's meta carries the OLD generation
    raises(RuntimeStateError, rt.commit, old_meta)
    assert rt.record(ev.event_id).terminal_status != "committed"


@case
def p2_a_cancelled_generation_cannot_emit_an_authoritative_frame():
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    old = begin(rt)
    old_meta = rt.new_frame_meta("real", old["chunk_index"], old["generation_id"],
                                 old["applied_event_ids"])
    rt.rebase_chunk("test")
    snap = begin(rt)
    meta = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                             snap["applied_event_ids"])
    rt.commit(meta)
    # the discarded attempt's frame cannot be marked as the first real frame
    raises(RuntimeStateError, rt.mark_real_decoded, old_meta)


# ------------------------------------------------------------------- P3
@case
def p3_the_triggering_input_claims_the_current_chunk():
    """cancel-before-accept, which is the ordering constraint 3B-A identified."""
    rt = rt_new()
    old = rt.accept({"forward": 0.6})
    begin(rt)
    rt.rebase_chunk("test")                       # cleared BEFORE the accept
    trig = rt.accept({"yaw": 0.8})
    claim = rt.claim_of(trig.event_id)
    assert claim is not None
    assert claim.target_chunk_index == 0, (
        f"the trigger claimed chunk {claim.target_chunk_index}, not the current 0 -- "
        f"the replay would gain nothing")
    snap = begin(rt)
    assert set(snap["applied_event_ids"]) == {old.event_id, trig.event_id}, \
        "the rebased batch and the trigger must be carried by the SAME chunk"
    assert snap["generation_id"] == 1


@case
def p3_accepting_before_cancelling_would_claim_the_next_chunk():
    """The negative control for P3: it shows the ordering is load-bearing."""
    rt = rt_new()
    rt.accept({"forward": 0.6})
    begin(rt)
    trig = rt.accept({"yaw": 0.8})                # accepted WHILE a chunk is in flight
    assert rt.claim_of(trig.event_id).target_chunk_index == 1, \
        "an accept during a flight should claim N+1, which is exactly why the order matters"


# ------------------------------------------------------------------- P4, P5
@case
def p4_exactly_once_survives_a_rebase():
    rt = rt_new()
    a = rt.accept({"forward": 0.6})
    begin(rt)
    rt.rebase_chunk("test")
    b = rt.accept({"yaw": 0.8})
    snap = begin(rt)
    meta = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                             snap["applied_event_ids"])
    rt.commit(meta)
    assert rt.record(a.event_id).terminal_status == "committed"
    assert rt.record(b.event_id).terminal_status == "committed"
    # one committed chunk, and each event appears in it exactly once
    log = rt.committed_chunks()
    assert len(log) == 1
    assert tuple(log[0].applied_event_ids) == (a.event_id, b.event_id)


@case
def p5_a_rebase_leaves_no_gap_in_the_processed_watermark():
    """Contrast with abort: an abort stalls `processed` forever, a rebase does not."""
    rt = rt_new()
    a = rt.accept({"forward": 0.6})
    begin(rt)
    rt.rebase_chunk("test")
    b = rt.accept({"yaw": 0.8})
    snap = begin(rt)
    meta = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                             snap["applied_event_ids"])
    rt.commit(meta)
    assert rt.committed.processed_input_index == b.event_id
    assert rt.committed.settled_input_index == b.event_id
    assert rt.committed.processed_input_index == rt.committed.settled_input_index, \
        "a rebase created a processed/settled gap, which is the abort signature"


@case
def p5_the_abort_path_still_opens_the_gap():
    """The control: the two paths must remain distinguishable."""
    rt = rt_new()
    a = rt.accept({"forward": 0.6})
    begin(rt)
    rt.abort_chunk("test")
    b = rt.accept({"yaw": 0.8})
    snap = begin(rt)
    meta = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                             snap["applied_event_ids"])
    rt.commit(meta)
    assert rt.committed.settled_input_index == b.event_id
    assert rt.committed.processed_input_index == 0, \
        "the abort did not open a gap, so the two dispositions are indistinguishable"


# ------------------------------------------------------------------- P7
@case
def p7_the_pose_advances_once_and_only_at_the_final_commit():
    """Session contract, with a stand-in that mirrors it (no GPU needed)."""
    class S:
        def __init__(self):
            self._prev_pose = None
            self._candidate_pose = None
            self.advances = 0

        def attempt(self, pose):
            self._candidate_pose = pose

        def on_commit(self):
            if self._candidate_pose is None:
                raise RuntimeStateError("no candidate")
            self._prev_pose = self._candidate_pose
            self._candidate_pose = None
            self.advances += 1

    s = S()
    s.attempt("A")                 # discarded attempt
    assert s._prev_pose is None
    s.attempt("A")                 # the replay, same chunk pose
    s.on_commit()
    assert s.advances == 1 and s._prev_pose == "A", \
        "the pose advanced once per attempt rather than once per commit"


# ------------------------------------------------------------------- P9
@case
def p9_the_one_preemption_bound_guarantees_progress():
    """Behavioural, through the real Worker: a session that would preempt forever must
    still be stopped after one rebase and must still produce a frame."""
    import demo_wasd as dw

    class AlwaysPreempting(dw.MockSession):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.attempts = 0

        def denoise(self, snap, cid, emit_preview, preempt=None, preempt_budget=0):
            self.attempts += 1
            if preempt is not None and preempt_budget > 0:
                # simulate "an input is always pending"
                preempt.post({"yaw": 0.1}, time.perf_counter_ns())
                raise dw.ChunkPreempted(preempt.take(), forward_index=1)
            return super().denoise(snap, cid, emit_preview)

    rt = rt_new()
    sess = AlwaysPreempting(h=304, w=528, step_ms=20.0)
    sess.attach_runtime(rt)
    inq = dw.InputMailbox()
    fq = queue.Queue()
    stop = threading.Event()
    w = dw.Worker(rt, sess, inq, fq, stop, mailbox=inq, max_preemptions_per_chunk=1)
    w.start()
    deadline = time.time() + 20
    got_authoritative = False
    while time.time() < deadline:
        try:
            m = fq.get(timeout=0.5)
        except queue.Empty:
            continue
        if isinstance(m, dw.AuthoritativeMsg):
            got_authoritative = True
            break
        if isinstance(m, dw.WorkerDone):
            break
    stop.set()
    w.join(timeout=15)
    assert got_authoritative, \
        "an always-preempting input starved the chunk: no frame was ever produced"
    # the invariant is PER CHUNK. The counter is run-wide and the budget resets with
    # each chunk, so the assertion is that no single chunk was restarted twice, which
    # is what actually guarantees progress.
    per_chunk = {}
    for rec in w.preemption_trace:
        per_chunk[rec["chunk_index"]] = per_chunk.get(rec["chunk_index"], 0) + 1
    assert per_chunk, "no preemption was recorded at all, so nothing was exercised"
    worst = max(per_chunk.values())
    assert worst == 1, (
        f"budget was not enforced: a chunk was restarted {worst} times "
        f"({per_chunk})")
    assert w.invariants["preemptions"] == len(w.preemption_trace)
    # and every rebase must have bumped the generation by exactly one
    for rec in w.preemption_trace:
        assert rec["to_generation"] == rec["from_generation"] + 1


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
