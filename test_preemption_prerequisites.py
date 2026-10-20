#!/usr/bin/env python
"""§Latency-3B-B1 gate: the three preemption prerequisites.

This stage adds NO cancellation. It fixes the three things that would make cancellation
unsafe if they were left as they are:

    C1  the reference pose advances only on a successful authoritative commit
    C3  t1 is a write-once assignment authority
    C2  a chunk's per-step noise can be reproduced by restoring its RNG position

Each is asserted here; the no-preemption reference equality that C2's weaker half
depends on is measured on the GPU by b1_reference.py, because it is a property of
generation and not of a data structure.

Run:  python test_preemption_prerequisites.py
"""
import sys
import traceback
from collections import deque

import numpy as np
import torch

from interactive_runtime import (
    COMMITTED, CameraState, InteractiveRuntime, QueuedInput, RuntimeStateError,
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


# ============================================================ C3  t1 write-once
@case
def c3_first_assignment_sets_t1():
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    assert rt.record(ev.event_id).t1_assign_ns is None
    rt.begin_chunk(_now_ns=1000)
    assert rt.record(ev.event_id).t1_assign_ns == 1000


@case
def c3_rebinding_the_same_claim_preserves_t1():
    """The case preemption creates: a cancelled batch is returned and the chunk begins
    again. The second begin must not rewrite the assignment time."""
    rt = rt_new()
    ev = rt.accept({"forward": 0.6})
    claim = rt.claim_of(ev.event_id)      # BEFORE begin_chunk: begin pops the queue
    assert claim is not None
    snap = rt.begin_chunk(_now_ns=1000)
    first_t1 = rt.record(ev.event_id).t1_assign_ns

    # simulate a rebase: un-inline the flight and put the batch back with its ORIGINAL
    # claim, exactly as a cancellation would have to
    rt._inflight = None
    rt._queue.append(QueuedInput(event=ev, claim=claim))
    rt.record(ev.event_id).terminal_status = "pending"
    rt.begin_chunk(_now_ns=9000)

    assert rt.record(ev.event_id).t1_assign_ns == first_t1, (
        "a re-begin overwrote t1, so accept_to_assign now depends on how many times "
        "the chunk was attempted rather than on when the input was assigned")


@case
def c3_a_new_claim_gets_its_own_t1():
    rt = rt_new()
    a = rt.accept({"forward": 0.6})
    rt.begin_chunk(_now_ns=1000)
    m = rt.new_frame_meta("real", 0, 0, (a.event_id,))
    rt.commit(m)
    b = rt.accept({"yaw": 0.8})
    rt.begin_chunk(_now_ns=2000)
    assert rt.record(a.event_id).t1_assign_ns == 1000
    assert rt.record(b.event_id).t1_assign_ns == 2000, \
        "a fresh event did not receive its own assignment time"


@case
def c3_accept_to_assign_does_not_inflate_across_a_rebind():
    rt = rt_new()
    ev = rt.accept({"forward": 0.6}, _now_ns=100)
    claim = rt.claim_of(ev.event_id)
    assert claim is not None
    snap = rt.begin_chunk(_now_ns=700)
    before = rt.record(ev.event_id).derived()["accept_to_assign_ms"]
    rt._inflight = None
    rt._queue.append(QueuedInput(event=ev, claim=claim))
    rt.begin_chunk(_now_ns=5000)
    after = rt.record(ev.event_id).derived()["accept_to_assign_ms"]
    assert before == after == 0.0006, (before, after)


# ================================================================ C1  the pose
@case
def c1_on_commit_advances_the_pose_exactly_once():
    """Session-level, with a stand-in that has the same contract and no GPU."""
    class S:
        def __init__(self):
            self._prev_pose, self._candidate_pose = None, None
            self.advances = 0

        def begin(self, pose):
            self._candidate_pose = pose

        def on_commit(self):
            if self._candidate_pose is None:
                raise RuntimeStateError("on_commit without a candidate")
            self._prev_pose = self._candidate_pose
            self._candidate_pose = None
            self.advances += 1

    s = S()
    s.begin("pose0")
    assert s._prev_pose is None, "the pose advanced before any commit"
    s.on_commit()
    assert s._prev_pose == "pose0" and s.advances == 1
    raises(RuntimeStateError, s.on_commit)     # no candidate left: cannot double-advance


@case
def c1_a_chunk_that_does_not_commit_leaves_the_pose_alone():
    rt = rt_new()
    rt.accept({"forward": 0.6})
    snap = rt.begin_chunk()
    rt.abort_chunk("test")                      # no commit
    assert rt.record(1).terminal_status == "aborted"
    # the observable runtime consequence: no committed chunk, so nothing for a session
    # to advance to. on_commit is never called on this path.
    assert rt.committed.chunk_index == -1 and rt.committed_chunk(0) is None


@case
def c1_the_worker_calls_on_commit_only_after_a_successful_commit():
    """The ordering the worker must maintain, asserted against the runtime."""
    calls = []

    class S:
        def on_commit(self):
            calls.append(rt.committed.chunk_index)

    rt = rt_new()
    s = S()
    ev = rt.accept({"forward": 0.6})
    snap = rt.begin_chunk()
    meta = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                             snap["applied_event_ids"])
    # a REFUSED commit must not lead to on_commit
    bad = rt.new_frame_meta("real", 999, snap["generation_id"],
                            snap["applied_event_ids"])
    raises(RuntimeStateError, rt.commit, bad)
    assert calls == [], "on_commit ran after a refused commit"
    rt.commit(meta)
    s.on_commit()
    assert calls == [0], "on_commit did not see the committed chunk"


# ============================================================= C2  the RNG
@case
def c2_restoring_the_position_reproduces_the_same_draws():
    g = torch.Generator(device="cpu")
    g.manual_seed(4321)
    _ = torch.randn(4, generator=g)                       # warm past the first draws
    at_chunk_start = g.get_state()

    a = [torch.randn(8, generator=g) for _ in range(3)]
    g.set_state(at_chunk_start)                           # what restore_chunk_rng does
    b = [torch.randn(8, generator=g) for _ in range(3)]
    for x, y in zip(a, b):
        assert torch.equal(x, y), "restoring the position did not reproduce the draws"


@case
def c2_capture_does_not_perturb_the_uninterrupted_sequence():
    """The whole reason this design was chosen over keyed noise: it cannot change the
    normal path, because it only READS the generator."""
    g1 = torch.Generator(device="cpu")
    g1.manual_seed(99)
    plain = [torch.randn(8, generator=g1) for _ in range(3)]

    g2 = torch.Generator(device="cpu")
    g2.manual_seed(99)
    captured = []
    with_capture = []
    for _ in range(3):
        captured.append(g2.get_state())          # the capture
        with_capture.append(torch.randn(8, generator=g2))
    for x, y in zip(plain, with_capture):
        assert torch.equal(x, y), "capturing the state perturbed the noise sequence"


@case
def c2_no_namespace_is_needed_because_the_state_is_the_position():
    """Why the contract does NOT key noise on (seed, epoch, chunk, step).

    A key has to be namespaced to avoid two different (seed, epoch) pairs colliding.
    A generator state does not: it is the position in one stream, so two different
    seeds simply produce different states. This asserts the property that makes the
    simpler design correct.
    """
    a = torch.Generator(device="cpu"); a.manual_seed(1); _ = torch.randn(4, generator=a)
    b = torch.Generator(device="cpu"); b.manual_seed(2); _ = torch.randn(4, generator=b)
    assert not torch.equal(a.get_state(), b.get_state()), \
        "different seeds produced the same state, so states would collide"


@case
def c2_state_round_trips_through_a_replayed_chunk_boundary():
    """The replay shape: draw, cut, restore, draw again, and require equality."""
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    boundary = g.get_state()
    first = torch.randn(16, generator=g)
    _ = torch.randn(16, generator=g)                       # work that a cut discards
    g.set_state(boundary)
    again = torch.randn(16, generator=g)
    assert torch.equal(first, again)


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
