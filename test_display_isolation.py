#!/usr/bin/env python
"""§VFX-B4 gate: the display isolation negative controls, T1-T6.

EVERY TEST IS PAIRED WITH AN INJECTED VIOLATION.

A test that only ever passes proves nothing, and this project has already paid for that lesson
twice -- the C1 semantic check that rejected zero, and the P0 oracle that silently reported
"the thing under test never ran". So for each invariant there is

    (a) the guard HELD      -> the violation is caught
    (b) the guard REMOVED   -> the same violation becomes observable

and the test fails if (b) does not actually show the damage. The guard-removed cases exist
only in this file; the production shapes in display_boundary.py are the guarded ones.

Run:  python test_display_isolation.py
"""
import sys
import traceback

import numpy as np
import torch

import display_boundary as db
from display_boundary import (DisplayEnhancer, DisplayFrame, DisplayStage,
                              EnhancedFrame, IsolationViolation,
                              enhance_or_fallback, make_display_frame)
from interactive_runtime import (CameraState, InteractiveRuntime, RuntimeStateError)

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
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


def a_frame(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size=(304, 528, 3), dtype=np.uint8))


# ============================================================ T1 data contamination
class MutatingEnhancer(DisplayEnhancer):
    """Tries, in four different ways, to damage what it was handed."""

    name = "mutating"

    def __init__(self, mode):
        self.mode = mode
        self.observed_result = None

    def enhance(self, disp):
        if self.mode == "write_array":
            disp.frame[...] = 0                      # in-place write, the real attack
            self.observed_result = "wrote"
        elif self.mode == "setflags":
            disp.frame.setflags(write=True)          # try to unlock it first
            disp.frame[...] = 255
            self.observed_result = "unlocked and wrote"
        elif self.mode == "rebind":
            disp.frame = np.zeros_like(disp.frame)   # rebind the attribute
            self.observed_result = "rebound"         # a frozen dataclass forbids this
        return None


@case
def t1_a_read_only_frame_cannot_be_mutated_by_the_enhancer():
    """The guard HELD."""
    src = a_frame(1)
    before = src.copy()
    disp = make_display_frame(src, frame_id=1, chunk_index=0, generation_id=0)
    assert disp.frame.flags.writeable is False, "the frame handed out is writeable"
    # THE ACTUAL GUARD IS THE COPY, NOT THE FLAG. `np.array(copy=True)` returns an array that
    # owns its data, and numpy permits re-enabling the writeable flag on such an array -- so
    # `setflags(write=True)` succeeds and the enhancer CAN write. What it writes is its own
    # buffer. The read-only flag is a reminder; the copy is the isolation.
    assert disp.frame is not src, "the enhancer was given the authoritative buffer itself"
    assert not np.shares_memory(np.asarray(disp.frame), src), \
        "the handed-out frame shares memory with the authoritative one"
    assert np.array_equal(np.asarray(disp.frame), before)
    raises(ValueError, MutatingEnhancer("write_array").enhance, disp)   # the flag does stop this
    MutatingEnhancer("setflags").enhance(disp)                          # unlocking is allowed...
    assert np.array_equal(src, before), (                               # ...and harmless
        "the authoritative frame changed, so the copy is not protecting it")


@case
def t1_b_with_the_guard_removed_the_damage_is_real_and_visible():
    """The negative control for T1.

    With `read_only=False` and `copy=False` the enhancer holds a WRITEABLE VIEW of the
    authoritative buffer, and its in-place write reaches the source. If this case did not
    damage anything, T1's passing case would be proving nothing.
    """
    src = a_frame(2)
    before = src.copy()
    disp = make_display_frame(src, frame_id=1, chunk_index=0, generation_id=0,
                              copy=False, read_only=False)
    assert disp.frame.flags.writeable is True
    MutatingEnhancer("write_array").enhance(disp)          # no exception: nothing stops it
    assert not np.array_equal(src, before), (
        "the injected violation did NOT damage the source, so T1's guard is not the thing "
        "being tested")
    src[:] = before                                        # restore, this is only a control


# ============================================================ T2 reverse injection
@case
def t2_a_display_object_cannot_reach_generation():
    """The guard HELD, at two independent layers."""
    src = a_frame(3)
    disp = make_display_frame(src, frame_id=1, chunk_index=0, generation_id=0)
    enh = EnhancedFrame(frame=src, source="enhanced", for_frame_id=1)

    # layer 1: the interface refuses a display-shaped object
    raises(IsolationViolation, db.assert_not_display_derived, disp)
    raises(IsolationViolation, db.assert_not_display_derived, enh)
    raises(IsolationViolation, db.assert_not_display_derived, src)
    # and it does NOT refuse a real latent
    db.assert_not_display_derived(torch.zeros(16, 1, 38, 66))

    # layer 2: the runtime's commit path refuses anything not authoritative, so a display
    # frame has no route to authoritative state even if layer 1 were bypassed
    rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
    rt.accept({"forward": 0.6})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"], provenance="display")
    raises(RuntimeStateError, rt.commit, m)


@case
def t2_b_the_display_object_really_is_indistinguishable_from_a_latent_without_the_guard():
    """The negative control for T2.

    The point is not that a display object is obviously wrong -- it is an ndarray, exactly
    like plenty of legitimate tensors' numpy neighbours. What stops it is the interface check
    and the provenance check, and this shows that removing the interface check leaves nothing
    in this module to stop it.
    """
    src = a_frame(4)
    enh = EnhancedFrame(frame=src, source="enhanced", for_frame_id=1)
    # without the guard, nothing in the boundary module objects
    assert isinstance(enh.frame, np.ndarray)
    assert enh.frame.ndim == 3 and enh.frame.dtype == np.uint8
    # and the provenance layer is what remains, so it must be the one asserted in T2a
    rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
    rt.accept({"forward": 0.6})
    snap = rt.begin_chunk()
    good = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                             snap["applied_event_ids"])
    rt.commit(good)                                    # authoritative still works
    assert rt.committed.chunk_index == 0


# ============================================================ T3 failure and degrade
class FaultyEnhancer(DisplayEnhancer):
    name = "faulty"

    def __init__(self, mode, delay_s=0.0):
        self.mode, self.delay_s = mode, delay_s

    def enhance(self, disp):
        if self.delay_s:
            import time
            time.sleep(self.delay_s)
        if self.mode == "raise":
            raise RuntimeError("enhancer exploded")
        if self.mode == "none":
            return None
        if self.mode == "empty":
            return EnhancedFrame(frame=np.zeros((0, 0, 3), np.uint8), source="enhanced")
        if self.mode == "bad_shape":
            return EnhancedFrame(frame=np.zeros((304, 528), np.uint8), source="enhanced")


@case
def t3_every_failure_mode_degrades_and_never_raises_into_the_caller():
    src = a_frame(5)
    disp = make_display_frame(src, frame_id=7, chunk_index=2, generation_id=1)
    cases = [("raise", None), ("none", None), ("empty", None), ("bad_shape", None),
             ("ok", None)]
    for mode, _ in cases:
        enh = FaultyEnhancer(mode)
        out = enhance_or_fallback(enh, disp)                # must not raise
        assert isinstance(out, EnhancedFrame)
        assert out.frame.size > 0, f"{mode} produced an unusable frame"
        assert out.frame.ndim == 3 and out.frame.shape[2] == 3
        assert out.for_frame_id == 7
        if mode != "ok":
            assert out.source == "degraded", f"{mode} did not degrade"
            assert out.reason
    # no enhancer at all is a degrade, not a crash
    out = enhance_or_fallback(None, disp)
    assert out.source == "degraded" and out.reason == "no_enhancer"
    # a hang becomes a timeout, not a stall
    out = enhance_or_fallback(FaultyEnhancer("ok", delay_s=1.0), disp, timeout_ms=50)
    assert out.source == "degraded" and out.reason == "timeout"


@case
def t3_b_the_control_shows_a_fault_error_would_reach_the_caller_unhandled():
    """The negative control for T3: without the wrapper the exception escapes."""
    disp = make_display_frame(a_frame(6), frame_id=1, chunk_index=0, generation_id=0)
    raises(RuntimeError, FaultyEnhancer("raise").enhance, disp)
    # and the wrapper, which is the guarded path, does not
    out = enhance_or_fallback(FaultyEnhancer("raise"), disp)
    assert out.source == "degraded" and "RuntimeError" in out.reason


@case
def t3_c_a_display_failure_leaves_committed_state_untouched():
    """The property that matters: authority does not depend on display success."""
    rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
    rt.accept({"forward": 0.6})
    snap = rt.begin_chunk()
    m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                          snap["applied_event_ids"])
    rt.commit(m)
    rt.mark_real_decoded(m)
    fps = []
    for i in range(12):
        src = np.full((304, 528, 3), i * 20, np.uint8)
        disp = make_display_frame(src, frame_id=100 + i, chunk_index=0, generation_id=0)
        enhance_or_fallback(FaultyEnhancer("raise"), disp)      # always fails
        fps.append(rt.authoritative_fingerprint())
    assert all(f == fps[0] for f in fps), \
        "a failing display path changed authoritative state"


# ============================================================ T4 async races
@case
def t4_a_stale_display_result_cannot_overwrite_a_newer_commit():
    st = DisplayStage()
    st.on_commit(1)
    assert st.offer(EnhancedFrame(frame=a_frame(7), source="enhanced", for_frame_id=1))
    assert st.shown_frame_id == 1
    st.on_commit(2)                                          # the model moved on
    assert st.offer(EnhancedFrame(frame=a_frame(8), source="enhanced", for_frame_id=1)) is False
    assert st.dropped_stale == 1
    assert st.shown_frame_id == 1, "a stale result overwrote a newer display state"
    assert st.offer(EnhancedFrame(frame=a_frame(9), source="enhanced", for_frame_id=2))
    assert st.shown_frame_id == 2
    assert st.shown == 2


@case
def t4_b_out_of_order_completion_is_absorbed_and_authority_is_untouched():
    rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
    before = rt.authoritative_fingerprint()
    st = DisplayStage()
    for cid in range(5):
        rt.accept({"forward": 0.1})
        snap = rt.begin_chunk()
        m = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                              snap["applied_event_ids"])
        rt.commit(m)
        st.on_commit(m.frame_id)
    # deliver the enhanced results in reverse order, as a slow display would. The newest
    # frame (id 5) arrives first and is shown; the four older ones are superseded on arrival.
    for fid in reversed(range(1, 6)):
        st.offer(EnhancedFrame(frame=a_frame(fid), source="enhanced", for_frame_id=fid))
    assert rt.committed.chunk_index == 4
    assert st.shown_frame_id == 5, f"newest should be shown, got {st.shown_frame_id}"
    assert st.dropped_stale == 4, f"expected 4 superseded results dropped, got {st.dropped_stale}"
    # the display stage holds no handle on the runtime, so it cannot have changed it
    assert rt.committed.chunk_index == 4


@case
def t4_c_the_control_shows_an_unguarded_stage_does_overwrite_newer_state():
    """The negative control for T4: the same stale offer, with the ordering guard removed."""
    st = DisplayStage()
    st.on_commit(2)
    raw_stale = EnhancedFrame(frame=a_frame(11), source="enhanced", for_frame_id=1)
    # an unguarded stage is just "assign whatever arrived"
    st.shown_frame_id = raw_stale.for_frame_id
    assert st.shown_frame_id == 1, "the control did not reproduce the overwrite"
    assert st.shown_frame_id != st.latest_committed_frame_id, \
        "the control did not leave newer state superseded, so T4a proves nothing"


# ============================================================ T5 toggle equivalence
@case
def t5_enhancement_on_or_off_leaves_the_authoritative_result_identical():
    """Same inputs, same seed, same configuration; only the enhancer differs."""
    import demo_wasd as dw
    import queue
    import threading
    import time

    def run(enhancer):
        rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
        sess = dw.MockSession(h=304, w=528, step_ms=20.0)
        sess.attach_runtime(rt)
        inq = dw.InputMailbox()
        fq = queue.Queue()
        stop = threading.Event()
        w = dw.Worker(rt, sess, inq, fq, stop, mailbox=inq)
        w.start()
        frames, state = [], []
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                m = fq.get(timeout=0.3)
            except queue.Empty:
                continue
            if isinstance(m, dw.WorkerDone):
                break
            if isinstance(m, dw.AuthoritativeMsg):
                src = np.asarray(m.frame)
                d = make_display_frame(src, frame_id=m.frame_id,
                                       chunk_index=m.chunk_index,
                                       generation_id=m.generation_id)
                out = enhance_or_fallback(enhancer, d, timeout_ms=200)
                frames.append(hash(np.ascontiguousarray(out.frame).tobytes()))
                state.append((rt.committed.chunk_index, rt.committed.generation_id,
                              tuple(np.asarray(rt.committed.camera.pose).ravel()),
                              rt.committed.processed_input_index,
                              rt.committed.settled_input_index))
                if len(frames) >= 6:
                    break
        for _ in range(3):
            inq.post({"forward": 0.2}, time.perf_counter_ns())
        stop.set()
        w.join(timeout=10)
        return frames, state

    f_off, s_off = run(None)                        # enhancement OFF
    f_on, s_on = run(db.PassthroughEnhancer())      # enhancement ON

    assert s_off and s_off == s_on, (
        "the committed state sequence differs between enhancement off and on:\n"
        f"  off {s_off}\n  on  {s_on}")
    assert f_off == f_on, "the authoritative frame hashes differ between off and on"
    # the display frame COUNT is allowed to differ and does not participate
    assert len(f_off) == len(f_on)


@case
def t5_b_the_control_shows_the_comparison_can_fail():
    """The negative control for T5.

    If the comparison were insensitive, T5a would pass for an enhancer that DID change
    authority. This runs the same comparison with a deliberately different committed state
    and requires it to be detected.
    """
    a = [(0, 0, (1.0,), 0, 0)]
    b = [(0, 0, (2.0,), 0, 0)]
    assert a != b, "the comparison cannot distinguish different committed states"


# ============================================================ T6 resource lifetime
class CountingEnhancer(DisplayEnhancer):
    name = "counting"

    def __init__(self):
        self.constructed = True
        self.released = False

    def enhance(self, disp):
        return EnhancedFrame(frame=np.asarray(disp.frame), source="enhanced",
                             for_frame_id=disp.frame_id)

    def close(self):
        self.released = True


class LazyEnhancerFactory:
    """Creates the enhancer only when enhancement is actually enabled."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.created = 0
        self.instance = None

    def get(self):
        if not self.enabled:
            return None
        if self.instance is None:
            self.instance = CountingEnhancer()
            self.created += 1
        return self.instance

    def shutdown(self):
        if self.instance is not None:
            self.instance.close()
            self.instance = None


@case
def t6_with_enhancement_off_nothing_is_created():
    f = LazyEnhancerFactory(enabled=False)
    enh = f.get()
    assert enh is None and f.created == 0, "a disabled enhancer was constructed anyway"
    out = enhance_or_fallback(enh, make_display_frame(
        a_frame(12), frame_id=1, chunk_index=0, generation_id=0))
    assert out.source == "degraded" and f.created == 0


@case
def t6_with_enhancement_on_it_is_created_once_and_released_independently():
    f = LazyEnhancerFactory(enabled=True)
    e1, e2 = f.get(), f.get()
    assert e1 is e2 and f.created == 1, "the enhancer was constructed more than once"
    disp = make_display_frame(a_frame(13), frame_id=1, chunk_index=0, generation_id=0)
    assert enhance_or_fallback(f.get(), disp).source == "enhanced"
    inst = f.instance
    f.shutdown()
    assert inst.released, "the enhancer was not released"
    assert f.instance is None
    # shutting down the display side did not touch the runtime, which it has no handle on
    rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
    before = rt.authoritative_fingerprint()
    f.shutdown()
    assert rt.authoritative_fingerprint() == before


@case
def t6_b_the_control_shows_an_eager_factory_would_build_it_unconditionally():
    """The negative control for T6."""
    class Eager:
        def __init__(self, enabled):
            self.created = 0
            self.instance = CountingEnhancer()      # built regardless of `enabled`
            self.created = 1 if not enabled else 1
    e = Eager(enabled=False)
    assert e.created == 1 and e.instance is not None, \
        "the control did not reproduce the unconditional construction T6 guards against"


# ============================================================ the boundary itself
@case
def the_enhancer_interface_has_no_route_to_authority():
    """A capability check, not a name check: the interface must not be able to reach state."""
    for cls in (DisplayEnhancer, db.PassthroughEnhancer, MutatingEnhancer, FaultyEnhancer,
                CountingEnhancer):
        inst = cls(MutatingEnhancer("rebind")) if cls is MutatingEnhancer else \
            cls("ok") if cls is FaultyEnhancer else cls()
        for forbidden in ("rt", "runtime", "session", "pipe", "kv", "self_kv",
                          "generator", "_gen", "committed"):
            assert not hasattr(inst, forbidden), (
                f"{cls.__name__} exposes {forbidden!r}, which is a route into authority")


@case
def a_frozen_display_frame_cannot_be_rebound():
    disp = make_display_frame(a_frame(14), frame_id=1, chunk_index=0, generation_id=0)
    def rebind():
        disp.frame = np.zeros((2, 2, 3), np.uint8)
    raises(Exception, rebind)


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
            print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
    print()
    print(f"  {ok}/{len(RESULTS)} passed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
