#!/usr/bin/env python
"""§VFX-B4: the boundary between the world model's authority and a display enhancer.

The point of this module is that isolation is ENFORCED, not documented. Two examples:

  * the frame handed to an enhancer is an independent buffer whose numpy writeable flag is
    False, so `frame[...] = x` raises rather than silently corrupting the authoritative
    frame. A test can therefore inject exactly that violation and be shown to catch it.
  * an enhanced frame has no route to authoritative state because the runtime's commit path
    already refuses anything whose provenance is not "authoritative", and the display side
    never holds a runtime handle at all.

WHY THIS EXISTS BEFORE THE SDK DOES

B4 decides whether a display enhancer can be attached safely; B0-B3 decide whether it is
worth attaching. They are independent, and the second does not need to be answered before
the first is frozen.

WHAT THIS MODULE IS NOT

It is not a display framework. There is no pipeline here, no threading model to commit to,
and no SDK. It is the smallest set of objects that lets the contract's invariants be tested
and, more importantly, lets the tests be shown to have teeth.

Nothing here is imported by the production path. rc2 at a875a0d is untouched.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class IsolationViolation(RuntimeError):
    """Raised when something crosses the display/model boundary in the wrong direction."""


@dataclass(frozen=True)
class DisplayFrame:
    """What a display enhancer is allowed to receive. Read-only by construction."""

    frame: np.ndarray              # HWC uint8, writeable flag False
    frame_id: int
    chunk_index: int
    generation_id: int
    t0_ns: Optional[int] = None
    t4_ns: Optional[int] = None


def make_display_frame(frame, *, frame_id, chunk_index, generation_id,
                       t0_ns=None, t4_ns=None, copy=True, read_only=True):
    """Build the object handed to an enhancer.

    `copy=True, read_only=True` is the production shape. The two flags exist so a test can
    construct the boundary with a specific guard REMOVED and then demonstrate that the
    violation becomes observable -- a negative control that cannot fail proves nothing.
    """
    arr = np.array(frame, copy=copy)
    if read_only:
        arr.setflags(write=False)
    return DisplayFrame(frame=arr, frame_id=frame_id, chunk_index=chunk_index,
                        generation_id=generation_id, t0_ns=t0_ns, t4_ns=t4_ns)


@dataclass(frozen=True)
class EnhancedFrame:
    """What an enhancer may return. It carries no authority and no provenance."""

    frame: np.ndarray
    source: str                    # enhanced | passthrough | degraded
    reason: str = ""
    for_frame_id: Optional[int] = None


def assert_not_display_derived(x, where="generation input"):
    """T2's interface guard.

    A display object or an HWC 8-bit image is not a generation input, and this refuses it at
    the interface rather than trusting callers to remember. The stronger guarantee lives in
    `InteractiveRuntime.commit()`, which refuses anything that is not an
    `authoritative` FrameMeta -- see the T2 test for both halves.
    """
    if isinstance(x, (DisplayFrame, EnhancedFrame)):
        raise IsolationViolation(
            f"a display object reached {where}: display data must not enter generation")
    if isinstance(x, np.ndarray) and x.ndim == 3 and x.dtype == np.uint8:
        raise IsolationViolation(
            f"an HWC uint8 image reached {where}: that is a display-shaped object, not a "
            f"latent")


class DisplayEnhancer:
    """The interface. Deliberately holds no runtime, session, KV cache or generator."""

    name = "abstract"

    def enhance(self, disp: DisplayFrame) -> Optional[EnhancedFrame]:
        raise NotImplementedError


class PassthroughEnhancer(DisplayEnhancer):
    """The honest baseline: returns exactly what it was given, marked as passthrough."""

    name = "passthrough"

    def enhance(self, disp: DisplayFrame) -> Optional[EnhancedFrame]:
        return EnhancedFrame(frame=np.asarray(disp.frame), source="passthrough",
                             for_frame_id=disp.frame_id)


def enhance_or_fallback(enhancer: Optional[DisplayEnhancer], disp: DisplayFrame,
                        timeout_ms: Optional[float] = None) -> EnhancedFrame:
    """V2: authority never depends on display success.

    Any failure -- exception, timeout, empty or wrong-shaped result, or no enhancer at all --
    degrades to a usable display frame. Nothing here can raise into the caller and nothing
    here can reach authoritative state, because this function has no access to it.
    """
    if enhancer is None:
        return EnhancedFrame(frame=np.asarray(disp.frame), source="degraded",
                             reason="no_enhancer", for_frame_id=disp.frame_id)

    box = {}

    def run():
        try:
            box["out"] = enhancer.enhance(disp)
        except BaseException as e:                      # a display fault is not an authority fault
            box["err"] = f"{type(e).__name__}: {e}"

    if timeout_ms is None:
        run()
    else:
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout_ms / 1000.0)
        if t.is_alive():
            return EnhancedFrame(frame=np.asarray(disp.frame), source="degraded",
                                 reason="timeout", for_frame_id=disp.frame_id)

    if "err" in box:
        return EnhancedFrame(frame=np.asarray(disp.frame), source="degraded",
                             reason=box["err"], for_frame_id=disp.frame_id)
    out = box.get("out")
    if out is None:
        return EnhancedFrame(frame=np.asarray(disp.frame), source="degraded",
                             reason="empty_result", for_frame_id=disp.frame_id)
    f = np.asarray(out.frame)
    if f.size == 0 or f.ndim != 3 or f.shape[2] not in (3, 4):
        return EnhancedFrame(frame=np.asarray(disp.frame), source="degraded",
                             reason=f"bad_shape:{f.shape}", for_frame_id=disp.frame_id)
    return out


@dataclass
class DisplayStage:
    """T4: orders enhanced results against the authoritative sequence.

    An enhanced result is displayable only while it still belongs to the newest committed
    frame. A late result for a superseded frame is dropped rather than shown, because showing
    it would present an older world state as if it were current -- a display error, and one
    that must not be able to reach authority either way.
    """

    latest_committed_frame_id: int = -1
    shown_frame_id: int = -1
    dropped_stale: int = 0
    shown: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def on_commit(self, frame_id: int):
        with self._lock:
            self.latest_committed_frame_id = max(self.latest_committed_frame_id, frame_id)

    def offer(self, enhanced: EnhancedFrame) -> bool:
        """Returns True if this result was shown, False if it was dropped."""
        with self._lock:
            if enhanced.for_frame_id is None:
                return False
            if enhanced.for_frame_id < self.latest_committed_frame_id:
                self.dropped_stale += 1
                return False
            self.shown_frame_id = enhanced.for_frame_id
            self.shown += 1
            return True
