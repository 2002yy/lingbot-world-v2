"""§Interactive-1 minimal runtime: event identity, tick assignment, commit seam.

SCOPE. Deliberately minimal. The Latency-1A audit found four missing structures
and this module supplies the three that are representable in the current
architecture:

    A. event identity      -> InputEvent with a monotonic event_id and t0
    B. commit seam         -> commit() gated on an exact metadata match
    D. frame lineage       -> FrameMeta carrying chunk/generation/event lineage

It does NOT supply a renderer or a present signal, so t4/t5 remain
unrepresentable (Latency-1A disposition: declare not measurable in the current
architecture). LatencyTrace therefore carries those fields as None and NEVER
infers them.

STATE DISCIPLINE, borrowed from the vLLM-Omni audit rather than from memory:

    committed state   the authoritative state; only commit() may write it
    in-flight state   a speculative copy for the chunk being generated

Correctness must not depend on remembering to bump something. A chunk that fails,
or whose returned metadata does not match what was submitted, leaves the committed
state untouched. This is the general form of the _CAM_EPOCH poisoning bug, which
was hit twice; here the separation is structural instead of procedural.

CLOCK. One clock domain throughout: time.perf_counter_ns(). Nothing is derived
from a different base, which is what made hotswap_loop's "relative elapsed minus
script timestamp" meaningless.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Optional


class RuntimeStateError(RuntimeError):
    """Raised when the runtime contract is violated."""


@dataclass(frozen=True)
class InputEvent:
    """An accepted input. t0 is the moment the authoritative queue took it."""
    event_id: int
    t0_ns: int
    kind: str                       # "control" | "reset" | "pause" | "quit"
    controls: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FrameMeta:
    """Lineage for one produced frame. This is what makes t3 meaningful."""
    frame_id: int
    frame_kind: str                 # "real" | "preview" | "warp"
    chunk_index: int
    generation_id: int
    applied_event_ids: tuple = ()


@dataclass
class CameraState:
    """The part of runtime state the committed/in-flight split must protect."""
    pose: Any = None                # 4x4, opaque to this module
    v: Any = None
    gate: float = 1.0

    def copy(self) -> "CameraState":
        import copy as _c
        return CameraState(pose=_c.deepcopy(self.pose), v=_c.deepcopy(self.v),
                           gate=self.gate)


@dataclass
class LatencyTrace:
    """One trace record per event. Missing fields stay None; never inferred."""
    event_id: int
    t0_input: Optional[int] = None
    t1_assigned: Optional[int] = None
    t2_committed: Optional[int] = None
    t3_real_decoded: Optional[int] = None
    t4_submitted: Optional[int] = None
    t5_presented: Optional[int] = None
    assigned_chunk: Optional[int] = None
    generation_id: Optional[int] = None
    first_real_frame_id: Optional[int] = None

    def derived(self) -> dict:
        """Derived figures. Each is None unless BOTH endpoints exist."""
        def d(a, b):
            return None if a is None or b is None else a - b
        return {
            "input_to_assign": d(self.t1_assigned, self.t0_input),
            "input_to_commit": d(self.t2_committed, self.t0_input),
            "commit_to_real": d(self.t3_real_decoded, self.t2_committed),
            "input_to_real": d(self.t3_real_decoded, self.t0_input),
            "real_to_submit": d(self.t4_submitted, self.t3_real_decoded),
            "submit_to_present": d(self.t5_presented, self.t4_submitted),
            # only this may be called measured input-to-display
            "control_to_real_display": d(self.t5_presented, self.t0_input),
        }


@dataclass
class CommittedState:
    """The authoritative state. Written only by commit()."""
    chunk_index: int = -1
    generation_id: int = 0
    camera: CameraState = field(default_factory=CameraState)
    applied_event_ids: tuple = ()


class InteractiveRuntime:
    """Minimal authoritative runtime: queue, event ids, assignment, commit.

    Usage per chunk:

        snap = rt.begin_chunk()                 # speculative in-flight snapshot
        ... generate with snap ...
        meta = FrameMeta(...)                   # from the actual output
        rt.commit(meta)                         # advances committed state, or raises
    """

    def __init__(self, camera: Optional[CameraState] = None):
        self._queue: deque[InputEvent] = deque()
        self._next_event_id = 1
        self._next_frame_id = 1
        self._traces: dict[int, LatencyTrace] = {}
        self.committed = CommittedState(
            camera=(camera.copy() if camera is not None else CameraState()))
        self._inflight: Optional[dict] = None
        self._pending_reset = False

    # ---------------------------------------------------------------- input
    def accept(self, controls: Optional[dict] = None, kind: str = "control",
               _now_ns: Optional[int] = None) -> InputEvent:
        """t0. Accept an input into the authoritative queue.

        event_id is monotonic for the whole session and keeps increasing across
        reset(), so an id is never reused and lineage can never alias.
        """
        if kind not in ("control", "reset", "pause", "quit"):
            raise RuntimeStateError(f"unknown event kind {kind!r}")
        ev = InputEvent(event_id=self._next_event_id,
                        t0_ns=_now_ns if _now_ns is not None
                        else time.perf_counter_ns(),
                        kind=kind, controls=dict(controls or {}))
        self._next_event_id += 1
        self._queue.append(ev)
        self._traces[ev.event_id] = LatencyTrace(event_id=ev.event_id,
                                                 t0_input=ev.t0_ns)
        if kind == "reset":
            self._pending_reset = True
        return ev

    def pending(self) -> list[InputEvent]:
        return list(self._queue)

    # ------------------------------------------------------------- lifecycle
    def begin_chunk(self, _now_ns: Optional[int] = None) -> dict:
        """t1. Bind the queued events to the next chunk and build the in-flight
        snapshot. Does NOT touch committed state."""
        if self._inflight is not None:
            raise RuntimeStateError(
                "begin_chunk called while a chunk is already in flight; "
                "commit or abort it first")
        if self._pending_reset:
            # reset re-anchors the chunk counter but keeps event ids monotonic
            self.committed = CommittedState(
                chunk_index=-1, generation_id=self.committed.generation_id + 1,
                camera=self.committed.camera.copy(), applied_event_ids=())
            self._pending_reset = False

        chunk_index = self.committed.chunk_index + 1
        assigned = list(self._queue)
        self._queue.clear()
        t1 = _now_ns if _now_ns is not None else time.perf_counter_ns()
        for ev in assigned:
            tr = self._traces[ev.event_id]
            tr.t1_assigned = t1
            tr.assigned_chunk = chunk_index
            tr.generation_id = self.committed.generation_id

        snapshot = dict(
            chunk_index=chunk_index,
            generation_id=self.committed.generation_id,
            applied_event_ids=tuple(ev.event_id for ev in assigned),
            camera=self.committed.camera.copy(),
            events=assigned,
        )
        self._inflight = snapshot
        return snapshot

    def abort_chunk(self) -> None:
        """Drop the in-flight chunk. Committed state is untouched by design."""
        self._inflight = None

    def commit(self, meta: FrameMeta,
               _now_ns: Optional[int] = None) -> CommittedState:
        """t2. Commit, but ONLY if the metadata matches the submitted snapshot.

        A mismatch means the output cannot be attributed to the events that were
        submitted, so committing would attach lineage to the wrong state. This is
        the fail-closed rule from the vLLM-Omni audit: commit only when the
        returned metadata exactly equals the submitted tick snapshot.
        """
        if self._inflight is None:
            raise RuntimeStateError("commit without an in-flight chunk")
        snap = self._inflight
        if meta.chunk_index != snap["chunk_index"]:
            raise RuntimeStateError(
                f"commit refused: chunk_index {meta.chunk_index} != submitted "
                f"{snap['chunk_index']}")
        if tuple(meta.applied_event_ids) != tuple(snap["applied_event_ids"]):
            raise RuntimeStateError(
                f"commit refused: applied_event_ids "
                f"{tuple(meta.applied_event_ids)} != submitted "
                f"{snap['applied_event_ids']}")
        if meta.generation_id != snap["generation_id"]:
            raise RuntimeStateError(
                f"commit refused: generation_id {meta.generation_id} != "
                f"submitted {snap['generation_id']}")

        t2 = _now_ns if _now_ns is not None else time.perf_counter_ns()
        self.committed = CommittedState(
            chunk_index=snap["chunk_index"],
            generation_id=snap["generation_id"],
            camera=snap["camera"],
            applied_event_ids=snap["applied_event_ids"],
        )
        for eid in snap["applied_event_ids"]:
            self._traces[eid].t2_committed = t2
        self._inflight = None
        return self.committed

    # ---------------------------------------------------------------- frames
    def new_frame_meta(self, frame_kind: str, chunk_index: int,
                       generation_id: int,
                       applied_event_ids: tuple = ()) -> FrameMeta:
        m = FrameMeta(frame_id=self._next_frame_id, frame_kind=frame_kind,
                      chunk_index=chunk_index, generation_id=generation_id,
                      applied_event_ids=tuple(applied_event_ids))
        self._next_frame_id += 1
        return m

    def mark_real_decoded(self, meta: FrameMeta,
                          _now_ns: Optional[int] = None) -> None:
        """t3. Only a real frame may set t3, and only for the events it carries."""
        if meta.frame_kind != "real":
            raise RuntimeStateError(
                f"t3 requires a real frame, got {meta.frame_kind!r}")
        t3 = _now_ns if _now_ns is not None else time.perf_counter_ns()
        for eid in meta.applied_event_ids:
            tr = self._traces.get(eid)
            if tr is None:
                raise RuntimeStateError(f"unknown event id {eid} in frame meta")
            if tr.t3_real_decoded is None:
                tr.t3_real_decoded = t3
                tr.first_real_frame_id = meta.frame_id

    # ---------------------------------------------------------------- traces
    def trace(self, event_id: int) -> LatencyTrace:
        try:
            return self._traces[event_id]
        except KeyError:
            raise RuntimeStateError(f"no trace for event id {event_id}") from None

    def traces(self) -> list[LatencyTrace]:
        return [self._traces[k] for k in sorted(self._traces)]

    def note(self, what: str) -> None:
        """Explicitly record that something is NOT measurable here.

        t4/t5 have no seam in this architecture (no renderer, no present signal).
        They stay None. This method exists so a caller cannot accidentally imply
        they were measured.
        """
        if what not in ("t4", "t5"):
            raise RuntimeStateError(f"note() is only for unmeasurable fields, "
                                    f"got {what!r}")
