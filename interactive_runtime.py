"""§Latency-1B: the formal latency trace record and its terminal semantics.

FROZEN HERE, BEFORE §Interactive-2 ADDS COMPLEXITY. Interactive-1 proved the chain
is real; this freezes what "one interaction, one attribution, one latency record"
means while the semantics are still simple. Adding WASD, stale detection and
prewarm isolation afterwards must not force a redefinition of the observation
contract at the same time as an expansion of the product semantics.

Deliberately NOT a metrics framework. One record type, one status enum, one
derived() function.

FOUR SEMANTICS FROZEN

1. A missing time is missing. No proxy, ever. t4/t5 have no seam in this
   architecture, so accept_to_renderer_ms and accept_to_present_ms are None and
   must not be filled from any other clock base.

2. A failed event still gets a terminal record. An event that is accepted, then
   assigned, then has its chunk aborted, must produce a complete record with
   terminal_status="aborted" and None timestamps for the stages it never reached.
   Otherwise a latency distribution silently contains only successful samples and
   the worst stalls, retries and rejections vanish from the statistics.

3. Raw timestamps are the authority; derived values are functions of them. Only
   raw fields are serialized. Persisting `accept_to_first_real_ms = 762` as an
   independent fact would create a second authority that can drift when a
   timestamp is corrected.

4. event_kind is reserved for §Interactive-2. Only "control" is used today and the
   payload stays in `controls`. No hierarchy is designed now for W/A/S/D, mouse
   look, joystick, continuous hold or key repeat; that must be driven by
   §Interactive-2's real requirements.

CANONICALITY: one event_id maps to at most one record. The runtime stores records
in a dict keyed by event_id, so a duplicate is impossible by construction rather
than by discipline.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from control_reduce import reduce_controls

# ---------------------------------------------------------------- status enum
PENDING = "pending"                      # accepted, not yet assigned
IN_FLIGHT = "in_flight"                  # assigned to a chunk being generated
COMMITTED = "committed"                  # t2 reached
ABORTED = "aborted"                      # the chunk was abandoned after assignment
REJECTED = "rejected"                    # refused before assignment
STALE = "stale"                          # tried to cross its application frontier
RESET_INVALIDATED = "reset_invalidated"  # invalidated by a reset

TERMINAL_STATUSES = frozenset({COMMITTED, ABORTED, REJECTED, STALE,
                               RESET_INVALIDATED})


class RuntimeStateError(RuntimeError):
    """Raised when the runtime contract is violated."""


@dataclass(frozen=True)
class InputEvent:
    event_id: int
    t0_ns: int
    kind: str
    controls: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FrameMeta:
    frame_id: int
    frame_kind: str
    chunk_index: int
    generation_id: int
    applied_event_ids: tuple = ()
    provenance: str = "authoritative"     # authoritative | prewarm | warmup


@dataclass
class CameraState:
    pose: Any = None
    v: Any = None
    gate: float = 1.0

    def copy(self) -> "CameraState":
        import copy as _c
        return CameraState(pose=_c.deepcopy(self.pose), v=_c.deepcopy(self.v),
                           gate=self.gate)


@dataclass
class LatencyTraceRecord:
    """One canonical record per accepted event.

    Only raw timestamps are serialized. `derived()` computes everything else, so
    there is exactly one authority for every latency figure.
    """
    # identity
    trace_id: str
    event_id: int
    event_kind: str
    # lineage
    assigned_chunk: Optional[int] = None
    generation_id: Optional[int] = None
    applied_event_ids: tuple = ()
    first_real_frame_id: Optional[int] = None
    # raw timestamps -- the only authority
    t0_accept_ns: Optional[int] = None
    t1_assign_ns: Optional[int] = None
    t2_commit_ns: Optional[int] = None
    t3_first_real_ns: Optional[int] = None
    t4_renderer_submit_ns: Optional[int] = None
    t5_present_ns: Optional[int] = None
    # terminal
    terminal_status: str = PENDING
    note: str = ""

    def derived(self) -> dict:
        """All figures, computed from raw timestamps only.

        A figure is None unless both endpoints exist. In particular
        accept_to_renderer_ms and accept_to_present_ms are None in this
        architecture, because there is no renderer and no present signal.
        """
        def d(a, b):
            return None if a is None or b is None else (a - b) / 1e6
        return {
            "accept_to_assign_ms": d(self.t1_assign_ns, self.t0_accept_ns),
            "accept_to_commit_ms": d(self.t2_commit_ns, self.t0_accept_ns),
            "accept_to_first_real_ms": d(self.t3_first_real_ns,
                                         self.t0_accept_ns),
            "accept_to_renderer_ms": d(self.t4_renderer_submit_ns,
                                       self.t0_accept_ns),
            "accept_to_present_ms": d(self.t5_present_ns, self.t0_accept_ns),
            "assign_to_commit_ms": d(self.t2_commit_ns, self.t1_assign_ns),
            "commit_to_first_real_ms": d(self.t3_first_real_ns,
                                         self.t2_commit_ns),
        }

    def to_dict(self) -> dict:
        """Serialize RAW FIELDS ONLY. No derived value is persisted."""
        d = asdict(self)
        for k in ("t0_accept_ns", "t1_assign_ns", "t2_commit_ns",
                  "t3_first_real_ns", "t4_renderer_submit_ns", "t5_present_ns"):
            if d[k] is not None:
                d[k] = int(d[k])
        d["applied_event_ids"] = list(d["applied_event_ids"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LatencyTraceRecord":
        d = dict(d)
        d["applied_event_ids"] = tuple(d.get("applied_event_ids") or ())
        return cls(**d)

    def is_terminal(self) -> bool:
        return self.terminal_status in TERMINAL_STATUSES

    def line(self) -> str:
        g = self.derived()

        def ms(x):
            return "None" if x is None else f"{x:.1f}"
        return (f"event={self.event_id} kind={self.event_kind} "
                f"status={self.terminal_status}\n"
                f"  t0={self.t0_accept_ns} t1={self.t1_assign_ns} "
                f"t2={self.t2_commit_ns} t3={self.t3_first_real_ns} "
                f"t4={self.t4_renderer_submit_ns} t5={self.t5_present_ns}\n"
                f"  chunk={self.assigned_chunk} gen={self.generation_id} "
                f"frame={self.first_real_frame_id}\n"
                f"  accept_to_assign={ms(g['accept_to_assign_ms'])} "
                f"accept_to_commit={ms(g['accept_to_commit_ms'])} "
                f"accept_to_first_real={ms(g['accept_to_first_real_ms'])} "
                f"accept_to_renderer={ms(g['accept_to_renderer_ms'])} "
                f"accept_to_present={ms(g['accept_to_present_ms'])}")


@dataclass
class CommittedState:
    chunk_index: int = -1
    generation_id: int = 0
    camera: CameraState = field(default_factory=CameraState)
    applied_event_ids: tuple = ()


class InteractiveRuntime:
    """Minimal authoritative runtime. Unchanged contract, formal records."""

    def __init__(self, camera: Optional[CameraState] = None):
        self._queue: deque[InputEvent] = deque()
        self._next_event_id = 1
        self._next_frame_id = 1
        # keyed by event_id => canonicality is structural, not procedural
        self._records: dict[int, LatencyTraceRecord] = {}
        self.committed = CommittedState(
            camera=(camera.copy() if camera is not None else CameraState()))
        self._inflight: Optional[dict] = None
        self._pending_reset = False
        self._rejected: list[tuple[InputEvent, str]] = []

    # ---------------------------------------------------------------- input
    def accept(self, controls: Optional[dict] = None, kind: str = "control",
               _now_ns: Optional[int] = None) -> InputEvent:
        if kind not in ("control", "reset", "pause", "quit"):
            raise RuntimeStateError(f"unknown event kind {kind!r}")
        ev = InputEvent(event_id=self._next_event_id,
                        t0_ns=(_now_ns if _now_ns is not None
                               else time.perf_counter_ns()),
                        kind=kind, controls=dict(controls or {}))
        self._next_event_id += 1
        self._records[ev.event_id] = LatencyTraceRecord(
            trace_id=f"ev{ev.event_id}", event_id=ev.event_id, event_kind=kind,
            t0_accept_ns=ev.t0_ns, terminal_status=PENDING)
        self._queue.append(ev)
        if kind == "reset":
            # Invalidate only what was accepted BEFORE the reset, at the moment
            # the reset is accepted. An earlier version cleared the whole queue in
            # begin_chunk(), which also discarded events accepted AFTER the reset
            # -- they had a legitimate claim on the new generation and were
            # wrongly marked reset_invalidated.
            for q in self._queue:
                if q.event_id == ev.event_id:
                    continue
                r = self._records[q.event_id]
                r.terminal_status = RESET_INVALIDATED
                r.note = "invalidated by reset"
            self._queue.clear()
            self._queue.append(ev)
            self._pending_reset = True
        return ev

    def reject(self, ev: InputEvent, reason: str = "rejected",
               _now_ns: Optional[int] = None) -> LatencyTraceRecord:
        """Refuse an event before assignment. Still produces a terminal record."""
        if reason not in (REJECTED, STALE):
            raise RuntimeStateError(f"reject reason must be {REJECTED!r} or "
                                    f"{STALE!r}, got {reason!r}")
        try:
            self._queue.remove(ev)
        except ValueError:
            pass
        rec = self._records[ev.event_id]
        rec.terminal_status = reason
        rec.note = f"refused before assignment at generation " \
                   f"{self.committed.generation_id}"
        self._rejected.append((ev, reason))
        return rec

    def pending(self) -> list[InputEvent]:
        return list(self._queue)

    # ------------------------------------------------------------- lifecycle
    def begin_chunk(self, _now_ns: Optional[int] = None) -> dict:
        """t1. Bind the queued events to the next chunk and MATERIALISE the
        immutable in-flight candidate camera.

        The reduction runs exactly once, here. A retry reads the stored candidate
        and never re-runs it, so retry double-apply is impossible by construction
        rather than prevented by a bookkeeping flag.
        """
        if self._inflight is not None:
            raise RuntimeStateError("begin_chunk while a chunk is in flight")
        if self._pending_reset:
            # Events queued after the reset are legitimate and stay; anything
            # accepted before it was already invalidated at accept() time.
            self.committed = CommittedState(
                chunk_index=-1, generation_id=self.committed.generation_id + 1,
                camera=self.committed.camera.copy(), applied_event_ids=())
            self._pending_reset = False

        chunk_index = self.committed.chunk_index + 1
        assigned = list(self._queue)
        self._queue.clear()
        t1 = _now_ns if _now_ns is not None else time.perf_counter_ns()
        for ev in assigned:
            r = self._records[ev.event_id]
            r.t1_assign_ns = t1
            r.assigned_chunk = chunk_index
            r.generation_id = self.committed.generation_id
            r.terminal_status = IN_FLIGHT

        base_camera = self.committed.camera.copy()
        # ---- the single materialisation of this chunk's camera authority ----
        candidate_camera = reduce_controls(base_camera, assigned)

        snapshot = dict(
            chunk_index=chunk_index,
            generation_id=self.committed.generation_id,
            applied_event_ids=tuple(ev.event_id for ev in assigned),
            base_camera=base_camera,
            candidate_camera=candidate_camera,
            events=assigned,
            attempts=0)
        self._inflight = snapshot
        return snapshot

    def fail_chunk(self, reason: str = "") -> dict:
        """A RETRYABLE failure. The flight, its candidate and the lineage all
        survive, and the trace stays in_flight.

        Deliberately NOT terminal: an attempt that failed must not consume the
        event, and must not be confused with abandoning the batch.
        """
        if self._inflight is None:
            raise RuntimeStateError("fail_chunk without an in-flight chunk")
        self._inflight["attempts"] += 1
        for eid in self._inflight["applied_event_ids"]:
            r = self._records[eid]
            r.terminal_status = IN_FLIGHT
            r.note = (reason or "attempt failed") + \
                     f" (attempt {self._inflight['attempts']}, retryable)"
        return self._inflight

    def abort_chunk(self, reason: str = "") -> list[LatencyTraceRecord]:
        """Abandon the batch for good. THIS is the only path to the terminal
        `aborted` status, and it is not retryable."""
        if self._inflight is None:
            raise RuntimeStateError("abort_chunk without an in-flight chunk")
        out = []
        for eid in self._inflight["applied_event_ids"]:
            r = self._records[eid]
            r.terminal_status = ABORTED
            r.note = reason or "chunk aborted"
            out.append(r)
        self._inflight = None
        return out

    def commit(self, meta: FrameMeta,
               _now_ns: Optional[int] = None) -> CommittedState:
        if self._inflight is None:
            raise RuntimeStateError("commit without an in-flight chunk")
        if meta.provenance != "authoritative":
            raise RuntimeStateError(
                f"refusing to commit a {meta.provenance!r} frame: only "
                f"authoritative frames may advance committed state")
        snap = self._inflight
        if meta.chunk_index != snap["chunk_index"]:
            raise RuntimeStateError(
                f"commit refused: chunk_index {meta.chunk_index} != "
                f"submitted {snap['chunk_index']}")
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
        # ---- ATOMIC ADOPTION. The candidate is adopted verbatim; camera state is
        # ---- never recomputed here, so commit cannot become a second apply seam.
        self.committed = CommittedState(
            chunk_index=snap["chunk_index"],
            generation_id=snap["generation_id"],
            camera=snap["candidate_camera"],
            applied_event_ids=snap["applied_event_ids"])
        for eid in snap["applied_event_ids"]:
            r = self._records[eid]
            r.t2_commit_ns = t2
            r.terminal_status = COMMITTED
        self._inflight = None
        return self.committed

    # ---------------------------------------------------------------- frames
    def new_frame_meta(self, frame_kind: str, chunk_index: int,
                       generation_id: int, applied_event_ids: tuple = (),
                       provenance: str = "authoritative") -> FrameMeta:
        m = FrameMeta(frame_id=self._next_frame_id, frame_kind=frame_kind,
                      chunk_index=chunk_index, generation_id=generation_id,
                      applied_event_ids=tuple(applied_event_ids),
                      provenance=provenance)
        self._next_frame_id += 1
        return m

    def mark_real_decoded(self, meta: FrameMeta,
                          _now_ns: Optional[int] = None) -> None:
        if meta.frame_kind != "real":
            raise RuntimeStateError(f"t3 requires a real frame, got "
                                    f"{meta.frame_kind!r}")
        if meta.provenance != "authoritative":
            raise RuntimeStateError(
                f"refusing t3 for a {meta.provenance!r} frame: prewarm and "
                f"warmup frames are a different provenance domain")
        t3 = _now_ns if _now_ns is not None else time.perf_counter_ns()
        for eid in meta.applied_event_ids:
            r = self._records.get(eid)
            if r is None:
                raise RuntimeStateError(f"unknown event id {eid} in frame meta")
            if r.t3_first_real_ns is None:
                r.t3_first_real_ns = t3
                r.first_real_frame_id = meta.frame_id

    # ---------------------------------------------------------------- traces
    def record(self, event_id: int) -> LatencyTraceRecord:
        try:
            return self._records[event_id]
        except KeyError:
            raise RuntimeStateError(f"no record for event id {event_id}") from None

    def records(self) -> list[LatencyTraceRecord]:
        return [self._records[k] for k in sorted(self._records)]

    def export(self) -> list[dict]:
        """Raw-only export. Derived values are recomputed on load."""
        return [r.to_dict() for r in self.records()]

    @staticmethod
    def import_records(blobs: list[dict]) -> list[LatencyTraceRecord]:
        return [LatencyTraceRecord.from_dict(b) for b in blobs]
