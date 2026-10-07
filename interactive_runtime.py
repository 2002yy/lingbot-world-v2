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

import numpy as np
from collections import deque
from contextlib import contextmanager
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

    @property
    def input_index(self) -> int:
        """The strictly increasing position of this input in the session's stream.

        DELIBERATELY NOT A SECOND STORED COUNTER. `_next_event_id` is assigned once
        and only ever incremented; `reset` does not touch it. So over the whole
        session `event_id` is already a strictly increasing input index, and a
        second field would be a second authority for one fact -- the same hazard as
        `delta_applied` and `_CAM_EPOCH`, both of which this codebase rejected on
        purpose.

        It becomes a stored counter, with a test that it strictly increases across
        a reset, on the day `event_id` stops being monotone (for example if it
        becomes a client-supplied UUID). Until then the two are equal by
        construction, so the API surface can exist without the drift.
        """
        return self.event_id


@dataclass(frozen=True)
class CommittedChunk:
    """The public record of one committed chunk: what it consumed, and what
    watermark it established.

    Appended inside `commit()`, so it cannot disagree with the records it
    summarises. `processed_input_index` here is the HISTORICAL value as of this
    chunk, which is what makes "does chunk 83 include my input?" answerable for a
    chunk that finished long ago -- the current watermark would answer a different
    question.
    """
    chunk_index: int
    generation_id: int
    applied_event_ids: tuple
    processed_input_index: int
    settled_input_index: int
    t2_committed_ns: int

    def includes(self, event_id: int) -> bool:
        """The strict question, answered as strictly as it can be."""
        return event_id in self.applied_event_ids

    def consumed_up_to(self, input_index: int) -> bool:
        """The coarse question. True means every input up to and including this one
        is in a committed state, so this chunk's world already has it."""
        return input_index <= self.processed_input_index


@dataclass(frozen=True)
class InputAcceptedAck:
    """The runtime has taken responsibility for this input.

    Says only one thing: it is queued, it has an identity, and it will not be
    silently dropped. It does NOT say the model has consumed it.
    """
    event_id: int
    input_index: int
    t0_accepted_ns: int


@dataclass(frozen=True)
class InputProcessedAck:
    """THIS INPUT IS IN A COMMITTED MODEL STATE.

    Much stronger than acceptance, and it is emitted at exactly one point: t2, after
    the containing chunk's commit has been validated. Never at queue insertion, and
    never at assignment -- at assignment the GPU may not have consumed the input at
    all, and the chunk that carries it can still be aborted.
    """
    event_id: int
    input_index: int
    chunk_index: int
    generation_id: int
    t2_committed_ns: int


@dataclass(frozen=True)
class FrameMeta:
    frame_id: int
    frame_kind: str
    chunk_index: int
    generation_id: int
    applied_event_ids: tuple = ()
    provenance: str = "authoritative"     # authoritative | prewarm | warmup

    # NOTE, and it is a real trap: a watermark field does NOT belong here. FrameMeta
    # is created BEFORE commit (commit validates against it), so any watermark
    # snapshotted at construction would carry the PREVIOUS chunk's value and would
    # answer "does this frame include my input?" with a confident no. The exact
    # answer is `event_id in meta.applied_event_ids`; the coarse per-frame answer is
    # served by the runtime's per-chunk log instead. See committed_chunk().


@dataclass(frozen=True)
class ApplicationClaim:
    """An immutable claim on exactly one authoritative application frontier.

    Staleness is a property of the CLAIM, never of the event's age. Nothing here
    may be derived from event_id ordering, queue residency time or wall-clock age:
    an old event whose frontier has not yet arrived is perfectly valid, and a
    brand-new event whose frontier has already passed is stale.
    """
    generation_id: int
    target_chunk_index: int

    def key(self) -> tuple:
        return (self.generation_id, self.target_chunk_index)


@dataclass
class QueuedInput:
    """A queued event together with its claim.

    Deliberately separate from LatencyTraceRecord: the claim is runtime
    correctness authority, not metrics authority, and the Latency-1B schema was
    just frozen. If stale forensics ever needs expected/observed frontier, that is
    a separate schema migration rather than a field smuggled in now.
    """
    event: InputEvent
    claim: ApplicationClaim


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


@dataclass(frozen=True)
class ReferenceStep:
    """One authoritative chunk's full projection.

    Reference reproducibility compares these step by step rather than comparing
    final poses: a right-then-left pair returns to where it started, so a
    final-state-only comparison would pass a broken sequence.
    """
    chunk_index: int
    generation_id: int
    applied_event_ids: tuple
    camera_pose: tuple
    camera_v: tuple


@dataclass
class CommittedState:
    chunk_index: int = -1
    generation_id: int = 0
    camera: CameraState = field(default_factory=CameraState)
    applied_event_ids: tuple = ()

    # ---- §Latency-1C input watermarks, both derived from the records ----
    # processed: every input_index <= it reached `committed`. This is the strong
    #            "the model consumed it" statement.
    # settled:   no input_index <= it is still pending or in_flight. Weaker: it says
    #            the runtime is done with those inputs, but they may have been
    #            aborted, rejected, gone stale or been reset away rather than
    #            processed.
    #
    # They differ exactly when an input takes a non-committed terminal path, and then
    # the gap is PERMANENT, because `processed` is a contiguous-prefix property and
    # one aborted input stops it forever. Reporting only `processed` would make a
    # healthy-looking deadline tag hide a dropped input; reporting only `settled`
    # would let a client believe an aborted input was consumed.
    processed_input_index: int = 0
    settled_input_index: int = 0


class InteractiveRuntime:
    """Minimal authoritative runtime. Unchanged contract, formal records."""

    def __init__(self, camera: Optional[CameraState] = None):
        self._queue: deque[QueuedInput] = deque()
        self._next_event_id = 1
        self._next_frame_id = 1
        self._next_prewarm_frame_id = 1
        # keyed by event_id => canonicality is structural, not procedural
        self._records: dict[int, LatencyTraceRecord] = {}
        self.committed = CommittedState(
            camera=(camera.copy() if camera is not None else CameraState()))
        self._inflight: Optional[dict] = None
        self._rejected: list[tuple[InputEvent, str]] = []
        # §Latency-1C: per-chunk public record, appended only in commit()
        self._chunk_log: list[CommittedChunk] = []
        # §Latency-3B-B2: explicit rebind bookkeeping. Kept out of
        # LatencyTraceRecord so the 1B schema is not migrated for a transition counter.
        self._rebinds: dict[int, int] = {}
        self._rebases: list[dict] = []

    # ------------------------------------------------------------- frontier
    def _next_free_chunk(self) -> int:
        """The frontier a newly accepted event legitimately targets.

        If a chunk is in flight, that chunk is already claimed, so a new event
        targets the one after it. This is what stops a retry from swallowing input
        that arrived during the retry window -- and it does so by construction,
        not by a check somewhere in the generation loop.
        """
        base = self.committed.chunk_index + 1
        if self._inflight is not None:
            base += 1
        return base

    def frontier(self) -> tuple:
        """The application frontier the NEXT begin_chunk() will realise."""
        return (self.committed.generation_id, self.committed.chunk_index + 1)

    # ------------------------------------------------- §Latency-1C watermarks
    def _input_watermarks(self) -> tuple:
        """(processed, settled), both computed from the records, never stored twice.

        The contiguous-prefix property holds because an event's claim is taken from
        `_next_free_chunk()` at accept time, which is monotone in accept order, and
        `begin_chunk()` assigns every event whose claim equals the frontier. So a
        lower-index input can never still be queued while a higher-index one has
        committed.
        """
        processed = settled = 0
        p_open = s_open = True
        for r in self.records():                 # ordered by event_id
            if p_open and r.terminal_status == COMMITTED:
                processed = r.event_id
            else:
                p_open = False
            if s_open and r.is_terminal():
                settled = r.event_id
            else:
                s_open = False
        return processed, settled

    def _refresh_watermarks(self) -> None:
        self.committed.processed_input_index, \
            self.committed.settled_input_index = self._input_watermarks()

    def committed_chunk(self, chunk_index: int) -> Optional["CommittedChunk"]:
        """What chunk `chunk_index` consumed and what watermark it established."""
        for c in self._chunk_log:
            if c.chunk_index == chunk_index:
                return c
        return None

    def committed_chunks(self) -> list:
        return list(self._chunk_log)

    def frame_lineage(self, meta: "FrameMeta") -> dict:
        """Everything needed to decide whether a frame a client is holding already
        contains a given input, without guessing from elapsed time.

        `strict` is exact and always available. `coarse_upto` is the historical
        watermark of the chunk this frame came from, and is None if that chunk is not
        in the log (a prewarm or warmup frame, or a chunk that never committed).
        """
        c = self.committed_chunk(meta.chunk_index)
        return {
            "frame_id": meta.frame_id,
            "frame_kind": meta.frame_kind,
            "provenance": meta.provenance,
            "chunk_index": meta.chunk_index,
            "generation_id": meta.generation_id,
            "strict": tuple(meta.applied_event_ids),
            "coarse_upto": (c.processed_input_index if c is not None else None),
        }

    # ------------------------------------------------------- §Latency-1C acks
    def accepted_ack(self, event_id: int) -> "InputAcceptedAck":
        """Acceptance is a property of having a record at all: every accepted event
        gets one before it is queued, so this cannot be missing."""
        r = self.record(event_id)
        return InputAcceptedAck(event_id=r.event_id, input_index=r.event_id,
                                t0_accepted_ns=r.t0_accept_ns)

    def processed_ack(self, event_id: int) -> "InputProcessedAck":
        """Raise rather than return a placeholder unless the input really is in a
        committed state. A caller must never be able to mistake "queued" or
        "assigned" for "processed"."""
        r = self.record(event_id)
        if r.terminal_status != COMMITTED or r.t2_commit_ns is None:
            raise RuntimeStateError(
                f"event {event_id} is not processed: status is "
                f"{r.terminal_status!r} (an InputProcessedAck requires t2 and the "
                f"committed status; acceptance or assignment is not processing)")
        return InputProcessedAck(event_id=r.event_id, input_index=r.event_id,
                                 chunk_index=r.assigned_chunk,
                                 generation_id=r.generation_id,
                                 t2_committed_ns=r.t2_commit_ns)

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

        if kind == "reset":
            # A reset is applied AT ACCEPT TIME, not deferred to begin_chunk.
            # Deferring it would let events accepted after the reset carry claims
            # in the old generation and be wrongly invalidated.
            if self._inflight is not None:
                raise RuntimeStateError(
                    "reset while a chunk is in flight: fail or abort it first, "
                    "otherwise the in-flight claim would be silently orphaned")
            for q in self._queue:
                r = self._records[q.event.event_id]
                r.terminal_status = RESET_INVALIDATED
                r.note = "invalidated by reset"
            self._queue.clear()
            self.committed = CommittedState(
                chunk_index=-1, generation_id=self.committed.generation_id + 1,
                camera=self.committed.camera.copy(), applied_event_ids=())
            # The input stream is session-wide, so a reset does NOT rewind the
            # watermark. The inputs it just invalidated are terminal, so `settled`
            # keeps advancing, while `processed` stops at the last real commit --
            # that permanent gap is the honest signal that those inputs were dropped
            # rather than consumed.
            self._refresh_watermarks()
            return ev

        # ---- the immutable claim, bound exactly once, here ----
        claim = ApplicationClaim(generation_id=self.committed.generation_id,
                                 target_chunk_index=self._next_free_chunk())
        self._queue.append(QueuedInput(event=ev, claim=claim))
        return ev

    def reject(self, ev: InputEvent, reason: str = "rejected",
               _now_ns: Optional[int] = None) -> LatencyTraceRecord:
        """Refuse an event before assignment. Still produces a terminal record."""
        if reason not in (REJECTED, STALE):
            raise RuntimeStateError(f"reject reason must be {REJECTED!r} or "
                                    f"{STALE!r}, got {reason!r}")
        self._queue = deque(q for q in self._queue
                            if q.event.event_id != ev.event_id)
        rec = self._records[ev.event_id]
        if rec.is_terminal():
            raise RuntimeStateError(
                f"event {ev.event_id} is already terminal "
                f"({rec.terminal_status}); terminal is terminal")
        rec.terminal_status = reason
        rec.note = f"refused before assignment at generation " \
                   f"{self.committed.generation_id}"
        self._rejected.append((ev, reason))
        self._refresh_watermarks()
        return rec

    def pending(self) -> list[InputEvent]:
        return [q.event for q in self._queue]

    def queued(self) -> list[QueuedInput]:
        """Queue with claims, for correctness assertions. Tests should inspect
        claims and runtime state directly rather than parsing `note`, which is
        human diagnostics and must not become a second authority."""
        return list(self._queue)

    def claim_of(self, event_id: int) -> Optional[ApplicationClaim]:
        for q in self._queue:
            if q.event.event_id == event_id:
                return q.claim
        return None

    # ------------------------------------------------------------- lifecycle
    def begin_chunk(self, _now_ns: Optional[int] = None) -> dict:
        """t1. Bind the events whose claim matches this frontier, and MATERIALISE
        the immutable in-flight candidate camera.

        The claim partition runs BEFORE the camera reduction, so a stale event can
        never contribute to a candidate even transiently.
        """
        if self._inflight is not None:
            raise RuntimeStateError("begin_chunk while a chunk is in flight")

        chunk_index = self.committed.chunk_index + 1
        current = (self.committed.generation_id, chunk_index)

        assigned, still_pending, stale = [], [], []
        for q in self._queue:
            c = q.claim.key()
            if c == current:
                assigned.append(q)
            elif c > current:
                still_pending.append(q)      # future frontier: not due yet
            else:
                stale.append(q)              # claim already expired

        # ---- fail closed on stale, BEFORE any camera work ----
        for q in stale:
            r = self._records[q.event.event_id]
            if r.is_terminal():
                # terminal is terminal: a canonical record is never reclassified
                raise RuntimeStateError(
                    f"event {q.event.event_id} is already terminal "
                    f"({r.terminal_status}); a terminal record must not be "
                    f"re-entered into the frontier")
            r.terminal_status = STALE
            r.note = "application frontier passed"
            self._rejected.append((q.event, STALE))
        self._queue = deque(still_pending)

        if stale:
            # stale events just became terminal, so `settled` advances past them and
            # `processed` does not: an input that expired at the frontier was never
            # consumed.
            self._refresh_watermarks()

        t1 = _now_ns if _now_ns is not None else time.perf_counter_ns()
        for q in assigned:
            r = self._records[q.event.event_id]
            # §Latency-3B-B1 / C3: t1 is WRITE-ONCE.
            #
            # Re-binding the same claim -- which is what a preemption replay does when
            # it returns a cancelled batch and begins the chunk again -- must not
            # overwrite the original assignment time. Overwriting would silently
            # lengthen accept_to_assign and, worse, would make the recorded assignment
            # time depend on how many times the chunk was attempted rather than on when
            # the input was actually assigned.
            #
            # This does not change generation output: t1 is observation, not behaviour.
            if r.t1_assign_ns is None:
                r.t1_assign_ns = t1
            r.assigned_chunk = chunk_index
            r.generation_id = self.committed.generation_id
            r.terminal_status = IN_FLIGHT

        base_camera = self.committed.camera.copy()
        # ---- the single materialisation of this chunk's camera authority ----
        candidate_camera = reduce_controls(base_camera,
                                           [q.event for q in assigned])

        snapshot = dict(
            chunk_index=chunk_index,
            generation_id=self.committed.generation_id,
            applied_event_ids=tuple(q.event.event_id for q in assigned),
            base_camera=base_camera,
            candidate_camera=candidate_camera,
            events=[q.event for q in assigned],
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
        self._refresh_watermarks()
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
        # §Latency-1C: this is THE point at which inputs become processed, so the
        # watermark is refreshed here and nowhere else claims to advance it.
        self._refresh_watermarks()
        # the public per-chunk record, appended from the same state, so it cannot
        # drift from the records it summarises
        self._chunk_log.append(CommittedChunk(
            chunk_index=snap["chunk_index"], generation_id=snap["generation_id"],
            applied_event_ids=tuple(snap["applied_event_ids"]),
            processed_input_index=self.committed.processed_input_index,
            settled_input_index=self.committed.settled_input_index,
            t2_committed_ns=t2))
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
        """t3. Only a real frame may set t3, only for the events it carries, and
        only AFTER the chunk that produced it has been successfully committed.

        The ordering is enforced here rather than left to the caller. Without it a
        refused commit could leave t3 already set, so a rejected frame would claim
        to be the first real frame -- and the measured input->first-real latency
        would describe a frame whose generation was never accepted.
        """
        if meta.frame_kind != "real":
            raise RuntimeStateError(f"t3 requires a real frame, got "
                                    f"{meta.frame_kind!r}")
        if meta.provenance != "authoritative":
            raise RuntimeStateError(
                f"refusing t3 for a {meta.provenance!r} frame: prewarm and "
                f"warmup frames are a different provenance domain")
        if meta.generation_id != self.committed.generation_id:
            raise RuntimeStateError(
                f"refusing t3: frame generation {meta.generation_id} != "
                f"committed generation {self.committed.generation_id}")
        if meta.chunk_index > self.committed.chunk_index:
            raise RuntimeStateError(
                f"refusing t3: chunk {meta.chunk_index} is not committed yet "
                f"(committed is {self.committed.chunk_index}). A real frame must "
                f"not be marked before its chunk commits.")
        t3 = _now_ns if _now_ns is not None else time.perf_counter_ns()
        for eid in meta.applied_event_ids:
            r = self._records.get(eid)
            if r is None:
                raise RuntimeStateError(f"unknown event id {eid} in frame meta")
            if r.t3_first_real_ns is None:
                r.t3_first_real_ns = t3
                r.first_real_frame_id = meta.frame_id

    def rebase_chunk(self, reason: str = "") -> dict:
        """§Latency-3B-B2. Logically cancel the in-flight attempt and open a NEW
        generation for the SAME chunk_index, carrying the same batch forward.

        WHAT THIS IS NOT: it is not a kernel kill. Python cannot interrupt a launched
        CUDA kernel. The caller must only invoke this at a forward boundary, after
        synchronising, so no work for the discarded attempt is still on the stream.

        WHY IT MUST TOUCH THE CLAIMS, and why that is a deliberate exception rather
        than a relaxation:

        `begin_chunk` partitions the queue by `claim.key() == (generation_id,
        chunk_index)`. Bumping the generation without re-claiming would make the batch's
        existing claims compare LESS than the new frontier, so they would be classified
        stale and dropped -- which is precisely what a rebase must not do. The claim
        therefore has to move with the generation.

        The invariant is restated rather than abandoned:

            accept()   binds a claim exactly once
            rebase()   rebinds it exactly once, explicitly, and is recorded
            nothing else ever rebinds

        `t1` is NOT rewritten (§Latency-3B-B1 / C3): t1 is when the input was first
        assigned, and that does not change because the chunk was attempted twice.
        """
        if self._inflight is None:
            raise RuntimeStateError("rebase_chunk without an in-flight chunk")
        snap = self._inflight
        events = list(snap["events"])
        stale_note = reason or "preempted at a forward boundary"

        for eid in snap["applied_event_ids"]:
            r = self._records[eid]
            if r.is_terminal():
                raise RuntimeStateError(
                    f"cannot rebase event {eid}: it is already terminal "
                    f"({r.terminal_status})")
            self._rebinds[eid] = self._rebinds.get(eid, 0) + 1
            # back into the pending pool: NOT terminal, NOT aborted. The input's
            # intent survives; only the attempt is discarded.
            r.terminal_status = PENDING
            r.note = f"{stale_note} (rebind {self._rebinds[eid]})"

        self._inflight = None
        # a FRESH generation for the new attempt, at the SAME chunk_index, so any
        # artefact of the discarded attempt fails a generation check rather than
        # silently matching
        self.committed.generation_id += 1
        target = self.committed.chunk_index + 1
        claim = ApplicationClaim(generation_id=self.committed.generation_id,
                                 target_chunk_index=target)
        for ev in events:
            self._queue.append(QueuedInput(event=ev, claim=claim))
        self._rebases.append(dict(chunk_index=snap["chunk_index"],
                                  from_generation=snap["generation_id"],
                                  to_generation=self.committed.generation_id,
                                  event_ids=tuple(snap["applied_event_ids"]),
                                  reason=stale_note))
        return self._rebases[-1]

    def rebase_count(self, event_id: int) -> int:
        """How many times this input has been carried across a rebase."""
        return self._rebinds.get(event_id, 0)

    def rebases(self) -> list:
        return list(self._rebases)

    def mark_renderer_submit(self, meta: FrameMeta,
                             _now_ns: Optional[int] = None) -> None:
        """t4. The authoritative frame was handed to a real, non-dummy display backend.

        §Latency-1D adds this as the first writer of `t4_renderer_submit_ns`, which
        §Latency-1B froze to None with no writer at all. The same shape as
        mark_real_decoded, and the same reasons:

        * only an authoritative frame, only after its chunk committed -- a submit
          timestamp for a frame whose generation was never accepted would describe a
          frame that does not exist;
        * WRITE-ONCE. One authoritative frame is redrawn many times (the blend emits
          several compositions), but only the first qualifying submission is t4.
          Later redraws are separate display events, and letting one overwrite t4
          would silently redefine the metric as "the last time it was drawn".

        `_now_ns` is supplied by the CALLER because the submit happens on the UI thread,
        which is the only place with a display, while this runtime is mutated by a
        single worker thread. The timestamp is therefore taken at the submit and merely
        recorded here: a t4 read by the worker when it got around to it would be wrong
        by however long the worker was busy.
        """
        if meta.frame_kind != "real":
            raise RuntimeStateError(f"t4 requires a real frame, got "
                                    f"{meta.frame_kind!r}")
        if meta.provenance != "authoritative":
            raise RuntimeStateError(
                f"refusing t4 for a {meta.provenance!r} frame: prewarm and warmup "
                f"frames never reach a display")
        if meta.generation_id != self.committed.generation_id:
            raise RuntimeStateError(
                f"refusing t4: frame generation {meta.generation_id} != committed "
                f"generation {self.committed.generation_id}")
        if meta.chunk_index > self.committed.chunk_index:
            raise RuntimeStateError(
                f"refusing t4: chunk {meta.chunk_index} is not committed yet "
                f"(committed is {self.committed.chunk_index})")
        for eid in meta.applied_event_ids:
            r = self._records.get(eid)
            if r is None:
                raise RuntimeStateError(f"unknown event id {eid} in frame meta")
            if r.t4_renderer_submit_ns is not None:
                raise RuntimeStateError(
                    f"refusing t4: frame {meta.frame_id} already has a first "
                    f"renderer submit (write-once); a redraw is a display event, "
                    f"not a new authoritative submit")
            if r.t3_first_real_ns is not None and _now_ns is not None \
                    and _now_ns < r.t3_first_real_ns:
                raise RuntimeStateError(
                    f"refusing t4 for event {eid}: submit {_now_ns} precedes its "
                    f"decode {r.t3_first_real_ns}")
        t4 = _now_ns if _now_ns is not None else time.perf_counter_ns()
        for eid in meta.applied_event_ids:
            self._records[eid].t4_renderer_submit_ns = t4

    def renderer_submit_record(self, meta: FrameMeta) -> dict:
        """The t4 lineage view. A view over existing authority, never a second copy."""
        t4s = [self._records[e].t4_renderer_submit_ns
               for e in meta.applied_event_ids if e in self._records]
        t4s = [t for t in t4s if t is not None]
        return {
            "frame_id": meta.frame_id,
            "generation_id": meta.generation_id,
            "source_chunk_index": meta.chunk_index,
            "applied_event_ids": tuple(meta.applied_event_ids),
            "t4_renderer_submit_ns": (min(t4s) if t4s else None),
        }

    # ---------------------------------------------------------------- traces
    def record(self, event_id: int) -> LatencyTraceRecord:
        try:
            return self._records[event_id]
        except KeyError:
            raise RuntimeStateError(f"no record for event id {event_id}") from None

    def records(self) -> list[LatencyTraceRecord]:
        return [self._records[k] for k in sorted(self._records)]

    # ---------------------------------------------------- prewarm provenance
    def prewarm_frame_meta(self, frame_kind: str = "real") -> FrameMeta:
        """A frame from a DIFFERENT provenance domain.

        It consumes a prewarm-local id, never the authoritative frame id space, so
        a prewarm pass cannot shift the numbering that lineage and reproducibility
        depend on.
        """
        m = FrameMeta(frame_id=self._next_prewarm_frame_id, frame_kind=frame_kind,
                      chunk_index=-1, generation_id=-1, applied_event_ids=(),
                      provenance="prewarm")
        self._next_prewarm_frame_id += 1
        return m

    def authoritative_fingerprint(self) -> dict:
        """Everything a prewarm pass is forbidden to touch.

        Includes the ID counters, because a prewarm that leaves the camera alone
        but advances `_next_frame_id` would still poison lineage.
        """
        c = self.committed
        def cam_bytes(cs):
            p = np.asarray(cs.pose) if cs.pose is not None else np.array([])
            v = np.asarray(cs.v) if cs.v is not None else np.array([])
            return (p.tobytes() if p.size else b"", v.tobytes() if v.size else b"")
        pose, vel = cam_bytes(c.camera)
        return dict(
            chunk_index=c.chunk_index,
            generation_id=c.generation_id,
            camera_pose=pose,
            camera_v=vel,
            camera_gate=c.camera.gate,
            applied_event_ids=tuple(c.applied_event_ids),
            # §Latency-1C: watermarks and chunk-log length belong to authoritative
            # state, so a prewarm pass must be unable to move them either.
            processed_input_index=c.processed_input_index,
            settled_input_index=c.settled_input_index,
            committed_chunks=len(self._chunk_log),
            queue=[(q.event.event_id, q.claim.key()) for q in self._queue],
            records=[(r.event_id, r.terminal_status, r.t1_assign_ns,
                      r.t2_commit_ns, r.t3_first_real_ns, r.t4_renderer_submit_ns,
                      r.assigned_chunk,
                      r.first_real_frame_id) for r in self.records()],
            next_event_id=self._next_event_id,
            next_frame_id=self._next_frame_id,
        )

    @contextmanager
    def prewarm_scope(self):
        """Run prewarm work in a separate provenance domain, and VERIFY it.

        On exit, normal or exceptional, the authoritative fingerprint must be
        unchanged. A partial prewarm that raised is therefore also proven to have
        polluted nothing -- which is the case most likely to be missed, because the
        happy path is what usually gets tested.
        """
        before = self.authoritative_fingerprint()
        try:
            yield self
        finally:
            after = self.authoritative_fingerprint()
            if before != after:
                diff = [k for k in before if before[k] != after[k]]
                raise RuntimeStateError(
                    f"prewarm polluted authoritative state: {diff}")

    def committed_projection(self) -> ReferenceStep:
        """The full projection of the current authoritative chunk."""
        c = self.committed
        pose = (tuple(np.asarray(c.camera.pose).ravel())
                if c.camera.pose is not None else ())
        vel = (tuple(np.asarray(c.camera.v).ravel())
               if c.camera.v is not None else ())
        return ReferenceStep(chunk_index=c.chunk_index,
                             generation_id=c.generation_id,
                             applied_event_ids=tuple(c.applied_event_ids),
                             camera_pose=pose, camera_v=vel)

    def export(self) -> list[dict]:
        """Raw-only export. Derived values are recomputed on load."""
        return [r.to_dict() for r in self.records()]

    @staticmethod
    def import_records(blobs: list[dict]) -> list[LatencyTraceRecord]:
        return [LatencyTraceRecord.from_dict(b) for b in blobs]
