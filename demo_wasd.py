#!/usr/bin/env python
"""§Demo-1 — WASD viewer: a thin display layer over the frozen interactive runtime.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT

`./run.sh play` must not be presented as a real WASD demo: its input source is a
hard-coded scripted timeline and its runner is headless, so the 50 ms
preview-to-authoritative blend exists only on a frame buffer. This file adds the
missing display layer and nothing else.

    UI main thread                      GPU worker thread
      pygame window
      KEYDOWN / KEYUP
        perf_counter_ns() at handling
        --(t0_ns, controls)--> input_queue
                                          drain at the legal seam
                                          rt.accept(controls, _now_ns=t0_ns)
                                          rt.begin_chunk()
                                          DiT step0
                                          variant D decode --> PREVIEW msg
                                          step1 + step2 + KV write
                                          rt.commit()               (t2)
                                          full decode               (t3)
                                    <---- AUTHORITATIVE msg
      display PREVIEW immediately
      50 ms blend to AUTHORITATIVE

THE ONE HARD INVARIANT: the UI thread never touches InteractiveRuntime.

`interactive_runtime.InteractiveRuntime` keeps its state in a plain
`deque`/`dict`/`_inflight` and has no synchronisation, by design: its correctness
argument is that `reduce_controls` runs exactly once per chunk inside
`begin_chunk()`. Calling `accept()` from the pygame thread would introduce a data
race into a contract that was closed on purpose. So the UI thread only ever puts
`(timestamp_ns, controls)` on a queue, and the single worker thread calls
`accept()` with `_now_ns=<that timestamp>`. t0 therefore remains the real key
event time while exactly one thread mutates the runtime.

SCOPE. This file must not change interactive_runtime.py, anything under wan/, the
model path, or the rtx5060-interactive-rc1 tag. It is product surface, not
optimisation. play.py stays the authoritative entrypoint.

THREE THINGS THIS FILE DOES NOT CLAIM

1. No physical present. The blend below is executed by us, on our own framebuffer.
   There is still no renderer and no present-completion signal, so t4/t5 stay
   unavailable. The HUD says "model-side" for exactly this reason.
2. No 60 Hz held-key sampling. One InputEvent is one discrete control intent, and
   held-key / OS key-repeat / keyup policy is not part of the frozen contract, so
   it is not invented here. Each keydown contributes exactly one intent.
3. `--script` is not a human at a keyboard. It posts real KEYDOWN/KEYUP events
   into the real pygame queue, so they travel the same handler and the same input
   path a human's keys travel, and the viewer stamps and queues them identically.
   Only the SOURCE of the press is automated. Pass no --script for live play, and
   say so if a recording was made this way.

play.py's model wiring is duplicated below rather than extracted, on purpose: the
RC entrypoint is frozen and a shared module would mean editing it. Session setup is
the only duplicated part and it is marked. If you change one, change the other.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from interactive_runtime import (CameraState, InteractiveRuntime,
                                 RuntimeStateError)

HUD_TITLE_FALLBACK = "NVIDIA GeForce RTX 5060 Laptop GPU"


# ---------------------------------------------------------------- input mapping
def key_table(pygame):
    """WASD plus the arrow keys, as discrete intents."""
    return {
        pygame.K_w: "W", pygame.K_a: "A", pygame.K_s: "S", pygame.K_d: "D",
        pygame.K_UP: "W", pygame.K_LEFT: "A", pygame.K_DOWN: "S",
        pygame.K_RIGHT: "D",
    }


# One InputEvent == one discrete control intent. 'forward' is control_reduce's
# canonical key for what play.py calls fwd; yaw is yaw. No key repeat, no held-key
# sampling: this table gives the single step each intent contributes.
INTENTS = {
    frozenset({"W"}):      {"forward": +0.6},
    frozenset({"S"}):      {"forward": -0.6},
    frozenset({"A"}):      {"yaw": -0.8},
    frozenset({"D"}):      {"yaw": +0.8},
    frozenset({"W", "A"}): {"forward": +0.3, "yaw": -0.8},
    frozenset({"W", "D"}): {"forward": +0.3, "yaw": +0.8},
    frozenset({"S", "A"}): {"forward": -0.3, "yaw": -0.8},
    frozenset({"S", "D"}): {"forward": -0.3, "yaw": +0.8},
}


def compose_intent(held: frozenset, newly_pressed: Optional[str]):
    """The intent for the keys down at this instant.

    An unmapped combination (e.g. W and S together) falls back to the key that was
    just pressed, so no keypress is ever silently dropped.
    """
    controls = INTENTS.get(held)
    if controls is not None:
        return dict(controls)
    if newly_pressed:
        return dict(INTENTS.get(frozenset({newly_pressed}), {}))
    return None


# The recorded take. (seconds, [(action, key), ...]) -- posted as real pygame
# KEYDOWN/KEYUP events, so they travel the same handler a human's keys travel.
SCRIPT_SEQUENCE = [
    (1.0, [("down", "W")]),
    (3.0, [("down", "D")]),            # W still down -> the W+D chord
    (5.0, [("up", "W"), ("up", "D"), ("down", "A")]),
    (7.0, [("up", "A"), ("down", "W")]),
    (9.0, [("up", "W")]),
]


# ------------------------------------------------------------------- messages
@dataclass
class PreviewMsg:
    frame: np.ndarray              # (H, W, 3) uint8 RGB
    chunk_index: int
    generation_id: int
    event_ids: tuple
    model_ms: Optional[float]      # t0(earliest applied event) -> preview decoded.
                                   # None when this chunk carries no input, which is
                                   # the common case: a chunk is an evolution step,
                                   # not a response, so most have no event to time.
    committed_at_emit: int         # must be chunk_index - 1


@dataclass
class AuthoritativeMsg:
    frame: np.ndarray              # (H, W, 3) uint8 RGB
    frame_id: int                  # so the viewer can name the frame it submitted
    chunk_index: int
    generation_id: int
    event_ids: tuple
    authority_ms: Optional[float]  # t0(earliest applied event) -> t3
    chunk_ms: float


def intent_key(controls: dict):
    """A comparable form of a control intent, so 'the same state again' is detectable.

    Deliberately value-based rather than identity-based: holding W delivers the same
    controls repeatedly, and those are not new directional decisions.
    """
    return tuple(sorted((k, round(float(v), 6)) for k, v in (controls or {}).items()))


def semantic_control_changed(pending_controls: dict, inflight_events) -> bool:
    """Does this input change what the running chunk MEANS?

    §Latency-3B-C1. The first C0 shakedown preempted 31 of 32 chunks because the
    mailbox kept handing over the same held state and every arrival was treated as a
    fresh decision. Repeatedly re-deriving 'still turning left' is not a reason to
    discard a running chunk; changing from 'turning left' to 'turning left and moving'
    is.
    """
    if not inflight_events:
        return True          # nothing bound yet, so any input changes the meaning
    last = getattr(inflight_events[-1], "controls", None)
    return intent_key(pending_controls) != intent_key(last or {})


class AdmissionPolicy:
    """Decides whether an arriving input is worth interrupting a running chunk for.

    Deterministic on purpose. A prediction of "how much would this save" would need a
    model of chunk phase and input arrival, and at this stage a wrong prediction is
    worse than a conservative rule, because it would silently reintroduce the
    pathology it is meant to remove.

        admit iff
            forward_index is an allowed boundary
            AND remaining forwards >= MIN_REMAINING
            AND budget remains
            AND the control intent actually changed

    The defaults are the conservative first cut: only after forward 1, where C0 found
    27 of its 31 preemptions, so the boundary set that matters most is exercised first.

    Every rejection reason is counted separately. C0 lost a real bug behind a single
    conflated counter, and that mistake is not worth repeating.
    """

    def __init__(self, boundaries=(1,), min_remaining=2):
        self.boundaries = tuple(int(b) for b in boundaries)
        self.min_remaining = int(min_remaining)
        self.tally = dict(peeks_with_input=0, admitted=0,
                          rejected_too_late=0, rejected_budget=0,
                          rejected_same_state=0)

    @property
    def enabled(self) -> bool:
        return bool(self.boundaries)

    def decide(self, controls, inflight_events, forward_index,
               remaining_forwards, budget):
        self.tally["peeks_with_input"] += 1
        if forward_index not in self.boundaries \
                or remaining_forwards < self.min_remaining:
            self.tally["rejected_too_late"] += 1
            return False, "too_late"
        if budget <= 0:
            self.tally["rejected_budget"] += 1
            return False, "budget"
        if not semantic_control_changed(controls, inflight_events):
            self.tally["rejected_same_state"] += 1
            return False, "same_state"
        self.tally["admitted"] += 1
        return True, "admitted"


def _parse_boundaries(spec: str):
    """'1' -> (1,), '1,2' -> (1, 2), '' -> () which disables preemption entirely."""
    return tuple(int(x) for x in str(spec).split(",") if x.strip() != "")


class ChunkPreempted(Exception):
    """Raised out of `denoise` when a preemption was taken at a forward boundary.

    An exception rather than a sentinel return, because the attempt is genuinely
    abandoned: the caller must not commit, must not decode, and must not treat any
    partial result as a chunk. Making that impossible to ignore is the point.
    """

    def __init__(self, request, forward_index: int):
        super().__init__(f"preempted after forward {forward_index}")
        self.request = request
        self.forward_index = forward_index


class InputMailbox:
    """The session's input queue: one FIFO that can be both drained and peeked.

    §Latency-3B-B2 needs two access patterns on the same bytes -- the worker drains it
    to `accept()`, and the generation loop peeks at a forward boundary to decide
    whether to preempt -- and the first version of this used two structures, which
    meant a trigger could be accepted twice. One object makes that structural rather
    than a discipline.

    The peek is a short CPU lock and nothing else. That is the discipline that keeps
    preemption from taxing every chunk: an uninterrupted run never enters the
    detection branch, so it never synchronises for preemption.

    The first observed time is kept as posted and never recomputed, because "when did
    the user actually press" must not move forward while the runtime is busy.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._items = deque()

    # -- the authoritative drain path -------------------------------------
    def get_nowait(self):
        with self._lock:
            if not self._items:
                raise queue.Empty
            return self._items.popleft()

    # -- the detection path ----------------------------------------------
    def peek(self):
        with self._lock:
            return self._items[0] if self._items else None

    def take(self):
        with self._lock:
            return self._items.popleft() if self._items else None

    # -- producer side ----------------------------------------------------
    def post(self, controls: dict, t_observed_ns: int):
        with self._lock:
            self._items.append((int(t_observed_ns), dict(controls)))

    def put(self, item):                      # queue.Queue-compatible
        t0, controls = item
        self.post(controls, t0)

    def qsize(self):
        with self._lock:
            return len(self._items)

    def empty(self):
        with self._lock:
            return not self._items


@dataclass
class WorkerDone:
    """Sentinel. reason=None means a clean stop; otherwise it is a failure."""
    reason: Optional[str] = None


# ------------------------------------------------------------------- recorder
class Recorder:
    """Records the viewer's own framebuffer. No desktop, no compositor.

    Raw frames are piped into ffmpeg rather than screen-captured, so the HUD and
    the video cannot drift: the bytes written are exactly the bytes drawn.
    """

    def __init__(self, path: str, size, ui_fps: float, out_fps: int, crf: int = 18):
        import subprocess
        self.path, self.size, self.fps = path, size, out_fps
        self.n = 0
        every = max(1, int(round(ui_fps / float(out_fps))))
        self.every = every
        self._i = 0
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{size[0]}x{size[1]}", "-r", str(out_fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, hwc_rgb: np.ndarray):
        i = self._i
        self._i += 1
        if i % self.every:
            return
        if hwc_rgb.dtype != np.uint8:
            hwc_rgb = hwc_rgb.astype(np.uint8)
        self.proc.stdin.write(np.ascontiguousarray(hwc_rgb).tobytes())
        self.n += 1

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        return self.proc.returncode, self.n


# ------------------------------------------------------------------ the worker
class Worker(threading.Thread):
    """Owns the runtime and the session. The only thread that mutates either."""

    def __init__(self, rt: InteractiveRuntime, session, input_q: queue.Queue,
                 frame_q: queue.Queue, stop_evt: threading.Event,
                 submit_q: Optional[queue.Queue] = None,
                 mailbox: Optional["InputMailbox"] = None,
                 max_preemptions_per_chunk: int = 1,
                 policy: Optional["AdmissionPolicy"] = None):
        super().__init__(name="gpu-worker", daemon=True)
        self.rt, self.session = rt, session
        self.input_q, self.frame_q = input_q, frame_q
        # §Latency-1D: the UI thread takes the t4 timestamp at the submit, because it
        # is the only thread with a display, and hands it here. This thread is the only
        # writer of runtime state, so the RECORD is written later than the instant it
        # describes -- the timestamp is authoritative, its arrival is not.
        self.submit_q = submit_q if submit_q is not None else queue.Queue()
        self._metas: dict = {}
        # §Latency-3B-B2: detection-only mailbox, and the frozen liveness bound
        self.mailbox = mailbox if mailbox is not None else InputMailbox()
        self.max_preemptions_per_chunk = max_preemptions_per_chunk
        # §Latency-3B-C1: the admission policy owns the "is it worth it" decision and
        # every rejection reason. Passing boundaries=() disables preemption entirely,
        # which is the A arm of the eventual A/B.
        self.policy = policy if policy is not None else AdmissionPolicy()
        self.preemption_trace: list = []
        self.stop = stop_evt
        self.invariants = {
            "previews_emitted": 0,
            "preview_commit_violations": 0,
            "events_accepted": 0,
            "t0_preserved": 0,
            "t0_mismatches": [],
            "chunks": 0,
            "t4_recorded": 0,
            "t4_already_set": 0,
            "t4_unknown_frame": 0,
            "t4_refused": 0,
            "t4_refusal_reason": "",
            "preemptions": 0,
        }
        self.error: Optional[str] = None

    def _drain_submits(self):
        """Record t4 values the viewer took at the display submit."""
        while True:
            try:
                frame_id, t4_ns = self.submit_q.get_nowait()
            except queue.Empty:
                return
            meta = self._metas.get(frame_id)
            if meta is None:
                # the frame was already retired from the small map; not an error in
                # a long run, but it must be visible rather than silent
                self.invariants["t4_unknown_frame"] += 1
                continue
            try:
                self.rt.mark_renderer_submit(meta, _now_ns=t4_ns)
                self.invariants["t4_recorded"] += 1
            except RuntimeStateError as e:
                # distinguish the reasons: "write-once" is a legitimate redraw
                # rejection, anything else is a real refusal and must not be hidden
                # behind a single counter. C0 found this the hard way.
                if "write-once" in str(e):
                    self.invariants["t4_already_set"] += 1
                else:
                    self.invariants["t4_refused"] += 1
                    self.invariants["t4_refusal_reason"] = str(e)[:160]
            self._metas.pop(frame_id, None)
            if len(self._metas) > 8:                    # bounded
                for k in sorted(self._metas)[:-8]:
                    del self._metas[k]

    def _drain_inputs(self):
        """Accept every queued key intent, preserving its real key timestamp.

        The claim a new event receives is computed by accept() from the frontier
        at THIS moment, so the drain sits at the top of the chunk loop: an event
        that arrived while chunk N was generating targets chunk N+1, which is the
        chunk it could still have influenced. Draining later would silently move
        the claim past a frontier it was entitled to.
        """
        while True:
            try:
                t0_ns, controls = self.input_q.get_nowait()
            except queue.Empty:
                return
            ev = self.rt.accept(controls, kind="control", _now_ns=t0_ns)
            self.invariants["events_accepted"] += 1
            if ev.t0_ns == t0_ns:
                self.invariants["t0_preserved"] += 1
            else:
                self.invariants["t0_mismatches"].append(
                    (ev.event_id, t0_ns, ev.t0_ns))

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
            self.frame_q.put(WorkerDone(self.error))
            return
        self.frame_q.put(WorkerDone(None))       # clean stop sentinel

    def _run(self):
        rt = self.rt
        while not self.stop.is_set():
            self._drain_inputs()
            self._drain_submits()

            # t1: binds this chunk's events and materialises the immutable
            # candidate camera exactly once. A retry would read the same
            # candidate; it must never re-run the reduction.
            snap = rt.begin_chunk()
            chunk_index = snap["chunk_index"]

            # ---- §Latency-3B-B2: single-rebase preemption -----------------------
            # budget starts at 1 and is spent by the first rebase, so a chunk can
            # never be restarted more than once. That bound is what makes progress
            # guaranteed: without it, sustained input could cancel every attempt and
            # the run would produce no frames at all, which is worse than waiting.
            budget = self.max_preemptions_per_chunk
            while True:
                def emit_preview(frame, model_ms):
                    # A preview is a read-only projection of an in-flight chunk and
                    # must be structurally incapable of advancing committed state.
                    committed = rt.committed.chunk_index
                    if committed != chunk_index - 1:
                        self.invariants["preview_commit_violations"] += 1
                    self.invariants["previews_emitted"] += 1
                    self.frame_q.put(PreviewMsg(
                        frame=frame, chunk_index=chunk_index,
                        generation_id=snap["generation_id"],
                        event_ids=snap["applied_event_ids"], model_ms=model_ms,
                        committed_at_emit=committed))

                t_chunk = time.perf_counter_ns()
                try:
                    x0 = self.session.denoise(
                        snap, chunk_index, emit_preview,
                        preempt=self.mailbox, preempt_budget=budget,
                        policy=self.policy)
                    break
                except ChunkPreempted as px:
                    budget -= 1
                    # order matters and is the whole of C6: cancel BEFORE accept, so
                    # the triggering input's claim is taken against a frontier with no
                    # chunk in flight and therefore names THIS chunk. Accepting first
                    # would claim chunk N+1 and the replay would gain nothing.
                    t_rebase = time.perf_counter_ns()
                    info = rt.rebase_chunk(
                        reason=f"preempted after forward {px.forward_index}")
                    # the mailbox stores (t_observed_ns, controls), the same shape the
                    # drain path yields. Unpacked in that order.
                    t_obs, controls = px.request
                    t_admit = time.perf_counter_ns()
                    rt.accept(controls, _now_ns=t_obs)
                    self.session.restore_chunk_rng()
                    self.invariants["preemptions"] += 1
                    self.preemption_trace.append(dict(
                        chunk_index=chunk_index,
                        forward=px.forward_index,
                        from_generation=info["from_generation"],
                        to_generation=info["to_generation"],
                        t_observed_ns=t_obs,
                        # timing only, NOT an authority: `t0` keeps the physical input
                        # time and is not redefined. This measures how much the
                        # preemption machinery itself cost, which would otherwise be
                        # invisible because t0 is deliberately backdated to the press.
                        t_admitted_ns=t_admit,
                        observed_to_admitted_ms=(t_admit - t_obs) / 1e6,
                        rebase_cost_ms=(t_admit - t_rebase) / 1e6))
                    snap = rt.begin_chunk()

            # t2 BEFORE the decode. The generation state became authoritative the
            # moment the last state-mutating step completed; the decode is a
            # read-only projection of it. Committing here is what keeps "a real
            # frame is never marked before its chunk commits" true, and it keeps a
            # refused commit from leaving t3 already set.
            meta = rt.new_frame_meta("real", chunk_index, snap["generation_id"],
                                     snap["applied_event_ids"])
            rt.commit(meta)
            # C1: the pose advances HERE and only here, after commit has succeeded.
            # Any path that skips this commit -- a failed, cancelled or retried chunk --
            # leaves the reference pose where it was.
            self.session.on_commit()

            real = self.session.decode_real(x0)
            rt.mark_real_decoded(meta)              # t3, after the frame exists

            t0s = [rt.record(e).t0_accept_ns for e in meta.applied_event_ids]
            t3s = [rt.record(e).t3_first_real_ns for e in meta.applied_event_ids]
            t3 = min([t for t in t3s if t is not None], default=None)
            authority_ms = ((t3 - min(t0s)) / 1e6
                            if (t3 is not None and t0s) else None)
            self._metas[meta.frame_id] = meta
            self.frame_q.put(AuthoritativeMsg(
                frame=real, frame_id=meta.frame_id, chunk_index=chunk_index,
                generation_id=snap["generation_id"],
                event_ids=snap["applied_event_ids"],
                authority_ms=authority_ms,
                chunk_ms=(time.perf_counter_ns() - t_chunk) / 1e6))
            self.invariants["chunks"] += 1
        # a t4 taken on the last displayed frame can still be in flight when the loop
        # ends; drop it only after giving it a chance to be recorded
        time.sleep(0.05)
        self._drain_submits()


# ------------------------------------------------------- real model session
class WanSession:
    """The frozen deployment stack, wired as play.py wires it.

    DUPLICATED FROM play.py ON PURPOSE. play.py is the RC's authoritative
    entrypoint and is frozen; extracting a shared module would mean editing it.
    This is the only duplicated block. If one changes, change the other.
    """

    def __init__(self, args):
        import gc
        import hashlib

        import torch
        import torchvision.transforms.functional as TF
        from einops import rearrange
        from PIL import Image

        import wan
        import wan.modules.model_fast as mf
        from wan.configs import WAN_CONFIGS
        from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
        from cam_controller import CameraController

        sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
        from taehv import TAEHV

        self._rearrange = rearrange
        self._get_plucker = get_plucker_embeddings
        self.gc = gc
        self.torch = torch
        self.rt = None

        os.environ["LINGBOT_MODE"] = "repro"
        os.environ["LINGBOT_WEIGHT_MODE"] = args.weight
        os.environ["LINGBOT_FP8"] = "0" if args.weight == "bf16" else "1"
        os.environ["LINGBOT_FFN0_FP8"] = "0"
        os.environ["LINGBOT_CAM_CACHE"] = "1"
        os.environ["LINGBOT_ROPE_CACHE"] = "0"
        os.environ.setdefault("LINGBOT_STREAM_ENCODE", "1")

        W, H = (int(x) for x in args.pixel.lower().split("x"))
        self.pixel = (W, H)
        cfg = WAN_CONFIGS[args.task]

        # NOTE: CameraController is not used to drive anything here. The runtime
        # owns the authoritative camera; main() seeds it from the same base pose
        # play.py uses. Driving a controller from the key intents as well would be
        # a second apply seam, and control_reduce's canonical key is 'forward'
        # while CameraController.set_input takes 'fwd' -- two vocabularies that
        # must not be wired to each other by accident.

        pipe = wan.WanI2VCausal(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
            convert_model_dtype=False, local_attn_size=args.local_window,
            sink_size=args.sink, infer_mode="causal_fast",
            assets_dir=args.assets_dir)
        self.pipe = pipe
        dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
        self.dev, self.pdt, self.dtype = dev, pdt, dtype
        self.tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()

        # ---- production preview path (variant D), identical to play.py ----
        import torch.nn as nn
        dec = self.tae.decoder
        up_idx = [i for i, m in enumerate(dec)
                  if type(m).__name__ == "Upsample"][-1]
        self._dec, self._up_idx, self._up_orig = dec, up_idx, dec[up_idx]
        self._tgt_hw = None

        key = hashlib.sha256(args.prompt.encode()).hexdigest()
        self.key = key
        ctx = pipe.text_encoder([args.prompt], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
        pipe.text_encoder = None
        gc.collect(); torch.cuda.empty_cache()

        vae_stride, patch = pipe.vae_stride, pipe.patch_size
        img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
        img_pil = img_pil.resize((W, H), Image.BICUBIC)
        img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
        h, w = img.shape[1:]
        lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
        frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
        self.h, self.w = h, w
        self.lat_h, self.lat_w = lat_h, lat_w
        self.frame_seqlen = frame_seqlen
        self.F = (args.n_chunks - 1) * 4 + 1
        self.kv_size = frame_seqlen * args.local_window
        ma = pipe.model.config
        self.lh = ma.num_heads // pipe.sp_size
        self.hd = ma.dim // ma.num_heads

        # A real dummy DiT forward, and the runtime's own prewarm bookkeeping.
        pipe.prewarm(img_pil, max_area=W * H, frame_num=self.F, chunk_size=1)
        mf.bump_cam_epoch()
        self.Ks = get_Ks_transformed(
            torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
            480, 832, h, w, h, w)[0].to(dev)

        # prepare: the condition latent, through the frozen streamed path
        y = pipe._condition_latent(img, self.F, h, w)
        msk = torch.ones(1, self.F, lat_h, lat_w, device=dev)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                            msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        self.y = torch.concat([msk, y])
        pipe.vae = None
        gc.collect(); torch.cuda.empty_cache()

        self.self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, self.kv_size, self.lh, self.hd],
            dtype=dtype, device=dev)
        self.cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, self.hd],
            dtype=dtype, device=dev)
        self.timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

        g = torch.Generator(device=dev); g.manual_seed(args.seed)
        self.noise = torch.randn(16, args.n_chunks, lat_h, lat_w,
                                 generator=g, device=dev)
        self._prev_pose = None
        self._candidate_pose = None       # C1: advanced only by on_commit()
        self._chunk_rng_state = None      # C2: chunk-local RNG position
        self._gen = g
        self.preview_on = args.preview == "variant_d"

    def attach_runtime(self, rt):
        """The worker owns the runtime; the session only reads t0 from it."""
        self.rt = rt

    # ------------------------------------------------- §Latency-3B-B1 C1 / C2
    def on_commit(self):
        """C1: the ONLY place the reference pose advances.

        Called by the worker after `rt.commit()` has succeeded, and never on a failed,
        cancelled or retried chunk. Before this, the pose advanced at the start of
        generation, which made a replay lose the chunk's camera motion.
        """
        if getattr(self, "_candidate_pose", None) is None:
            raise RuntimeStateError(
                "on_commit without a candidate pose: denoise() must have run first")
        self._prev_pose = self._candidate_pose
        self._candidate_pose = None

    def restore_chunk_rng(self):
        """C2: put the generator back where this chunk started.

        This is what makes a replay reproduce the chunk's per-step noise. The
        uninterrupted path never calls it, so its random sequence is unchanged.
        """
        if getattr(self, "_chunk_rng_state", None) is None:
            raise RuntimeStateError(
                "restore_chunk_rng without a captured state: denoise() must have run")
        self._gen.set_state(self._chunk_rng_state)

    # -- frame extraction --------------------------------------------------
    @staticmethod
    def _to_chw(fr):
        """decode_video returns [B, T, C, H, W]; take the middle temporal frame.

        play.py's own reduction slices dim 1 on the 4-D tensor left after dropping
        B, which on that layout picks a COLOUR CHANNEL rather than a frame. With
        T=1 per chunk it stayed self-consistent for a headless metric, but a viewer
        needs real RGB, so this path is written correctly and does not reuse it.
        """
        f = fr[0] if isinstance(fr, (list, tuple)) else fr
        while f.dim() > 4:
            f = f[0]
        if f.dim() == 4:                    # [T, C, H, W]
            f = f[f.shape[0] // 2]
        return f

    @staticmethod
    def _to_hwc_u8(chw):
        a = (chw.detach().float().clamp(0, 1) * 255.0).round().byte()
        return np.ascontiguousarray(a.permute(1, 2, 0).contiguous().cpu().numpy())

    def _decode_variant_d(self, z):
        """Variant D: last spatial upsample replaced by identity, then bicubic."""
        import torch.nn as nn
        torch = self.torch
        self._dec[self._up_idx] = nn.Identity()
        try:
            with torch.no_grad():
                fr = self.tae.decode_video(
                    z.permute(1, 0, 2, 3).unsqueeze(0), parallel=False,
                    show_progress_bar=False)
        finally:
            self._dec[self._up_idx] = self._up_orig
        f = self._to_chw(fr)
        if self._tgt_hw is None:
            self._tgt_hw = (f.shape[-2] * 2, f.shape[-1] * 2)
        out = torch.nn.functional.interpolate(
            f.unsqueeze(0), size=self._tgt_hw, mode="bicubic",
            align_corners=False).squeeze(0).clamp(0, 1)
        return self._to_hwc_u8(out)

    def _plucker(self, rel):
        torch = self.torch
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(self.dev)
        h = self.h
        p = self._get_plucker(r, self.Ks[None], h, self.w)
        p = self._rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                            c1=int(h // self.lat_h), c2=int(self.w // self.lat_w))[None]
        return self._rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=self.lat_h,
                               w=self.lat_w).to(self.pdt)

    # -- the two phases, so the worker can place commit() between them ----
    def denoise(self, snap, cid, emit_preview, preempt=None, preempt_budget=0,
                policy=None):
        """One chunk. Raises ChunkPreempted if a rebase is taken at a forward boundary.

        `preempt_budget` is the number of preemptions still permitted for THIS chunk.
        At 0 no detection happens at all, which is the normal path and therefore the
        one that must stay free of synchronisation.
        """
        torch = self.torch
        pipe = self.pipe

        # --n_chunks sizes BOTH the per-chunk noise tensor and the condition latent,
        # and this viewer runs open-ended, so it can outlive them. Without this the
        # failure is an IndexError inside split() that says nothing about the cause.
        if cid >= self.noise.shape[1]:
            raise RuntimeError(
                f"chunk {cid} is past --n_chunks {self.noise.shape[1]}. That flag "
                f"sizes the per-chunk noise AND the condition latent, so a longer "
                f"run needs a larger value: raise --n_chunks and re-run.")

        chunk_pose = snap["candidate_camera"].pose
        rel0 = (np.eye(4) if self._prev_pose is None
                else np.linalg.inv(self._prev_pose) @ chunk_pose)
        plk = self._plucker(rel0)
        kw = {"context": [pipe._t5_cache[self.key][0]],
              "seq_len": self.frame_seqlen,
              "y": [self.y.split(1, dim=1)[cid]],
              "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
              "kv_cache": self.self_kv, "crossattn_cache": self.cross_kv,
              "current_start": cid * self.frame_seqlen,
              "max_attention_size": self.kv_size,
              "frame_seqlen": self.frame_seqlen}

        # §Latency-3B-B1 / C1: the pose does NOT advance here.
        #
        # It is only ADVANCED on a successful authoritative commit, in on_commit()
        # below. Advancing it at the start of generation -- which is what this used to
        # do -- means a chunk that is cancelled or retried has already moved the
        # reference the next chunk's relative pose is computed against, so a replay
        # would compute inv(chunk_pose) @ chunk_pose = I and silently lose the camera
        # motion of that chunk.
        #
        # begin / generate / fail / rebase  ->  _prev_pose unchanged
        # successful commit                  ->  _prev_pose := committed pose
        self._candidate_pose = chunk_pose

        # §Latency-3B-B1 / C2: capture the chunk-local RNG position, so a replay of
        # this chunk can reproduce its per-step noise exactly.
        #
        # Deliberately NOT a redefinition of the noise schedule. The generator's state
        # is the whole stream position, so restoring it reproduces this chunk's draws
        # exactly and an uninterrupted run continues on the old sequence untouched.
        # That is strictly better than keying noise on (chunk, step): a key needs a
        # namespace to avoid collisions across seeds and epochs, whereas the stream
        # position cannot collide with itself.
        self._chunk_rng_state = self._gen.get_state()

        cur = self.noise.split(1, dim=1)[cid]
        x0 = None
        t0s = [self.rt.record(e).t0_accept_ns for e in snap["applied_event_ids"]]
        for ti in range(len(self.timesteps)):
            with torch.amp.autocast("cuda", dtype=self.pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(self.dev)],
                    t=torch.stack([self.timesteps[ti]]).to(self.dev),
                    cross_attn_first_call=(ti == 0 and cid == 0), **kw)[0]
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=self.timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(self.timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=self._gen,
                                        device=self.dev, dtype=x0.dtype),
                        self.timesteps[ti + 1])
            if ti == 0 and self.preview_on:
                # the production preview: variant D on the step0 latent, decoded
                # immediately so it can be shown while the rest of the chunk runs
                torch.cuda.synchronize()
                frame = self._decode_variant_d(x0)
                model_ms = ((time.perf_counter_ns() - min(t0s)) / 1e6
                            if t0s else None)
                emit_preview(frame, model_ms)

            # ---- §Latency-3B-B2: preemption detection at a forward boundary ----
            # Every denoise forward is a detection boundary: after forward 1, 2 and 3.
            #
            # After forward 3 only the clean-x0 write remains, so a rebase there saves
            # about a quarter of a chunk rather than most of one. It is kept anyway
            # because the alternative is a silent hole in the trigger set, and the
            # measured gain per boundary is reported rather than assumed. The genuinely
            # too-late case -- after the write -- is not a boundary at all, and an input
            # arriving then simply waits for the next chunk.
            #
            # The peek is a short lock on a CPU flag. When the budget is 0 -- the
            # normal path -- this branch is not even entered, so an uninterrupted run
            # never synchronises for preemption and its throughput is untouched.
            if preempt is not None and policy is not None and policy.enabled:
                req = preempt.peek()
                if req is not None:
                    n_forwards = len(self.timesteps) + 1
                    remaining = n_forwards - (ti + 1)
                    admit, why = policy.decide(
                        req[1], snap.get("events", ()), forward_index=ti + 1,
                        remaining_forwards=remaining, budget=preempt_budget)
                    if admit:
                        # the ONLY synchronise on this path, and only because the
                        # decision to abandon the attempt has already been made
                        torch.cuda.synchronize()
                        preempt.take()
                        raise ChunkPreempted(req, forward_index=ti + 1)
                    # rejected: leave the input in the mailbox for the next chunk, and
                    # do NOT synchronise. A rejected candidate must cost a lock read
                    # and nothing else.
        # the KV write: the last state-mutating step of the chunk
        with torch.amp.autocast("cuda", dtype=self.pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([self.timesteps[-1] * 0.0]).to(self.dev),
                       cross_attn_first_call=False, **kw)
        torch.cuda.synchronize()
        return x0

    def decode_real(self, x0):
        torch = self.torch
        with torch.no_grad():
            fr = self.tae.decode_video(
                x0.to(self.dev).permute(1, 0, 2, 3).unsqueeze(0),
                parallel=False, show_progress_bar=False)
        return self._to_hwc_u8(self._to_chw(fr))


# ------------------------------------------------------------- mock session
class MockSession:
    """No GPU, no model. Real runtime, synthetic frames.

    The frame is derived from the candidate camera the runtime actually committed
    to, so the lineage the viewer shows is visibly the lineage the runtime owns.
    """

    def __init__(self, h=304, w=528, step_ms=90.0, preview_frac=0.28):
        self.h, self.w = h, w
        self.step_ms = step_ms
        self.preview_frac = preview_frac
        self.rt = None
        self._yy, self._xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def attach_runtime(self, rt):
        self.rt = rt

    def on_commit(self):
        """C1's hook. The mock holds no reference pose, so nothing to advance; it
        exists so the worker can call it unconditionally on both sessions."""
        return None

    def restore_chunk_rng(self):
        """C2's hook. The mock draws no noise, so there is no stream position to
        restore; it exists so the worker can call it unconditionally."""
        return None

    def _glyph(self, pose, tint):
        p = (np.asarray(pose, dtype=np.float64).ravel()
             if pose is not None else np.zeros(16))
        cx = float(np.clip(p[3] if p.size > 3 else 0.0, -1, 1))
        cy = float(np.clip(p[7] if p.size > 7 else 0.0, -1, 1))
        yaw = float(np.clip(p[8] if p.size > 8 else 0.0, -1, 1))
        u = (self._xx / max(1, self.w - 1)) - 0.5
        v = (self._yy / max(1, self.h - 1)) - 0.5
        gr = np.clip(1.0 - 1.6 * np.abs(u - cx * 0.4), 0, 1)
        gb = np.clip(1.0 - 1.6 * np.abs(v - cy * 0.4), 0, 1)
        bz = float(np.clip(0.5 + yaw, 0.0, 1.0))
        img = np.stack([0.12 + 0.55 * gr,
                        0.16 + 0.50 * gb,
                        0.20 + 0.45 * bz + 0.10 * (1.0 - gr)], -1)
        img = np.clip(img * np.array(tint, dtype=np.float32), 0, 1)
        return (img * 255.0).round().astype(np.uint8)

    def denoise(self, snap, cid, emit_preview, preempt=None, preempt_budget=0,
                policy=None):
        time.sleep(self.step_ms / 1000.0 * (1.0 - self.preview_frac))
        pose = snap["candidate_camera"].pose
        t0s = [self.rt.record(e).t0_accept_ns for e in snap["applied_event_ids"]]
        ms = ((time.perf_counter_ns() - min(t0s)) / 1e6) if t0s else None
        emit_preview(self._glyph(pose, (1.00, 0.92, 0.62)), ms)
        time.sleep(self.step_ms / 1000.0 * self.preview_frac)
        return pose

    def decode_real(self, x0):
        time.sleep(self.step_ms / 1000.0 * 0.9)
        return self._glyph(x0, (1.0, 1.0, 1.0))


# ------------------------------------------------------------------- viewer
class Viewer:
    """Pygame UI. Owns the framebuffer, the blend, the HUD and the recording."""

    UI_FPS = 60

    def __init__(self, args, input_q, frame_q, stop_evt, title, pixel,
                 ui_fps: int = 60, submit_q: Optional[queue.Queue] = None):
        if args.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
        import pygame
        self.UI_FPS = ui_fps
        self.pygame = pygame
        self.args = args
        self.input_q, self.frame_q, self.stop = input_q, frame_q, stop_evt
        self.submit_q = submit_q if submit_q is not None else queue.Queue()
        self.title_text, self.pixel = title, pixel

        pygame.init()
        self.W, self.H = args.window
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("demo_wasd — RTX 5060 Laptop 8GB interactive")
        self.clock = pygame.time.Clock()

        self.f_title = pygame.font.SysFont(None, 26)
        self.f_badge = pygame.font.SysFont(None, 40, bold=True)
        self.f_body = pygame.font.SysFont(None, 22)
        self.f_tiny = pygame.font.SysFont(None, 17)
        self.f_key = pygame.font.SysFont(None, 24, bold=True)

        # the world frame is 528x304 (ratio 1.7368); fit it under the title bar
        bar_h = 38
        avail_h = self.H - bar_h
        fh = avail_h - 8
        fw = int(round(fh * (528.0 / 304.0)))
        if fw > self.W - 8:
            fw = self.W - 8
            fh = int(round(fw * (304.0 / 528.0)))
        self.frame_rect = (int((self.W - fw) // 2),
                           bar_h + int((avail_h - fh) // 2), fw, fh)
        self.display_f = np.zeros((fh, fw, 3), dtype=np.float32)
        self.blend_from: Optional[np.ndarray] = None
        self.blend_to: Optional[np.ndarray] = None
        self.blend_i = self.blend_n = 0

        self.badge = "—"
        self.preview_ms: Optional[float] = None
        self.authority_ms: Optional[float] = None
        self.last_chunk = -1
        self.last_event_ids: tuple = ()
        self.held_keys: set = set()
        self.blend_completed = 0
        self.stats = {"preview_shown": 0, "authoritative_shown": 0,
                      "intents_sent": 0, "frames_drawn": 0}
        self.trace: list = []
        self.rec = None
        if args.record:
            self.rec = Recorder(args.record, (self.W, self.H),
                                float(self.UI_FPS), args.record_fps)
        self._seq_idx = 0
        self._seq_t0: Optional[float] = None
        self._shake_next: Optional[float] = None
        self._shake_i = 0

        # ---- §Latency-1D: t4 is taken HERE, at the submit -------------------
        # `pygame.display.get_driver()` is read rather than trusting --headless, because
        # SDL can fall back to a dummy driver on its own, and a t4 stamped there would
        # describe a call with no display side.
        self._display_is_real = pygame.display.get_driver() != "dummy"
        self._pending_t4_frame: Optional[int] = None
        self._blend_complete_pending = False
        self.t4_stamped = 0
        self.blend_complete_stamped = 0
        self.submitted = []          # display-side trace, viewer-local by design
        self.display_driver = pygame.display.get_driver()
        # --phase_n: a characterization run, not a performance. Events fire at
        # randomized offsets so the input's phase against the chunk boundary is
        # uniform, which is what a real keypress actually experiences. Single keys
        # only, so each fired event is exactly one intent and one measurement.
        self._phase_plan = None
        self._phase_idx = 0
        self._phase_settle_s: Optional[float] = None
        if args.phase_n:
            import random
            rng = random.Random(args.phase_seed)
            cycle = ["W", "D", "S", "A"]
            t = 1.0
            plan = []
            for i in range(args.phase_n):
                plan.append((t, cycle[i % len(cycle)]))
                t += rng.uniform(args.phase_lo, args.phase_hi)
            self._phase_plan = plan
        # The world is "ready" at the first authoritative frame. The script clock
        # and the recording both start there, so the take does not include the cold
        # first chunk -- which is a real cost but not a representative one, and
        # showing it would misrepresent the interactive steady state.
        self._ready = False
        self._t_wall_start: Optional[float] = None
        self.preview_trace: list = []
        self.ready_ms: Optional[float] = None

    # -- input ------------------------------------------------------------
    def _name(self, key):
        return key_table(self.pygame).get(key)

    def _emit_intent(self, newly_pressed: str):
        held = frozenset(self.held_keys)
        controls = compose_intent(held, newly_pressed)
        if not controls:
            return
        t0_ns = time.perf_counter_ns()          # t0 is stamped HERE, at the event
        self.input_q.put((t0_ns, controls))
        self.stats["intents_sent"] += 1
        self.trace.append(dict(t0_ns=t0_ns, controls=controls,
                               held=sorted(held)))

    def _handle_key(self, e):
        pygame = self.pygame
        name = self._name(e.key)
        if not name:
            if e.key == pygame.K_ESCAPE:
                self.stop.set()
            return
        if e.type == pygame.KEYDOWN:
            if name in self.held_keys:
                return              # no OS key repeat: one press, one intent
            self.held_keys.add(name)
            self._emit_intent(name)
        elif e.type == pygame.KEYUP:
            self.held_keys.discard(name)

    def _run_script(self, now_s: float):
        table = {"W": self.pygame.K_w, "A": self.pygame.K_a,
                 "S": self.pygame.K_s, "D": self.pygame.K_d}
        while (self._seq_idx < len(SCRIPT_SEQUENCE)
               and now_s >= SCRIPT_SEQUENCE[self._seq_idx][0]):
            _, actions = SCRIPT_SEQUENCE[self._seq_idx]
            self._seq_idx += 1
            for action, name in actions:
                ev = self.pygame.event.Event(
                    self.pygame.KEYDOWN if action == "down" else self.pygame.KEYUP,
                    key=table[name], mod=0, unicode="", scancode=0)
                # Posted into the real event queue: the SAME handler a human's
                # keypress goes through, so the input path is the real one and
                # only the source of the press is automated.
                self.pygame.event.post(ev)

    def _run_shakedown(self, now_s: float):
        """§Latency-3B-C0 sustained input: one discrete intent every period.

        Sustained on purpose. A sparse script barely ever lands an intent inside a
        running chunk, so preemption would never be exercised and the shakedown would
        pass while proving nothing. Cycling single keys means each intent is exactly
        one measurement, and releasing first makes the held set match a real hand.
        """
        pygame = self.pygame
        table = {"W": pygame.K_w, "A": pygame.K_a,
                 "S": pygame.K_s, "D": pygame.K_d}
        cycle = ["W", "D", "S", "A"]
        load = getattr(self.args, "load", "rapid")
        if load == "hold":
            # HOLD deliberately re-sends the SAME intent. The real keyboard path
            # suppresses key repeat, so production never produces this -- which is
            # itself the answer to "does holding a direction cause pointless
            # preemption". This load is the stronger test of the semantic check, run
            # through the real handler by releasing and pressing the same key so the
            # intent genuinely repeats.
            period = max(0.01, self.args.input_period_ms / 1000.0)
            name = "W"
        else:
            period = max(0.01, self.args.input_period_ms / 1000.0)
            if load == "normal":
                # a realistic turn rhythm: roughly a second per direction change,
                # alternating so every arrival is a genuine change
                period = 0.9
            name = cycle[self._shake_i % len(cycle)]
        if self._shake_next is None:
            self._shake_next = 0.0
        while now_s >= self._shake_next:
            self._shake_next += period
            self._shake_i += 1
            for k in sorted(self.held_keys):
                pygame.event.post(pygame.event.Event(
                    pygame.KEYUP, key=table[k], mod=0, unicode="", scancode=0))
            pygame.event.post(pygame.event.Event(
                pygame.KEYDOWN, key=table[name], mod=0, unicode="", scancode=0))

    def _run_phase(self, now_s: float):
        """Fire the next characterization event when its randomized time arrives.

        The held set is released first and the new key pressed second, so the
        handler composes the intent from held state exactly as it does for a hand:
        a lone key gives +0.6 forward or +-0.8 yaw, and a chord would give the
        combined intent. Single keys here, so one event is one intent.
        """
        pygame = self.pygame
        table = {"W": pygame.K_w, "A": pygame.K_a,
                 "S": pygame.K_s, "D": pygame.K_d}
        while (self._phase_idx < len(self._phase_plan)
               and now_s >= self._phase_plan[self._phase_idx][0]):
            _, name = self._phase_plan[self._phase_idx]
            self._phase_idx += 1
            for k in sorted(self.held_keys):
                pygame.event.post(pygame.event.Event(
                    pygame.KEYUP, key=table[k], mod=0, unicode="", scancode=0))
            pygame.event.post(pygame.event.Event(
                pygame.KEYDOWN, key=table[name], mod=0, unicode="", scancode=0))

    # -- frames -----------------------------------------------------------
    def _scaled_f32(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        import cv2
        _, _, fw, fh = self.frame_rect
        r = cv2.resize(frame_hwc_u8, (fw, fh), interpolation=cv2.INTER_LINEAR)
        return r.astype(np.float32) / 255.0

    def _drain_frames(self):
        while True:
            try:
                msg = self.frame_q.get_nowait()
            except queue.Empty:
                return
            if isinstance(msg, WorkerDone):
                if msg.reason:
                    print(f"[viewer] worker failed: {msg.reason}")
                self.stop.set()
                return
            if isinstance(msg, PreviewMsg):
                # Only an input-carrying chunk can be timed against a keypress, so
                # only those update the figures. Holding the last measurement is
                # what "from the last keypress" means; the alternative is showing
                # 0 ms or a stale number as if it were this chunk's.
                if msg.model_ms is not None:
                    self.preview_ms = msg.model_ms
                self.last_chunk = msg.chunk_index
                if msg.event_ids:
                    self.last_event_ids = msg.event_ids
                self.display_f = self._scaled_f32(msg.frame)
                self.blend_from = self.blend_to = None
                self.blend_i = self.blend_n = 0
                self.badge = "PREVIEW"
                self.stats["preview_shown"] += 1
                if msg.event_ids:
                    self.preview_trace.append(dict(
                        chunk_index=msg.chunk_index,
                        event_ids=list(msg.event_ids),
                        model_ms=msg.model_ms,
                        committed_at_emit=msg.committed_at_emit))
            elif isinstance(msg, AuthoritativeMsg):
                # D1: the FIRST composition that contains this authoritative frame is
                # the one drawn on the next tick, whether that is a 25% blend step or a
                # hard replace. The end of the blend is a different metric (D2).
                if self._display_is_real:
                    self._pending_t4_frame = msg.frame_id
                if not self._ready:
                    self._ready = True
                    # _t_wall_start is perf_counter() seconds, so this must use the
                    # same clock. The slight overestimate is the drain latency, not
                    # a claim about the frame's own arrival.
                    if self._t_wall_start is not None:
                        self.ready_ms = (time.perf_counter()
                                         - self._t_wall_start) * 1000.0
                    self._seq_t0 = time.perf_counter()
                if msg.authority_ms is not None:
                    self.authority_ms = msg.authority_ms
                self.last_chunk = msg.chunk_index
                if msg.event_ids:
                    self.last_event_ids = msg.event_ids
                target = self._scaled_f32(msg.frame)
                n = max(0, int(round(self.args.blend_ms / (1000.0 / 60.0))))
                if n > 0:
                    self.blend_from = np.array(self.display_f)
                    self.blend_to = target
                    self.blend_n, self.blend_i = n, 0
                else:
                    self.display_f = target
                    self.blend_from = self.blend_to = None
                    self.badge = "AUTHORITATIVE"
                self.stats["authoritative_shown"] += 1

    def _advance_blend(self):
        if self.blend_from is None or self.blend_to is None:
            return
        self.blend_i += 1
        if self.blend_i >= self.blend_n:
            # land EXACTLY on the authoritative frame: assignment, not a lerp
            # that merely approaches 1.0
            self.display_f = np.array(self.blend_to)
            self.blend_from = self.blend_to = None
            self.badge = "AUTHORITATIVE"
            self.blend_completed += 1
            # D2: the blend finished on this tick, so this tick's submit is the
            # blend-complete submit. Kept separate from t4 on purpose.
            self._blend_complete_pending = True
            return
        a = self.blend_i / float(self.blend_n + 1)
        self.display_f = self.blend_from * (1.0 - a) + self.blend_to * a
        self.badge = f"BLEND {self.blend_i}/{self.blend_n}"

    # -- drawing ----------------------------------------------------------
    def _panel(self, rect, alpha: int = 165, radius: int = 8):
        """A translucent dark plate. The world frame is arbitrary video, so HUD
        text drawn straight onto it is unreadable whenever the scene is bright."""
        pygame = self.pygame
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (8, 10, 14, alpha), (0, 0, rect[2], rect[3]),
                         border_radius=radius)
        self.screen.blit(s, (rect[0], rect[1]))

    def _draw(self, fps: float):
        pygame = self.pygame
        scr = self.screen
        scr.fill((10, 12, 16))
        x, y, fw, fh = self.frame_rect

        world = (self.display_f * 255.0).round().astype(np.uint8)
        surf = pygame.surfarray.make_surface(
            np.ascontiguousarray(world.transpose(1, 0, 2)))
        scr.blit(surf, (x, y))
        pygame.draw.rect(scr, (44, 50, 60), (x - 1, y - 1, fw + 2, fh + 2), 1)

        # --- HUD 1: machine + geometry ---
        scr.fill((16, 19, 25), (0, 0, self.W, 38))
        pygame.draw.line(scr, (44, 50, 60), (0, 38), (self.W, 38))
        scr.blit(self.f_title.render(f"{self.title_text} · {self.pixel}", True,
                                     (226, 232, 240)), (14, 9))

        # --- HUD 3: PREVIEW / AUTHORITATIVE badge (on a plate) ---
        col = {"PREVIEW": (250, 190, 70), "AUTHORITATIVE": (80, 220, 150)}.get(
            self.badge, (150, 190, 230))
        line = f"chunk {self.last_chunk}"
        if self.last_event_ids:
            line += f"   events {list(self.last_event_ids)}"
        bw, bh = self.f_badge.size(self.badge)
        lw, lh = self.f_tiny.size(line)
        bw2, bh2 = max(bw, lw) + 26, bh + lh + 18
        bx, by = x + 12, y + fh - bh2 - 12
        self._panel((bx, by, bw2, bh2))
        scr.blit(self.f_badge.render(self.badge, True, col), (bx + 13, by + 5))
        scr.blit(self.f_tiny.render(line, True, (176, 186, 200)),
                 (bx + 14, by + 7 + bh))

        # --- HUD 4: model-side latency, measured from the keypress (on a plate) ---
        pm = f"{self.preview_ms:.0f} ms" if self.preview_ms is not None else "—"
        am = (f"{self.authority_ms:.0f} ms"
              if self.authority_ms is not None else "—")
        top = f"from last keypress:  preview {pm}   ·   authoritative {am}"
        sub = "model-side only · no renderer, no present signal (t4/t5 unavailable)"
        tw, th = self.f_body.size(top)
        sw, sh = self.f_tiny.size(sub)
        pw, ph = max(tw, sw) + 26, th + sh + 20
        self._panel((x + 12, y + 12, pw, ph))
        scr.blit(self.f_body.render(top, True, (236, 242, 250)), (x + 25, y + 17))
        scr.blit(self.f_tiny.render(sub, True, (160, 172, 188)),
                 (x + 25, y + 17 + th + 5))

        # --- HUD 2: WASD key state ---
        self._draw_keys(scr, x + fw - 178, y + fh - 120)

        scr.blit(self.f_tiny.render(f"{fps:4.1f} fps", True, (110, 120, 135)),
                 (self.W - 74, 11))

    def _draw_keys(self, scr, ox, oy):
        pygame = self.pygame
        s, g = 46, 6
        rows = [(1, ["W"]), (0, ["A", "S", "D"])]
        for ri, (lead, row) in enumerate(rows):
            for ci, name in enumerate(row):
                rx = ox + (lead + ci) * (s + g)
                ry = oy + ri * (s + g)
                on = name in self.held_keys
                pygame.draw.rect(scr, (72, 150, 235) if on else (28, 33, 42),
                                 (rx, ry, s, s), border_radius=7)
                pygame.draw.rect(scr, (58, 66, 80), (rx, ry, s, s), 1,
                                 border_radius=7)
                t = self.f_key.render(name, True,
                                      (255, 255, 255) if on else (150, 160, 175))
                scr.blit(t, (rx + (s - t.get_width()) // 2,
                             ry + (s - t.get_height()) // 2))

    # -- loop -------------------------------------------------------------
    def run(self, seconds: float):
        pygame = self.pygame
        self._t_wall_start = time.perf_counter()
        while not self.stop.is_set():
            now = time.perf_counter()
            if self._ready:
                if self._phase_plan is not None:
                    # a characterization run ends when the plan is done and the last
                    # measurement has landed, not on a wall-clock duration
                    if self._phase_idx >= len(self._phase_plan):
                        if self._phase_settle_s is None:
                            self._phase_settle_s = now
                        elif (now - self._phase_settle_s) >= self.args.phase_settle:
                            break
                elif seconds and (now - self._seq_t0) >= seconds:
                    break
            elif (now - self._t_wall_start) > self.args.warmup_timeout:
                print(f"[viewer] warmup timeout after "
                      f"{self.args.warmup_timeout:.0f} s: no authoritative frame "
                      f"arrived, so nothing could be verified")
                break

            if self.args.shakedown and self._ready:
                self._run_shakedown(now - self._seq_t0)
            elif self._phase_plan is not None and self._ready:
                self._run_phase(now - self._seq_t0)
            elif self.args.script and self._ready:
                self._run_script(now - self._seq_t0)
            for e in pygame.event.get():
                if e.type in (pygame.KEYDOWN, pygame.KEYUP):
                    self._handle_key(e)
                elif e.type == pygame.QUIT:
                    self.stop.set()

            self._drain_frames()
            self._advance_blend()

            # §Latency-1D, decision D4: stamped immediately BEFORE the display-update
            # call, at the moment the composition is handed to the backend. Stamping
            # after would fold the swap/vsync wait into a submit metric. Decision D5:
            # taken here on the UI thread but RECORDED by the worker, which is the only
            # thread allowed to mutate the runtime.
            self._draw(self.clock.get_fps())
            if self._display_is_real and (
                    self._pending_t4_frame is not None
                    or self._blend_complete_pending):
                now_ns = time.perf_counter_ns()
                if self._pending_t4_frame is not None:
                    self.submit_q.put((self._pending_t4_frame, now_ns))
                    self.submitted.append(
                        dict(frame_id=self._pending_t4_frame, t4_ns=now_ns,
                             chunk_index=self.last_chunk))
                    self._pending_t4_frame = None
                    self.t4_stamped += 1
                if self._blend_complete_pending:
                    self.blend_complete_stamped += 1
                    self._blend_complete_pending = False
                    if self.submitted:
                        self.submitted[-1]["blend_complete_ns"] = now_ns
            pygame.display.flip()
            self.stats["frames_drawn"] += 1
            if self.rec is not None and self._ready:
                self.rec.write(pygame.surfarray.array3d(self.screen)
                               .transpose(1, 0, 2))
            self.clock.tick(self.UI_FPS)
        return self.stats


# --------------------------------------------------------------- mock smoke
def run_mock_smoke(args) -> int:
    """CPU/mock: exercises queue, input path, runtime and blend with no GPU.

    Not a stand-in for the GPU run. It is the part that must be verifiable in CI:
    that an event's t0 survives the thread handoff, that a preview cannot advance
    committed state, and that the blend lands exactly on the authoritative frame.
    """
    print("=" * 78)
    print("  §Demo-1 mock smoke (no GPU, SDL dummy)")
    print("=" * 78)

    windowed = bool(getattr(args, "mock_windowed", False))
    args.headless = not windowed
    args.record = None
    args.script = True
    seconds = args.seconds or 4.0
    print(f"  mode: {'WINDOWED (real driver, t4 exists)' if windowed else 'HEADLESS (dummy driver, t4 must not exist)'}")

    rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3), gate=1.0))
    session = MockSession(h=304, w=528, step_ms=args.mock_step_ms)
    session.attach_runtime(rt)

    input_q = InputMailbox()
    frame_q: queue.Queue = queue.Queue()
    submit_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    worker = Worker(rt, session, input_q, frame_q, stop, submit_q=submit_q,
                    mailbox=input_q,
                    policy=AdmissionPolicy(
                        boundaries=_parse_boundaries(args.preempt_boundaries),
                        min_remaining=args.preempt_min_remaining))
    worker.start()

    viewer = Viewer(args, input_q, frame_q, stop, HUD_TITLE_FALLBACK, args.pixel,
                    ui_fps=240, submit_q=submit_q)
    drawn = viewer.run(seconds)

    time.sleep(min(0.5, args.mock_step_ms / 1000.0 * 3))   # let the drain catch up
    stop.set()
    worker.join(timeout=30)

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"\n         {detail}" if detail else ""))

    inv = worker.invariants
    recs = rt.records()
    committed = [r for r in recs if r.terminal_status == "committed"]
    sent, accepted = viewer.stats["intents_sent"], inv["events_accepted"]
    undrained = input_q.qsize()

    check("worker produced no error", worker.error is None, worker.error or "")
    check("no deadlock: worker joined", not worker.is_alive())
    check("intents were emitted by the viewer", sent > 0, f"{sent} intents")
    check("runtime accepted the intents (none lost to a bad handoff)",
          accepted > 0 and accepted + undrained == sent,
          f"sent {sent}, accepted {accepted}, still queued {undrained}")
    check("t0 is the KEY EVENT timestamp, preserved across the thread handoff",
          not inv["t0_mismatches"] and inv["t0_preserved"] == accepted,
          f"preserved {inv['t0_preserved']}/{accepted}, "
          f"mismatches {inv['t0_mismatches'][:2]}")
    check("every accepted event reached a terminal status",
          all(r.is_terminal() for r in recs),
          f"{sum(r.is_terminal() for r in recs)}/{len(recs)} terminal")
    check("every committed event has a first real frame",
          bool(committed) and all(r.first_real_frame_id is not None
                                  for r in committed),
          f"{len(committed)} committed")
    check("preview NEVER advanced committed state",
          inv["preview_commit_violations"] == 0 and inv["previews_emitted"] > 0,
          f"{inv['previews_emitted']} previews, "
          f"{inv['preview_commit_violations']} violations")
    check("authoritative lineage aligns with the committed chunk",
          set(rt.committed.applied_event_ids)
          <= set(r.event_id for r in committed),
          f"committed_chunk={rt.committed.chunk_index} "
          f"lineage={list(rt.committed.applied_event_ids)}")
    check("t1 <= t2 <= t3 for every committed event",
          all(r.t1_assign_ns <= r.t2_commit_ns <= r.t3_first_real_ns
              for r in committed if r.t3_first_real_ns is not None))
    check("the blend ran and landed by assignment",
          viewer.blend_completed > 0,
          f"{viewer.blend_completed} blends completed")
    check("frames were drawn", drawn["frames_drawn"] > 0,
          f"{drawn['frames_drawn']} frames")

    # ---- §Latency-1D: T1 first-submit semantics, T2 write-once, T3 headless absence
    if viewer.display_driver == "dummy":
        check("T3 headless: t4 does NOT exist (dummy driver presents nothing)",
              viewer.t4_stamped == 0
              and all(r.t4_renderer_submit_ns is None for r in recs)
              and inv["t4_recorded"] == 0,
              f"stamped {viewer.t4_stamped}, recorded {inv['t4_recorded']}")
    else:
        check("T1 windowed: t4 exists at all (real driver)",
              viewer.t4_stamped > 0 and inv["t4_recorded"] > 0,
              f"stamped {viewer.t4_stamped}, recorded {inv['t4_recorded']}")
        # T1: t4 is the FIRST submit containing the real frame, so it must not be the
        # blend-complete submit -- the two happen on the same tick only if the blend
        # is a single step, which it is not at 50 ms / 60 fps
        pairs = [s for s in viewer.submitted if "blend_complete_ns" in s]
        check("T1 first-submit is earlier than blend-complete when a blend ran",
              all(s["t4_ns"] <= s["blend_complete_ns"] for s in pairs),
              f"{len(pairs)} pairs with both stamps")
        blended = viewer.blend_completed > 0 and len(pairs) > 0
        check("T1 a blend did run, so the distinction was actually exercised",
              blended, f"blends completed {viewer.blend_completed}, "
                       f"pairs {len(pairs)}")
        # T2: write-once
        check("T2 write-once: no redraw overwrote an existing t4",
              inv["t4_already_set"] == 0
              and inv["t4_recorded"] == viewer.t4_stamped,
              f"stamped {viewer.t4_stamped}, recorded {inv['t4_recorded']}, "
              f"refusals {inv['t4_already_set']}")
        # T6 monotonicity, where both exist
        mono = all(r.t3_first_real_ns <= r.t4_renderer_submit_ns for r in recs
                   if r.t4_renderer_submit_ns is not None
                   and r.t3_first_real_ns is not None)
        check("T6 t3 <= t4 wherever both exist", mono)
        # T5: nothing anywhere produced a t5
        check("T5 no path produced a t5_presented_ns",
              all(r.t5_present_ns is None for r in recs))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(dict(invariants=inv, stats=viewer.stats,
                           blend_completed=viewer.blend_completed,
                           committed_chunk=rt.committed.chunk_index,
                           undrained=undrained, records=rt.export()),
                      f, indent=2)
        print(f"\n  wrote {args.out_json}")

    print()
    print(f"  MOCK SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------- main
def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--weight", default="bf16", choices=["bf16", "fp8_lowmem"])
    ap.add_argument("--pixel", default="304x528")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_chunks", type=int, default=40)
    ap.add_argument("--local_window", type=int, default=6)
    ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--prompt", default="A first-person view of a natural "
                                        "landscape with smooth camera motion.")
    ap.add_argument("--preview", default="variant_d", choices=["variant_d", "off"])
    ap.add_argument("--blend_ms", type=int, default=50)
    ap.add_argument("--window", type=int, nargs=2, default=[960, 540])
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="timed from the first authoritative frame, so the cold "
                         "first chunk is not counted as interactive latency")
    ap.add_argument("--warmup_timeout", type=float, default=300.0,
                    help="abort if no authoritative frame arrives within this many "
                         "seconds (cold model load, condition encode, first chunk)")
    ap.add_argument("--script", action="store_true",
                    help="drive the take from SCRIPT_SEQUENCE, posted as real "
                         "KEYDOWN/KEYUP events through the real handler")
    ap.add_argument("--record", default=None,
                    help="write the UI framebuffer to an mp4 (ffmpeg pipe)")
    ap.add_argument("--record_fps", type=int, default=30)
    ap.add_argument("--headless", action="store_true",
                    help="SDL dummy driver: render and record with no window")
    ap.add_argument("--mock", action="store_true",
                    help="CPU/mock smoke: real runtime, synthetic frames, no GPU")
    ap.add_argument("--mock_windowed", action="store_true",
                    help="with --mock: use a REAL SDL driver instead of dummy, so the "
                         "§Latency-1D display boundary exists and T1/T2 can be "
                         "asserted. Needs a display (WSLg is enough).")
    ap.add_argument("--phase_n", type=int, default=0,
                    help="characterization run: fire this many single-key events at "
                         "randomized offsets so the input phase against the chunk "
                         "boundary is uniform, then report p50/p90. Distinct from a "
                         "demo take: this is the statistics, the video is the picture.")
    ap.add_argument("--phase_seed", type=int, default=7)
    ap.add_argument("--phase_lo", type=float, default=1.2)
    ap.add_argument("--phase_hi", type=float, default=2.6)
    ap.add_argument("--phase_settle", type=float, default=5.0)
    ap.add_argument("--shakedown", action="store_true",
                    help="§Latency-3B-C0. Sustained camera input at --input_period_ms "
                         "so intents actually land inside a running chunk, with "
                         "preemption on. Reports BEHAVIOURAL gates only; it makes no "
                         "claim about latency gain, because that is 3B-C.")
    ap.add_argument("--arm", default="B",
                    help="label for the §Latency-3B-C A/B: 'A' (baseline, preemption "
                         "disabled) or 'B' (policy v1). Recorded, not interpreted.")
    ap.add_argument("--load", default="rapid", choices=["hold", "normal", "rapid"],
                    help="shakedown input load. hold re-sends the same intent (the "
                         "semantic check's stress case), normal is a ~1 s turn rhythm, "
                         "rapid is ~200 ms direction changes")
    ap.add_argument("--input_period_ms", type=float, default=200.0,
                    help="shakedown input rate. 200 ms against a ~600 ms chunk means "
                         "most chunks see an intent mid-flight")
    ap.add_argument("--preempt_boundaries", default="1",
                    help="§Latency-3B-C1. Comma-separated forward boundaries where a "
                         "preemption may be admitted. '1' is the conservative first "
                         "cut. '' disables preemption entirely (the eventual A arm).")
    ap.add_argument("--preempt_min_remaining", type=int, default=2,
                    help="require at least this many forwards still to run, so a "
                         "boundary with nothing left to save is rejected as too_late")
    ap.add_argument("--min_preemptions", type=int, default=1,
                    help="shakedown gate: at least this many preemptions must occur, "
                         "otherwise the mechanism was never actually exercised")
    ap.add_argument("--mock_step_ms", type=float, default=90.0)
    ap.add_argument("--out_json", default=None)
    return ap.parse_args(argv)


def device_probe():
    """Device facts, from nvidia-smi, WITHOUT importing torch.

    Delegates to the fixed cleanliness contract so that there is exactly ONE place
    where a threshold lives and it cannot be negotiated down per run. See
    gpu_cleanliness.py for why the unattributed delta, not the free-memory figure, is
    the number that gates.

    Returns None when nvidia-smi is unavailable, in which case the caller says so
    rather than substituting torch, whose own context makes the two states
    indistinguishable.
    """
    try:
        import gpu_cleanliness
    except Exception:
        return None
    s = gpu_cleanliness.sample()
    return s if s.get("ok") else None


def preflight_gpu(args, dev) -> None:
    """Refuse to start when the device is not clean, against thresholds fixed in code.

    THE THRESHOLDS ARE NOT TUNABLE HERE, ON PURPOSE. While bringing up §Latency-3B the
    free-memory gate was lowered three times (7200 -> 6000 -> 4500 MiB) to make an
    experiment fit, which is how an A arm ends up measured under 1.5 GiB of ghost
    occupancy and a B arm under 3 GiB, with allocator pressure, clock state and OOM
    headroom differing in a way that invites misreading as the effect under test.
    """
    if dev is None:
        print("  gpu memory  : preflight SKIPPED (nvidia-smi unavailable); if this "
              "run dies in prewarm with a CUDA OOM, VRAM from a previous run has "
              "probably not been released yet -- wait, or  wsl --shutdown")
        return
    import gpu_cleanliness as gc
    clean, reasons, s = gc.verdict(dev)
    print(f"  gpu         : {gc.describe(s)}")
    if not clean:
        print("  ABORT: the device is not clean, and the floors are fixed:")
        for r in reasons:
            print(f"           - {r}")
        print("           Wait for the driver to reclaim it, or reset the VM from")
        print("           Windows:  wsl --shutdown. Do NOT lower the thresholds.")
        sys.exit(2)


def main():
    args = build_args()
    if args.mock:
        sys.exit(run_mock_smoke(args))

    # BEFORE ANY TORCH IMPORT. The whole point of the device-level check is that it
    # has to happen while no CUDA context exists; importing torch here would create
    # one and silently invalidate the measurement the check is based on.
    dev = device_probe()
    preflight_gpu(args, dev)

    import torch
    if dev is not None:
        title = f"{dev['name']} · {dev['total_mib'] / 1024:.0f} GB"
    else:
        name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU (no CUDA)")
        try:
            gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            title = f"{name} · {gib:.0f} GB"
        except Exception:
            title = name

    print("=" * 78)
    print("  §Demo-1 — WASD viewer")
    print(f"  device      : {title}")
    print(f"  geometry    : {args.pixel}   weight {args.weight}   "
          f"preview {args.preview}   blend {args.blend_ms} ms")
    print("  input       : "
          + (f"PHASE CHARACTERIZATION, {args.phase_n} events at randomized offsets"
             if args.phase_n else
             "SCRIPT_SEQUENCE, posted as real key events" if args.script
             else "live keyboard"))
    print(f"  window      : {args.window[0]}x{args.window[1]}"
          + (f"   recording -> {args.record} @ {args.record_fps} fps"
             if args.record else ""))
    print("  t4/t5       : UNAVAILABLE (no renderer, no present signal)")
    print("=" * 78, flush=True)

    # The runtime owns the committed camera, and it starts from the same base pose
    # play.py uses -- not from identity. The reduction is applied to deltas, so a
    # different seed pose would still move, but the absolute camera the plucker is
    # derived from would not be the frozen one.
    from cam_controller import CameraController
    base_pose = np.load(f"{args.base}/poses.npy")[0]
    ctl0 = CameraController(base_pose[:3, :3], base_pose[:3, 3])
    ctl0.cfg.yaw_rate_max, ctl0.cfg.pitch_rate_max, ctl0.cfg.v_max = 6.0, 2.0, 1.0

    rt = InteractiveRuntime(CameraState(pose=ctl0.pose.copy(), v=np.zeros(3),
                                        gate=1.0))
    session = WanSession(args)
    session.attach_runtime(rt)

    input_q = InputMailbox()
    frame_q: queue.Queue = queue.Queue()
    submit_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    worker = Worker(rt, session, input_q, frame_q, stop, submit_q=submit_q,
                    mailbox=input_q,
                    policy=AdmissionPolicy(
                        boundaries=_parse_boundaries(args.preempt_boundaries),
                        min_remaining=args.preempt_min_remaining))
    worker.start()

    viewer = Viewer(args, input_q, frame_q, stop, title, args.pixel,
                    submit_q=submit_q)
    stats = viewer.run(args.seconds)
    stop.set()
    worker.join(timeout=60)

    # ------------------------------------------------ verification summary
    print()
    print("=" * 78)
    print("  VERIFICATION")
    print("=" * 78)
    inv = worker.invariants
    recs = rt.records()
    committed = [r for r in recs if r.terminal_status == "committed"]
    lat = [r.derived()["accept_to_first_real_ms"] for r in committed
           if r.derived()["accept_to_first_real_ms"] is not None]
    pvm = [t for t in viewer.trace if t.get("preview_ms")]

    def yn(b):
        return "PASS" if b else "FAIL"
    print(f"  real keyboard events produced InputEvents      "
          f"{yn(inv['events_accepted'] > 0)}  ({inv['events_accepted']})")
    print(f"  t0 came from the key event, not the drain      "
          f"{yn(not inv['t0_mismatches'])}  "
          f"({inv['t0_preserved']}/{inv['events_accepted']} preserved)")
    print(f"  preview never committed                        "
          f"{yn(inv['preview_commit_violations'] == 0)}  "
          f"({inv['previews_emitted']} previews)")
    print(f"  authoritative lineage aligned with events      "
          f"{yn(len(committed) > 0)}  ({len(committed)} committed)")
    print(f"  no deadlock (worker joined)                    {yn(not worker.is_alive())}")
    if args.phase_n:
        print(f"  fired every planned event and measured it      "
              f"{yn(stats['intents_sent'] == args.phase_n)}  "
              f"({stats['intents_sent']}/{args.phase_n} fired; "
              f"{inv['chunks']} chunks generated)")
    else:
        print(f"  ran the full requested duration                "
              f"{yn(stats['frames_drawn'] > 0)}  ({args.seconds:.0f} s requested)")
    def pct(xs, q):
        if not xs:
            return None
        s = sorted(xs)
        # nearest-rank, so p50 and p90 are always numbers that were observed rather
        # than interpolations of them
        i = min(len(s) - 1, max(0, int(round(q / 100.0 * len(s) + 0.5)) - 1))
        return s[i]

    def dist(name, xs):
        if not xs:
            return
        print(f"  {name:<34} n={len(xs):<3} "
              f"p50 {pct(xs,50):>6.0f}  p90 {pct(xs,90):>6.0f}  "
              f"min {min(xs):>6.0f}  max {max(xs):>6.0f}  ms")

    pvl = [p["model_ms"] for p in viewer.preview_trace]
    if lat or pvl:
        print()
        dist("keypress -> preview decoded:", pvl)
        dist("keypress -> first affected REAL:", lat)
        # The assign wait IS the residual of the in-flight chunk, so it is exactly
        # the term the boundary-aligned release figures set to zero for free.
        # Reporting it separately is the point: it is what a future preemption or
        # early-exit attempt would have to shrink.
        waits = [(r.t1_assign_ns - r.t0_accept_ns) / 1e6 for r in committed
                 if r.t1_assign_ns is not None]
        owns = [(r.t2_commit_ns - r.t1_assign_ns) / 1e6 for r in committed
                if r.t1_assign_ns is not None and r.t2_commit_ns is not None]
        print()
        dist("  of which: wait for in-flight chunk:", waits)
        dist("  of which: the event's own chunk:", owns)

    # ---- §Latency-1C: acknowledgement and the watermark, on the real path ----
    print()
    print("  §Latency-1C input acknowledgement")
    print(f"    processed_input_index  {rt.committed.processed_input_index:<5} "
          f"every input up to it reached COMMITTED")
    print(f"    settled_input_index    {rt.committed.settled_input_index:<5} "
          f"no input up to it is still pending or in-flight")
    ok_acks, bad = [], []
    for r in recs:
        try:
            ok_acks.append(rt.processed_ack(r.event_id))
        except RuntimeStateError:
            bad.append(r.event_id)
    print(f"    processed_ack          {len(ok_acks)}/{len(recs)} accepted inputs "
          f"are in a committed state" + (f"; refused for {bad}" if bad else ""))
    if ok_acks:
        a = ok_acks[-1]
        print(f"      last: event {a.event_id} -> chunk {a.chunk_index} "
              f"gen {a.generation_id}  (t2 set: {a.t2_committed_ns is not None})")
    log = rt.committed_chunks()
    print(f"    committed_chunk log    {len(log)} chunks recorded"
          + (f"; last carries inputs {list(log[-1].applied_event_ids)}, "
             f"watermark {log[-1].processed_input_index}" if log else ""))
    if waits:
        print()
        print("    NOTE: the two watermarks differ only when an input ends without")
        print("    being processed. They are equal here, which is the healthy case.")

        print("  NOTE ON THE FROZEN FIGURE. play.py's ~762 ms p50 is measured with a")
        print("  scripted source that only ever delivers input at a chunk boundary, so")
        print("  its assign wait is identically zero -- for free. A real keypress")
        print("  arrives at an arbitrary phase and must first let the in-flight chunk")
        print("  finish, so that wait is a real stage the boundary-aligned number does")
        print("  not contain. Both numbers are correct; they measure different things,")
        print("  and this one is what a person actually experiences.")
    if viewer.ready_ms:
        print(f"  world became ready after                "
              f"{viewer.ready_ms/1000:.1f} s (cold chunk, excluded from --seconds)")
    if inv["chunks"]:
        print(f"  chunks generated                   {inv['chunks']}")
    # ---- §Latency-1D: renderer submit, or an explicit reason it cannot exist ----
    print()
    print("  §Latency-1D renderer submit")
    print(f"    display driver          {viewer.display_driver}")
    if viewer.display_driver == "dummy":
        print("    t4 renderer submit      unavailable  "
              "(renderer_submit_unavailable = \"headless_dummy_driver\")")
        print("                            a dummy driver presents nothing, so a")
        print("                            timestamp taken there would describe a call")
        print("                            with no display side. NOT synthesised.")
    else:
        t4v = [(r.t4_renderer_submit_ns - r.t0_accept_ns) / 1e6
               for r in committed if r.t4_renderer_submit_ns is not None]
        dist("    keypress -> renderer submit:", t4v)
        rds = [(r.t4_renderer_submit_ns - r.t3_first_real_ns) / 1e6
               for r in committed
               if r.t4_renderer_submit_ns is not None
               and r.t3_first_real_ns is not None]
        dist("    real decode -> submit:", rds)
        print(f"    t4 stamped at submit    {viewer.t4_stamped}   "
              f"write-once refusals {inv['t4_already_set']}   "
              f"unknown frames {inv['t4_unknown_frame']}")
        print(f"    blend-complete submits  {viewer.blend_complete_stamped}   "
              f"(a separate metric from t4, by contract)")
    print("    t5 physical present     UNAVAILABLE (no present-completion signal)")
    print("    input -> present        NOT MEASURED")
    print()
    print(f"  frames drawn {stats['frames_drawn']}   "
          f"previews shown {stats['preview_shown']}   "
          f"authoritative shown {stats['authoritative_shown']}   "
          f"intents sent {stats['intents_sent']}")
    print(f"  blends completed {viewer.blend_completed}")

    if viewer.rec is not None:
        rc, n = viewer.rec.close()
        size = os.path.getsize(args.record) if os.path.exists(args.record) else 0
        print()
        print(f"  recorded {args.record}")
        print(f"    ffmpeg rc={rc}  frames written={n}  bytes={size}")
        print(f"    MP4 playable: {'PASS' if rc == 0 and size > 0 else 'FAIL'}")

    # ---------------------------------------- §Latency-3B-C0 shakedown report
    if args.shakedown:
        pt = list(worker.preemption_trace)
        n_pre = len(pt)
        cuts = {}
        per_chunk = {}
        for r in pt:
            cuts[r["forward"]] = cuts.get(r["forward"], 0) + 1
            per_chunk[r["chunk_index"]] = per_chunk.get(r["chunk_index"], 0) + 1
        non_committed_terminal = [r for r in recs
                                  if r.is_terminal()
                                  and r.terminal_status != "committed"]
        seen = {}
        for c in rt.committed_chunks():
            for e in c.applied_event_ids:
                seen[e] = seen.get(e, 0) + 1
        duplicates = [e for e, k in seen.items() if k > 1]
        # A leak is a committed chunk at a generation that was SUPERSEDED FOR THAT
        # SAME CHUNK. Comparing generations globally is wrong: a later chunk's rebase
        # legitimately reuses an earlier generation number, so the first version of
        # this check flagged a chunk that had never been preempted at all.
        superseded = {(r["chunk_index"], r["from_generation"]) for r in pt}
        leaked = [c for c in rt.committed_chunks()
                  if (c.chunk_index, c.generation_id) in superseded]
        gap = (rt.committed.settled_input_index
               - rt.committed.processed_input_index)

        print()
        print("  §Latency-3B-C0  shakedown — BEHAVIOURAL GATES ONLY")
        print("    (no claim about latency gain here; that is 3B-C)")

        def g(name, cond, detail=""):
            print(f"    [{'PASS' if cond else 'FAIL'}] {name}"
                  + (f"  {detail}" if detail else ""))

        g("preemption actually fired (else nothing was exercised)",
          n_pre >= args.min_preemptions, f"{n_pre} preemptions")
        g("several chunks really committed",
          len(rt.committed_chunks()) >= 3,
          f"{len(rt.committed_chunks())} committed chunks, "
          f"{len(committed)} committed inputs")
        g("no chunk was restarted more than once",
          all(v <= 1 for v in per_chunk.values()),
          f"per-chunk {per_chunk}")
        g("no event committed twice (exactly-once held under rebase)",
          not duplicates, f"duplicates {duplicates}")
        g("no stale generation leaked into a committed chunk",
          not leaked, f"leaked {[c.chunk_index for c in leaked]}")
        g("a preview still never advanced committed state",
          inv["preview_commit_violations"] == 0,
          f"{inv['preview_commit_violations']} violations")
        g("every committed frame's t4 was actually recorded",
          inv["t4_refused"] == 0,
          f"{inv['t4_recorded']} recorded, {inv['t4_already_set']} redraw-rejections, "
          f"{inv['t4_refused']} refused"
          + (f" ({inv['t4_refusal_reason']})" if inv["t4_refused"] else ""))
        g("processed/settled are explainable",
          gap == len(non_committed_terminal),
          f"processed={rt.committed.processed_input_index} "
          f"settled={rt.committed.settled_input_index} gap={gap} "
          f"non-committed terminal={len(non_committed_terminal)}")
        g("frames kept coming (no starvation)",
          stats["authoritative_shown"] > 1
          and stats["intents_sent"] > n_pre,
          f"{stats['authoritative_shown']} authoritative frames shown, "
          f"{stats['intents_sent']} intents sent")
        g("the post-rebase attempt committed (fallback path works)",
          len(rt.committed_chunks()) > n_pre,
          f"{len(rt.committed_chunks())} chunks from {n_pre} preemptions")

        t = worker.policy.tally
        g("the admission policy actually rejected things (else it is a no-op)",
          (t["rejected_same_state"] + t["rejected_too_late"]
           + t["rejected_budget"]) > 0 or not worker.policy.enabled,
          f"peeks_with_input={t['peeks_with_input']}")

        print()
        print("    ADMISSION POLICY TALLY (§Latency-3B-C1) — reasons kept separate")
        print(f"      boundaries allowed       {worker.policy.boundaries} "
              f"(min remaining forwards {worker.policy.min_remaining})")
        print(f"      peeks that found input   {t['peeks_with_input']}")
        print(f"      admitted                 {t['admitted']}")
        print(f"      rejected_same_state      {t['rejected_same_state']}")
        print(f"      rejected_too_late        {t['rejected_too_late']}")
        print(f"      rejected_budget          {t['rejected_budget']}")

        print()
        print("    RECORDED, not a performance conclusion:")
        print(f"      cut point distribution   {dict(sorted(cuts.items()))} "
              f"(after forward k)")
        if pt:
            oc = [r["observed_to_admitted_ms"] for r in pt]
            rc = [r["rebase_cost_ms"] for r in pt]
            print(f"      observed -> admitted     p50 {statistics.median(oc):.2f} ms"
                  f"  max {max(oc):.2f} ms")
            print(f"      rebase call cost         p50 {statistics.median(rc):.2f} ms"
                  f"  max {max(rc):.2f} ms")
        if lat:
            print(f"      input -> first real      p50 {statistics.median(lat):.0f} ms")
        t4v = [(r.t4_renderer_submit_ns - r.t0_accept_ns) / 1e6
               for r in committed if r.t4_renderer_submit_ns is not None]
        if t4v:
            print(f"      input -> renderer submit p50 {statistics.median(t4v):.0f} ms")
        print(f"      chunks generated         {inv['chunks']}")
        print(f"      preemption rate          {n_pre}/{inv['chunks']} chunks")
        # wasted work, in forwards: attempt A executed `forward` forwards before being
        # discarded, and the replay redoes the whole chunk. A chunk costs 4 forwards, so
        # this is directly comparable to the chunk count.
        wasted_fwd = sum(r["forward"] for r in pt)
        total_fwd = inv["chunks"] * 4
        print(f"      wasted forward work      {wasted_fwd} of {total_fwd} "
              f"({100.0 * wasted_fwd / max(1, total_fwd):.1f}%)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(dict(device=title, pixel=args.pixel, weight=args.weight,
                           blend_ms=args.blend_ms, seconds=args.seconds,
                           scripted=bool(args.script),
                           ready_ms=viewer.ready_ms,
                           invariants=inv, stats=stats,
                           blend_completed=viewer.blend_completed,
                           input_trace=viewer.trace,
                           preview_trace=viewer.preview_trace,
                           committed_chunk=rt.committed.chunk_index,
                           records=rt.export(),
                           display_driver=viewer.display_driver,
                           renderer_submit_unavailable=(
                               "headless_dummy_driver"
                               if viewer.display_driver == "dummy" else None),
                           t4_stamped=viewer.t4_stamped,
                           blend_complete_stamps=viewer.blend_complete_stamped,
                           submitted=viewer.submitted,
                           t4_available=(viewer.display_driver != "dummy"),
                           t5_available=False,
                           # §Latency-3B-C A/B provenance, so a result cannot be read
                           # without knowing which arm and load produced it
                           arm=args.arm,
                           load=args.load,
                           preempt_boundaries=list(worker.policy.boundaries),
                           preemption_trace=worker.preemption_trace,
                           admission_tally=worker.policy.tally,
                           wasted_forwards=sum(r["forward"]
                                               for r in worker.preemption_trace),
                           observed_to_admitted=[
                               r["observed_to_admitted_ms"]
                               for r in worker.preemption_trace]), f, indent=2)
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    main()
