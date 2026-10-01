"""§Preview-1A: step0 preview cost and its interference with the authoritative path.

WHAT IS BEING TESTED. Not whether a preview is conceptually nice, but whether it is
economically viable as a product path. The authoritative chunk is ~766 ms and its
decomposition is DiT steps 71.8% + KV update 22.8% + decode 4.8%, so the question is
whether an extra decode can be inserted at step0 for roughly its own cost, or
whether it drags the authoritative path by far more.

THREE ARMS, and C is only worth running if B shows value.

    A  baseline: no preview
    B  step0 + SYNCHRONOUS preview decode   (simplest, most conservative)
    C  step0 snapshot + SEPARATE-STREAM preview decode (overlap candidate)

LATENT HANDOFF is measured explicitly rather than assumed. It is tempting to report
"preview costs 36 ms" from TAE(latent) alone, but if the latent must be cloned to
survive later steps, the clone's device-to-device copy, its VRAM and its bandwidth
contention are part of the cost. Arm B and C are therefore run both with and
without the clone so the handoff price is visible.

CORRECTNESS BOUNDARY. A preview is NON-AUTHORITATIVE lineage. It may carry
chunk_index, generation_id, applied_event_ids, source_step and frame_kind="preview"
so a user-facing preview can point back at the control event it predicts. It may NOT
commit, mark_real_decoded, write t3, or pass itself off as an authoritative frame.

PreviewTrace is a SEPARATE record from LatencyTraceRecord, for the same reason
ApplicationClaim and ChunkPhaseTrace are: `t3 = first affected REAL frame` must not
be diluted by a speculative response.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PreviewTrace:
    """A speculative visual response. Never authoritative.

    Not frozen: the trace is filled in as the chunk progresses. Only the lineage
    fields are set at construction, and they are never reassigned.
    """
    chunk_index: int
    generation_id: int
    applied_event_ids: tuple
    source_step: int
    frame_kind: str = "preview"

    # host-clock seams (monotonic)
    accept_ns: Optional[int] = None
    step0_done_ns: Optional[int] = None
    handoff_done_ns: Optional[int] = None
    preview_decoded_ns: Optional[int] = None

    # costs, milliseconds
    handoff_ms: Optional[float] = None      # clone / copy, if any
    preview_decode_ms: Optional[float] = None
    overlap: bool = False

    def ready_ms(self) -> Optional[float]:
        """input t0 -> preview decode complete."""
        if self.accept_ns is None or self.preview_decoded_ns is None:
            return None
        return (self.preview_decoded_ns - self.accept_ns) / 1e6

    def step0_to_preview_ms(self) -> Optional[float]:
        if self.step0_done_ns is None or self.preview_decoded_ns is None:
            return None
        return (self.preview_decoded_ns - self.step0_done_ns) / 1e6


def cuda_phase(dev):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()

    def stop():
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)
    return stop


def assert_non_authoritative(rt, meta) -> None:
    """A preview must be refused by every authoritative seam. Asserted, not assumed."""
    from interactive_runtime import RuntimeStateError
    for name, fn in (("commit", rt.commit), ("mark_real_decoded",
                                             rt.mark_real_decoded)):
        try:
            fn(meta)
        except RuntimeStateError:
            continue
        raise AssertionError(
            f"the runtime accepted a preview frame at the {name} seam; preview "
            f"must be non-authoritative")
