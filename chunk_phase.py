"""§Latency-2A: authoritative chunk critical-path decomposition.

SEPARATE FROM LatencyTraceRecord ON PURPOSE. LatencyTrace answers "how long did the
user's event take"; ChunkPhaseTrace answers "where did this chunk spend its time".
Two authorities, two schemas -- the same reasoning that kept ApplicationClaim out
of the trace record.

THREE MEASUREMENT DISCIPLINES, frozen before any number is produced.

1. Both exclusive and wall-clock are reported. If overlap ever appears (KV update
   in parallel with decode), summing exclusive phase durations can exceed the chunk
   wall time. Route decisions must rest on the critical path, never on a sum of
   percentages.

2. CUDA time and host time are never mixed. GPU phases are measured with CUDA
   events; host seams use a monotonic host clock. Subtracting one from the other is
   forbidden here by construction: the two live in different fields and
   `accounting()` never combines them.

3. Instrumentation overhead must pass its own gate. Profiling off and on are
   compared on the same frozen preset; if the difference is material, the
   instrument is measuring itself.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

import torch

MS = 1e6


@dataclass
class ChunkPhaseTrace:
    """One chunk's internal breakdown. Host seams and GPU durations are separate."""
    chunk_index: int
    generation_id: int

    # ---- host-clock seams (monotonic; never combined with CUDA durations) ----
    begin_ns: Optional[int] = None
    control_done_ns: Optional[int] = None
    conditioning_done_ns: Optional[int] = None
    commit_ns: Optional[int] = None
    first_real_ns: Optional[int] = None
    end_ns: Optional[int] = None

    # ---- GPU phases, measured with CUDA events (milliseconds) ----
    dit_step_ms: list = field(default_factory=list)
    kv_update_ms: Optional[float] = None
    decode_ms: Optional[float] = None

    # ---- host-side exclusive durations (milliseconds) ----
    host_ms: dict = field(default_factory=dict)

    def host(self, a: str, b: str) -> Optional[float]:
        x, y = getattr(self, a + "_ns"), getattr(self, b + "_ns")
        if x is None or y is None:
            return None
        return (y - x) / MS

    def wall_ms(self) -> Optional[float]:
        if self.begin_ns is None or self.end_ns is None:
            return None
        return (self.end_ns - self.begin_ns) / MS

    def gpu_total_ms(self) -> float:
        t = sum(self.dit_step_ms)
        if self.kv_update_ms is not None:
            t += self.kv_update_ms
        if self.decode_ms is not None:
            t += self.decode_ms
        return t

    def accounting(self) -> dict:
        """The account. Percentages are against chunk wall time, and the
        unattributed remainder is reported rather than hidden."""
        wall = self.wall_ms() or 0.0
        parts = []
        g = self.host("begin", "control_done")
        if g is not None:
            parts.append(("control / reduction", g))
        g = self.host("control_done", "conditioning_done")
        if g is not None:
            parts.append(("conditioning / state prep", g))
        for i, d in enumerate(self.dit_step_ms):
            parts.append((f"DiT step {i}", d))
        if self.kv_update_ms is not None:
            parts.append(("KV / state update", self.kv_update_ms))
        if self.decode_ms is not None:
            parts.append(("TAE decode", self.decode_ms))
        g = self.host("first_real", "end")
        if g is not None:
            parts.append(("post-decode / bookkeeping", g))
        accounted = sum(v for _, v in parts)
        return dict(wall_ms=wall, parts=parts, accounted_ms=accounted,
                    accounted_pct=(accounted / wall * 100.0) if wall else 0.0,
                    unattributed_ms=wall - accounted,
                    unattributed_pct=((wall - accounted) / wall * 100.0)
                    if wall else 0.0)


def cuda_phase(dev):
    """Context manager yielding a callable that returns elapsed milliseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    def stop():
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    return stop


def summarise(traces: list) -> dict:
    """Aggregate accounting across chunks, excluding the first (warmup) chunk."""
    warm = traces[1:] if len(traces) > 1 else traces
    if not warm:
        return {}
    keys = []
    for t in warm:
        for name, _ in t.accounting()["parts"]:
            if name not in keys:
                keys.append(name)
    out = {"n": len(warm), "phases": {}}
    for k in keys:
        vals = []
        for t in warm:
            for name, v in t.accounting()["parts"]:
                if name == k:
                    vals.append(v)
        if vals:
            out["phases"][k] = dict(
                p50=statistics.median(vals), mean=statistics.mean(vals),
                min=min(vals), max=max(vals))
    walls = [t.wall_ms() for t in warm if t.wall_ms()]
    if walls:
        out["wall"] = dict(p50=statistics.median(walls),
                           mean=statistics.mean(walls),
                           min=min(walls), max=max(walls))
    accs = [t.accounting()["accounted_pct"] for t in warm]
    out["accounted_pct_p50"] = statistics.median(accs)
    return out


def print_accounting(s: dict, label: str = "") -> None:
    if not s:
        print("  (no traces)")
        return
    w = s.get("wall", {}).get("p50", 0.0)
    print(f"  chunk wall (p50)            {w:8.1f} ms   {label}")
    print("  " + "-" * 58)
    total = 0.0
    for name, st in s["phases"].items():
        pct = (st["p50"] / w * 100.0) if w else 0.0
        print(f"  {name:<26} {st['p50']:8.1f} ms  {pct:5.1f}%")
        total += st["p50"]
    print("  " + "-" * 58)
    pct = (total / w * 100.0) if w else 0.0
    print(f"  {'accounted':<26} {total:8.1f} ms  {pct:5.1f}%")
    print(f"  {'unattributed':<26} {w - total:8.1f} ms  "
          f"{100.0 - pct:5.1f}%")
    print()
    print(f"  n={s['n']}  accounted p50 {s['accounted_pct_p50']:.1f}%  "
          f"(target >= 95%)")
