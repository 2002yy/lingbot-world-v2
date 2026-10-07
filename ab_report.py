#!/usr/bin/env python
"""§Latency-3B-C: the formal A/B for preemption policy v1.

A   preemption budget = 0        the frozen baseline
B   policy v1                    semantic change, forward 1 only, max one rebase/chunk

Everything else is held identical: geometry, weight, blend, window, input load, seed,
wall-clock duration, and -- crucially -- the GPU cleanliness state, which is checked
against FIXED thresholds before each arm and which refuses the arm rather than being
tuned to fit it.

Three input loads, because one is not enough to say anything:

    hold    the same intent re-sent. The real keyboard path suppresses repeats, so
            production never reaches this; it is here as the semantic check's stress
            case, and B should preempt essentially never.
    normal  a ~1 s turn rhythm, representing actual play
    rapid   ~200 ms direction changes, the pressure case

Reads the per-arm JSON that demo_wasd.py writes and prints one table. Deliberately does
NOT decide what the numbers mean; it reports them, with p50/p90/p95 and the cadence and
waste figures side by side, because a latency win that costs cadence is not obviously a
win.

Usage:  python ab_report.py out/c_ab/*.json
"""
from __future__ import annotations

import glob
import json
import statistics
import sys


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q / 100.0 * len(s) + 0.5)) - 1))
    return s[i]


def load_arm(path):
    d = json.load(open(path))
    recs = d["records"]
    committed = [r for r in recs if r["terminal_status"] == "committed"]
    t0 = [r["t0_accept_ns"] for r in committed if r["t0_accept_ns"] is not None]
    first_real = [r["t3_first_real_ns"] for r in committed
                  if r["t3_first_real_ns"] is not None]
    submits = [r["t4_renderer_submit_ns"] for r in committed
               if r["t4_renderer_submit_ns"] is not None]
    # observed -> ... : t0 IS the observed time in this implementation (the input's own
    # timestamp is what accept() is given), so the two are the same series and are not
    # reported twice.
    o2r = [(b - a) / 1e6 for a, b in zip(
        sorted(t0), sorted(first_real))][:len(first_real)]
    o2r = []
    for r in committed:
        if r["t3_first_real_ns"] is not None:
            o2r.append((r["t3_first_real_ns"] - r["t0_accept_ns"]) / 1e6)
    o2s = [(r["t4_renderer_submit_ns"] - r["t0_accept_ns"]) / 1e6
           for r in committed if r["t4_renderer_submit_ns"] is not None]
    pre = d["invariants"].get("preemptions", 0)
    chunks = d["invariants"].get("chunks", 0)
    return dict(
        path=path, load=d.get("load", "?"), arm=d.get("arm", "?"),
        seconds=d.get("seconds"), chunks=chunks, preemptions=pre,
        submitted=len(submits), inputs=len(committed),
        o2r=o2r, o2s=o2s,
        cadence=chunks / max(0.001, d.get("seconds") or 1),
        wasted=d.get("wasted_forwards", 0),
        observed_to_admitted=d.get("observed_to_admitted") or [],
        t4_refused=d["invariants"].get("t4_refused", 0),
        starvation=d["invariants"].get("preview_commit_violations", 0),
    )


def main(paths):
    arms = sorted((load_arm(p) for p in paths),
                  key=lambda a: (a["load"], a["arm"]))
    print("=" * 100)
    print("  §Latency-3B-C   A/B    A = budget 0 (baseline)   B = policy v1")
    print("=" * 100)
    print(f"  {'load':7} {'arm':4} {'in':>5} {'chunks':>7} {'pre':>4} "
          f"{'o->real p50':>12} {'p90':>6} {'p95':>6} "
          f"{'o->submit p50':>14} {'cadence':>8} {'waste':>7}")
    print("  " + "-" * 96)
    for a in arms:
        print(f"  {a['load']:<7} {a['arm']:<4} {a['inputs']:>5} {a['chunks']:>7} "
              f"{a['preemptions']:>4} "
              f"{(pct(a['o2r'],50) or 0):>12.0f} {(pct(a['o2r'],90) or 0):>6.0f} "
              f"{(pct(a['o2r'],95) or 0):>6.0f} "
              f"{(pct(a['o2s'],50) or 0):>14.0f} {a['cadence']:>8.2f} "
              f"{a['wasted']:>7}")
    print()
    print("  per-load delta (B - A), milliseconds. Negative is faster.")
    for load in sorted({a["load"] for a in arms}):
        A = next((a for a in arms if a["load"] == load and a["arm"] == "A"), None)
        B = next((a for a in arms if a["load"] == load and a["arm"] == "B"), None)
        if not A or not B:
            continue
        d50 = (pct(B["o2r"], 50) or 0) - (pct(A["o2r"], 50) or 0)
        d90 = (pct(B["o2r"], 90) or 0) - (pct(A["o2r"], 90) or 0)
        d95 = (pct(B["o2r"], 95) or 0) - (pct(A["o2r"], 95) or 0)
        dcad = B["cadence"] - A["cadence"]
        print(f"    {load:<7} o->real  p50 {d50:>+7.0f}  p90 {d90:>+7.0f}  "
              f"p95 {d95:>+7.0f}   cadence {dcad:>+6.2f} chunks/s   "
              f"preemptions {B['preemptions']}")
    print()
    bad = [a for a in arms if a["t4_refused"] or a["starvation"]]
    print(f"  arms with a t4 refusal or a preview-commit violation: {len(bad)}"
          + ("  <- SHOULD BE ZERO" if bad else ""))
    print("  NOTE: observed and accepted coincide in this implementation -- accept() is")
    print("  given the input's own timestamp -- so they are not reported as two series.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or glob.glob("output/c_ab/*.json")))
