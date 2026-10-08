#!/usr/bin/env python
"""Verify a published live-keyboard take.

A take that claims to be human-operated has to be checkable by someone who was not
there. This reads the run JSON and the recorded media and reports what can and cannot be
established from them.

WHAT IT CAN ESTABLISH

    input_source == "live_keyboard"     neither --script nor --shakedown drove it
    scancode present on every intent    synthetic events in this tree carry scancode=0
    unicode / mod present               likewise only set by a real SDL event
    direction changes covered           more than one distinct control intent
    inputs traced to real frames        every committed event reached a t3 and a frame
    no starvation                       no preview advanced committed state
    media present and playable          the mp4 exists and ffprobe reads it

WHAT IT CANNOT ESTABLISH, AND DOES NOT CLAIM

    that a person was physically present. Nothing in a JSON can prove that. The
    scancode check rules out the two synthetic paths that exist in this repository; it
    does not rule out a determined forgery, and saying otherwise would be exactly the
    kind of overclaim this project has spent its effort avoiding.

Usage:  python verify_live_take.py output/live/take.json [output/live/take.mp4]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def main(argv):
    if not argv:
        print("usage: verify_live_take.py <take.json> [media.mp4]")
        return 2
    path = argv[0]
    media = argv[1] if len(argv) > 1 else None
    d = json.load(open(path))

    fail = []

    def check(name, cond, detail=""):
        if not cond:
            fail.append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""))

    print("=" * 78)
    print("  LIVE KEYBOARD TAKE — verification")
    print("=" * 78)
    print(f"  file          {path}")
    print(f"  input_source  {d.get('input_source')!r}")
    print(f"  device        {d.get('device')}")
    print(f"  geometry      {d.get('pixel')}  weight {d.get('weight')}")
    print(f"  seconds       {d.get('seconds')}")
    print()

    check("the take was NOT driven by --script or --shakedown",
          d.get("input_source") == "live_keyboard",
          f"input_source={d.get('input_source')!r}")
    check("the display was a real driver, not the dummy one",
          bool(d.get("display_driver_live")),
          f"driver={d.get('display_driver')}")

    trace = d.get("input_trace") or []
    check("intents were recorded", len(trace) > 0, f"{len(trace)} intents")
    sc = [t.get("scancode") for t in trace]
    check("every intent carries a real scancode (synthetic paths set 0)",
          bool(sc) and all(s not in (None, 0) for s in sc),
          f"scancodes {sc}")
    uni = [bool(t.get("unicode")) for t in trace]
    check("real event payload present (unicode / key code)",
          bool(uni) and any(uni) or all(t.get("key") not in (None, 0) for t in trace),
          f"{sum(uni)}/{len(trace)} with unicode")

    controls = {json.dumps(t.get("controls"), sort_keys=True) for t in trace}
    check("more than one distinct control intent (a real direction change)",
          len(controls) > 1, f"{len(controls)} distinct: {sorted(controls)}")

    recs = d.get("records") or []
    committed = [r for r in recs if r.get("terminal_status") == "committed"]
    check("inputs reached committed state", len(committed) > 0, f"{len(committed)}")
    traced = [r for r in committed
              if r.get("t3_first_real_ns") and r.get("first_real_frame_id")]
    check("every committed input traces to a real generation frame",
          len(traced) == len(committed),
          f"{len(traced)}/{len(committed)} with t3 and a frame id")
    sub = [r for r in committed if r.get("t4_renderer_submit_ns")]
    check("renderer-submit timestamps recorded for committed frames",
          len(sub) == len(committed), f"{len(sub)}/{len(committed)}")

    inv = d.get("invariants") or {}
    check("no starvation during the take",
          inv.get("preview_commit_violations", 1) == 0,
          f"{inv.get('preview_commit_violations')} preview-commit violations")
    check("chunks were generated throughout",
          inv.get("chunks", 0) > 0, f"{inv.get('chunks')} chunks")
    stats = d.get("stats") or {}
    check("frames were actually shown",
          stats.get("authoritative_shown", 0) > 0,
          f"{stats.get('authoritative_shown')} authoritative frames")

    if media:
        exists = os.path.exists(media)
        check("recorded media exists", exists, media)
        if exists:
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "stream=codec_name,width,height,r_frame_rate,nb_frames",
                     "-show_entries", "format=duration", "-of", "json", media],
                    capture_output=True, text=True, timeout=60)
                info = json.loads(r.stdout)
                st = info["streams"][0]
                print(f"   media         {st.get('codec_name')} "
                      f"{st.get('width')}x{st.get('height')} "
                      f"{st.get('r_frame_rate')} {st.get('nb_frames')} frames "
                      f"{float(info['format']['duration']):.2f} s")
                check("media is a readable video", True)
            except Exception as e:
                check("media is a readable video", False, f"{type(e).__name__}: {e}")

    print()
    print("  NOT established, and not claimed: that a person was physically present.")
    print("  The scancode check rules out the two synthetic input paths that exist in")
    print("  this tree; it does not rule out a determined forgery.")
    print()
    print(f"  VERDICT: {'PASS' if not fail else 'FAIL (' + ', '.join(fail) + ')'}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
