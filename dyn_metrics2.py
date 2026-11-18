#!/usr/bin/env python
"""§41B metric correction: measure trail / background recovery at positions the
object has ACTUALLY LEFT (>= 1 object width behind), not at overlapping ROIs.

The first pass used hist[-2] (one chunk back). With 0.5-2 latent/chunk of
motion and an 8-latent-wide object, that ROI still contains the object, so both
metrics were measuring overlap, not trailing. Recomputed here from the saved
frames -- no model re-run needed.
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from object_permanence import patch  # noqa: E402
from door_state_probe import surround_brightness_of  # noqa: E402
from eval_two_layer import load_models, dino_feat  # noqa: E402

DOOR_BB = (0.70, 0.50, 0.84, 0.66)
VAE_S = 8
SPEEDS = {"slow": 0.5, "med": 1.0, "fast": 2.0}
PERIODS = {"P1": 1, "P2": 2, "P4": 4}


def main():
    load_models()

    def dino_np(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    D0 = np.load("output/dyn/D0_frames.npy")
    n = D0.shape[0]
    tail = 14
    revisit = [n - tail, n - tail + 1]
    observe = list(range(revisit[-1] + 1, n))
    ref_chunk = 2
    ref_door = patch(D0[ref_chunk], DOOR_BB)
    H, W = D0[0].shape[:2]
    dh, dw = int((DOOR_BB[3] - DOOR_BB[1]) * H), int((DOOR_BB[2] - DOOR_BB[0]) * W)

    def bb_at(cid, v):
        t = 0.0 if cid in revisit else float(cid - revisit[-1])
        sx = -int(round(v * t)) * VAE_S      # px, moves LEFT
        x0 = int(DOOR_BB[0] * W) + sx
        y0 = int(DOOR_BB[1] * H)
        return x0, y0, x0 + dw, y0 + dh

    # the canonical OPEN patch was not persisted by dyn_transform.py, so use the
    # D0 reference door as the "object-likeness" template. It is a valid object
    # reference for detecting a TRAIL (an object-like feature left behind).
    canon_door = ref_door

    print(f"object = {dw}x{dh} px; observe = {observe[0]}..{observe[-1]}")
    print(f"\n{'tag':>9s} {'speed':>6s} {'P':>3s} {'trail@left':>11s} "
          f"{'bg_rec@left':>12s} {'trail@now':>10s} {'collat':>7s}")
    out = {}
    for f in sorted(glob.glob("output/dyn/*_frames.npy")):
        tag = os.path.basename(f).replace("_frames.npy", "")
        if tag in ("D0", "canon"):
            continue
        sname, pname = tag.split("_")
        v = SPEEDS[sname]
        F = np.load(f)
        trails, bgs, nows, coll = [], [], [], []
        for cid in observe:
            fr = F[cid]
            x0, y0, x1, y1 = bb_at(cid, v)
            # current position similarity
            nows.append(float((dino_np(fr[y0:y1, x0:x1]) *
                               dino_np(canon_door)).sum()))
            # find the most recent position at least one object width behind
            past = None
            for c2 in observe:
                if c2 >= cid:
                    break
                px0, py0, px1, py1 = bb_at(c2, v)
                if abs(px0 - x0) >= dw and 0 <= px0 and px1 <= W:
                    past = (px0, py0, px1, py1)
            if past is not None:
                px0, py0, px1, py1 = past
                trails.append(float((dino_np(fr[py0:py1, px0:px1]) *
                                     dino_np(canon_door)).sum()))
                d0p = D0[cid][py0:py1, px0:px1].astype(float)
                bgs.append(float(np.abs(fr[py0:py1, px0:px1].astype(float)
                                        - d0p).mean()))
            # accumulated collateral outside the current roi
            diff = np.abs(fr.astype(float) - D0[cid].astype(float)).mean(-1)
            m = np.ones(diff.shape, bool)
            m[y0:y1, x0:x1] = False
            coll.append(float(diff[m].mean()))
        out[tag] = dict(speed=v, period=PERIODS[pname],
                        trail_left=float(np.mean(trails)) if trails else float("nan"),
                        bg_rec_left=float(np.mean(bgs)) if bgs else float("nan"),
                        trail_now=float(np.mean(nows)),
                        collateral=float(np.mean(coll)))
        o = out[tag]
        print(f"{tag:>9s} {v:6.1f} {PERIODS[pname]:3d} {o['trail_left']:11.3f} "
              f"{o['bg_rec_left']:12.1f} {o['trail_now']:10.3f} "
              f"{o['collateral']:7.2f}")

    print("\n=== interpretation ===")
    print("  trail_left  : object-likeness at a position >=1 object-width behind")
    print("                (should drop toward the D0 background baseline)")
    print("  bg_rec_left : pixel diff vs D0 at that vacated position")
    print("  trail_now   : object-likeness at the CURRENT predicted position")
    base = None
    for tag, o in out.items():
        if base is None or o["trail_left"] < base[1]:
            base = (tag, o["trail_left"])
    print(f"\n  lowest trail_left = {base[0]} ({base[1]:.3f})")
    json.dump(out, open("output/dyn/dyn_metrics2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
