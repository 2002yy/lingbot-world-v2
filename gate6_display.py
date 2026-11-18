#!/usr/bin/env python
"""Gate 6: does CV conditioning actually improve the DISPLAY layer?

Runs the SAME adaptive display loop over
    A_current clip  (ref_offset = 0)
    B_cv clip       (ref_offset = +1, because a CV chunk reflects the pose one
                     chunk ahead of the intended timeline)
and compares refresh jump, state shares and mean warp gain.

Criterion: refresh mean jump must improve >= 5% AND must NOT be bought by a
lower mean warp gain (more conservative display).

  python gate6_display.py
"""
import glob
import json
import os

import cv2
import numpy as np

import adaptive_display as AD
from eval_two_layer import load_models
from wan.utils.cam_utils import get_Ks_transformed


def run(clip, traj, n_lat, Ks, h, w, ref_offset):
    rows, ev, _ = AD.simulate(clip, "adaptive", {}, traj, n_lat, Ks, h, w,
                              ref_offset=ref_offset)
    j = np.array([e["jump"] for e in ev])
    n = len(rows)
    sh = {s: sum(1 for x in rows if x["state"] == s) / n for s in AD.NAMES}
    gain = float(np.mean([x["gain"] for x in rows]))
    return j, sh, gain, rows


def main():
    load_models()
    print(f"{'scene':6s} {'kind':12s} {'cond':10s} {'mean_jump':>10s} "
          f"{'p90':>7s} {'mean_gain':>10s} {'reset':>6s} {'fail':>6s} {'Δjump%':>8s}")
    out = []
    for scene in ["04", "01"]:
        for kind in ["steady", "changepoint"]:
            ex = f"examples/{scene}"
            base_traj = np.load(f"examples/pred_{kind}_{scene}_intended.npy")
            n_lat = (len(base_traj) - 1) // 4 + 1
            cap = cv2.VideoCapture(f"output/replicate/{scene}_{kind}_A_current_h1.0_s42.mp4")
            ok, fr0 = cap.read(); cap.release()
            h, w = fr0.shape[:2]
            Ks = get_Ks_transformed(
                __import__('torch').from_numpy(
                    np.load(f"{ex}/intrinsics.npy")).float(),
                480, 832, h, w, h, w)[0].numpy()
            res = {}
            for cond, off in (("A_current", 0), ("B_cv", 1)):
                clip = f"output/replicate/{scene}_{kind}_{cond}_h1.0_s42.mp4"
                if not os.path.exists(clip):
                    continue
                j, sh, gain, rows = run(clip, base_traj, n_lat, Ks, h, w, off)
                res[cond] = (j, sh, gain)
                print(f"{scene:6s} {kind:12s} {cond:10s} {j.mean():10.4f} "
                      f"{np.percentile(j,90):7.4f} {gain:10.3f} "
                      f"{sh['reset']:6.2f} {sh['fail']:6.2f}")
            if "A_current" in res and "B_cv" in res:
                ja, sa, ga = res["A_current"]
                jb, sb, gb = res["B_cv"]
                dj = 100 * (jb.mean() / ja.mean() - 1)
                dg = 100 * (gb / ga - 1)
                print(f"{'':6s} {'':12s} {'Δ(CV-A)':10s} {dj:+9.1f}% "
                      f"{'':7s} {dg:+9.1f}% {100*(sb['reset']-sa['reset']):+6.2f} "
                      f"{100*(sb['fail']-sa['fail']):+6.2f}")
                out.append(dict(scene=scene, kind=kind, djump=dj, dgain=dg))
    print()
    dj = np.array([o["djump"] for o in out])
    dg = np.array([o["dgain"] for o in out])
    print(f"  mean Δjump = {dj.mean():+.1f}%   mean Δgain = {dg.mean():+.1f}%")
    ok = (dj.mean() <= -5.0) and (dg.mean() >= -1.0)
    print(f"  Gate 6 criterion (jump <= -5% AND gain not lower): "
          f"{'PASS' if ok else 'FAIL'}")
    with open("output/gate6.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    import torch
    main()
