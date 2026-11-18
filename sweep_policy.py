#!/usr/bin/env python
"""Sweep reset-policy thresholds: refresh-jump vs display responsiveness.

  python sweep_policy.py --clips output/motions
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

import adaptive_display as AD
from eval_two_layer import load_models
from wan.utils.cam_utils import get_Ks_transformed

SETTINGS = [
    dict(soft=0.20, reset=0.30, fail=0.45),
    dict(soft=0.25, reset=0.35, fail=0.50),
    dict(soft=0.30, reset=0.40, fail=0.55),
    dict(soft=0.35, reset=0.45, fail=0.60),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="output/motions")
    args = ap.parse_args()
    load_models()

    clips = []
    for mp4 in sorted(glob.glob(f"{args.clips}/*.mp4")):
        tag = os.path.basename(mp4)[:-4]
        scene = tag.split("_")[0]
        ex = f"examples/mot_{scene}"
        traj = np.load(f"{ex}/poses.npy")
        n_lat = (len(traj) - 1) // 4 + 1
        cap = cv2.VideoCapture(mp4); ok, fr0 = cap.read(); cap.release()
        h, w = fr0.shape[:2]
        Ks = get_Ks_transformed(
            torch.from_numpy(np.load(f"{ex}/intrinsics.npy")).float(),
            480, 832, h, w, h, w)[0].numpy()
        clips.append((mp4, tag, traj, n_lat, Ks, h, w))
        if len(clips) >= 2:
            break

    print(f"{'setting':22s} {'mean_jump':>10s} {'p90':>7s} {'mean_gain':>10s} "
          f"{'ok':>5s} {'soft':>5s} {'reset':>6s} {'fail':>5s}")
    for st in SETTINGS:
        jumps, gains, states = [], [], []
        for mp4, tag, traj, n_lat, Ks, h, w in clips:
            rows, ev, _ = AD.simulate(mp4, "adaptive", {}, traj, n_lat, Ks, h, w,
                                      cfg=st)
            jumps += [e["jump"] for e in ev]
            gains += [r["gain"] for r in rows]
            states += [r["state"] for r in rows]
        j = np.array(jumps)
        n = len(states)
        sh = {s: states.count(s) / n for s in AD.NAMES}
        print(f"soft{st['soft']:.2f}/res{st['reset']:.2f}/fail{st['fail']:.2f} "
              f"{j.mean():10.4f} {np.percentile(j,90):7.4f} "
              f"{np.mean(gains):10.3f} {sh['ok']:5.2f} {sh['soft']:5.2f} "
              f"{sh['reset']:6.2f} {sh['fail']:5.2f}")


if __name__ == "__main__":
    import torch
    main()
