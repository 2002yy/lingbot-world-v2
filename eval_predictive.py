#!/usr/bin/env python
"""Evaluate predictive conditioning from the generated clips (no regeneration).

Definitions (n = output frame index, chunk c = n//4):
  generated flow  g[n] = flow between generated frames n-1 and n
  intended  flow  a[n] = analytic flow of the controller's smooth frame-rate
                         trajectory between frames n-1 and n

corr_at(a, g, lag) = corr(a[k], g[k+lag]).
  lag = 0    : generated motion matches the control at GENERATION time
  lag = -4   : generated motion matches the control at DISPLAY time (T0 + L)
"""
import argparse
import glob
import os

import numpy as np

from run_predictive import read_frames, gen_flow, analytic_flow, corr_at
from wan.utils.cam_utils import get_Ks_transformed

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/predictive")
    ap.add_argument("--scene", default="04")
    args = ap.parse_args()

    print(f"{'kind':12s} {'mode':11s} {'lag0':>7s} {'lag-4':>7s} "
          f"{'lag-8':>7s} {'best_lag':>9s} {'best':>7s}")
    summary = {}
    for kind in ("steady", "changepoint"):
        intended = np.load(f"examples/pred_{kind}_{args.scene}_intended.npy")
        mp4s = sorted(glob.glob(f"{args.dir}/{kind}_{args.scene}_*_s42.mp4"))
        if not mp4s:
            continue
        f0 = read_frames(mp4s[0])
        h, w = f0[0].shape[:2]
        Ks = get_Ks_transformed(
            torch.from_numpy(np.load(f"examples/{args.scene}/intrinsics.npy")).float(),
            480, 832, h, w, h, w)[0].numpy()
        from wan.utils.cam_utils import interpolate_camera_poses
        P = interpolate_camera_poses(
            np.linspace(0, len(intended) - 1, len(intended)),
            torch.from_numpy(intended[:, :3, :3]).float(),
            torch.from_numpy(intended[:, :3, 3]).float(),
            np.linspace(0, len(intended) - 1, len(f0))).numpy()
        a = analytic_flow(P, Ks, h, w)

        for mp4 in mp4s:
            base = os.path.basename(mp4)[:-4]
            mode = base.split(f"{kind}_{args.scene}_")[1].rsplit("_h", 1)[0]
            g = gen_flow(read_frames(mp4))
            c0 = corr_at(a, g, 0)
            cm4 = corr_at(a, g, -4)
            cm8 = corr_at(a, g, -8)
            bl, bc = 0, -9
            for L in range(-12, 13):
                c = corr_at(a, g, L)
                if not np.isnan(c) and c > bc:
                    bc, bl = c, L
            summary[(kind, mode)] = dict(lag0=c0, lagm4=cm4, lagm8=cm8,
                                         best_lag=bl, best=bc)
            print(f"{kind:12s} {mode:11s} {c0:7.3f} {cm4:7.3f} {cm8:7.3f} "
                  f"{bl:9d} {bc:7.3f}")
        print()

    print("Display-time match (lag -4) ranking:")
    for kind in ("steady", "changepoint"):
        rows = [(m, v["lagm4"]) for (k, m), v in summary.items() if k == kind]
        rows.sort(key=lambda x: -x[1])
        print(f"  {kind:12s} " + "  ".join(f"{m}={v:.3f}" for m, v in rows))


if __name__ == "__main__":
    main()
