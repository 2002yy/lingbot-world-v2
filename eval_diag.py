#!/usr/bin/env python
"""Diagnostics for the CV replication: Oracle / h sweep / E_scaled control.

  python eval_diag.py
"""
import os
import numpy as np
import torch

from run_predictive import read_frames, gen_flow, analytic_flow, corr_at
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses

OUT = "output/replicate"
CASES = [("A_current", 1.0), ("B_cv", 1.0), ("D_oracle", 1.0),
         ("B_cv", 0.75), ("B_cv", 1.25), ("E_scaled", 1.0)]
SCENES = ["04", "01"]
KINDS = ["steady", "changepoint"]

print(f"{'scene':6s} {'kind':12s} " + " ".join(f"{m+'@'+str(h):>14s}"
                                               for m, h in CASES))
summary = {}
for scene in SCENES:
    for kind in KINDS:
        intended = np.load(f"examples/pred_{kind}_{scene}_intended.npy")
        vals = []
        for mode, h in CASES:
            mp4 = f"{OUT}/{scene}_{kind}_{mode}_h{h}_s42.mp4"
            if not os.path.exists(mp4):
                vals.append(float("nan"))
                continue
            frames = read_frames(mp4)
            hh, ww = frames[0].shape[:2]
            Ks = get_Ks_transformed(
                torch.from_numpy(np.load(f"examples/{scene}/intrinsics.npy")).float(),
                480, 832, hh, ww, hh, ww)[0].numpy()
            P = interpolate_camera_poses(
                np.linspace(0, len(intended) - 1, len(intended)),
                torch.from_numpy(intended[:, :3, :3]).float(),
                torch.from_numpy(intended[:, :3, 3]).float(),
                np.linspace(0, len(intended) - 1, len(frames))).numpy()
            a = analytic_flow(P, Ks, hh, ww)
            g = gen_flow(frames)
            vals.append(corr_at(a, g, -4))
            summary[(scene, kind, mode, h)] = vals[-1]
        print(f"{scene:6s} {kind:12s} " + " ".join(f"{v:14.3f}" for v in vals))

print()
print("E_scaled vs CV (does the benefit come from motion gain or temporal lead?)")
for scene in SCENES:
    for kind in KINDS:
        cv = summary.get((scene, kind, "B_cv", 1.0), float("nan"))
        es = summary.get((scene, kind, "E_scaled", 1.0), float("nan"))
        ac = summary.get((scene, kind, "A_current", 1.0), float("nan"))
        print(f"  {scene}/{kind:12s} A={ac:.3f}  E_scaled={es:.3f}  CV={cv:.3f}  "
              f"CV-E={cv-es:+.3f}")
