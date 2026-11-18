#!/usr/bin/env python
"""Print the full flow cross-correlation curve for one clip (lag diagnosis)."""
import sys
import numpy as np
import cv2

from pose_tracking_lag import read_frames, analytic_flow, gen_flow
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses
import torch

DIRS = ["output/motions", "output/motions_off2", "output/motions_off4"]
TAG = "04_s42"

for d in DIRS:
    mp4 = f"{d}/{TAG}.mp4"
    ex = "examples/mot_04"
    traj = np.load(f"{ex}/poses.npy")
    frames = read_frames(mp4)
    h, w = frames[0].shape[:2]
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{ex}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].numpy()
    L = (len(traj) - 1) // 4 + 1
    lat = interpolate_camera_poses(
        np.linspace(0, len(traj) - 1, len(traj)),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, len(traj) - 1, L)).numpy()
    P = np.stack([lat[0 if k == 0 else (k + 3) // 4] for k in range(len(frames))])
    a = analytic_flow(P, Ks, h, w)
    g = gen_flow(frames)
    n = min(len(a), len(g))
    a, g = a[:n], g[:n]
    a = (a - a.mean()) / a.std()
    g = (g - g.mean()) / g.std()
    print(f"--- {d} (n={n}) ---")
    for Lg in range(-6, 7):
        if Lg >= 0:
            x, y = a[:n - Lg], g[Lg:]
        else:
            x, y = a[-Lg:], g[:n + Lg]
        c = float(np.corrcoef(x, y)[0, 1])
        bar = "#" * int(max(0, c) * 40)
        print(f"  lag {Lg:+d}: r={c:+.3f} {bar}")
