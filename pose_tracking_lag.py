#!/usr/bin/env python
"""Does the generated motion track the intended camera trajectory, and by how much does it lag?

For each motion clip:
  a[k] = analytic mean |flow| of the homography between the INTENDED per-output-
         frame poses (poses[k-1] -> poses[k])
  g[k] = measured mean |flow| between GENERATED frames k-1 and k
Cross-correlate the z-scored series to find the lag that best aligns them.

A positive lag means the generated motion trails the intended trajectory.

  python pose_tracking_lag.py --clips output/motions
"""
import argparse
import glob
import os

import cv2
import numpy as np

from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def analytic_flow(poses, Ks, h, w):
    fx, fy, cx, cy = [float(x) for x in Ks]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    xs = np.arange(0, w, 4)
    ys = np.arange(0, h, 4)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 0).astype(np.float64)
    out = []
    for k in range(1, len(poses)):
        R = poses[k][:3, :3] @ poses[k - 1][:3, :3].T
        H = K @ R @ Ki
        q = H @ pts
        q = q[:2] / q[2]
        out.append(float(np.mean(np.linalg.norm(q - pts[:2], axis=0))))
    return np.array(out)


def gen_flow(frames):
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    out = []
    for i in range(1, len(g)):
        fl = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        out.append(float(np.mean(np.linalg.norm(fl, axis=2))))
    return np.array(out)


def best_lag(a, g, max_lag=8):
    a = (a - a.mean()) / max(a.std(), 1e-9)
    g = (g - g.mean()) / max(g.std(), 1e-9)
    lags, cors = [], []
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            x, y = a[:len(a) - L], g[L:]
        else:
            x, y = a[-L:], g[:len(g) + L]
        if len(x) < 10:
            continue
        lags.append(L)
        cors.append(float(np.corrcoef(x, y)[0, 1]))
    lags, cors = np.array(lags), np.array(cors)
    return lags, cors, lags[int(np.argmax(cors))], float(cors.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="output/motions")
    args = ap.parse_args()

    print(f"{'clip':10s} {'best_lag':>9s} {'corr':>7s} {'zero_lag_corr':>14s}")
    allrows = []
    for mp4 in sorted(glob.glob(f"{args.clips}/*.mp4")):
        tag = os.path.basename(mp4)[:-4]
        scene = tag.split("_")[0]
        ex = f"examples/mot_{scene}"
        traj = np.load(f"{ex}/poses.npy")
        frames = read_frames(mp4)
        h, w = frames[0].shape[:2]
        Ks = get_Ks_transformed(
            __import__('torch').from_numpy(np.load(f"{ex}/intrinsics.npy")).float(),
            480, 832, h, w, h, w)[0].numpy()
        L = (len(traj) - 1) // 4 + 1
        # INTENDED per-output-frame poses = the smooth trajectory resampled to the
        # output frame count. (Using the piecewise-constant latent poses instead
        # makes the analytic flow a 4-frame-periodic spike train, which
        # contaminates the correlation with the latent period.)
        P = interpolate_camera_poses(
            np.linspace(0, len(traj) - 1, len(traj)),
            __import__('torch').from_numpy(traj[:, :3, :3]).float(),
            __import__('torch').from_numpy(traj[:, :3, 3]).float(),
            np.linspace(0, len(traj) - 1, len(frames))).numpy()
        a = analytic_flow(P, Ks, h, w)
        g = gen_flow(frames)
        n = min(len(a), len(g))
        lags, cors, bl, bc = best_lag(a[:n], g[:n])
        zero = float(cors[list(lags).index(0)]) if 0 in list(lags) else float('nan')
        print(f"{tag:10s} {bl:9d} {bc:7.3f} {zero:14.3f}")
        allrows.append((bl, bc, zero))
    bl = np.mean([r[0] for r in allrows])
    print(f"{'MEAN':10s} {bl:9.2f}")
    print()
    print("[lag] positive => generated motion trails the intended trajectory by that many frames")
    print("[note] 4 frames == 1 latent == one camera-conditioning period")


if __name__ == "__main__":
    main()
