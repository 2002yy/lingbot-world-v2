#!/usr/bin/env python
"""Audit camera-conditioning temporal granularity.

Part 1 (tensor): shapes and distinctness of the per-latent Plucker slices.
Part 2 (motion): does the generated motion move smoothly inside a latent, or is
        it a staircase that only updates at latent boundaries?

Latent temporal compression is 4x, so 1 latent governs 4 output frames.

  python cond_granularity_audit.py --clips output/motions
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from wan.utils.cam_utils import (compute_relative_poses, get_Ks_transformed,
                                 get_plucker_embeddings, interpolate_camera_poses)


def part1(ex, h, w):
    traj = np.load(f"{ex}/poses.npy")
    N = len(traj)
    L = (N - 1) // 4 + 1
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{ex}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0]
    c2w = interpolate_camera_poses(
        np.linspace(0, N - 1, N), torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, N - 1, L))
    rel = compute_relative_poses(c2w, framewise=True)
    emb = get_plucker_embeddings(rel, Ks[None].repeat(len(rel), 1), h, w)
    print(f"[p1] trajectory frames N={N}  ->  latent poses L={L}")
    print(f"[p1] plucker emb shape {tuple(emb.shape)}  "
          f"(f = LATENT frames, not output frames)")
    # distinctness: adjacent latent poses must differ
    d = (c2w[1:, :3, 3] - c2w[:-1, :3, 3]).norm(dim=-1)
    print(f"[p1] adjacent latent translation step: "
          f"min={float(d.min()):.6f} median={float(d.median()):.6f} "
          f"max={float(d.max()):.6f}")
    # chunk_size=1 -> each chunk is exactly one latent slice
    chunks = emb.split(1, dim=0)
    print(f"[p1] chunk_size=1 -> {len(chunks)} chunks, "
          f"each slice shape {tuple(chunks[0].shape)}")
    same = [bool(torch.equal(chunks[0], c)) for c in chunks[:5]]
    print(f"[p1] first 5 chunks identical to chunk0? {same}  "
          f"(all False => no broadcast bug)")
    # per-output-frame pose: 4 output frames map to the SAME latent pose
    print(f"[p1] output frame k -> latent k//4 ; "
          f"so frames 4i..4i+3 share ONE conditioning pose")
    print(f"[p1] camera-condition period = 4 output frames")


def part2(mp4):
    cap = cv2.VideoCapture(mp4)
    g = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    mags = []
    for i in range(1, len(g)):
        fl = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None,
                                          0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.mean(np.linalg.norm(fl, axis=2))))
    mags = np.array(mags)
    # mags[k-1] is the flow between frame k-1 and k
    within = mags[[k - 1 for k in range(1, len(g)) if k % 4 != 0]]
    across = mags[[k - 1 for k in range(1, len(g)) if k % 4 == 0]]
    return within, across


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="output/motions")
    args = ap.parse_args()

    cap = cv2.VideoCapture(sorted(glob.glob(f"{args.clips}/*.mp4"))[0])
    ok, fr0 = cap.read()
    cap.release()
    h, w = fr0.shape[:2]
    part1("examples/mot_04", h, w)
    print()

    print(f"{'clip':10s} {'within-latent':>14s} {'across-latent':>14s} "
          f"{'ratio':>7s} {'jerk':>7s}")
    allw, alla = [], []
    for mp4 in sorted(glob.glob(f"{args.clips}/*.mp4")):
        tag = os.path.basename(mp4)[:-4]
        wi, ac = part2(mp4)
        allw.append(wi); alla.append(ac)
        print(f"{tag:10s} {wi.mean():14.3f} {ac.mean():14.3f} "
              f"{ac.mean()/max(wi.mean(),1e-6):7.2f} "
              f"{ac.mean()-wi.mean():7.3f}")
    wi = np.concatenate(allw); ac = np.concatenate(alla)
    print(f"{'ALL':10s} {wi.mean():14.3f} {ac.mean():14.3f} "
          f"{ac.mean()/max(wi.mean(),1e-6):7.2f} {ac.mean()-wi.mean():7.3f}")
    print()
    print("[p2] within-latent  = flow between frames that share one conditioning pose")
    print("[p2] across-latent = flow at a latent boundary (new conditioning pose)")
    print("[p2] ratio >> 1 means the motion is a STAIRCASE (jerk at latent boundaries)")


if __name__ == "__main__":
    main()
