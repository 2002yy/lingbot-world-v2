#!/usr/bin/env python
"""Stage A: conditioning symmetry audit (0 generations).

For perfectly matched +yaw / -yaw trajectories, check whether the asymmetry
already exists BEFORE the DiT, at any of:
    controller -> SE(3) -> Plucker
and whether the translation floor breaks the mirror symmetry.

  python cond_symmetry_audit.py --base examples/01
"""
import argparse
import os
import sys

import numpy as np
import torch

from cam_controller import CameraController
from wan.utils.cam_utils import (compute_relative_poses, get_Ks_transformed,
                                 get_plucker_embeddings)

M = np.diag([-1.0, 1.0, 1.0])   # world / camera x-mirror


def mirror_pose(P):
    """Horizontal mirror of a c2w pose: R' = M R M, t' = M t."""
    R = P[:3, :3]
    return np.block([[M @ R @ M, (M @ P[:3, 3])[:, None]], [np.zeros((1, 3)), 1.0]])


def traj(base, yaw, n=6, fwd=0.0, yaw_rate=6.0, v_max=0.6):
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max = yaw_rate
    ctl.cfg.v_max = v_max
    ctl.set_input(fwd=fwd, yaw=yaw)
    out = []
    for _ in range(n * 4):
        ctl.step(dt=0.25)
        out.append(ctl.pose.copy())
    return np.stack(out)


def rel_of(P):
    rel = np.stack([np.linalg.inv(P[i - 1]) @ P[i] for i in range(1, len(P))])
    return rel


def rodrigues(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-9:
        return th, np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th, ax / (2 * np.sin(th))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="examples/01")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    base = np.load(f"{args.base}/poses.npy")[0]
    Pp = traj(base, +1.0, args.n)
    Pm = traj(base, -1.0, args.n)

    print("=" * 78)
    print("1) CONTROLLER / SE(3)")
    print("=" * 78)
    rp, rm = rel_of(Pp), rel_of(Pm)
    for tag, rel in (("yaw+", rp), ("yaw-", rm)):
        ang = np.array([np.degrees(rodrigues(r[:3, :3])[0]) for r in rel])
        tn = np.linalg.norm(rel[:, :3, 3], axis=1)
        print(f"  {tag}: yaw angle mean={ang.mean():7.4f} deg  "
              f"|t| mean={tn.mean():.6f} max={tn.max():.6f}")
    # mirror check: is rel(-yaw) the mirror of rel(+yaw)?
    mir = np.stack([mirror_pose(np.linalg.inv(np.eye(4)) @ r) for r in rp])
    dR = np.abs(mir[:, :3, :3] - rm[:, :3, :3]).max()
    dt = np.abs(mir[:, :3, 3] - rm[:, :3, 3]).max()
    print(f"  mirror test  max|R_mirror(-yaw)-R(-yaw)| = {dR:.3e}")
    print(f"               max|t_mirror(-yaw)-t(-yaw)| = {dt:.3e}")

    # translation floor
    print(f"  translation floor: v_max*floor = {0.6*0.18:.4f} "
          f"(applied as +forward for BOTH yaw signs -> not mirrored)")
    print(f"  rel t (yaw+) sample: {rp[5,:3,3]}")
    print(f"  rel t (yaw-) sample: {rm[5,:3,3]}")

    print()
    print("=" * 78)
    print("2) PLUCKER CONDITIONING")
    print("=" * 78)
    W, H = 528, 288
    lat_h, lat_w = 36, 66
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, H, W, H, W)[0]

    def plucker(P):
        c2w = torch.from_numpy(P).float()
        rel = compute_relative_poses(c2w, framewise=True)
        emb = get_plucker_embeddings(rel, Ks[None].repeat(len(rel), 1), H, W)
        return rel, emb

    relp, embp = plucker(Pp)
    relm, embm = plucker(Pm)
    print(f"  plucker shape {tuple(embp.shape)}  channels={embp.shape[-1]}")
    d = embp.shape[-1] // 2
    dirp, dir_m = embp[..., :d], embm[..., :d]
    momp, mom_m = embp[..., d:], embm[..., d:]
    print(f"  |dir| mean  yaw+={dirp.norm(dim=-1).mean():.6f}  "
          f"yaw-={dir_m.norm(dim=-1).mean():.6f}  "
          f"ratio={float(dirp.norm(dim=-1).mean()/dir_m.norm(dim=-1).mean()):.6f}")
    print(f"  |mom| mean  yaw+={momp.norm(dim=-1).mean():.6f}  "
          f"yaw-={mom_m.norm(dim=-1).mean():.6f}  "
          f"ratio={float(momp.norm(dim=-1).mean()/mom_m.norm(dim=-1).mean()):.6f}")
    # mirror equivariance of the plucker features
    # direction should mirror (x negated), moment likewise; magnitudes equal
    mir_dir = torch.stack([dirp[..., 0], -dirp[..., 1], dirp[..., 2]], -1)
    print(f"  mirror check |dir| yaw+ vs yaw-: "
          f"{float((dirp.norm(dim=-1)-dir_m.norm(dim=-1)).abs().max()):.3e}")
    print(f"  dir x-sign flip: mean|x+|={float(dirp[...,0].abs().mean()):.6f} "
          f"mean|x-|={float(dir_m[...,0].abs().mean()):.6f}")

    print()
    print("=" * 78)
    print("3) ANALYTIC ROTATION HOMOGRAPHY PREDICTION")
    print("=" * 78)
    fx, fy, cx, cy = [float(x) for x in Ks]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    xs = np.arange(W, dtype=np.float64)[None, :] - cx
    for tag, rel in (("yaw+", relp), ("yaw-", relm)):
        r = rel[-1][:3, :3].numpy()
        Hm = K @ r @ Ki
        # project the pixel grid, measure mean dx
        pts = np.stack([np.broadcast_to(np.arange(W), (H, W)).ravel(),
                        np.repeat(np.arange(H), W).ravel(),
                        np.ones(H * W)], 0).astype(np.float64)
        q = Hm @ pts
        q = q[:2] / q[2]
        dx = (q[0].reshape(H, W) - pts[0].reshape(H, W))
        dy = (q[1].reshape(H, W) - pts[1].reshape(H, W))
        print(f"  {tag}: analytic mean dx={dx.mean():8.4f}  dy={dy.mean():8.4f}  "
              f"mean|flow|={np.mean(np.hypot(dx,dy)):8.4f}")
    print()
    print("  NOTE: rotation-only homography is symmetric by construction, so a")
    print("  symmetric analytic prediction + asymmetric model response localises")
    print("  the asymmetry to the DiT (Stage B).")


if __name__ == "__main__":
    main()
