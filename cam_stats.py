#!/usr/bin/env python
"""Measure per-latent relative-motion statistics of official example trajectories.
This defines the in-distribution envelope for the interactive camera controller."""
import numpy as np, torch, os, sys
sys.path.insert(0, '.')
from wan.utils.cam_utils import compute_relative_poses, interpolate_camera_poses


def angle_deg(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def stats(ex):
    p = f'examples/{ex}/poses.npy'
    if not os.path.exists(p):
        return None
    c2ws = np.load(p)
    N = len(c2ws)
    L = int((N - 1) // 4) + 1
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, N - 1, N),
        src_rot_mat=c2ws[:, :3, :3], src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, N - 1, L)).numpy()
    rel = compute_relative_poses(
        torch.from_numpy(c2ws_infer).float(), framewise=True,
        normalize_trans=False).numpy()
    T = np.linalg.norm(rel[:, :3, 3], axis=1)
    rot = np.array([angle_deg(rel[i, :3, :3]) for i in range(L)])
    # decompose rotation into yaw/pitch/roll magnitudes (approx via euler)
    from scipy.spatial.transform import Rotation
    eul = np.degrees(Rotation.from_matrix(rel[:, :3, :3]).as_euler('yxz'))
    return dict(N=N, L=L, T=T, rot=rot, eul=eul)


exs = ['00', '01', '02', '03', '04', '05']
print(f"{'ex':>3} {'N':>4} {'lat':>4} | {'T/fr max':>9} {'med':>7} {'p90':>7} | "
      f"{'rot/fr max':>10} {'med':>6} {'p90':>6} | {'yaw max':>8} {'pitch max':>9} {'cumYaw':>8}")
for ex in exs:
    s = stats(ex)
    if s is None:
        continue
    T, rot, eul = s['T'], s['rot'], s['eul']
    print(f"{ex:>3} {s['N']:>4} {s['L']:>4} | {T.max():9.4f} {np.median(T):7.4f} {np.percentile(T,90):7.4f} | "
          f"{rot.max():10.2f} {np.median(rot):6.2f} {np.percentile(rot,90):6.2f} | "
          f"{np.abs(eul[:,0]).max():8.2f} {np.abs(eul[:,1]).max():9.2f} {np.abs(eul[:,0]).sum():8.1f}")
