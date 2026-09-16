#!/usr/bin/env python
"""Offline check of the frustum-membership projection used by §42C.

No model needed: derive world points exactly as outfov.py does and print the
projected (u,v) per chunk, so we can see whether the camera motion actually
takes an object out of frame.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from cam_controller import CameraController  # noqa: E402
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses  # noqa: E402

SCENE = "04"
OUT_CHUNKS = 12
TAIL = 30
AREA = (512, 320)
OBJECTS = {
    "A_left":  (0.28, 0.56, 0.52, 0.90),
    "B_right": (0.68, 0.42, 0.88, 0.72),
    "C_ctrl":  (0.86, 0.10, 1.00, 0.40),
}


def build_traj_tail(scene, total_out, tail):
    T = max(1, total_out // 2)
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    start = ctl.pose.copy()
    out = []
    ctl.set_input(yaw=1.0)
    for _ in range(T * 4):
        ctl.step(dt=0.25)
        out.append(ctl.pose.copy())
    fr = [start] * 8 + out + out[::-1] + [start] * (tail * 4)
    traj = np.stack(fr)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n], 2


def main():
    W, H = AREA
    traj, ref_chunk = build_traj_tail(SCENE, OUT_CHUNKS, TAIL)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    revisit = [n_lat - TAIL, n_lat - TAIL + 1]
    observe = list(range(revisit[-1] + 1, n_lat))
    print(f"frames_n={frames_n} n_lat={n_lat} ref={ref_chunk} "
          f"observe={observe[0]}..{observe[-1]}")

    vs = (4, 8, 8)
    aspect = 480 / 832
    lat_h = round(math.sqrt(W * H * aspect) // vs[1] // 8 * 8)
    lat_w = round(math.sqrt(W * H / aspect) // vs[2] // 8 * 8)
    h, w = lat_h * vs[1], lat_w * vs[2]
    print(f"frame {h}x{w} latent {lat_h}x{lat_w}")

    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"examples/{SCENE}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0]
    K = Ks.cpu().numpy().astype(np.float64)
    fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
    print(f"intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat))
    c2w_np = c2w.numpy()
    ref_pose = c2w_np[ref_chunk]
    print(f"ref pose t={np.round(ref_pose[:3,3],3)}")
    print(f"yaw at observe: first {np.round(c2w_np[observe[0]][:3,3],2)} "
          f"last {np.round(c2w_np[observe[-1]][:3,3],2)}")

    DEPTH = 5.0
    WORLD = {}
    for nm, bb in OBJECTS.items():
        uc = (bb[0] + bb[2]) / 2 * w
        vc = (bb[1] + bb[3]) / 2 * h
        xn = (uc - cx) / fx
        yn = (vc - cy) / fy
        Xc = np.array([xn, yn, 1.0]) * DEPTH
        WORLD[nm] = ref_pose[:3, :3] @ Xc + ref_pose[:3, 3]
        print(f"  {nm}: Xc={np.round(Xc,3)} Xw={np.round(WORLD[nm],3)}")

    def project(Xw, cid):
        P = c2w_np[cid]
        Xc = P[:3, :3].T @ (Xw - P[:3, 3])
        if Xc[2] <= 1e-3:
            return None
        return ((fx * Xc[0] / Xc[2] + cx) / w, (fy * Xc[1] / Xc[2] + cy) / h)

    print(f"\n{'chunk':>5s} " + " ".join(f"{n:>18s}" for n in OBJECTS))
    for cid in observe:
        line = f"{cid:5d} "
        for nm in OBJECTS:
            uv = project(WORLD[nm], cid)
            if uv is None:
                line += f"{'behind':>18s} "
            else:
                vis = 0.02 <= uv[0] <= 0.98 and 0.02 <= uv[1] <= 0.98
                line += f"({uv[0]:+.2f},{uv[1]:+.2f}){'V' if vis else 'x':>1s} "
        print(line)


if __name__ == "__main__":
    main()
