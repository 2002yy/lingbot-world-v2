#!/usr/bin/env python
"""Offline sweep of the §42C return phase length (no model).

The depart takes the target C_ctrl out of frame; the return must bring it back
while still landing on a pose that differs from start. Sweep the return length
and report where C lands.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from cam_controller import CameraController  # noqa: E402
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses  # noqa: E402

SCENE = "04"
AREA = (512, 320)
DEPTH = 5.0
VIS_LO, VIS_HI = 0.02, 0.98
OUT_LO, OUT_HI = 0.00, 1.00
TARGET = "C_ctrl"
OBJECTS = {
    "A_left":  (0.28, 0.56, 0.52, 0.90),
    "B_right": (0.68, 0.42, 0.88, 0.72),
    "C_ctrl":  (0.86, 0.10, 1.00, 0.40),
}


def build(scene, depart_n, hold_n, return_n, obs_n, return_rate=1.0):
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    frames = []
    for _ in range(8):
        frames.append(ctl.pose.copy())
    ctl.set_input(yaw=-1.0)
    for _ in range(depart_n * 4):
        ctl.step(dt=0.25); frames.append(ctl.pose.copy())
    cur = ctl.pose.copy()
    for _ in range(hold_n * 4):
        frames.append(cur.copy())
    ctl.set_input(yaw=return_rate)
    for _ in range(return_n * 4):
        ctl.step(dt=0.25); frames.append(ctl.pose.copy())
    cur = ctl.pose.copy()
    for _ in range(obs_n * 4):
        frames.append(cur.copy())
    traj = np.stack(frames)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n]


def probe(traj):
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    vs = (4, 8, 8)
    aspect = 480 / 832
    lat_h = round(math.sqrt(AREA[0] * AREA[1] * aspect) // vs[1] // 8 * 8)
    lat_w = round(math.sqrt(AREA[0] * AREA[1] / aspect) // vs[2] // 8 * 8)
    h, w = lat_h * vs[1], lat_w * vs[2]
    K = get_Ks_transformed(
        torch.from_numpy(np.load(f"examples/{SCENE}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].cpu().numpy().astype(np.float64)
    fx, fy, cx, cy = K[0], K[1], K[2], K[3]
    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).numpy()
    ref = c2w[2]
    bb = OBJECTS[TARGET]
    uc = (bb[0] + bb[2]) / 2 * w
    vc = (bb[1] + bb[3]) / 2 * h
    Xc = np.array([(uc - cx) / fx, (vc - cy) / fy, 1.0]) * DEPTH
    Xw = ref[:3, :3] @ Xc + ref[:3, 3]

    def proj(cid):
        P = c2w[cid]
        X = P[:3, :3].T @ (Xw - P[:3, 3])
        if X[2] <= 1e-3:
            return None
        return ((fx * X[0] / X[2] + cx) / w, (fy * X[1] / X[2] + cy) / h)

    def st(uv):
        if uv is None:
            return "out"
        u, v = uv
        if u < OUT_LO or u > OUT_HI or v < OUT_LO or v > OUT_HI:
            return "out"
        if VIS_LO <= u <= VIS_HI and VIS_LO <= v <= VIS_HI:
            return "vis"
        return "edge"

    states = [(c, st(proj(c)), proj(c)) for c in range(n_lat)]
    vis = [c for c, s, _ in states if s == "vis"]
    best, cur = [], []
    for c, s, _ in states:
        if s == "out":
            cur.append(c)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    base_vis = [c for c in vis if not best or c < best[0]]
    ret_vis = [c for c in vis if best and c > best[-1]]
    uv_b = next((uv for c, s, uv in states if s == "vis"), None)
    uv_r = next((uv for c, s, uv in states if s == "vis" and best and
                 c > best[-1]), None)
    return dict(n_lat=n_lat, out_run=len(best), base=len(base_vis),
                ret=len(ret_vis), uv_b=uv_b, uv_r=uv_r,
                out_span=(best[0], best[-1]) if best else None)


def main():
    print(f"{'depart':>7s} {'hold':>5s} {'ret':>4s} {'rate':>5s} | "
          f"{'out':>4s} {'base':>5s} {'ret':>4s} | uv_base        uv_return     "
          f"d(px)")
    combos = [(d, hh, r) for d in (2, 3, 4) for hh in (0, 40)
              for r in (2, 3, 4, 5, 6)]
    for (dep, hh, return_n) in combos:
        traj = build(SCENE, dep, hh, return_n, 6, 1.0)
        r = probe(traj)
        if r["uv_b"] and r["uv_r"]:
            d = math.hypot(r["uv_r"][0] - r["uv_b"][0],
                           r["uv_r"][1] - r["uv_b"][1]) * 512
            uvbs = f"({r['uv_b'][0]:+.2f},{r['uv_b'][1]:+.2f})"
            uvrs = f"({r['uv_r'][0]:+.2f},{r['uv_r'][1]:+.2f})"
        else:
            d, uvbs, uvrs = float("nan"), "n/a", "n/a"
        print(f"{dep:7d} {hh:5d} {return_n:4d} {1.0:5.1f} | {r['out_run']:4d} "
              f"{r['base']:5d} {r['ret']:4d} | {uvbs:<13s} {uvrs:<13s} {d:6.0f}")


if __name__ == "__main__":
    main()
