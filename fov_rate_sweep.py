#!/usr/bin/env python
"""§42C-2 offline: find return-phase settings that produce a controlled Δyaw.

Everything else is held fixed (same depart, same 40-chunk hold-out); only the
return rate is swept, and we record
    Δyaw  = angle between the baseline pose and the return pose
    C visibility on return  (must still come back into frame)
    screen displacement Δu/Δv
so we can pick settings that sample Δyaw = 0..10 degrees.

  python fov_rate_sweep.py
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
OBJECTS = {
    "A_left":  (0.28, 0.56, 0.52, 0.90),
    "B_right": (0.68, 0.42, 0.88, 0.72),
    # C moved AWAY from the frame edge (was 0.86..1.00 -> u=0.93, where the
    # visible camera-pose window is extremely narrow). An interior ROI gives a
    # wide visibility window so the sweep can pass smoothly through dYaw ~ 0.
    "C_ctrl":  (0.62, 0.10, 0.76, 0.40),
}
TARGET = "C_ctrl"


def build_mirror(scene, K, hold_n, ft_rate, ft_n, obs_n):
    """depart K / hold / return K at the MIRRORED rate -> lands at dYaw ~ 0, then
    a small fine-tune samples the tolerance range."""
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    frames = [ctl.pose.copy() for _ in range(16)]
    ctl.set_input(yaw=-1.0)
    for _ in range(K * 4):
        ctl.step(dt=0.25); frames.append(ctl.pose.copy())
    cur = ctl.pose.copy()
    for _ in range(hold_n * 4):
        frames.append(cur.copy())
    ctl.set_input(yaw=+1.0)
    for _ in range(K * 4):
        ctl.step(dt=0.25); frames.append(ctl.pose.copy())
    if ft_n > 0:
        ctl.set_input(yaw=ft_rate)
        for _ in range(ft_n):
            ctl.step(dt=0.25); frames.append(ctl.pose.copy())
    cur = ctl.pose.copy()
    for _ in range(obs_n * 4):
        frames.append(cur.copy())
    traj = np.stack(frames)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n]


def build(scene, depart_rate, depart_n, hold_n, return_rate, return_n, obs_n,
          ft_rate=0.0, ft_frames=0):
    """Adds an optional fine-tune phase after the main return.

    The return rate is saturated by yaw_rate_max, so sweeping it does nothing
    (every setting lands on the same 7.5 deg residual). A small fine-tune
    rotation is the only way to sample the Δyaw range controllably.
    """
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    frames = [ctl.pose.copy() for _ in range(16)]          # P0 baseline 4 chunks
    ctl.set_input(yaw=depart_rate)
    for _ in range(depart_n * 4):
        ctl.step(dt=0.25); frames.append(ctl.pose.copy())
    cur = ctl.pose.copy()
    for _ in range(hold_n * 4):
        frames.append(cur.copy())
    ctl.set_input(yaw=return_rate)
    for _ in range(return_n * 4):
        ctl.step(dt=0.25); frames.append(ctl.pose.copy())
    if ft_frames > 0:
        ctl.set_input(yaw=ft_rate)
        for _ in range(ft_frames):
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
    fx, fy, cx, cy = K
    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).numpy()
    base_i = 0
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

    # SIGNED yaw difference. acos((trace-1)/2) loses the sign, so a 9 deg yaw
    # to the left and 9 deg to the right look identical -- which is exactly how
    # a small-|dYaw| setting can still leave C out of frame.
    def yaw_of(R):
        return math.degrees(math.atan2(R[0, 2], R[2, 2]))

    R0 = c2w[base_i][:3, :3]
    R1 = c2w[n_lat - 1][:3, :3]
    dy = yaw_of(R1) - yaw_of(R0)
    while dy > 180.0:
        dy -= 360.0
    while dy < -180.0:
        dy += 360.0
    ang = dy
    return dict(n_lat=n_lat, out=len(best), base=len(base_vis), ret=len(ret_vis),
                uv_b=uv_b, uv_r=uv_r, dyaw=ang, ret_start=n_lat-1)


def main():
    print(f"{'K':>3s} {'hold':>4s} {'ft_r':>6s} {'ft_n':>4s} | {'out':>4s} {'base':>4s} "
          f"{'ret':>4s} | {'dYaw':>7s} {'du':>5s} {'dv':>5s}")
    rows = []
    # MIRROR return: depart K chunks, return K chunks at the mirrored rate, so the
    # camera lands back at dYaw ~ 0 by construction. Only then does a small
    # fine-tune sample the 0..7.5 deg range. (A rate-swept return saturates on
    # yaw_rate_max and can never land precisely.)
    for K in (10, 14, 18, 24):
        for ft_r, ft_n in ((0.0, 0), (0.02, 4), (0.04, 4), (0.06, 4),
                           (0.08, 4), (0.10, 4), (0.12, 4), (0.20, 4),
                           (0.30, 4)):
            traj = build_mirror(SCENE, K, 40, ft_r, ft_n, 8)
            r = probe(traj)
            du = dv = float("nan")
            if r["uv_b"] and r["uv_r"]:
                du = (r["uv_r"][0] - r["uv_b"][0]) * 512
                dv = (r["uv_r"][1] - r["uv_b"][1]) * 256
            rows.append((K, ft_r, ft_n, r))
            print(f"{K:3d} {40:4d} {ft_r:6.2f} {ft_n:4d} | {r['out']:4d} "
                  f"{r['base']:4d} {r['ret']:4d} | {r['dyaw']:7.2f} "
                  f"{du:5.0f} {dv:5.0f}")
    print("\nusable (out >= 32, base >= 4, ret >= 4), sorted by |dYaw|:")
    ok = [(K, fr, fn, r) for K, fr, fn, r in rows
          if r["out"] >= 32 and r["base"] >= 4 and r["ret"] >= 4]
    for K, fr, fn, r in sorted(ok, key=lambda z: abs(z[3]["dyaw"])):
        print(f"  K={K:2d} ft_rate={fr:.2f} ft_n={fn} -> dYaw={r['dyaw']:+6.2f} deg"
              f", out={r['out']}, ret={r['ret']}")
    if not ok:
        print("  (none)")


if __name__ == "__main__":
    main()
