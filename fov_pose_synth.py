#!/usr/bin/env python
"""§42C-2 offline: synthesize poses DIRECTLY so Δyaw is exact.

Why the controller cannot do this: CameraController integrates a rate-limited
yaw, and its response is ~4x asymmetric between directions (measured depart
~5.5 deg/chunk vs return ~1.3 deg/chunk). Consequences:
  * sweeping the return rate does nothing (it saturates)
  * a "mirror" return (same K) under-rotates by 13-17 deg
  * every setting with out>=32 AND a return visible has |dYaw| >= 23.5 deg
  * every setting with a small |dYaw| has no out period or no return

A trajectory is just an array of [4,4] camera-to-world poses, so we can build
it analytically instead:

    R0    = start rotation
    Rout  = R0 @ RotY(+theta_out)        the turned pose
    Rret  = R0 @ RotY(-theta_ret)        return pose, theta_ret small

    [R0]*baseline  |  slerp(R0 -> Rout)  |  [Rout]*hold
    |  slerp(Rout -> Rret)  |  [Rret]*observe

theta_ret is then the sweep variable and dYaw is exact by construction.

  python fov_pose_synth.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses  # noqa: E402
import torch  # noqa: E402

SCENE = "04"
AREA = (512, 320)
DEPTH = 5.0
VIS_LO, VIS_HI = 0.02, 0.98
OUT_LO, OUT_HI = 0.00, 1.00
OBJECTS = {
    "A_left":  (0.28, 0.56, 0.52, 0.90),
    "B_right": (0.68, 0.42, 0.88, 0.72),
    # C pulled slightly inward (was 0.86..1.00 -> u=0.93). At the extreme edge
    # the visible camera-pose window is so narrow that the return fails beyond
    # dYaw=3 deg, which is exactly the range we need to sample.
    "C_ctrl":  (0.80, 0.10, 0.94, 0.40),
}
TARGET = "C_ctrl"


def rot_y(deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], float)


def slerp_R(Ra, Rb, t):
    dR = Ra.T @ Rb
    tr = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(tr)
    if ang < 1e-8:
        return Ra.copy()
    ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                   dR[1, 0] - dR[0, 1]]) / (2.0 * math.sin(ang))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Rt = np.eye(3) + math.sin(ang * t) * K + (1 - math.cos(ang * t)) * (K @ K)
    return Ra @ Rt


def synth(scene, theta_out, theta_ret, base_n, dep_n, hold_n, ret_n, obs_n):
    p = np.load(f"examples/{scene}/poses.npy")
    R0 = p[0, :3, :3].copy()
    t0 = p[0, :3, 3].copy()
    Rout = R0 @ rot_y(theta_out)
    Rret = R0 @ rot_y(-theta_ret)

    def pose(R):
        P = np.eye(4)
        P[:3, :3] = R
        P[:3, 3] = t0
        return P

    frames = [pose(R0) for _ in range(base_n * 4)]
    for i in range(dep_n * 4):
        frames.append(pose(slerp_R(R0, Rout, (i + 1) / (dep_n * 4))))
    frames += [pose(Rout) for _ in range(hold_n * 4)]
    for i in range(ret_n * 4):
        frames.append(pose(slerp_R(Rout, Rret, (i + 1) / (ret_n * 4))))
    frames += [pose(Rret) for _ in range(obs_n * 4)]
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

    def yaw_of(R):
        return math.degrees(math.atan2(R[0, 2], R[2, 2]))

    dy = yaw_of(c2w[n_lat - 1][:3, :3]) - yaw_of(c2w[0][:3, :3])
    while dy > 180:
        dy -= 360
    while dy < -180:
        dy += 360
    return dict(n_lat=n_lat, out=len(best), base=len(base_vis), ret=len(ret_vis),
                uv_b=uv_b, uv_r=uv_r, dyaw=dy)


def main():
    print(f"{'theta_out':>9s} {'theta_ret':>9s} | {'out':>4s} {'base':>4s} "
          f"{'ret':>4s} | {'dYaw':>7s} {'du(px)':>7s} {'dv(px)':>7s}")
    rows = []
    for theta_out in (-55.0, 55.0):
        for theta_ret in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5,
                          8.0, 10.0):
            traj = synth(SCENE, theta_out, theta_ret, 4, 6, 40, 6, 8)
            r = probe(traj)
            du = dv = float("nan")
            if r["uv_b"] and r["uv_r"]:
                du = (r["uv_r"][0] - r["uv_b"][0]) * 512
                dv = (r["uv_r"][1] - r["uv_b"][1]) * 256
            rows.append((theta_out, theta_ret, r))
            print(f"{theta_out:9.1f} {theta_ret:9.1f} | {r['out']:4d} "
                  f"{r['base']:4d} {r['ret']:4d} | {r['dyaw']:7.2f} "
                  f"{du:7.0f} {dv:7.0f}")
    print("\nusable (out >= 32, base >= 4, ret >= 4):")
    for to, tr, r in rows:
        if r["out"] >= 32 and r["base"] >= 4 and r["ret"] >= 4:
            print(f"  theta_out={to:.0f} theta_ret={tr:.1f} -> "
                  f"dYaw={r['dyaw']:+.2f} deg, out={r['out']}, ret={r['ret']}")


if __name__ == "__main__":
    main()
