#!/usr/bin/env python
"""§42C trajectory probe -- the HARD GATE before any model run.

The previous trajectory could not produce a real out-of-FOV: its final segment
is [start]*(tail*4), so the camera is completely static during the observe
window ("yaw first == yaw last"), and every object projects to a constant
screen point. That is why the first attempt reported "visible 28/28".

New four-phase trajectory:
    P0 baseline          [start] * 8                     C clearly visible
    P1 depart            single-direction yaw for K1     C sweeps out
    P2 hold-out          keep the turned pose for K2     C continuously out
    P3 return-different  yaw back but STOP != start      C returns at a
                                                         DIFFERENT screen
                                                         position / angle
    P4 observe           hold a few chunks               reacquisition

Visibility hysteresis (as requested): a single threshold makes this an
edge-jitter test. So:
    visible = 0.02 <= u,v <= 0.98
    truly_out = u < 0.00 or u > 1.00 or v < 0.00 or v > 1.00
    anything between is "edge" and counts as NEITHER

PASS conditions (all must hold):
    baseline_visible      >= 4 chunks
    continuous_out        >= 32 chunks   (strictly out, no edge chunks inside)
    return_visible        >= 4 chunks
    first_return uv       != baseline uv
    return pose           != start pose

Reported: last_visible / first_out / last_out / first_return,
continuous_out_length, uv at baseline and at first return, yaw start/out/return.

  python fov_probe.py
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

# phases: (name, kind, chunks)  kind = hold | yaw(pos rate) | yaw(neg rate)
PHASES = [
    ("P0_baseline", "hold", 6),
    ("P1_depart", -1.0, 2),
    ("P2_holdout", "hold", 40),
    ("P3_return", +1.0, 6),
    ("P4_observe", "hold", 8),
]

OBJECTS = {
    "A_left":  (0.28, 0.56, 0.52, 0.90),
    "B_right": (0.68, 0.42, 0.88, 0.72),
    "C_ctrl":  (0.86, 0.10, 1.00, 0.40),
}
TARGET = "C_ctrl"


def build_traj_phases(scene, phases):
    """Returns (traj, ref_chunk, chunk_phase) with chunk granularity of 4."""
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    start_pose = ctl.pose.copy()
    frames, cphase = [], []
    for name, kind, nch in phases:
        if kind == "hold":
            cur = ctl.pose.copy()
            for _ in range(nch * 4):
                frames.append(cur.copy())
                cphase.append(name)
        else:
            ctl.set_input(yaw=float(kind))
            for _ in range(nch * 4):
                ctl.step(dt=0.25)
                frames.append(ctl.pose.copy())
                cphase.append(name)
    traj = np.stack(frames)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n], start_pose, cphase[:n]


def yaw_of(pose):
    """Yaw angle (deg) from the rotation matrix, for reporting."""
    R = pose[:3, :3]
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))


def main():
    W, H = AREA
    traj, start_pose, cphase = build_traj_phases(SCENE, PHASES)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    print(f"trajectory: {frames_n} frames -> {n_lat} chunks")
    print("phases: " + ", ".join(
        f"{nm}:{sum(1 for c in cphase[::4] if c == nm)}ch"
        for nm, _, _ in PHASES))

    vs = (4, 8, 8)
    aspect = 480 / 832
    lat_h = round(math.sqrt(W * H * aspect) // vs[1] // 8 * 8)
    lat_w = round(math.sqrt(W * H / aspect) // vs[2] // 8 * 8)
    h, w = lat_h * vs[1], lat_w * vs[2]
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"examples/{SCENE}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0]
    K = Ks.cpu().numpy().astype(np.float64)
    fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat))
    c2w_np = c2w.numpy()
    ref_chunk = 2          # inside P0

    ref_pose = c2w_np[ref_chunk]
    WORLD = {}
    for nm, bb in OBJECTS.items():
        uc = (bb[0] + bb[2]) / 2 * w
        vc = (bb[1] + bb[3]) / 2 * h
        Xc = np.array([(uc - cx) / fx, (vc - cy) / fy, 1.0]) * DEPTH
        WORLD[nm] = ref_pose[:3, :3] @ Xc + ref_pose[:3, 3]

    def project(Xw, cid):
        P = c2w_np[cid]
        Xc = P[:3, :3].T @ (Xw - P[:3, 3])
        if Xc[2] <= 1e-3:
            return None
        return ((fx * Xc[0] / Xc[2] + cx) / w, (fy * Xc[1] / Xc[2] + cy) / h)

    def state(uv):
        if uv is None:
            return "out"
        u, v = uv
        if u < OUT_LO or u > OUT_HI or v < OUT_LO or v > OUT_HI:
            return "out"
        if VIS_LO <= u <= VIS_HI and VIS_LO <= v <= VIS_HI:
            return "vis"
        return "edge"

    print(f"\nintrinsics fx={fx:.1f} cx={cx:.1f} w={w}  "
          f"→ horizontal half-FOV {math.degrees(math.atan(cx/fx)):.1f} deg")

    print(f"\n{'chunk':>5s} {'phase':>13s} " +
          " ".join(f"{n:>16s}" for n in OBJECTS) + "   yaw")
    seq = {}
    for cid in range(n_lat):
        line = f"{cid:5d} {cphase[min(cid*4, len(cphase)-1)]:>13s} "
        for nm in OBJECTS:
            uv = project(WORLD[nm], cid)
            st = state(uv)
            seq.setdefault(nm, []).append((cid, st, uv))
            if uv is None:
                line += f"{'out(behind)':>16s} "
            else:
                mark = {"vis": "V", "out": "O", "edge": "~"}[st]
                line += f"({uv[0]:+.2f},{uv[1]:+.2f}){mark:>1s} "
        line += f" {yaw_of(c2w_np[cid][:3, :3]):+7.1f}"
        print(line)

    # ---------- PASS evaluation for the target ----------
    s = seq[TARGET]
    vis_chunks = [c for c, st, _ in s if st == "vis"]
    out_chunks = [c for c, st, _ in s if st == "out"]
    edge_chunks = [c for c, st, _ in s if st == "edge"]

    # longest strictly-out run with NO edge/vis chunk inside
    best_run, cur = [], []
    for c, st, _ in s:
        if st == "out":
            cur.append(c)
        else:
            if len(cur) > len(best_run):
                best_run = cur
            cur = []
    if len(cur) > len(best_run):
        best_run = cur

    baseline_vis = [c for c in vis_chunks if c < (best_run[0] if best_run else 1e9)]
    return_vis = [c for c in vis_chunks if best_run and c > best_run[-1]]
    uv_base = next((uv for c, st, uv in s if st == "vis"), None)
    uv_ret = next((uv for c, st, uv in s if st == "vis" and best_run and
                   c > best_run[-1]), None)
    yaw_start = yaw_of(c2w_np[0][:3, :3])
    yaw_out = yaw_of(c2w_np[best_run[0] if best_run else 0][:3, :3])
    yaw_ret = yaw_of(c2w_np[return_vis[0] if return_vis else 0][:3, :3])

    print(f"\n===== §42C trajectory gate for target {TARGET} =====")
    print(f"  baseline visible chunks   : {len(baseline_vis)} "
          f"(need >= 4)  {'OK' if len(baseline_vis) >= 4 else 'FAIL'}")
    print(f"  continuous OUT chunks     : {len(best_run)} "
          f"(need >= 32) {'OK' if len(best_run) >= 32 else 'FAIL'}")
    print(f"    run = "
          f"{best_run[0] if best_run else 'n/a'}.."
          f"{best_run[-1] if best_run else 'n/a'}")
    print(f"  edge chunks (ambiguous)   : {len(edge_chunks)} "
          f"{edge_chunks if len(edge_chunks) < 20 else '(many)'}")
    print(f"  return visible chunks     : {len(return_vis)} "
          f"(need >= 4)  {'OK' if len(return_vis) >= 4 else 'FAIL'}")
    print(f"  uv @ baseline             : "
          f"{('(%.2f,%.2f)' % uv_base) if uv_base else 'n/a'}")
    print(f"  uv @ first return         : "
          f"{('(%.2f,%.2f)' % uv_ret) if uv_ret else 'n/a'}")
    if uv_base and uv_ret:
        d = math.hypot(uv_ret[0] - uv_base[0], uv_ret[1] - uv_base[1])
        print(f"  screen displacement       : {d:.3f} normalized "
              f"({d*512:.0f} px)  {'OK' if d > 0.02 else 'FAIL (too small)'}")
    print(f"  yaw start / out / return  : {yaw_start:+.1f} / {yaw_out:+.1f} / "
          f"{yaw_ret:+.1f}")
    print(f"  return pose != start pose : "
          f"{abs(yaw_ret - yaw_start) > 1.0}")

    ok = (len(baseline_vis) >= 4 and len(best_run) >= 32
          and len(return_vis) >= 4 and uv_base is not None and uv_ret is not None
          and math.hypot(uv_ret[0] - uv_base[0], uv_ret[1] - uv_base[1]) > 0.02
          and abs(yaw_ret - yaw_start) > 1.0)
    print(f"\n  TRAJECTORY GATE: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  -> run outfov.py with these phase settings")
    else:
        print("  -> adjust PHASES and re-probe; do NOT run the model yet")


if __name__ == "__main__":
    main()
