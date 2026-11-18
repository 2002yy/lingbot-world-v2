#!/usr/bin/env python
"""Adaptive display loop: reset policy -> realtime_display, A/B vs fixed warp.

Three separate poses (display never feeds back into control):
    control pose  -- authoritative, always integrated faithfully (drives model)
    model pose    -- equals control pose (what goes to Plucker)
    display pose  -- control pose damped toward the reference by `warp_gain`

States (hysteresis: escalate fast, recover slowly):
    ok -> soft -> reset -> fail

Modes:
    fixed    : warp_gain = 1 always, fixed 6-frame correction blend
    adaptive : policy-driven warp_gain + state-dependent blend

  python adaptive_display.py --clips output/motions --out_dir output/adaptive
"""
import argparse
import csv
import glob
import json
import os
import time

import cv2
import numpy as np

from eval_two_layer import load_models, lpips_d
from pose_warp import PoseWarper
from reset_policy import predict_lpips
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses

FPS = 60.0
CADENCE = 1.25           # seconds per chunk
D = int(FPS * CADENCE)   # display frames per chunk
NAMES = ["ok", "soft", "reset", "fail"]
DEESCALATE_HOLD = 0.35   # seconds below threshold before recovering one level
CFG = dict(soft=0.30, reset=0.40, fail=0.55)


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


def slerp_pose(a, b, u):
    """Interpolate between two 4x4 c2w poses."""
    if u <= 0:
        return a
    if u >= 1:
        return b
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
    rots = r.as_rotvec()
    rv = rots[0] * (1 - u) + rots[1] * u
    out = np.eye(4)
    out[:3, :3] = Rot.from_rotvec(rv).as_matrix()
    out[:3, 3] = a[:3, 3] * (1 - u) + b[:3, 3] * u
    return out


def level_of(pred, valid, cfg=None):
    c = cfg or CFG
    if valid < 0.90 or pred >= c["fail"]:
        return 3
    if pred >= c["reset"]:
        return 2
    if pred >= c["soft"]:
        return 1
    return 0


def warp_gain(pred, state, cfg=None):
    c = cfg or CFG
    if state in ("ok", "soft"):
        return 1.0
    if state == "reset":
        span = max(c["fail"] - c["reset"], 1e-6)
        return float(np.clip(1.0 - (pred - c["reset"]) / span * 0.5, 0.5, 1.0))
    return float(np.clip(0.5 - (pred - c["fail"]) / 0.30 * 0.35, 0.15, 0.5))


def blend_len(state):
    return {"ok": 6, "soft": 8, "reset": 12, "fail": 18}[state]


class Machine:
    def __init__(self):
        self.state = "ok"
        self.t_change = 0.0

    def update(self, lvl, t):
        cur = NAMES.index(self.state)
        if lvl > cur:
            self.state = NAMES[lvl]
            self.t_change = t
        elif lvl < cur and (t - self.t_change) > DEESCALATE_HOLD:
            self.state = NAMES[cur - 1]
            self.t_change = t
        return self.state


def simulate(clip, mode, marks, traj, n_lat, Ks, h, w, depth=4.0, cfg=None,
             ref_offset=0):
    cfg = cfg or CFG
    frames = read_frames(clip)
    P = np.asarray(traj, float)
    N = len(P)
    total = n_lat * D
    ctrl = interpolate_camera_poses(
        np.linspace(0, N - 1, N), P[:, :3, :3], P[:, :3, 3],
        np.linspace(0, N - 1, total)).numpy()

    warper = PoseWarper(Ks, h, w, nominal_depth=depth)
    mach = Machine()
    ref_idx = -1
    ref_frame = None
    ref_pose = None
    pending = None
    blend_left = 0
    blend_total = 1

    rows = []
    disp_frames = []
    refresh_events = []
    prev_disp = None
    for f in range(total):
        t = f / FPS
        chunk = f // D
        avail = max(0, chunk - 1)          # one chunk of generation latency
        if avail != ref_idx:
            if ref_frame is not None:
                pending = ref_frame
                blend_total = (blend_len(mach.state) if mode == "adaptive" else 6)
                blend_left = blend_total
                refresh_events.append(dict(f=f, t=t, state=mach.state,
                                           blend=blend_total))
            ref_idx = avail
            ref_frame = frames[min(4 * avail, len(frames) - 1)]
            ref_pose = ctrl[min((avail + ref_offset) * D, total - 1)]

        # --- metrics from the raw (undamped) extrapolation ---
        warper.set_reference(ref_frame, ref_pose)
        H = warper.homography(ctrl[f])
        pts = np.stack([np.arange(0, w, 8).repeat(len(range(0, h, 8))),
                        np.tile(np.arange(0, h, 8), len(range(0, w, 8))),
                        np.ones(len(range(0, w, 8)) * len(range(0, h, 8)))], 0).astype(np.float64)
        q = H @ pts
        q = q[:2] / q[2]
        disp = float(np.mean(np.linalg.norm(q - pts[:2], axis=0)))
        mask = cv2.warpPerspective(np.ones((h, w), np.uint8) * 255, H, (w, h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = float((mask > 127).mean())
        t_rel = np.linalg.inv(ref_pose) @ ctrl[f]
        yaw = np.degrees(np.arccos(np.clip((np.trace(t_rel[:3, :3]) - 1) / 2, -1, 1)))
        pred = predict_lpips(disp, yaw)

        if mode == "adaptive":
            state = mach.update(level_of(pred, valid, cfg), t)
            gain = warp_gain(pred, state, cfg)
        else:
            state, gain = "ok", 1.0

        # display pose: damped extrapolation (control pose untouched)
        disp_pose = slerp_pose(ref_pose, ctrl[f], gain) if gain < 1.0 else ctrl[f]
        out = warper.warp(disp_pose)
        if blend_left > 0 and pending is not None:
            a = 1.0 - blend_left / (blend_total + 1.0)
            out = cv2.addWeighted(pending, a, out, 1.0 - a, 0.0)
            blend_left -= 1
        disp_frames.append(out)
        rows.append(dict(f=f, t=t, chunk=chunk, state=state, gain=gain,
                         disp=disp, valid=valid, yaw=yaw, pred=pred))
        prev_disp = out

    # visible jump at each refresh: LPIPS across the refresh window
    for ev in refresh_events:
        f0 = ev["f"]
        lo = max(0, f0 - 1)
        hi = min(len(disp_frames) - 1, f0 + ev["blend"] + 1)
        best = 0.0
        for k in range(lo + 1, hi + 1):
            d = lpips_d(cv2.cvtColor(disp_frames[k - 1], cv2.COLOR_BGR2RGB),
                        cv2.cvtColor(disp_frames[k], cv2.COLOR_BGR2RGB))
            best = max(best, d)
        ev["jump"] = best
    return rows, refresh_events, disp_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="output/motions")
    ap.add_argument("--out_dir", default="output/adaptive")
    ap.add_argument("--save_video", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    load_models()
    all_rows, all_ev = [], []
    for mp4 in sorted(glob.glob(f"{args.clips}/*.mp4")):
        tag = os.path.basename(mp4)[:-4]
        scene = tag.split("_")[0]
        ex = f"examples/mot_{scene}"
        marks = json.load(open(f"{ex}/marks.json"))
        traj = np.load(f"{ex}/poses.npy")
        n_lat = (len(traj) - 1) // 4 + 1
        cap = cv2.VideoCapture(mp4)
        ok, fr0 = cap.read()
        cap.release()
        h, w = fr0.shape[:2]
        Ks = torch.from_numpy(np.load(f"{ex}/intrinsics.npy")).float()
        from wan.utils.cam_utils import get_Ks_transformed as gkt
        Ks = gkt(Ks, 480, 832, h, w, h, w)[0].numpy()

        for mode in ("fixed", "adaptive"):
            t0 = time.perf_counter()
            rows, ev, disp_frames = simulate(mp4, mode, marks, traj, n_lat,
                                             Ks, h, w)
            dt = time.perf_counter() - t0
            for r in rows:
                all_rows.append(dict(clip=tag, mode=mode, **r))
            for e in ev:
                all_ev.append(dict(clip=tag, mode=mode, **e))
            if args.save_video:
                vw = cv2.VideoWriter(f"{args.out_dir}/{tag}_{mode}.mp4",
                                     cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                for f in disp_frames:
                    vw.write(f)
                vw.release()
            print(f"[ad] {tag} {mode}: {len(rows)} frames, "
                  f"{len(ev)} refreshes, {dt:.1f}s", flush=True)

    with open(f"{args.out_dir}/frames.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        wr.writeheader()
        for r in all_rows:
            wr.writerow(r)
    with open(f"{args.out_dir}/refresh.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(all_ev[0].keys()))
        wr.writeheader()
        for r in all_ev:
            wr.writerow(r)

    print("\n[ad] ===== refresh jump (LPIPS) by clip x mode =====")
    print(f"{'clip':10s} {'mode':9s} {'refreshes':>9s} {'mean_jump':>10s} "
          f"{'p90_jump':>9s} {'max_jump':>9s}")
    for tag in sorted({r['clip'] for r in all_ev}):
        for mode in ("fixed", "adaptive"):
            e = [r for r in all_ev if r["clip"] == tag and r["mode"] == mode]
            if not e:
                continue
            j = np.array([x["jump"] for x in e])
            print(f"{tag:10s} {mode:9s} {len(e):9d} {j.mean():10.4f} "
                  f"{np.percentile(j,90):9.4f} {j.max():9.4f}")

    print("\n[ad] ===== state time share + gain (adaptive) =====")
    for tag in sorted({r['clip'] for r in all_rows}):
        r = [x for x in all_rows if x["clip"] == tag and x["mode"] == "adaptive"]
        n = len(r)
        share = {s: sum(1 for x in r if x["state"] == s) / n for s in NAMES}
        g = np.mean([x["gain"] for x in r])
        print(f"{tag:10s} ok={share['ok']:.2f} soft={share['soft']:.2f} "
              f"reset={share['reset']:.2f} fail={share['fail']:.2f} "
              f"mean_gain={g:.3f}")

    print("\n[ad] ===== per-segment mean jump (fixed -> adaptive) =====")
    marks_ref = {}
    for mp4 in sorted(glob.glob(f"{args.clips}/*.mp4")):
        tag = os.path.basename(mp4)[:-4]
        marks_ref[tag] = json.load(open(f"examples/mot_{tag.split('_')[0]}/marks.json"))
    for tag in sorted(marks_ref):
        for seg, (lo, hi) in marks_ref[tag].items():
            lo_ms, hi_ms = lo * CADENCE / 4, hi * CADENCE / 4
            vals = {}
            for mode in ("fixed", "adaptive"):
                e = [x for x in all_ev if x["clip"] == tag and x["mode"] == mode
                     and lo_ms <= x["t"] < hi_ms]
                vals[mode] = np.mean([x["jump"] for x in e]) if e else float("nan")
            if not (np.isnan(vals["fixed"]) and np.isnan(vals["adaptive"])):
                print(f"{tag:10s} {seg:10s} fixed={vals['fixed']:.4f} "
                      f"adaptive={vals['adaptive']:.4f} "
                      f"delta={vals['adaptive']-vals['fixed']:+.4f}")


if __name__ == "__main__":
    import torch
    main()
