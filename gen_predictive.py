#!/usr/bin/env python
"""Predictive-conditioning trajectories: A current / B CV / C CA / D oracle.

The model's own time coordinate is left untouched (offset stays 0). Only the
camera state fed in is replaced by the state predicted for the DISPLAY time
(control time + horizon), reusing the same SE(3) integrator as the controller.

  python gen_predictive.py --kind steady   --scene 04 --horizon 1.0
  python gen_predictive.py --kind changepoint --scene 04 --horizon 1.0
"""
import argparse
import json
import os
import shutil

import numpy as np

from cam_controller import CameraController

LATENT_SEC = 1.25
MODES = ["A_current", "B_cv", "C_ca", "D_oracle"]

# (name, n_chunks, inputs)  -- steady, predictable
STEADY = [
    ("settle", 2, dict(fwd=0.0)),
    ("fwd", 6, dict(fwd=0.6)),
    ("yaw", 5, dict(yaw=0.6, fwd=0.3)),
    ("walk_turn", 8, dict(fwd=0.6, yaw=-0.5)),
    ("settle", 2, dict()),
]

# change points placed at chunk 0/25/50/75% of the moving window
CHANGEPOINT = [
    ("settle", 3, dict()),
    ("fwd_on", 4, dict(fwd=0.7)),          # stop -> W
    ("fwd_off", 4, dict()),                # W -> release
    ("rev", 4, dict(fwd=-0.7)),            # W -> S
    ("yaw_swap", 4, dict(yaw=-0.7, fwd=0.4)),  # yaw+ -> yaw-
    ("flick", 4, dict(yaw=+1.0, strafe=0.8)),  # sudden mouse flick
    ("strafe_rev", 4, dict(strafe=-0.8)),  # strafe reversal
    ("settle", 2, dict()),
]


def build_base(scene, kind):
    """Run the controller at DISPLAY rate (dt=0.25, 4 steps per chunk).

    Returns (chunk_states, frame_poses, marks):
      chunk_states[c] = controller state at the START of chunk c (frame 4c)
      frame_poses[k]  = intended pose at output frame k  (the reference the
                        generated motion must be compared against)
    """
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max = 6.0
    ctl.cfg.pitch_rate_max = 2.0
    ctl.cfg.v_max = 1.0
    script = STEADY if kind == "steady" else CHANGEPOINT
    frame_poses, chunk_states, marks = [], [], {}
    n_chunks = 0
    for name, n, inp in script:
        lo = n_chunks
        ctl.set_input(**inp)
        for _ in range(n):
            chunk_states.append(ctl.clone())
            n_chunks += 1
            for _ in range(4):
                ctl.step(dt=0.25)
                frame_poses.append(ctl.pose.copy())
        marks[name] = (lo, n_chunks)
    return chunk_states, np.stack(frame_poses), marks


def pose_at(states, idx):
    if idx < 0:
        return states[0].pose
    if idx >= len(states):
        return states[-1].pose
    return states[idx].pose


def scale_se3(T, g):
    """Scale an SE(3) element (log -> *g -> exp)."""
    from scipy.spatial.transform import Rotation as Rot
    out = np.eye(4)
    out[:3, :3] = Rot.from_rotvec(Rot.from_matrix(T[:3, :3]).as_rotvec() * g).as_matrix()
    out[:3, 3] = T[:3, 3] * g
    return out


def predict(states, c, mode, horizon_lat, gain=1.0):
    st = states[c]
    if mode == "A_current":
        return st.pose
    if mode == "B_cv":
        return st.rollout_constant_velocity(horizon_lat)
    if mode == "C_ca":
        return st.rollout_constant_input(horizon_lat)
    if mode == "D_oracle":
        return pose_at(states, int(round(c + horizon_lat)))
    if mode == "E_scaled":
        # past-only control: extrapolate the PREVIOUS step, gain-matched to CV
        prev = states[max(c - 1, 0)]
        step = np.linalg.inv(prev.pose) @ st.pose
        return st.pose @ scale_se3(step, gain)
    raise ValueError(mode)


def cv_gain(states, horizon_lat):
    """Mean |CV displacement| / mean |previous-step displacement|."""
    num, den = [], []
    for c in range(1, len(states)):
        cv = states[c].rollout_constant_velocity(horizon_lat)
        num.append(np.linalg.norm(cv[:3, 3] - states[c].pose[:3, 3]))
        den.append(np.linalg.norm(states[c].pose[:3, 3] - states[c - 1].pose[:3, 3]))
    n, d = float(np.mean(num)), float(np.mean(den))
    return n / max(d, 1e-9)


def write_traj(base_pose, latent_poses, outdir, src):
    """Place latent pose i at trajectory index 4i and interpolate between."""
    from wan.utils.cam_utils import interpolate_camera_poses
    import torch
    L = len(latent_poses)
    N = 4 * (L - 1) + 1
    R = torch.from_numpy(np.stack([p[:3, :3] for p in latent_poses])).float()
    t = torch.from_numpy(np.stack([p[:3, 3] for p in latent_poses])).float()
    full = interpolate_camera_poses(np.linspace(0, L - 1, L), R, t,
                                    np.linspace(0, L - 1, N)).numpy()
    os.makedirs(outdir, exist_ok=True)
    np.save(f"{outdir}/poses.npy", full)
    shutil.copy(f"{src}/intrinsics.npy", f"{outdir}/intrinsics.npy")
    shutil.copy(f"{src}/image.jpg", f"{outdir}/image.jpg")
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="steady", choices=["steady", "changepoint"])
    ap.add_argument("--scene", default="04")
    ap.add_argument("--horizon", type=float, default=1.0,
                    help="in units of the generation latency L (1.25s)")
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args()

    states, frame_poses, marks = build_base(args.scene, args.kind)
    np.save(f"examples/pred_{args.kind}_{args.scene}_intended.npy", frame_poses)
    horizon_lat = args.horizon * (LATENT_SEC / LATENT_SEC)   # latent units
    out = {}
    for mode in args.modes.split(","):
        lats = [predict(states, c, mode, horizon_lat) for c in range(len(states))]
        d = f"examples/pred_{args.kind}_{args.scene}_{mode}_h{args.horizon}"
        N = write_traj(None, lats, d, f"examples/{args.scene}")
        out[mode] = dict(dir=d, frames=N)
        print(f"{mode:11s} h={args.horizon}  {N} frames -> {d}")
    with open(f"examples/pred_{args.kind}_{args.scene}_h{args.horizon}_meta.json", "w") as fh:
        json.dump(dict(kind=args.kind, scene=args.scene, horizon=args.horizon,
                       marks=marks, n_chunks=len(states), modes=out), fh, indent=1)


if __name__ == "__main__":
    main()
