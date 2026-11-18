#!/usr/bin/env python
"""Build one motion-showcase trajectory covering all motion classes.

Segments (4 frames per chunk, 4 chunks each unless noted):
    settle (1 chunk) -> rot_only(fast, floor off) -> rot_med -> rot_fast
    -> fwd_med -> fwd_fast -> str_med -> str_fast -> walk_med -> walk_fast
    -> settle (1 chunk)

  python gen_motions.py --scene 04
"""
import argparse
import json
import os
import shutil

import numpy as np

from cam_controller import CameraController

# name, n_chunks, inputs, floor (None = leave default)
MOTIONS = [
    ("rot_only", 4, dict(yaw=+1.0), 0.0),
    ("rot_med",  4, dict(yaw=+0.5), None),
    ("rot_fast", 4, dict(yaw=+1.0), None),
    ("fwd_med",  4, dict(fwd=+0.5), None),
    ("fwd_fast", 4, dict(fwd=+1.0), None),
    ("str_med",  4, dict(strafe=+0.5), None),
    ("str_fast", 4, dict(strafe=+1.0), None),
    ("walk_med", 4, dict(fwd=+0.5, yaw=-0.4), None),
    ("walk_fast", 4, dict(fwd=+1.0, yaw=-0.8), None),
]


def build(scene, outdir):
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max = 6.0
    ctl.cfg.pitch_rate_max = 2.0
    ctl.cfg.v_max = 1.0
    ctl.poses = []
    frames, marks = [], {}
    def run(n_chunks, **inp):
        ctl.set_input(**inp)
        for _ in range(n_chunks * 4):
            ctl.step(dt=0.25)
            frames.append(ctl.pose.copy())
    run(1)                                   # settle
    for name, n, inp, floor in MOTIONS:
        lo = len(frames)
        ctl.cfg.turn_trans_floor = 0.18 if floor is None else floor
        run(n, **inp)
        marks[name] = (lo, len(frames))
    ctl.cfg.turn_trans_floor = 0.18
    run(1)                                   # settle
    traj = np.stack(frames)
    os.makedirs(outdir, exist_ok=True)
    np.save(f"{outdir}/poses.npy", traj)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{outdir}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{outdir}/image.jpg")
    with open(f"{outdir}/marks.json", "w") as fh:
        json.dump(marks, fh, indent=1)
    return len(traj), marks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    n, marks = build(args.scene, f"examples/mot_{args.scene}")
    print(f"examples/mot_{args.scene}: {n} frames")
    for k, v in marks.items():
        print(f"  {k:10s} {v}")
