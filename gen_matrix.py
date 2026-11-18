#!/usr/bin/env python
"""Build a memory-matrix trajectory: forward / hold-out / return / hold + yaw probe.

Layout (frames), K = interval in chunks (4 frames per chunk):
  fwd        [0, 4K)
  hold_out   [4K, 8K)
  return     [8K, 12K-1)     -> reaches the start pose at 12K-2
  hold       [12K-1, 12K+3)
  yaw_plus   [12K+3, 12K+15)
  hold       [12K+15, 12K+19)
  yaw_minus  [12K+19, 12K+43)
  hold       [12K+43, 12K+47)
Total = 12K + 47 frames.

  python gen_matrix.py --scene 01 --K 8
"""
import argparse
import os
import shutil

import numpy as np

from cam_controller import CameraController

YAW_RATE = 6.0    # deg per latent (== per chunk)
FWD_SPEED = 0.6


def build(scene, K, outdir):
    p = np.load(f"examples/{scene}/poses.npy")
    fwd = p[:4 * K]
    far = p[4 * K - 1]
    hold_out = np.tile(far, (4 * K, 1, 1))
    bwd = p[4 * K - 2::-1]
    hold = np.tile(p[0], (4, 1, 1))
    traj = np.concatenate([fwd, hold_out, bwd, hold], 0)

    # yaw probe seeded at the start pose (history already written above)
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max = YAW_RATE
    ctl.cfg.v_max = FWD_SPEED
    ctl.poses = []                      # we only want the probe frames
    probe = []
    def run(n_chunks, **inp):
        ctl.set_input(**inp)
        for _ in range(n_chunks * 4):
            ctl.step(dt=0.25)           # 4 frames == 1 latent
            probe.append(ctl.pose.copy())
    run(1, fwd=0.0)                     # settle (kept in 'hold')
    run(3, yaw=+1.0)
    run(1)
    run(6, yaw=-1.0)
    run(1)
    probe = np.stack(probe)
    # drop the settle chunk (already covered by 'hold')
    traj = np.concatenate([traj, probe[4:]], 0)

    os.makedirs(outdir, exist_ok=True)
    np.save(f"{outdir}/poses.npy", traj)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{outdir}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{outdir}/image.jpg")

    marks = dict(
        fwd=(0, 4 * K), hold_out=(4 * K, 8 * K), ret=(8 * K, 12 * K - 1),
        hold=(12 * K - 1, 12 * K + 3),
        yaw_plus=(12 * K + 3, 12 * K + 15),
        yaw_minus=(12 * K + 19, 12 * K + 43),
        return_idx=12 * K - 2, n_frames=len(traj))
    return marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or f"examples/mx_{args.scene}_K{args.K}"
    m = build(args.scene, args.K, outdir)
    print(f"{outdir}: {m['n_frames']} frames, return_idx={m['return_idx']}")
    print("marks:", {k: v for k, v in m.items() if k != "n_frames"})


if __name__ == "__main__":
    main()
