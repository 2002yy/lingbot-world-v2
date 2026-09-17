#!/usr/bin/env python
"""In-distribution SE(3) camera controller for LingBot-World.

Turns keyboard/mouse-style inputs into a smooth camera trajectory whose
per-step statistics match the official example trajectories. Increments are
composed in the CAMERA-LOCAL frame (verified from official data: yaw about
local Y dominates, pitch about local X is secondary, roll ~ 0).

Envelope (per latent = 4 frames), measured from examples/00..04:
    translation  normalized median 0.19-0.78, p90 0.61-0.92, max 1.0
    yaw          mean |Y| 0.9-6.3 deg, max 36 deg
    pitch        mean |X| 0.29-0.98 deg, max ~6.8 deg
    roll         negligible

Usage (offline):
    python cam_controller.py --demo yaw_forward --out examples/ctl_demo
"""
import argparse
import math
import os
import shutil

import numpy as np

from wan.utils.cam_utils import interpolate_camera_poses


def so3_exp(w):
    """Rodrigues: w (rad, 3) -> R (3,3)."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


class EnvelopeConfig:
    """Per-latent motion limits, derived from the official trajectory envelope."""
    v_max = 1.0            # normalized translation per latent (model units)
    v_boost = 1.0          # Shift multiplier (capped at v_max)
    a_max = 0.20           # translation accel limit (per latent^2)
    yaw_rate_max = 6.0     # deg per latent
    pitch_rate_max = 2.0   # deg per latent
    roll_rate_max = 0.5    # deg per latent
    yaw_accel_max = 3.0    # deg per latent^2
    pitch_accel_max = 1.5  # deg per latent^2
    roll_accel_max = 0.5   # deg per latent^2
    input_alpha = 0.35     # EMA smoothing on raw inputs (0 = none)
    # In-distribution requirement (calibration 2026-09-12): pure-rotation
    # trajectories are ignored in strongly anchored scenes. Rotation only
    # responds when a concurrent translation is present. Keep a translation
    # floor whenever the camera is turning.
    turn_trans_floor = 0.18


class CameraController:
    """First-person SE(3) camera driven by (forward, strafe, yaw, pitch)."""

    def __init__(self, R0, t0, cfg=None):
        self.cfg = cfg or EnvelopeConfig()
        self.pose = np.eye(4)
        self.pose[:3, :3] = np.asarray(R0, float)
        self.pose[:3, 3] = np.asarray(t0, float)
        self.v = np.zeros(3)     # local [right, down, forward]
        self.w = np.zeros(3)     # local [pitch(x), yaw(y), roll(z)] rad/latent
        self.inp = np.zeros(4)   # raw [fwd, strafe, yaw, pitch] in [-1,1]
        self.sm = np.zeros(4)    # smoothed inputs
        self.boost = 1.0
        self.poses = [self.pose.copy()]

    def set_input(self, fwd=0.0, strafe=0.0, yaw=0.0, pitch=0.0, boost=False):
        self.inp = np.clip(np.array([fwd, strafe, yaw, pitch], float), -1.0, 1.0)
        self.boost = self.cfg.v_boost if boost else 1.0

    def step(self, dt=1.0):
        """Advance one step. dt is in latent units (1.0 = one latent = 4 frames);
        use dt=1/60 to drive the controller at display rate."""
        c = self.cfg
        a = c.input_alpha
        self.sm = (1.0 - a) * self.sm + a * self.inp
        fwd, strafe, yaw, pitch = self.sm

        v_target = np.array([strafe, 0.0, fwd]) * c.v_max * self.boost
        # rotation needs a concurrent translation to be in-distribution
        if (abs(yaw) > 0.05 or abs(pitch) > 0.05):
            need = c.turn_trans_floor * c.v_max
            if np.linalg.norm(v_target) < need:
                s = fwd if abs(fwd) > 1e-6 else 1.0
                v_target[2] = math.copysign(
                    math.sqrt(max(need * need - v_target[0] * v_target[0], 0.0)), s)
        w_target = np.radians(np.array([
            pitch * c.pitch_rate_max,
            yaw * c.yaw_rate_max,
            0.0,
        ]))

        self.v += np.clip(v_target - self.v, -c.a_max * dt, c.a_max * dt)
        w_lim = np.radians([c.pitch_accel_max, c.yaw_accel_max, c.roll_accel_max]) * dt
        self.w += np.clip(w_target - self.w, -w_lim, w_lim)

        Td = np.eye(4)
        Td[:3, :3] = so3_exp(self.w * dt)
        Td[:3, 3] = self.v * dt
        self.pose = self.pose @ Td
        self.poses.append(self.pose.copy())
        return self.pose

    # ---- prediction support (predictive conditioning) ----
    def clone(self):
        """Deep-copy the controller state so it can be rolled out without
        disturbing the authoritative control state."""
        import copy as _copy
        c = _copy.deepcopy(self)
        return c

    def rollout_constant_velocity(self, latents):
        """Integrate with FROZEN v and w for `latents` latent periods (CV)."""
        P = self.pose.copy()
        Td = np.eye(4)
        Td[:3, :3] = so3_exp(self.w * latents)
        Td[:3, 3] = self.v * latents
        return P @ Td

    def rollout_constant_input(self, latents):
        """Keep the current input held and step with the normal integrator
        (acceleration-limited) for `latents` latent periods (CA)."""
        c = self.clone()
        c.step(dt=latents)
        return c.pose

    def to_frames(self):
        """Upsample latent-rate poses to the frame rate the pipeline expects."""
        P = np.stack(self.poses)  # [L,4,4]
        L = len(P)
        N = 4 * (L - 1) + 1
        return interpolate_camera_poses(
            np.linspace(0, L - 1, L), P[:, :3, :3], P[:, :3, 3],
            np.linspace(0, L - 1, N)).numpy()

    def save(self, outdir, base_example='examples/00'):
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, 'poses.npy'), self.to_frames())
        shutil.copy(os.path.join(base_example, 'intrinsics.npy'),
                    os.path.join(outdir, 'intrinsics.npy'))
        shutil.copy(os.path.join(base_example, 'image.jpg'),
                    os.path.join(outdir, 'image.jpg'))
        print(f'saved {outdir}  latents={len(self.poses)}  frames={4*(len(self.poses)-1)+1}')


DEMOS = {
    # (n_latents, fwd, strafe, yaw, pitch, boost)
    'yaw_right':   [(3, 0, 0, 0, 0, 0), (9, 0, 0, 0.6, 0, 0), (4, 0, 0, 0, 0, 0)],
    'yaw_left':    [(3, 0, 0, 0, 0, 0), (9, 0, 0, -0.6, 0, 0), (4, 0, 0, 0, 0, 0)],
    'pitch_up':    [(3, 0, 0, 0, 0, 0), (6, 0, 0, 0, 0.6, 0), (4, 0, 0, 0, 0, 0)],
    'pitch_down':  [(3, 0, 0, 0, 0, 0), (6, 0, 0, 0, -0.6, 0), (4, 0, 0, 0, 0, 0)],
    'forward':     [(3, 0, 0, 0, 0, 0), (9, 0.6, 0, 0, 0, 0), (4, 0, 0, 0, 0, 0)],
    'backward':    [(3, 0, 0, 0, 0, 0), (9, -0.6, 0, 0, 0, 0), (4, 0, 0, 0, 0, 0)],
    'strafe_r':    [(3, 0, 0, 0, 0, 0), (9, 0, 0.6, 0, 0, 0), (4, 0, 0, 0, 0, 0)],
    'strafe_l':    [(3, 0, 0, 0, 0, 0), (9, 0, -0.6, 0, 0, 0), (4, 0, 0, 0, 0, 0)],
    'walk_turn':   [(2, 0, 0, 0, 0, 0), (6, 0.5, 0, 0.3, 0, 0),
                    (6, 0.5, 0, -0.3, 0, 0), (4, 0, 0, 0, 0, 0)],
}


def run_demo(name, base_example, outdir):
    base = np.load(os.path.join(base_example, 'poses.npy'))[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    for n, fwd, strafe, yaw, pitch, boost in DEMOS[name]:
        ctl.set_input(fwd, strafe, yaw, pitch, boost)
        for _ in range(n):
            ctl.step()
    ctl.set_input(0, 0, 0, 0)
    ctl.save(outdir, base_example)
    return ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', default='walk_turn', choices=list(DEMOS))
    ap.add_argument('--base', default='examples/00')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    outdir = args.out or f'examples/ctl_{args.demo}'
    run_demo(args.demo, args.base, outdir)


if __name__ == '__main__':
    main()
