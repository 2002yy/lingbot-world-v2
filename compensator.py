#!/usr/bin/env python
"""Camera Response Compensator: component-wise pre-emphasis.

The plant (per §29/§30) is a stateful dynamic system with two clearly separated
channels:

    rotation    (yaw, walk_turn)  T50 ~1.0-1.17 chunk, REVERSE flips in 1 chunk
    translation (fwd, strafe)     T50 ~2.0 chunk,      REVERSE takes 4-5 chunks

So we do NOT push the whole SE(3) pose into the future with one horizon. We
pre-emphasise each twist component separately:

    orientation <- w * LEAD_ROT
    position    <- v * LEAD_TRANS * gate

where `gate` is a reversal guard: it drops immediately when the translation
direction reverses or the command is released, and recovers slowly, so we never
aggressively lead a channel that cannot come back.

This module only rewrites the *conditioning* pose. The authoritative control
pose and the model pose pipeline are untouched.
"""
import math

import numpy as np

from cam_controller import so3_exp

LEAD_ROT = 1.0        # chunks
LEAD_TRANS = 2.0      # chunks
GATE_RECOVER = 0.25   # per-chunk recovery fraction (slow)
STOP_EPS = 0.02       # |v| below this counts as released


class Compensator:
    def __init__(self, lead_rot=LEAD_ROT, lead_trans=LEAD_TRANS,
                 recover=GATE_RECOVER):
        self.lead_rot = lead_rot
        self.lead_trans = lead_trans
        self.recover = recover
        self.gate = 1.0
        self.log = []

    def _gate_target(self, states, c):
        """1 = same direction and moving; 0 = released or reversed."""
        if c == 0:
            return 1.0
        vp = states[c - 1].v
        vn = states[c].v
        n_prev = float(np.linalg.norm(vp))
        n_now = float(np.linalg.norm(vn))
        if n_now < STOP_EPS:
            return 0.0                     # release / hard deceleration
        if n_prev < STOP_EPS:
            return 1.0                     # starting from rest: no old momentum
        # bring the previous velocity into the current local frame
        R = states[c - 1].pose[:3, :3].T @ states[c].pose[:3, :3]
        vp_l = R @ vp
        cos = float(np.dot(vp_l, vn) / (n_prev * n_now))
        if cos <= 0.0:
            return 0.0                     # reversal: shut the lead off
        return float(min(1.0, cos / 0.7))  # partial credit while turning

    def step(self, states, c):
        tgt = self._gate_target(states, c)
        if tgt < self.gate:
            self.gate = tgt                 # drop fast
        else:
            self.gate += self.recover * (tgt - self.gate)   # recover slowly
        self.log.append(dict(c=c, gate=self.gate, target=tgt))
        return self.gate

    def pose_at(self, states, c):
        """Conditioning pose for chunk c (rotation and translation led apart)."""
        g = self.step(states, c)
        st = states[c]
        T = np.eye(4)
        T[:3, :3] = so3_exp(st.w * self.lead_rot)
        T[:3, 3] = st.v * (self.lead_trans * g)
        return st.pose @ T

    def trajectory(self, states):
        return [self.pose_at(states, c) for c in range(len(states))]
