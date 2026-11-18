#!/usr/bin/env python
"""Immediate Camera Response Layer (Level 0 pose reprojection).

Rotation is geometrically exact (infinite-depth homography). Translation is a
conservative fronto-parallel-plane approximation with a nominal depth; the
world model is responsible for the final correction.

  H = K (R - t n^T / d) K^-1 ,  n = e3 (plane perpendicular to the optical axis)

Also provides the four-metric reporting口径:
  control-to-warp latency | control-to-real-frame latency |
  world-model throughput | display FPS
"""
import argparse
import os
import time

import cv2
import numpy as np


def K_from_vec(k):
    fx, fy, cx, cy = [float(x) for x in k[:4]]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


class PoseWarper:
    """Reprojects a reference frame to the current pose."""

    def __init__(self, k_vec, h, w, nominal_depth=4.0):
        self.K = K_from_vec(k_vec)
        self.Ki = np.linalg.inv(self.K)
        self.h, self.w = int(h), int(w)
        self.d = float(nominal_depth)
        self.ref_frame = None
        self.ref_pose = None

    def set_reference(self, frame, pose):
        self.ref_frame = frame
        self.ref_pose = np.asarray(pose, float).reshape(4, 4)

    def homography(self, pose):
        T = np.linalg.inv(self.ref_pose) @ np.asarray(pose, float).reshape(4, 4)
        R = T[:3, :3].copy()
        t = T[:3, 3]
        R[:, 2] -= t / self.d          # subtract t n^T / d  (n = e3)
        H = self.K @ R @ self.Ki
        return H / H[2, 2]

    def warp(self, pose):
        H = self.homography(pose)
        return cv2.warpPerspective(self.ref_frame, H, (self.w, self.h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)


def _demo(args):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cam_controller import CameraController
    from wan.utils.cam_utils import get_Ks_transformed
    import torch
    from PIL import Image

    W, H = (int(x) for x in args.area.lower().split("x"))
    img = np.array(Image.open(f"{args.base}/image.jpg").convert("RGB"))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    Ks = torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float()
    Ks = get_Ks_transformed(Ks, 480, 832, H, W, H, W)[0].numpy()

    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    warper = PoseWarper(Ks, H, W, nominal_depth=args.depth)
    warper.set_reference(img, ctl.pose)

    # scripted mouse turn at DISPLAY rate; reference frame refreshed on the
    # real-frame cadence (one real chunk every ~1.25s), as the live system does.
    real_cadence = args.real_cadence          # seconds between real frames
    disp_fps = args.fps
    dt = 1.0 / (disp_fps * real_cadence)      # controller dt in latent units
    seg = int(round(disp_fps * real_cadence))  # display frames per real frame
    script = [(seg, 0.0), (seg, 1.0), (seg, 0.0), (seg, -1.0), (seg, 0.0)]
    frames, warp_ms, ref_updates = [], [], 0
    since_ref = 0
    for n, yaw in script:
        ctl.set_input(fwd=args.fwd, yaw=yaw)
        for _ in range(n):
            ctl.step(dt=dt)
            t0 = time.perf_counter()
            out = warper.warp(ctl.pose)
            warp_ms.append((time.perf_counter() - t0) * 1000.0)
            frames.append(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            since_ref += 1
            if since_ref >= seg:
                # a new real frame arrived: it becomes the new reference
                warper.set_reference(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), ctl.pose)
                since_ref = 0
                ref_updates += 1

    disp_fps = args.fps
    os.makedirs(args.out_dir, exist_ok=True)
    mp4 = f"{args.out_dir}/immediate_warp.mp4"
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), disp_fps, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    wm = np.array(warp_ms)
    print(f"[warp] {len(frames)} frames, warp cost mean={wm.mean():.2f}ms "
          f"p90={np.percentile(wm,90):.2f}ms max={wm.max():.2f}ms")
    print(f"[warp] control-to-warp latency = {wm.mean():.2f}ms (target < 10ms)")
    print(f"[warp] reference refreshed {ref_updates}x on {real_cadence}s cadence")
    print(f"[warp] -> {mp4}")
    sheet = frames[::max(1, len(frames)//6)][:6]
    tile = np.concatenate([np.concatenate(sheet[:3], axis=1),
                           np.concatenate(sheet[3:6], axis=1)], axis=0)
    out_png = f"{args.out_dir}/immediate_warp_sheet.png"
    cv2.imwrite(out_png, tile)
    print(f"[warp] sheet -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="examples/00")
    ap.add_argument("--area", default="528x288")
    ap.add_argument("--depth", type=float, default=4.0)
    ap.add_argument("--fwd", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--real_cadence", type=float, default=1.25)
    ap.add_argument("--out_dir", default="output/warp")
    args = ap.parse_args()
    _demo(args)


if __name__ == "__main__":
    main()
