#!/usr/bin/env python
"""Display decoupling PoC: raise the world-model output rate to a display rate.

World generation runs at ~0.8 chunk/s (= 4 frames/chunk -> ~3.2 fps). This
script synthesises in-between frames with bidirectional Farneback optical flow
so the display can run at N x the generation rate, decoupling "how fast the
world is computed" from "how fast it is shown".

  python display_decouple.py --video output/interactive/interactive.mp4 --mult 4
"""
import argparse
import os
import time

import cv2
import numpy as np


def flow_warp(img, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (gx + flow[..., 0]).astype(np.float32)
    mapy = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def interp(a, b, f01):
    """Synthesise the frame at fraction f01 between a and b (BGR uint8)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    fwd = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 21, 3, 7, 1.2, 0)
    bwd = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 21, 3, 7, 1.2, 0)
    # approximate intermediate flow (linear, no occlusion handling)
    fa = flow_warp(a, fwd * f01)
    fb = flow_warp(b, bwd * (1.0 - f01))
    return cv2.addWeighted(fa, 1.0 - f01, fb, f01, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="output/interactive/interactive.mp4")
    ap.add_argument("--mult", type=int, default=4)
    ap.add_argument("--out", default="output/interactive/interactive_display.mp4")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    src, fps = [], cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        src.append(f)
    cap.release()
    h, w = src[0].shape[:2]
    print(f"[dd] src {len(src)} frames @ {fps:.1f}fps ({w}x{h})")

    out_fps = fps * args.mult
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    t0 = time.perf_counter()
    n_out = 0
    for i in range(len(src) - 1):
        vw.write(src[i])
        n_out += 1
        for k in range(1, args.mult):
            vw.write(interp(src[i], src[i + 1], k / args.mult))
            n_out += 1
    vw.write(src[-1])
    n_out += 1
    vw.release()
    dt = time.perf_counter() - t0
    print(f"[dd] wrote {n_out} frames @ {out_fps:.1f}fps -> {args.out}")
    print(f"[dd] interpolation cost {dt:.1f}s for {n_out-len(src)} synth frames "
          f"= {1000*dt/max(1, n_out-len(src)):.1f} ms/frame")


if __name__ == "__main__":
    main()
