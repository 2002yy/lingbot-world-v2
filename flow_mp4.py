#!/usr/bin/env python
"""Mean optical-flow magnitude of an mp4 (or a directory of them)."""
import sys
import glob
import os
import numpy as np
import cv2


def flow_mp4(path, idle=2, nact=8):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    mags = [float(np.mean(np.linalg.norm(cv2.calcOpticalFlowFarneback(
        frames[i - 1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2)))
        for i in range(1, len(frames))]
    mags = np.array(mags)
    act = mags[4 * idle:4 * (idle + nact)]
    return float(act.mean()), float(np.median(act)), len(frames)


def main():
    paths = sys.argv[1:] or ["output/rottrans"]
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.mp4"))) if os.path.isdir(p) else [p]
    print(f"{'clip':24s} {'mean':>8s} {'median':>8s} {'frames':>7s}")
    for f in files:
        m, md, n = flow_mp4(f)
        print(f"{os.path.basename(f)[:-4]:24s} {m:8.3f} {md:8.3f} {n:7d}")


if __name__ == "__main__":
    main()
