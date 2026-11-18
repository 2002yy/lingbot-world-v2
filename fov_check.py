#!/usr/bin/env python
"""Is the door ROI still showing the same door across chunks?
If the ROI content drifts a lot, the door is out of FOV there."""
import sys
import numpy as np
import cv2

sys.path.insert(0, ".")
from object_permanence import patch  # noqa: E402

BB = (0.70, 0.50, 0.84, 0.66)


def feat(p):
    g = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.array([g.mean(), g.std(),
                     np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean()])


for tag in ("A", "B"):
    A = np.load(f"output/commit/{tag}_frames.npy")
    f0 = feat(patch(A[0], BB))
    print(f"--- branch {tag} ---")
    print("chunk  mean   std   |gx|   drift_vs_chunk0")
    for i in range(len(A)):
        f = feat(patch(A[i], BB))
        print(f"{i:5d} {f[0]:6.3f} {f[1]:6.3f} {f[2]:6.3f}   {np.abs(f - f0).sum():6.3f}")
