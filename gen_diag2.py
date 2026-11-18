#!/usr/bin/env python
import numpy as np, math, os, shutil, sys
sys.path.insert(0, '.')
base = np.load('examples/03/poses.npy')[0]
R0 = base[:3, :3]; t0 = base[:3, 3]
fwd = R0 @ np.array([0, 0, 1.0])   # view dir
right = R0 @ np.array([1, 0, 0.0]) # camera right
up = R0 @ np.array([0, 1, 0.0])
def build(name, dirvec, dist, hold_frac=0.45):
    N = 81; poses = np.zeros((N, 4, 4))
    for t in range(N):
        c = t / 4.0
        # move out over c=0..4, hold c=4..13, come back c=13..17, hold
        if c <= 4:   amt = c / 4.0
        elif c <= 13: amt = 1.0
        elif c <= 17: amt = 1.0 - (c - 13) / 4.0
        else:         amt = 0.0
        poses[t, :3, :3] = R0
        poses[t, :3, 3] = t0 + dirvec * (dist * amt)
        poses[t, 3, 3] = 1.0
    outdir = f'examples/{name}'; os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'poses.npy'), poses)
    shutil.copy('examples/03/intrinsics.npy', os.path.join(outdir, 'intrinsics.npy'))
    print(name, 'max |t-t0| =', np.abs(poses[:, :3, 3] - t0).max())

build('03_strafe_r', right, 6.0)
build('03_strafe_l', right, -6.0)
build('03_dolly_past', fwd, 15.0)
print('generated strafe_r / strafe_l / dolly_past')
