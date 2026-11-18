#!/usr/bin/env python
import numpy as np, math, os, shutil, sys
base_dir = sys.argv[1]          # e.g. examples/00
name = sys.argv[2]              # output name
base = np.load(f'{base_dir}/poses.npy')[0]
R0 = base[:3, :3]; t0 = base[:3, 3]
def Ry(deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def ss(a, b, x):
    u = max(0.0, min(1.0, (x - a) / (b - a))); return u * u * (3 - 2 * u)
def yaw(c):
    if c <= 3:  return 90.0 * ss(0, 3, c)
    if c <= 9:  return 90.0
    if c <= 13: return 90.0 * (1 - ss(9, 13, c))
    return 0.0
N = 81; poses = np.zeros((N, 4, 4))
for t in range(N):
    poses[t, :3, :3] = R0 @ Ry(yaw(t / 4.0)); poses[t, :3, 3] = t0; poses[t, 3, 3] = 1.0
outdir = f'examples/{name}'; os.makedirs(outdir, exist_ok=True)
np.save(f'{outdir}/poses.npy', poses)
shutil.copy(f'{base_dir}/intrinsics.npy', f'{outdir}/intrinsics.npy')
shutil.copy(f'{base_dir}/image.jpg', f'{outdir}/image.jpg')
print('built', name)
