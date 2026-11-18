#!/usr/bin/env python
import numpy as np, math, os, shutil, sys
sys.path.insert(0, '.')
base = np.load('examples/03/poses.npy')[0]
R0 = base[:3, :3]; t0 = base[:3, 3]
def Ry(deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def build(yaw_fn, name, trans_fn=None):
    N = 81; poses = np.zeros((N, 4, 4))
    for t in range(N):
        c = t / 4.0
        poses[t, :3, :3] = R0 @ yaw_fn(c)
        poses[t, :3, 3] = t0 if trans_fn is None else t0 + trans_fn(c)
        poses[t, 3, 3] = 1.0
    outdir = f'examples/{name}'; os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'poses.npy'), poses)
    shutil.copy('examples/03/intrinsics.npy', os.path.join(outdir, 'intrinsics.npy'))
    print(name, 'yaw@chunks:', [round(math.degrees(math.atan2(yaw_fn(c)[0,2], yaw_fn(c)[0,0])),0) for c in range(0,21,2)])

# full spin: yaw 0 -> 360
build(lambda c: Ry(c * 360.0 / 20.0), '03_spin360')
# pitch (rotate around X) 0 -> 90 : look up to sky
build(lambda c: Rz(c * 90.0 / 20.0), '03_pitch90')
# pure translation forward along view dir (world -Z-ish), 0 -> 6 units
fwd = R0 @ np.array([0, 0, 1.0])
build(lambda c: np.eye(3), '03_fwd6', trans_fn=lambda c: fwd * (c * 6.0 / 20.0))
print('generated spin360 / pitch90 / fwd6')
