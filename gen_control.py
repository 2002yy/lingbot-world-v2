#!/usr/bin/env python
import numpy as np, os, shutil, sys
ex = 'examples/00'
base = np.load(f'{ex}/poses.npy')[0]
N = 81
def mk(name, poses):
    d = f'examples/{name}'; os.makedirs(d, exist_ok=True)
    np.save(f'{d}/poses.npy', poses)
    shutil.copy(f'{ex}/intrinsics.npy', f'{d}/intrinsics.npy')
    shutil.copy(f'{ex}/image.jpg', f'{d}/image.jpg')
    print('built', name, poses.shape)

mk('00_static', np.tile(base, (N, 1, 1)))
mk('00_official', np.load(f'{ex}/poses.npy'))   # full 269, denoise_ab slices to frames
print('yaw90 already at examples/00_yaw90')
