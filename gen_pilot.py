#!/usr/bin/env python
import numpy as np, os, shutil, sys
def pilot(e, Kf=32, H=32, hold=4):
    p = np.load(f'examples/{e}/poses.npy')
    fwd = p[:Kf]
    hld_out = np.tile(p[Kf - 1], (H, 1, 1))
    bwd = p[Kf - 2::-1]
    hld = np.tile(p[0], (hold, 1, 1))
    traj = np.concatenate([fwd, hld_out, bwd, hld], 0)
    d = f'examples/{e}_pilot'; os.makedirs(d, exist_ok=True)
    np.save(f'{d}/poses.npy', traj)
    shutil.copy(f'examples/{e}/intrinsics.npy', f'{d}/intrinsics.npy')
    shutil.copy(f'examples/{e}/image.jpg', f'{d}/image.jpg')
    ret = Kf + H + len(bwd) - 1
    print(f'{e}_pilot: {len(traj)} frames, return idx {ret}, out-hold {H//4} chunks')

for e in sys.argv[1:] or ['01', '04']:
    pilot(e)
