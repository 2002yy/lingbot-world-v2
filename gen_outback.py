#!/usr/bin/env python
import numpy as np, os, shutil
def outback(e, K, hold=6):
    p = np.load(f'examples/{e}/poses.npy')
    fwd = p[:K]                     # 0..K-1
    bwd = p[K-2::-1]                # K-2..0  (return to start)
    hld = np.tile(p[0], (hold, 1, 1))
    traj = np.concatenate([fwd, bwd, hld], axis=0)
    d = f'examples/{e}_outback'; os.makedirs(d, exist_ok=True)
    np.save(f'{d}/poses.npy', traj)
    shutil.copy(f'examples/{e}/intrinsics.npy', f'{d}/intrinsics.npy')
    shutil.copy(f'examples/{e}/image.jpg', f'{d}/image.jpg')
    ret = len(fwd) + len(bwd) - 1
    print(f'{e}_outback: {len(traj)} frames, return at idx {ret}, chunks {(len(traj)-1)//4+1}')

outback('04', 40)
outback('01', 40)
