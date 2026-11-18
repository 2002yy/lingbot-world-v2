#!/usr/bin/env python
import numpy as np, os, shutil
ex = 'examples/02'
d = 'examples/02_official'; os.makedirs(d, exist_ok=True)
shutil.copy(f'{ex}/poses.npy', f'{d}/poses.npy')
shutil.copy(f'{ex}/intrinsics.npy', f'{d}/intrinsics.npy')
shutil.copy(f'{ex}/image.jpg', f'{d}/image.jpg')
print('built 02_official', np.load(f'{d}/poses.npy').shape)
