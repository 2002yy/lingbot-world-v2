#!/usr/bin/env python
import numpy as np, os, shutil
ex = 'examples/00'
p = np.load(f'{ex}/poses.npy')
# moving segment: frames 128..191 (rotation ramps up)
seg = p[128:192]
d = 'examples/00_official_mv'; os.makedirs(d, exist_ok=True)
np.save(f'{d}/poses.npy', seg)
shutil.copy(f'{ex}/intrinsics.npy', f'{d}/intrinsics.npy')
shutil.copy(f'{ex}/image.jpg', f'{d}/image.jpg')
print('built 00_official_mv', seg.shape)
