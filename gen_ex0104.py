#!/usr/bin/env python
import numpy as np, os, shutil
for e in ['01', '04']:
    d = f'examples/{e}_official'; os.makedirs(d, exist_ok=True)
    shutil.copy(f'examples/{e}/poses.npy', f'{d}/poses.npy')
    shutil.copy(f'examples/{e}/intrinsics.npy', f'{d}/intrinsics.npy')
    shutil.copy(f'examples/{e}/image.jpg', f'{d}/image.jpg')
    print('built', d)
