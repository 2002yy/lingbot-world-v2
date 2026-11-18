#!/usr/bin/env python
import numpy as np, math, torch
import sys
sys.path.insert(0, '.')
from wan.utils.cam_utils import compute_relative_poses

c2ws = np.load('examples/03/poses.npy')
Ks = np.load('examples/03/intrinsics.npy')
print('poses shape', c2ws.shape, 'intrinsics', Ks.shape)
print('Ks[0]:', Ks[0])
fx = Ks[0,0]; fy = Ks[0,1]; cx = Ks[0,2]; cy = Ks[0,3]
W, H = 832, 480
print(f'hFOV = {2*math.degrees(math.atan(W/(2*fx))):.1f} deg')
print(f'vFOV = {2*math.degrees(math.atan(H/(2*fy))):.1f} deg')
print('pose[0]:\n', c2ws[0])
print('translation (t) across poses:')
t = c2ws[:, :3, 3]
print('  min', t.min(0), 'max', t.max(0), 'range', (t.max(0)-t.min(0)))
print('  |t - t0| max:', np.abs(t - t[0]).max())
print('rotation det range:', np.linalg.det(c2ws[:, :3, :3]).min(), np.linalg.det(c2ws[:, :3, :3]).max())
# how much does the example rotate?
R0 = c2ws[0,:3,:3]
ang = [math.degrees(math.acos(min(1,max(-1,((np.trace(R0.T@c2ws[i,:3,:3])-1)/2))))) for i in range(len(c2ws))]
print(f'example total rotation: {ang[0]:.1f} -> {ang[-1]:.1f} deg')
