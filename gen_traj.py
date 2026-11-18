#!/usr/bin/env python
import numpy as np, math, os, shutil, sys
sys.path.insert(0, '.')
from wan.utils.cam_utils import compute_relative_poses, interpolate_camera_poses

base = np.load('examples/03/poses.npy')[0]
R0 = base[:3, :3]; t0 = base[:3, 3]

def Ry(deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def ss(a, b, x):
    if b == a: return 0.0
    u = max(0.0, min(1.0, (x - a) / (b - a)))
    return u * u * (3 - 2 * u)

def yaw_latent(c):
    if c <= 0:  return 0.0
    if c <= 3:  return 90.0 * ss(0, 3, c)          # turn away (c 0->3)
    if c <= 9:  return 90.0                          # hold out (c 4..9 = 6 chunks)
    if c <= 13: return 90.0 * (1 - ss(9, 13, c))     # turn back (c 10->13)
    return 0.0                                       # visible again (c 14..20)

N = 81
poses = np.zeros((N, 4, 4))
for t in range(N):
    yaw = yaw_latent(t / 4.0)
    poses[t, :3, :3] = R0 @ Ry(yaw)
    poses[t, :3, 3] = t0
    poses[t, 3, 3] = 1.0

outdir = 'examples/03_leave_return'
os.makedirs(outdir, exist_ok=True)
np.save(os.path.join(outdir, 'poses.npy'), poses)
shutil.copy('examples/03/intrinsics.npy', os.path.join(outdir, 'intrinsics.npy'))

print('=== trajectory ===')
print('latent yaw:', [round(yaw_latent(c), 1) for c in range(21)])
lat = poses[0::4]  # frames 0,4,...,80 == latent poses
print('=== geometry gate (latent-level) ===')
print('max|latent[0]-latent[14]| =', np.abs(lat[0] - lat[14]).max())
print('max|latent[0]-latent[20]| =', np.abs(lat[0] - lat[20]).max())
print('translation const?  |t-t0| max =', np.abs(poses[:, :3, 3] - t0).max())

# reproduce pipeline's relative-pose path for the return frame
c2ws = torch_poses = None
import torch
c2ws_t = torch.from_numpy(poses).float()
len_c2ws_ = int((N - 1) // 4) + 1
c2ws_infer = interpolate_camera_poses(
    src_indices=np.linspace(0, N - 1, N),
    src_rot_mat=poses[:, :3, :3], src_trans_vec=poses[:, :3, 3],
    tgt_indices=np.linspace(0, N - 1, len_c2ws_))
rel = compute_relative_poses(c2ws_infer, framewise=True)
print('=== pipeline relative-pose check ===')
print('return chunk index = 14, rel[14] vs identity max =',
      float((rel[14] - torch.eye(4)).abs().max()))
print('chunks:', len_c2ws_, '  return frame =', 4 * 14)
print('saved ->', outdir)
