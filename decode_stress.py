#!/usr/bin/env python
import os, time, torch
from wan.modules.vae2_1 import Wan2_1_VAE

ASSETS = os.path.expanduser('~/ai/models/lingbot-shared-assets')
vae = Wan2_1_VAE(z_dim=16, vae_pth=os.path.join(ASSETS, 'Wan2.1_VAE.pth'),
                 dtype=torch.float32, device='cuda')
print('vae loaded', flush=True)

for N in [6, 11, 21]:
    z = torch.randn(16, N, 36, 66, device='cuda', dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    print(f'--- N={N} full decode ---', flush=True)
    t0 = time.time()
    with torch.no_grad():
        full = vae.decode([z])[0]
    torch.cuda.synchronize()
    print(f'N={N} full {time.time()-t0:.1f}s shape={tuple(full.shape)} '
          f'peak={torch.cuda.max_memory_allocated()/2**20:.0f}MiB', flush=True)

    print(f'--- N={N} stream decode ---', flush=True)
    vae.stream_decode_start()
    parts = []
    for i in range(N):
        t0 = time.time()
        with torch.no_grad():
            parts.append(vae.stream_decode_step(z[:, i:i+1]))
        torch.cuda.synchronize()
        print(f'  N={N} step {i} {(time.time()-t0)*1000:.0f}ms', flush=True)
    vae.stream_decode_end()
    s = torch.cat(parts, dim=1)
    print(f'N={N} stream shape={tuple(s.shape)} maxdiff={(full-s).abs().max().item():.3e}',
          flush=True)
print('DONE', flush=True)
