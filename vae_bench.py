#!/usr/bin/env python
"""
vae_bench.py — VAE streaming decode 隔离 benchmark（不加载 DiT）

每个配置独立进程运行；输出 median/P90 decode_ms、peak alloc/reserved、以及
与 fp32 基线的 maxdiff/cosine。

用法: python vae_bench.py --config B_bf16_NCDHW --out /tmp/vae_B.pt
"""
import argparse
import os
import statistics
import time

import torch

from wan.modules.vae2_1 import Wan2_1_VAE

ASSETS = os.path.expanduser('~/ai/models/lingbot-shared-assets')
N = 21


def make(dtype, layout):
    vae = Wan2_1_VAE(z_dim=16,
                     vae_pth=os.path.join(ASSETS, 'Wan2.1_VAE.pth'),
                     dtype=dtype, device='cuda')
    if dtype != torch.float32:
        vae.model.to(dtype)
    if layout == 'cl3d':
        for m in vae.model.modules():
            if isinstance(m, torch.nn.Conv3d):
                m.to(memory_format=torch.channels_last_3d)
    return vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='/tmp/vae_out.pt')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    name = args.config
    dtype = torch.float32 if 'fp32' in name else (
        torch.bfloat16 if 'bf16' in name else torch.float16)
    layout = 'cl3d' if 'cl3d' in name else 'ncdhw'

    vae = make(dtype, layout)
    g = torch.Generator(device='cuda')
    g.manual_seed(args.seed)
    z = torch.randn(16, N, 36, 66, generator=g, device='cuda',
                    dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats()
    vae.stream_decode_start()
    ms = []
    parts = []
    for i in range(N):
        zi = z[:, i:i + 1]
        if layout == 'cl3d':
            zi = zi.to(memory_format=torch.channels_last_3d)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        with torch.no_grad():
            parts.append(vae.stream_decode_step(zi))
        e1.record()
        torch.cuda.synchronize()
        ms.append(e0.elapsed_time(e1))
    vae.stream_decode_end()
    out = torch.cat(parts, dim=1)

    med = statistics.median(ms)
    p90 = sorted(ms)[min(len(ms) - 1, int(0.9 * len(ms)))]
    print(f"[vae] {name:16s} median={med:7.1f}ms  p90={p90:7.1f}ms  "
          f"peak_alloc={torch.cuda.max_memory_allocated()/2**20:6.0f}MiB  "
          f"peak_reserved={torch.cuda.max_memory_reserved()/2**20:6.0f}MiB  "
          f"out={tuple(out.shape)}", flush=True)
    torch.save({'ms': ms, 'out': out.half().cpu()}, args.out)


if __name__ == '__main__':
    main()
