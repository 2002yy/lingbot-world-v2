#!/usr/bin/env python
"""VAE tiled-decode bench: peak VRAM / latency / fidelity vs tile size.

Verifies that spatial tiling does not change the output (MAE/PSNR vs the
untiled reference) while lowering the decode peak enough to overlap with DiT.

  python vae_tile_bench.py --chunks 6
"""
import argparse
import os
import time

import numpy as np
import torch

from wan.modules.vae2_1 import Wan2_1_VAE


def peak_mib():
    return torch.cuda.max_memory_allocated() / 2**20, \
           torch.cuda.max_memory_reserved() / 2**20


def run(vae, latents, tile=None):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated() / 2**20
    if tile is None:
        vae.stream_decode_start()
    else:
        vae.stream_decode_start_tiled(*tile)
    outs, ms = [], []
    for z in latents:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs.append(vae.stream_decode_step_tiled(z) if tile is not None
                    else vae.stream_decode_step(z))
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000.0)
    vae.stream_decode_end()
    alloc, reserved = peak_mib()
    return torch.cat(outs, dim=1), np.array(ms), alloc - base, reserved - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_pth", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets/Wan2.1_VAE.pth"))
    ap.add_argument("--lat_h", type=int, default=36)
    ap.add_argument("--lat_w", type=int, default=66)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out_dir", default="output/vae_tile")
    args = ap.parse_args()

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = Wan2_1_VAE(vae_pth=args.vae_pth, dtype=dt, device="cuda")
    vae.model.eval()
    print(f"[vt] VAE loaded dtype={args.dtype}", flush=True)

    g = torch.Generator(device="cuda").manual_seed(0)
    latents = [torch.randn(16, 1, args.lat_h, args.lat_w, generator=g,
                           device="cuda", dtype=torch.float32)
               for _ in range(args.chunks)]

    ref, ref_ms, ref_a, ref_r = run(vae, latents, tile=None)
    print(f"[vt] full      dec={ref_ms[1:].mean():.0f}ms "
          f"alloc+={ref_a:.0f}MiB reserved+={ref_r:.0f}MiB", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    cfgs = [
        ("large", (args.lat_h, 34, 6)),
        ("medium", (20, 34, 6)),
        ("small", (20, 18, 6)),
        ("tiny", (12, 18, 6)),
    ]
    rows = [("full", None, ref_ms, ref_a, ref_r, 0.0, float("inf"))]
    for name, tile in cfgs:
        out, ms, a, r = run(vae, latents, tile=tile)
        d = (out - ref).abs()
        mae = float(d.mean())
        mse = float((d ** 2).mean())
        psnr = 10 * np.log10(4.0 / max(mse, 1e-12))  # range [-1,1] -> 4
        rows.append((name, tile, ms, a, r, mae, psnr))
        print(f"[vt] {name:8s} tiles={len(vae.model._tile_grid)} "
              f"dec={ms[1:].mean():.0f}ms alloc+={a:.0f}MiB reserved+={r:.0f}MiB "
              f"MAE={mae:.5f} PSNR={psnr:.1f}dB", flush=True)
        # seam check: max |diff| along tile boundaries vs global max
        last = d[-1][0]
        np.save(f"{args.out_dir}/{name}_diff.npy", last.mean(0).cpu().numpy())

    with open(f"{args.out_dir}/summary.txt", "w") as fh:
        fh.write("cfg tiles dec_ms alloc_MiB reserved_MiB MAE PSNR_dB\n")
        for r in rows:
            tiles = 1 if r[1] is None else len(vae.model._tile_grid)
            fh.write(f"{r[0]} {tiles} {r[2][1:].mean():.0f} {r[3]:.0f} "
                     f"{r[4]:.0f} {r[5]:.5f} {r[6]:.1f}\n")
    print("[vt] ->", f"{args.out_dir}/summary.txt")


if __name__ == "__main__":
    main()
