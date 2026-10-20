#!/usr/bin/env python
"""Zero-tail probe: does the conditioning VAE encode of a zero tail reach a
fixed point, so the tail can be broadcast instead of encoded?

Claim under test (from the LingBot-World V2 MPS port):
    after latent frame ~25, the BF16 output of the zero-padded tail becomes
    constant, so the remaining latent frames can be broadcast without running
    the VAE. They report byte-identical final video and ~3.0x on conditioning
    encode for a 361-frame config.

Our conditioning input is  [first_frame, zeros(3, N-1)]  -- the tail is 100%
zeros from frame 1, which is MORE extreme than their case. So if the claim
holds at all, it should hold for us, and possibly from much earlier.

This probe measures, for the REAL VAE used in our pipeline:
    1. encode latency vs sequence length (does it scale with N?)
    2. whether the tail latent frames are (near) constant
    3. from which latent frame the tail can be broadcast within a tolerance
    4. the error introduced by broadcasting vs a full encode

Loads ONLY the VAE (no DiT), so it is cheap and does not need the full
pipeline in memory.

  LINGBOT_FP8=1 python zerotail_probe.py --frames 224 --area 512x320
"""
import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import torch

import wan
from wan.configs import WAN_CONFIGS


def stats(x):
    return dict(mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), max=float(x.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--frames", default="41,81,161,224")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    lengths = [int(x) for x in args.frames.split(",")]

    print("[zt] building pipe (need only the VAE)", flush=True)
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev = pipe.device
    vs = pipe.vae_stride
    print(f"[zt] vae_stride={vs}", flush=True)

    # a plausible first frame (the model's own conditioning source)
    gen = torch.Generator(device="cpu").manual_seed(0)
    aspect = 480 / 832
    lat_h = round(math.sqrt(W * H * aspect) // vs[1] // 8 * 8)
    lat_w = round(math.sqrt(W * H / aspect) // vs[2] // 8 * 8)
    h = lat_h * vs[1]
    w = lat_w * vs[2]
    print(f"[zt] target frame {h}x{w} -> latent {lat_h}x{lat_w} "
          f"(1 latent frame = {vs[0]} pixel frames)", flush=True)

    results = {}
    for N in lengths:
        # ---- full-length encode: [first frame, zeros...] ----
        first = (torch.rand(3, 1, h, w, generator=gen) - 0.5) / 0.5
        tail = torch.zeros(3, N - 1, h, w)
        x = torch.concat([first, tail], dim=1).to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            z = pipe.vae.encode([x])[0]
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1000.0
        z = z.float()
        T = z.shape[1]

        # ---- per-latent-frame deviation from the LAST frame ----
        last = z[:, -1:, :, :]
        d = (z - last).abs().amax(dim=(0, 2, 3)).cpu().numpy()   # [T]
        # ---- is the tail constant? first latent frame where dev < tol ----
        hold_from = None
        for i in range(1, T):
            if np.all(d[i:] < args.tol):
                hold_from = i
                break

        # ---- single-frame encode for comparison ----
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            z1 = pipe.vae.encode([first[:, :1].contiguous().to(dev)])[0]
        torch.cuda.synchronize()
        one_ms = (time.perf_counter() - t1) * 1000.0
        z1 = z1.float()

        # ---- error if we broadcast from `hold_from` ----
        if hold_from is not None:
            zb = z.clone()
            zb[:, hold_from:] = z[:, hold_from:hold_from + 1]
            err = (z - zb).abs()
            max_err = float(err.max())
            mean_err = float(err.mean())
            # does the broadcast version match the first latent exactly?
            first_ok = float((z[:, :1] - zb[:, :1]).abs().max())
        else:
            max_err = mean_err = first_ok = float("nan")

        results[N] = dict(T=T, full_ms=full_ms, one_ms=one_ms, hold_from=hold_from,
                          max_err=max_err, mean_err=mean_err, first_ok=first_ok,
                          d=d.tolist())
        print(f"\n[zt] N={N} pixel-frames -> T={T} latent frames", flush=True)
        print(f"     full encode {full_ms:8.1f} ms   single-frame encode "
              f"{one_ms:6.1f} ms   ratio {full_ms/max(one_ms,1e-9):6.2f}x", flush=True)
        print(f"     tail becomes constant from latent frame "
              f"{hold_from if hold_from is not None else 'never'} "
              f"(tol {args.tol})", flush=True)
        print(f"     broadcast error: max {max_err:.3e} mean {mean_err:.3e} "
              f"| frame0 preserved: {first_ok:.3e}", flush=True)
        print(f"     deviation from last frame (first 12): "
              f"{['%.2e' % v for v in d[:12]]}", flush=True)

    print("\n[zt] ===== verdict =====")
    for N, r in results.items():
        if r["hold_from"] is not None:
            # latency if we encode only up to hold_from and broadcast the rest
            keep = r["hold_from"] + 1
            frac = keep / max(r["T"], 1)
            est = r["one_ms"] + (r["full_ms"] - r["one_ms"]) * frac
            print(f"  N={N}: broadcast from latent {r['hold_from']}/{r['T']} "
                  f"-> est {est:.1f} ms vs full {r['full_ms']:.1f} ms "
                  f"({r['full_ms']/max(est,1e-9):.2f}x), "
                  f"max_err {r['max_err']:.2e}")
        else:
            print(f"  N={N}: tail NEVER becomes constant within tol "
                  f"{args.tol} -> broadcast NOT valid here")
    json_out = {str(k): {kk: vv for kk, vv in v.items() if kk != "d"}
                for k, v in results.items()}
    import json
    os.makedirs("output/zerotail", exist_ok=True)
    json.dump(json_out, open("output/zerotail/zerotail.json", "w"), indent=1)


if __name__ == "__main__":
    main()
