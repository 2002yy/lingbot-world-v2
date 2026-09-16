#!/usr/bin/env python
"""§41E-1: prefix invariance gate.

Before touching build_y we must prove the safe structure:

    pixel:  [first_frame, 0, 0, ...]
                    |
            encode only enough pixel frames to yield >= 30 latent frames
                    |
            z[0..28]  +  c = z[29]
                    |
            y = cat(z[0..28], expand(c, target_T - 29))

The 179.8x ratio measured in zerotail_probe.py is NOT a production speedup,
because production still needs z[0..28]. What must hold is:

    encode(N=117)[:, :29]  ==  encode(N=161)[:, :29]  ==  encode(N=224)[:, :29]
    bitwise

and for each N:

    z[29] == z[30] == ... == z[T-1]      (the constant tail)

`ZERO_TAIL_CONST_START = 29` is a property of THIS VAE checkpoint, verified
here, not a universal law -- it is recorded as such.

  LINGBOT_FP8=1 python zerotail_prefix_gate.py
"""
import argparse
import gc
import json
import math
import os
import time

import numpy as np
import torch

import wan
from wan.configs import WAN_CONFIGS

CONST_START = 29          # first latent frame that is constant (to be verified)
PREFIX_PIXEL = 117        # ceil(117/4) = 30 latent frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--lengths", default="117,161,224")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--out_dir", default="output/zerotail")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    lengths = [int(x) for x in args.lengths.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    print("[pg] building pipe (VAE only path)", flush=True)
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev = pipe.device
    vs = pipe.vae_stride
    aspect = 480 / 832
    lat_h = round(math.sqrt(W * H * aspect) // vs[1] // 8 * 8)
    lat_w = round(math.sqrt(W * H / aspect) // vs[2] // 8 * 8)
    h = lat_h * vs[1]
    w = lat_w * vs[2]
    print(f"[pg] frame {h}x{w} -> latent {lat_h}x{lat_w}, vae_stride={vs}",
          flush=True)

    # one deterministic first frame, shared by every length
    gen = torch.Generator(device="cpu").manual_seed(1234)
    first = (torch.rand(3, 1, h, w, generator=gen) - 0.5) / 0.5

    zs, times = {}, {}
    for N in lengths:
        x = torch.concat([first, torch.zeros(3, N - 1, h, w)], dim=1).to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            z = pipe.vae.encode([x])[0]
        torch.cuda.synchronize()
        times[N] = (time.perf_counter() - t0) * 1000.0
        zs[N] = z.float().cpu()
        print(f"[pg] N={N:4d} -> T={z.shape[1]:3d} latent, "
              f"{times[N]:8.1f} ms", flush=True)
        del x, z
        gc.collect(); torch.cuda.empty_cache()

    print("\n[pg] ===== GATE A: prefix invariance across lengths =====")
    ok_a = True
    ref_N = lengths[-1]
    for N in lengths[:-1]:
        T = min(zs[N].shape[1], zs[ref_N].shape[1], CONST_START)
        a = zs[N][:, :T].numpy()
        b = zs[ref_N][:, :T].numpy()
        max_err = float(np.abs(a - b).max())
        bitwise = bool(np.array_equal(a, b))
        ok_a &= bitwise
        print(f"    z[:{T}]  N={N} vs N={ref_N}: max_abs={max_err:.3e}  "
              f"bitwise_equal={bitwise}", flush=True)

    print("\n[pg] ===== GATE B: tail constant from latent "
          f"{CONST_START} =====")
    ok_b = True
    for N in lengths:
        T = zs[N].shape[1]
        c = zs[N][:, CONST_START:CONST_START + 1]
        tail = zs[N][:, CONST_START:]
        d = (tail - c).abs().max().item()
        bit = bool(torch.equal(tail, c.expand_as(tail)))
        ok_b &= bit
        print(f"    N={N:4d} T={T:3d}: z[{CONST_START}:] vs z[{CONST_START}] "
              f"max_abs={d:.3e} bitwise_equal={bit}", flush=True)

    print("\n[pg] ===== GATE C: reconstruction from prefix only =====")
    ok_c = True
    for N in lengths:
        T = zs[N].shape[1]
        if T <= CONST_START:
            print(f"    N={N}: T={T} <= {CONST_START}, prefix path not applicable")
            continue
        # encode only the prefix length
        xp = torch.concat([first, torch.zeros(3, PREFIX_PIXEL - 1, h, w)],
                          dim=1).to(dev)
        with torch.no_grad():
            zp = pipe.vae.encode([xp])[0].float().cpu()
        del xp
        gc.collect(); torch.cuda.empty_cache()
        assert zp.shape[1] >= CONST_START + 1, \
            f"prefix encode yielded T={zp.shape[1]}, need >= {CONST_START+1}"
        prefix = zp[:, :CONST_START]
        c = zp[:, CONST_START:CONST_START + 1]
        y = torch.concat([prefix, c.expand(-1, T - CONST_START, -1, -1)], dim=1)
        ref = zs[N]
        max_err = float((y - ref).abs().max())
        bit = bool(torch.equal(y, ref))
        ok_c &= bit
        print(f"    N={N:4d} T={T:3d}: prefix({PREFIX_PIXEL}px -> "
              f"{zp.shape[1]} lat) rebuild max_abs={max_err:.3e} "
              f"bitwise_equal={bit}", flush=True)

    print("\n[pg] ===== latency (context, not the production number) =====")
    print(f"    {'N':>5s} {'full ms':>9s}")
    for N in lengths:
        print(f"    {N:5d} {times[N]:9.1f}")
    print("\n    NOTE: the headline 'N=224 vs single-frame = 179.8x' from "
          "zerotail_probe.py\n    is NOT a production speedup: production still "
          f"needs z[0..{CONST_START-1}].\n    Honest expectation = the prefix "
          "encode cost, i.e. roughly the N~117 number.")

    verdict = ok_a and ok_b and ok_c
    print(f"\n[pg] §41E-1 GATE: {'PASS' if verdict else 'FAIL'}")
    if verdict:
        print(f"    ZERO_TAIL_CONST_START = {CONST_START}")
        print(f"    PREFIX_PIXEL_FRAMES   = {PREFIX_PIXEL}")
        print("    (recorded as a property of THIS VAE checkpoint, "
              "not a universal law)")
    json.dump(dict(const_start=CONST_START, prefix_pixel=PREFIX_PIXEL,
                   times_ms={str(k): v for k, v in times.items()},
                   gate_a=bool(ok_a), gate_b=bool(ok_b), gate_c=bool(ok_c),
                   pass_=bool(verdict)),
              open(f"{args.out_dir}/prefix_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
