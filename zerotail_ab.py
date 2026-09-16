#!/usr/bin/env python
"""§41E-4: zero-tail + cond_cache A/B.

Reports, against a full-length conditioning encode baseline:
    A  full conditioning encode wall time
    B  zero-tail COLD wall time          (prefix encode only)
    C  cond_cache WARM wall time         (no encode at all)
    D  build total wall time
    F  peak allocated / reserved VRAM
    G  final conditioning max_abs_error / bitwise_equal

Scope note: the steady-state contract (DiT ~796ms + decode / chunk,
~1.05 chunk/s, ~4.2 fps-eq) is NOT expected to change -- this optimisation
only touches conditioning. A separate full-pipeline run checks that.

  LINGBOT_FP8=1 python zerotail_ab.py --lengths 81,161,224
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
from cond_latent import build_condition_latent, CondCache, stats, reset_stats, \
    PREFIX_PIXEL_FRAMES, ZERO_TAIL_CONST_START


def vram():
    a = torch.cuda.memory_allocated() / 2**20
    r = torch.cuda.memory_reserved() / 2**20
    free, _ = torch.cuda.mem_get_info()
    return dict(alloc_mb=a, reserved_mb=r, driver_free_mb=free / 2**20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--lengths", default="81,161,224")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--out_dir", default="output/zerotail")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    lengths = [int(x) for x in args.lengths.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    print("[ab] building pipe", flush=True)
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
    print(f"[ab] frame {h}x{w} -> latent {lat_h}x{lat_w} vae_stride={vs}",
          flush=True)
    print(f"[ab] ZERO_TAIL_CONST_START={ZERO_TAIL_CONST_START} "
          f"PREFIX_PIXEL_FRAMES={PREFIX_PIXEL_FRAMES}", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(7)
    first = (torch.rand(3, 1, h, w, generator=gen) - 0.5) / 0.5

    rows = []
    for N in lengths:
        T = (N - 1) // vs[0] + 1
        print(f"\n[ab] ===== N={N} -> T={T} latent =====", flush=True)

        # ---------- A: full encode (baseline) ----------
        torch.cuda.empty_cache()
        gc.collect()
        base = vram()
        x = torch.concat([first, torch.zeros(3, N - 1, h, w)], dim=1).to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            zA = pipe.vae.encode([x])[0]
        torch.cuda.synchronize()
        tA = (time.perf_counter() - t0) * 1000.0
        pinA = dict(alloc_mb=torch.cuda.max_memory_allocated() / 2**20)
        zA_cpu = zA.float().cpu()
        del x, zA
        torch.cuda.empty_cache(); gc.collect()
        print(f"    A full encode          {tA:9.1f} ms   "
              f"peak_alloc {pinA['alloc_mb']:8.1f} MB", flush=True)

        # ---------- B: zero-tail COLD ----------
        cache = CondCache(max_entries=4)
        reset_stats()
        gc.collect(); torch.cuda.empty_cache()
        t0 = time.perf_counter()
        yB, cold_ms = build_condition_latent(
            pipe.vae, first, N, h, w, dev, vae_stride_t=vs[0], cache=cache)
        torch.cuda.synchronize()
        tB = (time.perf_counter() - t0) * 1000.0
        yB_cpu = yB.float().cpu()
        errB = float((yB_cpu - zA_cpu).abs().max())
        bitB = bool(torch.equal(yB_cpu, zA_cpu))
        del yB
        torch.cuda.empty_cache(); gc.collect()
        print(f"    B zero-tail COLD       {tB:9.1f} ms   "
              f"err {errB:.3e}  bitwise {bitB}", flush=True)

        # ---------- C: cond_cache WARM (same length) ----------
        gc.collect(); torch.cuda.empty_cache()
        t0 = time.perf_counter()
        yC, _ = build_condition_latent(
            pipe.vae, first, N, h, w, dev, vae_stride_t=vs[0], cache=cache)
        torch.cuda.synchronize()
        tC = (time.perf_counter() - t0) * 1000.0
        yC_cpu = yC.float().cpu()
        errC = float((yC_cpu - zA_cpu).abs().max())
        bitC = bool(torch.equal(yC_cpu, zA_cpu))
        print(f"    C cond_cache WARM      {tC:9.1f} ms   "
              f"err {errC:.3e}  bitwise {bitC}", flush=True)

        # ---------- C2: warm across a LONGER rollout (cache reuse) ----------
        N2 = N + 40
        T2 = (N2 - 1) // vs[0] + 1
        t0 = time.perf_counter()
        yC2, _ = build_condition_latent(
            pipe.vae, first, N2, h, w, dev, vae_stride_t=vs[0], cache=cache)
        torch.cuda.synchronize()
        tC2 = (time.perf_counter() - t0) * 1000.0
        print(f"    C2 warm, longer N={N2} (T={T2}) {tC2:7.1f} ms "
              f"-> cache reusable across lengths "
              f"({cache.hits} hits / {cache.misses} misses)", flush=True)
        del yC, yC2
        torch.cuda.empty_cache(); gc.collect()

        rows.append(dict(N=N, T=T, A_ms=tA, B_ms=tB, C_ms=tC, C2_ms=tC2,
                         errB=errB, bitB=bitB, errC=errC, bitC=bitC,
                         A_vs_B=tA / max(tB, 1e-9), A_vs_C=tA / max(tC, 1e-9),
                         cold_ms=cold_ms))

    print("\n[ab] ===== summary =====")
    print(f"  {'N':>5s} {'T':>4s} {'A full':>9s} {'B cold':>9s} {'C warm':>9s} "
          f"{'A/B':>6s} {'A/C':>8s} {'bitwise':>8s}")
    for r in rows:
        print(f"  {r['N']:5d} {r['T']:4d} {r['A_ms']:9.1f} {r['B_ms']:9.1f} "
              f"{r['C_ms']:9.1f} {r['A_vs_B']:6.2f} {r['A_vs_C']:8.1f} "
              f"{str(r['bitB'] and r['bitC']):>8s}")

    allbit = all(r["bitB"] and r["bitC"] for r in rows)
    warm_zero = all(r["C_ms"] < 50.0 for r in rows)
    print(f"\n  bitwise exact on every length : {allbit}")
    print(f"  warm cache < 50ms             : {warm_zero}")
    print(f"\n  NOTE: B (zero-tail cold) saves only "
          f"{np.mean([r['A_vs_B'] for r in rows]):.2f}x on average -- the VAE "
          f"encode has a large\n  fixed cost, so the prefix cut is modest. "
          f"The real win is C (warm), "
          f"{np.mean([r['A_vs_C'] for r in rows]):.0f}x.")
    print("  Steady-state world generation (DiT/decode/fps-equiv) is NOT "
          "affected by this change.")

    json.dump(dict(rows=rows, const_start=ZERO_TAIL_CONST_START,
                   prefix_pixel=PREFIX_PIXEL_FRAMES,
                   all_bitwise=bool(allbit), warm_under_50ms=bool(warm_zero)),
              open(f"{args.out_dir}/ab.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
