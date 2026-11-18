#!/usr/bin/env python
"""§35D: TAEW2.1 drop-in audit against the full Wan 2.1 VAE.

TAEHV's decode_video expects an NTCHW latent with ~Gaussian values (i.e. exactly
the DiT latent, no mean/std un-normalisation) and returns NTCHW RGB in [0,1].

Compares, on the same LingBot DiT latents:
    full Wan VAE decode   vs   TAEW2.1 decode
on latency, peak VRAM, LPIPS, DINO distance and temporal consistency.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python tae_audit.py
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from eval_two_layer import load_models, dino_feat, lpips_d

sys.path.insert(0, "/tmp/taehv")
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default="/tmp/taehv/taew2_1.pth")
    ap.add_argument("--trajs", default="examples/mot_04,examples/mot_01")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--n_chunks", type=int, default=8)
    ap.add_argument("--out_dir", default="output/tae")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[ta] pipe built", flush=True)
    dev = pipe.device

    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    n_params = sum(p.numel() for p in tae.decoder.parameters())
    print(f"[ta] TAEW2.1 loaded, decoder params={n_params/1e6:.2f}M "
          f"(arch={tae.arch_name})", flush=True)

    load_models_flag = False   # metrics are loaded AFTER all generation

    latents = []
    orig_decode = pipe.vae.decode

    def decode_patched(zs_in):
        for z in zs_in:
            latents.append(z.detach().clone())
        return orig_decode(zs_in)
    pipe.vae.decode = decode_patched

    rows = []
    pairs = []          # (scene, seed, full_np, tiny_np, meta)
    for traj in args.trajs.split(","):
        if not os.path.exists(f"{traj}/image.jpg"):
            continue
        scene = os.path.basename(traj)
        img_pil = Image.open(f"{traj}/image.jpg").convert("RGB")
        for sd in [int(x) for x in args.seeds.split(",")]:
            latents.clear()
            video = pipe.generate(PROMPT, img_pil, action_path=traj, chunk_size=1,
                                  max_area=528 * 288,
                                  frame_num=args.n_chunks * 4 + 1,
                                  timesteps_index=[0, 250, 750], shift=5.0,
                                  seed=sd, offload_model=False)
            if not latents:
                continue
            z = latents[0].to(dev).float()          # [C,T,H,W]
            C, T, H, W = z.shape
            print(f"[ta] {scene} s{sd}: latent {tuple(z.shape)}", flush=True)

            # ---- full Wan VAE ----
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                full = pipe.vae.decode([z])[0]
            torch.cuda.synchronize()
            t_full = time.perf_counter() - t0
            peak_full = torch.cuda.max_memory_allocated() / 2**20
            full_np = ((full.permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5
                       ).clip(0, 255).astype(np.uint8)

            # ---- TAEW2.1 ----
            zt = z.permute(1, 0, 2, 3).unsqueeze(0)          # [1,T,C,H,W]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                tiny = tae.decode_video(zt, parallel=False,
                                        show_progress_bar=False)
            torch.cuda.synchronize()
            t_tiny = time.perf_counter() - t0
            peak_tiny = torch.cuda.max_memory_allocated() / 2**20
            tiny = tiny[0].permute(0, 2, 3, 1)               # [T,H,W,C]
            tiny_np = (tiny.float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            n = min(len(full_np), len(tiny_np))
            Hf, Wf = full_np.shape[1:3]
            pairs.append((scene, sd, full_np, tiny_np, Hf, Wf,
                          t_full * 1000, t_tiny * 1000, peak_full, peak_tiny))
            print(f"[ta] {scene} s{sd}: full={t_full*1000:.0f}ms "
                  f"tiny={t_tiny*1000:.0f}ms (metrics deferred)", flush=True)
    pipe.vae.decode = orig_decode

    # ---- metrics after all generation (frees the DiT path first) ----
    del pipe
    gc.collect(); torch.cuda.empty_cache()
    load_models()

    for scene, sd, full_np, tiny_np, Hf, Wf, t_full, t_tiny, pk_f, pk_t in pairs:
        n = min(len(full_np), len(tiny_np))
        lps, dns = [], []
        for k in range(min(16, n)):
            a = cv2.resize(tiny_np[k], (Wf, Hf))
            lps.append(lpips_d(a, full_np[k]))
            dns.append(1.0 - float((dino_feat(a) * dino_feat(full_np[k])).sum()))

        def flicker(seq):
            g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq[:n]]
            m = []
            for i in range(1, len(g)):
                fl = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None,
                                                  0.5, 3, 15, 3, 5, 1.2, 0)
                m.append(float(np.mean(np.linalg.norm(fl, axis=2))))
            return float(np.mean(m)), float(np.std(m))
        fm, fs = flicker(full_np)
        tm, ts = flicker([cv2.resize(f, (Wf, Hf)) for f in tiny_np])
        rows.append(dict(scene=scene, seed=sd,
                         t_full_ms=t_full, t_tiny_ms=t_tiny,
                         peak_full=pk_f, peak_tiny=pk_t,
                         lpips=float(np.mean(lps)), dino=float(np.mean(dns)),
                         full_flow_mean=fm, full_flow_std=fs,
                         tiny_flow_mean=tm, tiny_flow_std=ts))
        print(f"[ta] {scene} s{sd}: LPIPS={np.mean(lps):.4f} "
              f"DINO={np.mean(dns):.4f}", flush=True)

    # timing table from the earlier pass
    timings = []
    for traj in args.trajs.split(","):
        pass
    print("\n[ta] ===== TAEW2.1 drop-in summary =====")
    print(f"{'scene':14s} {'seed':>5s} {'LPIPS':>7s} {'DINO':>7s} "
          f"{'full_flow':>10s} {'tiny_flow':>10s}")
    for r in rows:
        print(f"{r['scene']:14s} {r['seed']:5d} {r['lpips']:7.4f} {r['dino']:7.4f} "
              f"{r['full_flow_mean']:10.3f} {r['tiny_flow_mean']:10.3f}")
    if rows:
        print(f"\n  mean LPIPS={np.mean([r['lpips'] for r in rows]):.4f}  "
              f"DINO={np.mean([r['dino'] for r in rows]):.4f}")
    json.dump(rows, open(f"{args.out_dir}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

