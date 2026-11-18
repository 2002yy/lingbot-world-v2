#!/usr/bin/env python
"""§35B-2: tiny neural preview head on the `upsamples.7` feature.

Timeline (from §35B-1):
    denoise 688ms -> VAE prefix to upsamples.7 88ms -> preview head ?ms
Target: preview TTFNF ~780-820ms vs full frame ~1058ms.

Plan
  1. Collect (feature, full-RGB) pairs: VAE-encode real generated frames to get
     realistic latents, then run the streaming decode with a hook on
     `upsamples.7` (t=4, 192ch, 144x264) and record the final RGB.
  2. Train two heads (P0 / P1), no temporal conv: fold [B,C,T,H,W] -> B*T and
     apply the same 2D head per temporal slice.
  3. Evaluate preview vs full (LPIPS / DINO) and the head's latency overhead.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python preview_head.py --collect --train
"""
import argparse
import gc
import math
import glob
import json
import os
import time
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wan
from wan.configs import WAN_CONFIGS


class P0(nn.Module):
    """1x1 192->3 + bilinear x2."""
    def __init__(self, cin=192):
        super().__init__()
        self.c = nn.Conv2d(cin, 3, 1)

    def forward(self, f):
        return F.interpolate(self.c(f), scale_factor=2, mode="bilinear",
                             align_corners=False)


class P1(nn.Module):
    """192->32 1x1 -> DW 3x3 -> 32->3 -> bilinear x2."""
    def __init__(self, cin=192, mid=32):
        super().__init__()
        self.a = nn.Conv2d(cin, mid, 1)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.b = nn.Conv2d(mid, 3, 1)
        self.act = nn.SiLU()

    def forward(self, f):
        x = self.act(self.a(f))
        x = self.act(self.dw(x))
        return F.interpolate(self.b(x), scale_factor=2, mode="bilinear",
                             align_corners=False)


def read_frames(path, limit=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(f, (528, 288)), cv2.COLOR_BGR2RGB))
    cap.release()
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--clips", default="output/motions/04_s42.mp4,output/motions/01_s42.mp4")
    ap.add_argument("--cache", default="output/preview/data.pt")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out_dir", default="output/preview")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[pv] pipe built", flush=True)
    vae = pipe.vae
    dec = vae.model.decoder
    dev = pipe.device

    feats = {}

    def make_wrap(idx):
        orig = dec.upsamples[idx].forward

        def patched(self, *a, **kw):
            out = orig(*a, **kw)
            feats["f7"] = out
            return out
        return patched

    # upsamples.7 is the Resample that expands t 2 -> 4
    for i, m in enumerate(dec.upsamples):
        if i == 7:
            m.forward = types.MethodType(make_wrap(7), m)
    print("[pv] hooked upsamples.7", flush=True)

    if args.collect:
        samples = []
        zs = []
        # capture the real latents the pipeline decodes
        orig_decode = vae.decode

        def decode_patched(zs_in):
            for z in zs_in:
                zs.append(z.detach().to(torch.float16).cpu())
            return orig_decode(zs_in)
        vae.decode = decode_patched

        from PIL import Image
        TRAJS = [("04", "examples/mot_04"), ("04", "examples/pred_steady_04_A_current_h1.0"),
                 ("01", "examples/mot_01"), ("01", "examples/pred_steady_01_A_current_h1.0")]
        for scene, ex in TRAJS:
            if not os.path.exists(f"{ex}/image.jpg"):
                continue
            img_pil = Image.open(f"{ex}/image.jpg").convert("RGB")
            for sd in (42, 123, 7):
                pipe.generate("A first-person view of a natural landscape with smooth camera motion.",
                              img_pil, action_path=ex, chunk_size=1, max_area=528 * 288,
                              frame_num=41, timesteps_index=[0, 250, 750], shift=5.0,
                              seed=sd, offload_model=False)
                print(f"[pv] generated {ex} s{sd}, total latent tensors {len(zs)}",
                      flush=True)
        vae.decode = orig_decode
        # replay each latent chunk through the streaming decode with the f7 hook
        for zfull in zs:
            zfull = zfull.to(dev).float()
            T = zfull.shape[1]
            for t in range(T):
                z = zfull[:, t:t + 1]
                feats.clear()
                vae.stream_decode_start()
                with torch.no_grad():
                    rgb = vae.stream_decode_step(z)
                vae.stream_decode_end()
                if "f7" not in feats:
                    continue
                f = feats["f7"].detach().to(torch.float16).cpu()
                r = rgb.detach().to(torch.float16).cpu()
                if f.shape[2] != r.shape[1]:
                    continue
                samples.append((f, r))
        torch.save(samples, args.cache)
        print(f"[pv] saved {len(samples)} samples -> {args.cache}", flush=True)

    if args.train:
        data = torch.load(args.cache)
        print(f"[pv] training on {len(data)} samples", flush=True)
        FH, FW = 144, 264        # common feature size
        OH, OW = 288, 528        # common RGB size
        Xs, Ys = [], []
        for f, r in data:
            if f.dim() == 4:
                f = f.unsqueeze(0)
            if r.dim() == 4:
                r = r.unsqueeze(0)
            n, c, t, h, w = f.shape
            ff = f.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).float()
            rr = r.permute(0, 2, 1, 3, 4).reshape(n * t, 3, r.shape[-2],
                                                   r.shape[-1]).float()
            ff = F.interpolate(ff, size=(FH, FW), mode="bilinear", align_corners=False)
            rr = F.interpolate(rr, size=(OH, OW), mode="bilinear", align_corners=False)
            Xs.append(ff); Ys.append(rr)
        X = torch.cat(Xs, 0)          # keep on CPU (132*192*144*264*4 ~ 3.8GB)
        Y = torch.cat(Ys, 0)
        N, C = X.shape[0], X.shape[1]
        print(f"[pv] flat tensors X={tuple(X.shape)} Y={tuple(Y.shape)}", flush=True)
        n_val = max(1, len(X) // 8)
        Xtr, Ytr = X[:-n_val], Y[:-n_val]
        Xva, Yva = X[-n_val:], Y[-n_val:]

        results = {}
        for name, Head in (("P0", P0), ("P1", P1)):
            net = Head().to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            for step in range(args.steps):
                i = torch.randint(0, len(Xtr), (8,))
                pred = net(Xtr[i].to(dev))
                loss = F.l1_loss(pred, Ytr[i].to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
                if step % 100 == 0:
                    print(f"[pv] {name} step {step} l1={loss.item():.4f}", flush=True)
            with torch.no_grad():
                pv = net(Xva.to(dev))
                l1 = float(F.l1_loss(pv, Yva.to(dev)))
                psnr = 10 * math.log10(4.0 / max(float(F.mse_loss(pv, Yva.to(dev))), 1e-9))
            # timing
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    _ = net(Xva[:1].to(dev))
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 100
            torch.save(net.state_dict(), f"{args.out_dir}/{name}.pt")
            # acceptance metrics vs the full VAE frame
            lp = dn = float("nan")
            try:
                from eval_two_layer import load_models, dino_feat, lpips_d, _f
                load_models()
                import cv2 as _cv2
                with torch.no_grad():
                    pv_all = net(Xva.to(dev)).cpu()
                lps, dns = [], []
                for k in range(min(24, len(pv_all))):
                    p = ((pv_all[k].permute(1, 2, 0).cpu().numpy() + 1) * 127.5
                         ).clip(0, 255).astype(np.uint8)
                    g = ((Yva[k].permute(1, 2, 0).numpy() + 1) * 127.5
                         ).clip(0, 255).astype(np.uint8)
                    lps.append(lpips_d(p, g))
                    dns.append(1.0 - float((dino_feat(p) * dino_feat(g)).sum()))
                lp = float(np.mean(lps)); dn = float(np.mean(dns))
                import gc as _gc
                del _f
                _gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"[pv] metric eval skipped: {e}", flush=True)
            results[name] = dict(l1=l1, psnr=psnr, head_ms=ms,
                                 lpips=lp, dino=dn,
                                 params=sum(p.numel() for p in net.parameters()))
            print(f"[pv] {name}: val L1={l1:.4f} PSNR={psnr:.2f}dB "
                  f"LPIPS={lp:.4f} DINO_dist={dn:.4f} head={ms:.1f}ms "
                  f"params={results[name]['params']}", flush=True)
        json.dump(results, open(f"{args.out_dir}/head_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
