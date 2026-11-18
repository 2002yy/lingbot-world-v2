#!/usr/bin/env python
"""§35C: does the neural preview actually improve the final DISPLAY?

Three display paths over the same generated clip:
    A  warp -> full
    B  warp -> preview -> full
    C  warp -> preview -> full + adaptive blending

Preview timing inside a chunk (from §35A/§35B):
    t=0      denoise starts (warp keeps extrapolating from the previous ref)
    t=776ms  preview available  (VAE prefix 88ms + P1 head 0.23ms after denoise 688ms)
    t=1058ms full frame available
The preview becomes an intermediate reference, so the hard end-of-chunk
extrapolation only has to survive ~280ms instead of ~1058ms.

Async copy: the preview tensor is copied to a pinned host buffer on a side
stream and consumed via a CUDA event, so the main VAE stream is never blocked
by a synchronizing `.cpu()`.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python display_preview_ab.py
"""
import argparse
import gc
import hashlib
import json
import math
import os
import time
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from eval_two_layer import load_models, lpips_d
from pose_warp import PoseWarper
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses
from preview_head import P1

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
FPS = 60.0
CADENCE = 1.25
D = int(FPS * CADENCE)
T_PREVIEW = 0.776
T_FULL = 1.058
BLEND = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--traj", default="examples/mot_04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_chunks", type=int, default=8)
    ap.add_argument("--head", default="output/preview/P1.pt")
    ap.add_argument("--out_dir", default="output/display_ab")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[ab] pipe built", flush=True)
    vae = pipe.vae
    dev = pipe.device
    dec = vae.model.decoder

    # hook upsamples.7 to grab the preview feature
    feats = {}

    def make_wrap(idx):
        orig = dec.upsamples[idx].forward

        def patched(self, *a, **kw):
            out = orig(*a, **kw)
            feats["f7"] = out
            return out
        return patched

    for i, m in enumerate(dec.upsamples):
        if i == 7:
            m.forward = types.MethodType(make_wrap(7), m)

    # capture the latents the pipeline decodes
    latents = []
    orig_decode = vae.decode

    def decode_patched(zs_in):
        for z in zs_in:
            latents.append(z.detach().clone())
        return orig_decode(zs_in)
    vae.decode = decode_patched

    img_pil = Image.open(f"{args.traj}/image.jpg").convert("RGB")
    t0 = time.perf_counter()
    video = pipe.generate(PROMPT, img_pil, action_path=args.traj, chunk_size=1,
                          max_area=528 * 288, frame_num=args.n_chunks * 4 + 1,
                          timesteps_index=[0, 250, 750], shift=5.0,
                          seed=args.seed, offload_model=False)
    vae.decode = orig_decode
    # the decode receives the whole latent sequence at once -> split per chunk
    lats = []
    for z in latents:
        for t in range(z.shape[1]):
            lats.append(z[:, t:t + 1].detach().clone())
    latents = lats
    print(f"[ab] generated in {time.perf_counter()-t0:.0f}s, "
          f"latent chunks={len(latents)}", flush=True)
    full = ((video.permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5
            ).clip(0, 255).astype(np.uint8)      # [N,H,W,C] RGB
    H, W = full.shape[1:3]

    # ---- previews via VAE prefix + P1 head ----
    net = P1()
    net.load_state_dict(torch.load(args.head, map_location="cpu"))
    net = net.to(dev).eval()
    load_models()
    previews = []
    stream = torch.cuda.Stream()
    pinned = None
    ev = torch.cuda.Event()
    for z in latents:
        feats.clear()
        vae.stream_decode_start()
        with torch.no_grad():
            _ = vae.stream_decode_step(z[0])
        vae.stream_decode_end()
        if "f7" not in feats:
            previews.append(None)
            continue
        f7 = feats["f7"]
        # fold temporal and run the head on the main stream (0.23ms)
        b, c, t, h, w = f7.shape
        ff = f7.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
        with torch.no_grad():
            pv = net(ff)[:1]                     # first temporal slice
        # async copy to pinned buffer (no sync on the main stream)
        if pinned is None:
            pinned = torch.empty_like(pv[0], dtype=torch.float32,
                                      device="cpu", pin_memory=True)
        with torch.cuda.stream(stream):
            pinned.copy_(pv[0], non_blocking=True)
            ev.record(stream)
        ev.synchronize()
        arr = (pinned.permute(1, 2, 0).numpy() + 1) * 127.5
        pv_img = cv2.resize(arr.clip(0, 255).astype(np.uint8), (W, H))
        previews.append(pv_img)
    print(f"[ab] previews: {sum(p is not None for p in previews)}/{len(latents)}",
          flush=True)
    pv_gap = []
    for k, p in enumerate(previews):
        if p is None:
            continue
        g = real_frame(k)
        pv_gap.append(lpips_d(cv2.cvtColor(p, cv2.COLOR_BGR2RGB),
                              cv2.cvtColor(g, cv2.COLOR_BGR2RGB)))
    print(f"[ab] intrinsic preview<->full LPIPS: mean={np.mean(pv_gap):.4f} "
          f"max={np.max(pv_gap):.4f}  (this is the floor the two-step path "
          f"must pay twice)", flush=True)

    # ---- pose timeline at display rate ----
    P = np.load(f"{args.traj}/poses.npy")
    n_lat = min(len(latents), len(previews), args.n_chunks)
    total = n_lat * D
    ctrl = interpolate_camera_poses(
        np.linspace(0, len(P) - 1, len(P)),
        torch.from_numpy(P[:, :3, :3]).float(),
        torch.from_numpy(P[:, :3, 3]).float(),
        np.linspace(0, len(P) - 1, total)).numpy()
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.traj}/intrinsics.npy")).float(),
        480, 832, H, W, H, W)[0].numpy()

    def real_frame(chunk):
        k = min(4 * chunk, len(full) - 1)
        return full[k]

    def simulate(mode):
        """mode in {'A','B','C'}."""
        warper = PoseWarper(Ks, H, W, nominal_depth=4.0)
        out_frames, rows = [], []
        prev = None
        blend_left = 0
        blend_src = None
        for f in range(total):
            tic = (f % D) / FPS
            chunk = f // D
            # choose the current reference for this moment
            if mode == "A":
                ref = max(0, chunk - 1) if tic < T_FULL else chunk
                ref_img, ref_pose = real_frame(ref), ctrl[min(ref * D, total - 1)]
            else:
                if tic < T_PREVIEW:
                    ref = max(0, chunk - 1)
                    ref_img, ref_pose = real_frame(ref), ctrl[min(ref * D, total - 1)]
                elif tic < T_FULL and previews[chunk] is not None:
                    ref = chunk
                    ref_img, ref_pose = previews[chunk], ctrl[min(chunk * D, total - 1)]
                else:
                    ref = chunk
                    ref_img, ref_pose = real_frame(chunk), ctrl[min(chunk * D, total - 1)]
            # blend when the reference changes
            key = (ref, mode, 0 if tic < T_PREVIEW else (1 if tic < T_FULL else 2))
            if prev is not None and key != prev:
                blend_left = BLEND
                blend_src = out_frames[-1].copy()
            prev = key
            warper.set_reference(ref_img, ref_pose)
            out = warper.warp(ctrl[f])
            if blend_left > 0 and blend_src is not None:
                a = 1.0 - blend_left / (BLEND + 1.0)
                out = cv2.addWeighted(blend_src, 1.0 - a, out, a, 0.0)
                blend_left -= 1
            out_frames.append(out)
            rows.append(dict(f=f, tic=tic, ref=ref))
        return out_frames, rows

    results = {}
    for mode in ("A", "B", "C"):
        frames, rows = simulate(mode)
        jumps = []
        for k in range(1, len(frames)):
            jumps.append(lpips_d(cv2.cvtColor(frames[k - 1], cv2.COLOR_BGR2RGB),
                                 cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB)))
        jumps = np.array(jumps)
        # per-refresh transitions: measure at the reference-change boundaries
        peak = float(jumps.max())
        results[mode] = dict(mean_jump=float(jumps.mean()), peak_jump=peak,
                             total_energy=float(jumps.sum()))
        print(f"[ab] {mode}: mean_jump={jumps.mean():.4f} peak={peak:.4f} "
              f"total={jumps.sum():.1f}", flush=True)
        vw = cv2.VideoWriter(f"{args.out_dir}/{mode}.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for fr in frames:
            vw.write(fr)
        vw.release()

    print("\n[ab] ===== display A/B/C =====")
    a = results["A"]
    for mode in ("A", "B", "C"):
        r = results[mode]
        print(f"  {mode}: mean={r['mean_jump']:.4f} peak={r['peak_jump']:.4f} "
              f"total={r['total_energy']:.1f}  "
              f"(peak {100*(r['peak_jump']/a['peak_jump']-1):+.1f}%, "
              f"total {100*(r['total_energy']/a['total_energy']-1):+.1f}%)")
    json.dump(results, open(f"{args.out_dir}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
