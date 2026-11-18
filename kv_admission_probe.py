#!/usr/bin/env python
"""§34 KV-update camera-admission audit.

Part 1 (code): the KV-update forward runs a full forward with the camera
condition injected but DISCARDS the return value; its only side effect is
writing K/V into self_kv_cache. So the camera condition at this stage should
affect the NEXT chunk's cache, not the current chunk's latent. This script
verifies that empirically.

Part 2 (probe): same seed / same noise, switch the condition A->B at
    - before denoise step 2   (known from §32: effect_current ~0.69)
    - before the KV update
    - both
and measure TWO observation targets:
    effect_current : distance of chunk 0's final latent to the A/B anchors
    effect_next    : distance of chunk 1's final latent (condition fixed to A
                     for chunk 1) to the A/B anchors

Part 3: closed-window occupancy for the different admission windows.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python kv_admission_probe.py --base examples/04
"""
import argparse
import gc
import hashlib
import math
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yaw_deg", type=float, default=25.0)
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[kv] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch[2] * patch[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = 9
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * pipe.local_attn_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads

    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                        mode='bicubic').transpose(0, 1),
        torch.zeros(3, F - 1, h, w)], dim=1).to(dev)])[0]
    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y])
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, 3, lat_h, lat_w, generator=g, device=dev)

    def plucker(angle_deg):
        th = math.radians(angle_deg)
        R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0],
                      [-math.sin(th), 0, math.cos(th)]])
        rel = torch.eye(4)[None].to(dev)
        rel[0, :3, :3] = torch.from_numpy(R).float().to(dev)
        p = get_plucker_embeddings(rel, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                         h=lat_h, w=lat_w).to(pdt)

    P_A = plucker(0.0)
    P_B = plucker(args.yaw_deg)

    def fresh_kv():
        sk = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
        ck = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
        return sk, ck

    def run(swap_step, swap_kv, cond1=P_A):
        """swap_step: denoise steps < swap_step use A. swap_kv: KV update uses B."""
        g.manual_seed(args.seed)
        sk, ck = fresh_kv()
        pipe._cross_attn_initialized = False
        cur = noise.split(1, dim=1)[0].clone()
        for ti in range(len(timesteps)):
            plk = P_A if ti < swap_step else P_B
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[0]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": sk, "crossattn_cache": ck, "current_start": 0,
                  "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=not pipe._cross_attn_initialized, **kw)[0]
                pipe._cross_attn_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=g, device=x0.device,
                                        dtype=x0.dtype), timesteps[ti + 1])
        # KV-update forward (return value discarded; writes sk/ck)
        plk_kv = P_B if swap_kv else P_A
        kw_kv = dict(kw)
        kw_kv["dit_cond_dict"] = {"c2ws_plucker_emb": plk_kv.chunk(1, dim=0)}
        kw_kv["current_start"] = 0
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw_kv)
        x0_cur = x0.clone()
        # ---- chunk 1 with a FIXED condition, observing the cache effect ----
        cur1 = noise.split(1, dim=1)[1].clone()
        for ti in range(len(timesteps)):
            kw1 = dict(kw)
            kw1["dit_cond_dict"] = {"c2ws_plucker_emb": cond1.chunk(1, dim=0)}
            kw1["current_start"] = frame_seqlen
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur1.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=not pipe._cross_attn_initialized, **kw1)[0]
                pipe._cross_attn_initialized = True
                x0n = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur1, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    cur1 = pipe.scheduler.add_noise(
                        x0n, torch.randn(x0n.shape, generator=g, device=x0n.device,
                                         dtype=x0n.dtype), timesteps[ti + 1])
        return x0_cur, x0n

    print("\n[kv] ===== Part 2: perturbation with current/next observation =====")
    aA_cur, aA_next = run(3, False)
    aB_cur, aB_next = run(0, True)
    d_cur = float((aA_cur - aB_cur).abs().mean())
    d_next = float((aA_next - aB_next).abs().mean())
    print(f"[kv] anchor distances: |A-B| current={d_cur:.5f}  next={d_next:.5f}")

    cases = [("A->B before step2", 2, False), ("A->B before KV", 3, True),
             ("A->B before step2 + KV", 2, True), ("all A", 3, False),
             ("all B", 0, True)]
    print(f"\n{'case':24s} {'eff_current':>12s} {'eff_next':>10s}")
    for name, ss, sk in cases:
        xc, xn = run(ss, sk)
        dcA = float((xc - aA_cur).abs().mean())
        dcB = float((xc - aB_cur).abs().mean())
        dnA = float((xn - aA_next).abs().mean())
        dnB = float((xn - aB_next).abs().mean())
        eff_c = dcA / max(dcA + dcB, 1e-9)
        eff_n = dnA / max(dnA + dnB, 1e-9)
        print(f"{name:24s} {eff_c:12.3f} {eff_n:10.3f}")

    # ---- Part 3: closed-window occupancy ----
    print("\n[kv] ===== Part 3: closed-window occupancy =====")
    denoise, kv, dec = 688.0, 229.0, 370.0
    total = denoise + kv + dec
    # Part 2 shows the KV-update camera condition has ZERO effect on the CURRENT
    # chunk (its latent is already fixed) and only writes the future cache, so a
    # KV snapshot does NOT extend the current chunk's admission window.
    print("  [current-chunk influence]")
    for label, closed in (("chunk-start only (baseline)", total),
                          ("+ step hot-swap", kv + dec),
                          ("+ step + KV hot-swap", kv + dec)):
        print(f"    {label:30s} closed={closed:6.0f}ms  "
              f"occupancy={100*closed/total:5.1f}%")
    print("  [next-chunk state freshness]  (KV snapshot DOES help here)")
    for label, closed in (("chunk-start only (baseline)", kv + dec),
                          ("+ step hot-swap", kv + dec),
                          ("+ step + KV hot-swap", dec)):
        print(f"    {label:30s} closed={closed:6.0f}ms  "
              f"occupancy={100*closed/total:5.1f}%")
    print(f"  (chunk period = denoise {denoise:.0f} + KV {kv:.0f} + decode {dec:.0f}"
          f" = {total:.0f}ms)")


if __name__ == "__main__":
    main()
