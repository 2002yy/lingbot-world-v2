#!/usr/bin/env python
"""Measure the real VRAM budget for DiT||VAE overlap.

  python overlap_probe.py --base examples/00 --tile_h 12 --tile_w 18 --tile_ov 6
"""
import argparse
import gc
import hashlib
import math
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import (compute_relative_poses, get_Ks_transformed,
                                 get_plucker_embeddings)

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def mem(tag):
    free, total = torch.cuda.mem_get_info()
    a = torch.cuda.memory_allocated() / 2**20
    r = torch.cuda.memory_reserved() / 2**20
    print(f"[op] {tag:26s} free={free/2**20:7.0f}MiB total={total/2**20:6.0f}MiB "
          f"alloc={a:7.0f} reserved={r:7.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/00")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--tile_h", type=int, default=12)
    ap.add_argument("--tile_w", type=int, default=18)
    ap.add_argument("--tile_ov", type=int, default=6)
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    mem("startup")
    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    mem("pipe built")
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect()
    mem("after T5 free")

    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch[2] * patch[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = 5
    lat_f = 2
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * args.local_attn_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads

    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    mem("after prewarm")
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
    mem("after condition latent")

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    mem("after KV caches")

    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(42)
    noise = torch.randn(16, lat_f, lat_h, lat_w, generator=g, device=dev)

    plk = torch.zeros(1, 4 * (h // 8) * (w // 8) // (patch[1] * patch[2]) * 1,
                      1, 1, dtype=pdt, device=dev)

    def dit_step(cid, cur):
        rel = torch.eye(4)[None].to(dev)
        p = get_plucker_embeddings(rel, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        p = rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h, w=lat_w).to(pdt)
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
              "y": [y.split(1, dim=1)[cid]],
              "dit_cond_dict": {"c2ws_plucker_emb": p.chunk(1, dim=0)},
              "kv_cache": self_kv, "crossattn_cache": cross_kv,
              "current_start": cid * frame_seqlen,
              "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            for ti in range(len(timesteps)):
                npred = pipe.model(x=[cur.to(dev)],
                                   t=torch.stack([timesteps[ti]]).to(dev),
                                   cross_attn_first_call=not pipe._cross_attn_initialized,
                                   **kw)[0]
                pipe._cross_attn_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=g, device=x0.device,
                                        dtype=x0.dtype), timesteps[ti + 1])
                else:
                    break
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        return x0

    torch.cuda.reset_peak_memory_stats()
    z = dit_step(0, noise.split(1, dim=1)[0])
    mem("after DiT chunk0")
    print(f"[op] DiT peak so far = {torch.cuda.max_memory_allocated()/2**20:.0f}MiB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    pipe.vae.stream_decode_start_tiled(args.tile_h, args.tile_w, args.tile_ov)
    out = pipe.vae.stream_decode_step_tiled(z)
    mem("after tiled decode")
    print(f"[op] tiled decode peak = {torch.cuda.max_memory_allocated()/2**20:.0f}MiB", flush=True)
    pipe.vae.stream_decode_end()
    print(f"[op] out {tuple(out.shape)}", flush=True)


if __name__ == "__main__":
    main()
