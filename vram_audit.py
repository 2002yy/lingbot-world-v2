#!/usr/bin/env python
"""§36C: VRAM headroom / allocator stability audit.

Samples driver free/total, torch allocated/reserved and the allocator's
inactive-split / segment counts at every setup stage, then dumps a memory
snapshot summary so we can tell apart:
    A. reserved ~= 8GB while allocated ~= 3.5GB  -> torch allocator / fragmentation
    B. torch reserved is modest but driver free ~0 -> cuDNN/cuBLAS workspace or
       other CUDA libraries / processes

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python vram_audit.py --chunks 4
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from cam_controller import CameraController
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
FPS, CADENCE = 60.0, 1.25


def snap(tag):
    free, total = torch.cuda.mem_get_info()
    a = torch.cuda.memory_allocated() / 2**20
    r = torch.cuda.memory_reserved() / 2**20
    st = torch.cuda.memory_stats()
    inact = st.get("inactive_split_bytes.all.current", 0) / 2**20
    segs = st.get("segment.all.current", 0)
    print(f"[vram] {tag:26s} free={free/2**20:7.0f} "
          f"alloc={a:7.0f} reserved={r:7.0f} inact_split={inact:7.0f} "
          f"segs={segs}", flush=True)
    return dict(tag=tag, free=free / 2**20, total=total / 2**20, alloc=a,
                reserved=r, inact=inact, segs=segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--drop_vae", type=int, default=1,
                    help="delete the full VAE after the condition latent is built")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/vram")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    stages = [snap("startup")]

    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    n_ctrl = int((args.chunks * CADENCE + 2.0) * FPS)
    cps = []
    for k in range(n_ctrl):
        ctl.set_input(fwd=0.6)
        ctl.step(dt=1.0 / (FPS * CADENCE))
        cps.append(ctl.pose.copy())
    cps = np.stack(cps)

    def slot(t):
        return cps[int(np.clip(round(t * FPS), 0, len(cps) - 1))]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    stages.append(snap("pipe built"))
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    stages.append(snap("TAE loaded"))
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    stages.append(snap("T5 freed + empty_cache"))

    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch[2] * patch[2])
    h = lat_h * vae_stride[1]; w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = (args.chunks - 1) * 4 + 1
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * 6
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads
    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    stages.append(snap("prewarm"))
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
    stages.append(snap("condition latent (VAE encode)"))

    if args.drop_vae:
        pipe.vae = None
        gc.collect(); torch.cuda.empty_cache()
        stages.append(snap("full VAE dropped"))

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    stages.append(snap("KV caches"))
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, args.chunks, lat_h, lat_w, generator=g, device=dev)
    stages.append(snap("noise"))

    def plucker(rel):
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(dev)
        p = get_plucker_embeddings(r, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h, w=lat_w).to(pdt)

    prev_pose = None
    t0 = time.perf_counter()
    for cid in range(args.chunks):
        cur = noise.split(1, dim=1)[cid]
        kw = None
        tc = time.perf_counter()
        for ti in range(len(timesteps)):
            now = time.perf_counter() - t0
            pn = slot(now)
            rel = np.eye(4) if prev_pose is None else np.linalg.inv(prev_pose) @ pn
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plucker(rel).chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * frame_seqlen,
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
        t_dit = time.perf_counter() - tc
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        zt = x0.permute(1, 0, 2, 3).unsqueeze(0)
        t1 = time.perf_counter()
        with torch.no_grad():
            fr = tae.decode_video(zt, parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        t_tae = time.perf_counter() - t1
        print(f"[vram] chunk {cid}: dit={t_dit*1000:.0f}ms tae={t_tae*1000:.0f}ms",
              flush=True)
        prev_pose = pn
        if cid in (0, 1, 2, args.chunks - 1):
            stages.append(snap(f"after chunk {cid}"))

    # deliberate small allocation stress: can the allocator recover?
    try:
        probe = torch.empty(300 * 2**20 // 4, device=dev, dtype=torch.float32)
        stages.append(snap("+300MB probe"))
        del probe
        gc.collect(); torch.cuda.empty_cache()
        stages.append(snap("probe released"))
    except Exception as e:
        print(f"[vram] probe FAILED: {e}", flush=True)

    print("\n[vram] ===== allocator snapshot summary =====")
    try:
        s = torch.cuda.memory_snapshot()
        from collections import Counter
        c = Counter()
        for seg in s:
            c[seg.get("segment_type", "?")] += seg.get("allocated_size", 0) / 2**20
        for k, v in c.most_common():
            print(f"  {k:12s} {v:8.1f} MiB allocated")
        print(f"  segments total = {len(s)}")
    except Exception as e:
        print(f"  snapshot failed: {e}")

    json.dump(stages, open(f"{args.out_dir}/stages.json", "w"), indent=1)


if __name__ == "__main__":
    main()
