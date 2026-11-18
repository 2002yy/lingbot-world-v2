#!/usr/bin/env python
"""
coresident_probe.py — DiT + KV + streaming VAE 同驻测试

对每个 (local, sink)：交替执行 DiT chunk -> streaming decode chunk，连续 N 个 chunk，
记录每 chunk 的 dit_ms / decode_ms / allocated / reserved，并在整个联合循环前
reset_peak_memory_stats()，循环后打印真实联合峰值。

用法:
  LINGBOT_FP8=1 python coresident_probe.py --local 12 --sink 4 --steps 6
"""
import argparse
import hashlib
import math
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import (
    compute_relative_poses, interpolate_camera_poses,
    get_plucker_embeddings, get_Ks_transformed,
)


def log(*a):
    print(*a, flush=True)


def parse_area(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--assets_dir", default=None)
    ap.add_argument("--image", default="examples/03/image.jpg")
    ap.add_argument("--action_path", default="examples/03")
    ap.add_argument("--prompt",
                    default="A serene lakeside scene with a lone tree standing in calm water, "
                            "surrounded by distant snow-capped mountains under a bright blue sky "
                            "with drifting white clouds — gentle ripples reflect the tree and sky, "
                            "creating a tranquil, meditative atmosphere.")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local", type=int, default=12)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=1)
    ap.add_argument("--frames", type=int, default=41)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--timesteps_index", default="0,250,500,750")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    W, H = parse_area(args.area)
    max_area = W * H
    ts_idx = [int(x) for x in args.timesteps_index.split(",")]
    cfg = WAN_CONFIGS[args.task]

    log(f"[co] building pipe local={args.local} sink={args.sink} ...")
    t0 = time.perf_counter()
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local,
        sink_size=args.sink, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    log(f"[co] pipe built in {time.perf_counter()-t0:.1f}s")

    img_pil = Image.open(args.image).convert("RGB")
    dev = pipe.device
    pdt = pipe.param_dtype
    dtype = pipe.pipe_dtype
    vae_stride = pipe.vae_stride
    patch = pipe.patch_size

    key = hashlib.sha256(args.prompt.encode()).hexdigest()
    if key not in pipe._t5_cache:
        log("[co] T5 encode ...")
        ctx = pipe.text_encoder([args.prompt], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(dev) for t in ctx]
    context = pipe._t5_cache[key]
    # free T5 (prompt already cached) to relieve host RAM
    import gc
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
        gc.collect()

    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(max_area * aspect) // vae_stride[1] //
                  patch[1] * patch[1])
    lat_w = round(math.sqrt(max_area / aspect) // vae_stride[2] //
                  patch[2] * patch[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    F = ((args.frames - 1) // 4) * 4 + 1
    lat_f = (F - 1) // vae_stride[0] + 1
    lat_f = int(lat_f - (lat_f % args.chunk_size))
    F = (lat_f - 1) * 4 + 1
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    max_seq_len = int(math.ceil(
        args.chunk_size * frame_seqlen / pipe.sp_size)) * pipe.sp_size
    log(f"[co] area={W}x{H} -> lat {lat_h}x{lat_w}, lat_f={lat_f}, "
        f"frame_seqlen={frame_seqlen}")

    c2ws = np.load(os.path.join(args.action_path, "poses.npy"))
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    c2ws = c2ws[:min(F, len_c2ws)]
    Ks = torch.from_numpy(
        np.load(os.path.join(args.action_path, "intrinsics.npy"))).float()
    Ks = get_Ks_transformed(Ks, height_org=480, width_org=832,
                            height_resize=h, width_resize=w,
                            height_final=h, width_final=w)[0]
    len_c2ws_ = int((len(c2ws) - 1) // 4) + 1
    len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % args.chunk_size))
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len(c2ws) - 1, len(c2ws)),
        src_rot_mat=c2ws[:, :3, :3], src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len(c2ws) - 1, len_c2ws_))
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks = Ks.repeat(len(c2ws_infer), 1)
    c2ws_infer = c2ws_infer.to(dev)
    Ks = Ks.to(dev)
    plucker = get_plucker_embeddings(c2ws_infer, Ks, h, w)
    plucker = rearrange(plucker, "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                        c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
    plucker = rearrange(plucker, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h,
                        w=lat_w).to(pdt)

    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

    log("[co] VAE encode condition ...")
    y = pipe.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
            torch.zeros(3, F - 1, h, w)
        ], dim=1).to(dev)
    ])[0]
    y = torch.concat([msk, y])

    model_args = pipe.model.config
    kv_size = frame_seqlen * args.local
    head_dim = model_args.dim // model_args.num_heads
    local_heads = model_args.num_heads // pipe.sp_size
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=model_args.num_layers,
        shape=[1, kv_size, local_heads, head_dim], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=model_args.num_layers,
        shape=[1, 512, model_args.num_heads, head_dim], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False

    pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=args.shift)
    timesteps = pipe.scheduler.timesteps[ts_idx]

    log("[co] prewarm ...")
    pipe.prewarm(img_pil, max_area=max_area, frame_num=args.frames,
                 chunk_size=args.chunk_size)

    seed_g = torch.Generator(device=dev)
    seed_g.manual_seed(args.seed)
    noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32,
                        generator=seed_g, device=dev)
    latents_chunk = noise.split(args.chunk_size, dim=1)
    cond_chunk = y.split(args.chunk_size, dim=1)
    plk_chunk = plucker.split(args.chunk_size, dim=2)
    n_chunks = min(len(latents_chunk), args.steps)

    @contextmanager
    def noop():
        yield

    no_sync = getattr(pipe.model, "no_sync", noop)

    log(f"[co] === co-resident loop: {n_chunks} chunks (local={args.local} "
        f"sink={args.sink}, offload=0) ===")
    torch.cuda.reset_peak_memory_stats()
    pipe.vae.stream_decode_start()
    dit_ms, dec_ms = [], []
    for cid in range(n_chunks):
        cur = latents_chunk[cid]
        kwargs = {
            "context": [context[0]], "seq_len": max_seq_len,
            "y": [cond_chunk[cid]],
            "dit_cond_dict": {"c2ws_plucker_emb": plk_chunk[cid].chunk(1, dim=0)},
            "kv_cache": self_kv, "crossattn_cache": cross_kv,
            "current_start": cid * args.chunk_size * frame_seqlen,
            "max_attention_size": kv_size, "frame_seqlen": frame_seqlen,
        }
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad(), no_sync():
            for ti in range(len(timesteps)):
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=not pipe._cross_attn_initialized,
                    **kwargs)[0]
                pipe._cross_attn_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=seed_g,
                                        device=x0.device, dtype=x0.dtype),
                        timesteps[ti + 1])
                else:
                    break
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kwargs)
        e1.record()
        torch.cuda.synchronize()
        d_it = e0.elapsed_time(e1)
        dit_ms.append(d_it)

        d0 = torch.cuda.Event(enable_timing=True)
        d1 = torch.cuda.Event(enable_timing=True)
        d0.record()
        with torch.no_grad():
            _ = pipe.vae.stream_decode_step(x0)
        d1.record()
        torch.cuda.synchronize()
        d_dec = d0.elapsed_time(d1)
        dec_ms.append(d_dec)

        log(f"[co] chunk {cid+1}/{n_chunks}  dit={d_it:6.0f}ms  dec={d_dec:8.0f}ms  "
            f"alloc={torch.cuda.memory_allocated()/2**20:6.0f}MiB  "
            f"reserved={torch.cuda.memory_reserved()/2**20:6.0f}MiB")
    pipe.vae.stream_decode_end()

    log(f"[co] peak allocated = {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    log(f"[co] peak reserved  = {torch.cuda.max_memory_reserved()/2**20:.0f} MiB")
    log(f"[co] dit avg = {sum(dit_ms)/len(dit_ms):.0f}ms  "
        f"dec avg = {sum(dec_ms)/len(dec_ms):.0f}ms  "
        f"max dec = {max(dec_ms):.0f}ms")
    log("[co] DONE")


if __name__ == "__main__":
    main()
