#!/usr/bin/env python
"""
denoise_ab.py — causal-fast denoise 步数 A/B（同首帧/同 prompt/同 poses/同 seed）

固定: 512x320-area, FP8, chunk=1, local=2。只改 timesteps_index。
一次建模型，多个 schedule 复用；用 streaming decode 出 21 帧视频。

用法:
  LINGBOT_FP8=1 python denoise_ab.py \
      --ckpt_dir ~/ai/models/lingbot-world-v2-1.3b-causal-fast \
      --assets_dir ~/ai/models/lingbot-shared-assets \
      --out_dir output/denoise_ab
"""
import argparse
import gc
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
from wan.utils.utils import save_video
from wan.utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)


def log(*a):
    print(*a, flush=True)


def parse_area(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


SCHEDULES = {
    "4step": [0, 250, 500, 750],
    "3step_A": [0, 250, 750],
    "3step_B": [0, 500, 750],
    "2step": [0, 750],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--assets_dir", default=None)
    ap.add_argument("--image", default="examples/03/image.jpg")
    ap.add_argument("--action_path", default="examples/03")
    ap.add_argument("--prompt",
                    default="A serene lakeside landscape with smooth forward camera movement.")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=2)
    ap.add_argument("--sink_size", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=1)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="output/denoise_ab")
    ap.add_argument("--only", default="", help="comma list to run subset")
    ap.add_argument("--offload_decode", type=int, default=0,
                    help="move DiT to CPU before VAE decode (for local18)")
    ap.add_argument("--ts", default="", help="override: comma-separated timesteps")
    ap.add_argument("--name", default="custom", help="name when --ts is used")
    args = ap.parse_args()

    if args.ts:
        SCHEDULES.clear()
        SCHEDULES[args.name] = [int(x) for x in args.ts.split(",")]

    W, H = parse_area(args.area)
    max_area = W * H
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    log(f"[ab] building pipe (FP8={os.getenv('LINGBOT_FP8','0')}) ...")
    t0 = time.perf_counter()
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    log(f"[ab] pipe built in {time.perf_counter()-t0:.1f}s")

    img_pil = Image.open(args.image).convert("RGB")
    dev = pipe.device
    pdt = pipe.param_dtype
    dtype = pipe.pipe_dtype
    vae_stride = pipe.vae_stride
    patch = pipe.patch_size

    key = hashlib.sha256(args.prompt.encode()).hexdigest()
    if key not in pipe._t5_cache:
        log("[ab] T5 encode ...")
        ctx = pipe.text_encoder([args.prompt], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(dev) for t in ctx]
    context = pipe._t5_cache[key]

    # T5 is only needed for prompt encoding; free it (drops the 10.6GiB mmap)
    # to relieve host RAM pressure during long runs.
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
        gc.collect()
        log("[ab] freed T5 encoder")

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
    log(f"[ab] area={W}x{H} -> lat {lat_h}x{lat_w}, lat_f={lat_f}, "
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

    log("[ab] VAE encode condition ...")
    y = pipe.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
            torch.zeros(3, F - 1, h, w)
        ], dim=1).to(dev)
    ])[0]
    y = torch.concat([msk, y])

    model_args = pipe.model.config
    kv_size = frame_seqlen * args.local_attn_size
    head_dim = model_args.dim // model_args.num_heads
    local_heads = model_args.num_heads // pipe.sp_size

    pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=args.shift)

    @contextmanager
    def noop():
        yield

    no_sync = getattr(pipe.model, "no_sync", noop)

    log("[ab] prewarm ...")
    pipe.prewarm(img_pil, max_area=max_area, frame_num=args.frames,
                 chunk_size=args.chunk_size)

    names = [n for n in SCHEDULES if not args.only or n in args.only.split(",")]
    results = {}
    for name in names:
        ts_idx = SCHEDULES[name]
        timesteps = pipe.scheduler.timesteps[ts_idx]
        n_fwd = len(timesteps) + 1

        self_kv = pipe._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=[1, kv_size, local_heads, head_dim], dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=model_args.num_layers,
            shape=[1, 512, model_args.num_heads, head_dim], dtype=dtype,
            device=dev)
        pipe._cross_attn_initialized = False

        seed_g = torch.Generator(device=dev)
        seed_g.manual_seed(args.seed)
        noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32,
                            generator=seed_g, device=dev)
        latents_chunk = noise.split(args.chunk_size, dim=1)
        cond_chunk = y.split(args.chunk_size, dim=1)
        plk_chunk = plucker.split(args.chunk_size, dim=2)
        n_chunks = len(latents_chunk)

        torch.cuda.reset_peak_memory_stats()
        pred = []
        chunk_ms = []
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
            chunk_ms.append(e0.elapsed_time(e1))
            pred.append(x0)
            log(f"[ab] {name} chunk {cid+1}/{n_chunks} dit={chunk_ms[-1]:.0f}ms")

        # streaming decode -> video. Optionally offload DiT to CPU first so
        # the VAE decode has room on an 8GB card (DiT weights + KV can be ~4GB).
        if args.offload_decode:
            pipe.model.to('cpu')
            torch.cuda.empty_cache()
        pipe.vae.stream_decode_start()
        parts = []
        dec_ms = []
        for i, c in enumerate(pred):
            d0 = torch.cuda.Event(enable_timing=True)
            d1 = torch.cuda.Event(enable_timing=True)
            d0.record()
            with torch.no_grad():
                parts.append(pipe.vae.stream_decode_step(c))
            d1.record()
            torch.cuda.synchronize()
            dec_ms.append(d0.elapsed_time(d1))
            log(f"[ab] {name} decode {i+1}/{len(pred)} {dec_ms[-1]:.0f}ms")
        pipe.vae.stream_decode_end()
        if args.offload_decode:
            pipe.model.to(dev)
        rgb = torch.cat(parts, dim=1)

        out = os.path.join(args.out_dir, f"{name}.mp4")
        save_video(tensor=rgb[None], save_file=out, fps=cfg.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        dit_avg = sum(chunk_ms[1:]) / max(1, len(chunk_ms) - 1)
        dec_avg = sum(dec_ms) / len(dec_ms)
        peak = torch.cuda.max_memory_allocated() / 2**20
        results[name] = (dit_avg, dec_avg, dit_avg + dec_avg, peak)
        log(f"[ab] {name:8s} ts={ts_idx} fwd/chunk={n_fwd} | "
            f"dit={dit_avg:.0f}ms dec={dec_avg:.0f}ms "
            f"total={dit_avg+dec_avg:.0f}ms peak={peak:.0f}MiB -> {out}")

    log("\n[ab] ===== summary (steady, excl chunk0) =====")
    for n in names:
        d, v, t, p = results[n]
        log(f"[ab] {n:8s} dit={d:.0f}ms  vae={v:.0f}ms  total={t:.0f}ms  "
            f"~{1000.0/t*4:.1f} fps-equiv  peak={p:.0f}MiB")
    log("[ab] DONE")


if __name__ == "__main__":
    main()
