#!/usr/bin/env python
"""
realtime_probe.py — LingBot-World 2.0 1.3B causal-fast 实时引擎核心测量器

加载一次模型 -> 持久 KV 会话 -> prewarm -> 逐 chunk 计时(拆 5 次 DiT forward)
-> VAE streaming decode 正确性 + 性能。

用法:
  LINGBOT_FP8=1 python realtime_probe.py \
      --ckpt_dir ~/ai/models/lingbot-world-v2-1.3b-causal-fast \
      --assets_dir ~/ai/models/lingbot-shared-assets \
      --area 512x320 --local_attn_size 2 --sink_size 0 \
      --chunk_size 1 --frames 21 --steps 6
"""
import argparse
import hashlib
import math
import os
import sys
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


def cos_sim(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)


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
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--timesteps_index", default="0,250,500,750")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stream_decode", type=int, default=1)
    ap.add_argument("--compile", default="eager",
                    choices=["eager", "reduce-overhead", "max-autotune",
                             "max-autotune-no-cudagraphs"])
    ap.add_argument("--discard", type=int, default=5,
                    help="chunks to drop before steady stats")
    args = ap.parse_args()

    W, H = parse_area(args.area)
    max_area = W * H
    ts_idx = [int(x) for x in args.timesteps_index.split(",")]

    cfg = WAN_CONFIGS[args.task]
    log(f"[probe] building pipe (FP8={os.getenv('LINGBOT_FP8','0')}) ...")
    t0 = time.perf_counter()
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    log(f"[probe] pipe built in {time.perf_counter()-t0:.1f}s")

    if args.compile != "eager":
        log(f"[probe] torch.compile(mode={args.compile}, dynamic=False) ...")
        t0 = time.perf_counter()
        pipe.model = torch.compile(pipe.model, mode=args.compile, dynamic=False)
        log(f"[probe] compile wrap done in {time.perf_counter()-t0:.1f}s")

    img_pil = Image.open(args.image).convert("RGB")
    dev = pipe.device
    pdt = pipe.param_dtype
    dtype = pipe.pipe_dtype
    vae_stride = pipe.vae_stride
    patch = pipe.patch_size

    key = hashlib.sha256(args.prompt.encode()).hexdigest()
    if key in pipe._t5_cache:
        context = pipe._t5_cache[key]
    else:
        log("[probe] T5 encode (CPU) ...")
        t0 = time.perf_counter()
        context = pipe.text_encoder([args.prompt], torch.device("cpu"))
        context = [t.to(dev) for t in context]
        pipe._t5_cache[key] = context
        log(f"[probe] T5 encode {time.perf_counter()-t0:.1f}s")

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
    max_seq_len = args.chunk_size * frame_seqlen
    max_seq_len = int(math.ceil(max_seq_len / pipe.sp_size)) * pipe.sp_size
    log(f"[probe] area={W}x{H} -> lat {lat_h}x{lat_w}, lat_f={lat_f}, "
        f"frame_seqlen={frame_seqlen}, max_seq_len={max_seq_len}")

    c2ws = np.load(os.path.join(args.action_path, "poses.npy"))
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    frame_num = min(F, len_c2ws)
    c2ws = c2ws[:frame_num]
    Ks = torch.from_numpy(
        np.load(os.path.join(args.action_path, "intrinsics.npy"))).float()
    Ks = get_Ks_transformed(Ks, height_org=480, width_org=832,
                            height_resize=h, width_resize=w,
                            height_final=h, width_final=w)
    Ks = Ks[0]
    len_c2ws_ = int((len(c2ws) - 1) // 4) + 1
    len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % args.chunk_size))
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len(c2ws) - 1, len(c2ws)),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len(c2ws) - 1, len_c2ws_))
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks = Ks.repeat(len(c2ws_infer), 1)
    c2ws_infer = c2ws_infer.to(dev)
    Ks = Ks.to(dev)
    c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb, "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
        c1=int(h // lat_h), c2=int(w // lat_w))
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h,
        w=lat_w).to(pdt)

    seed_g = torch.Generator(device=dev)
    seed_g.manual_seed(args.seed)
    noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32,
                        generator=seed_g, device=dev)

    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    log("[probe] VAE encode first-frame condition ...")
    y = pipe.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
            torch.zeros(3, F - 1, h, w)
        ], dim=1).to(dev)
    ])[0]
    y = torch.concat([msk, y])

    pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=args.shift)
    timesteps = pipe.scheduler.timesteps[ts_idx]

    model_args = pipe.model.config
    kv_size = frame_seqlen * args.local_attn_size if args.local_attn_size > -1 \
        else frame_seqlen * lat_f
    head_dim = model_args.dim // model_args.num_heads
    local_num_heads = model_args.num_heads // pipe.sp_size
    self_kv_cache = pipe._initialize_self_kv_cache(
        num_layers=model_args.num_layers,
        shape=[1, kv_size, local_num_heads, head_dim],
        dtype=dtype, device=dev)
    cross_kv_cache = pipe._initialize_crossattn_cache(
        num_layers=model_args.num_layers,
        shape=[1, 512, model_args.num_heads, head_dim],
        dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False

    log("[probe] prewarm ...")
    t0 = time.perf_counter()
    pipe.prewarm(img_pil, max_area=max_area, frame_num=args.frames,
                 chunk_size=args.chunk_size)
    log(f"[probe] prewarm {time.perf_counter()-t0:.1f}s")

    latents_chunk = noise.split(args.chunk_size, dim=1)
    condition_chunk = y.split(args.chunk_size, dim=1)
    plucker_chunk = c2ws_plucker_emb.split(args.chunk_size, dim=2)
    n_chunks = min(len(latents_chunk), args.steps)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_model = getattr(pipe.model, "no_sync", noop_no_sync)
    n_fwd = len(timesteps) + 1  # denoise forwards + 1 KV update

    log(f"[probe] === DiT timing: {n_chunks} chunks x {n_fwd} forwards "
        f"(local={args.local_attn_size}) ===")
    torch.cuda.reset_peak_memory_stats()
    pred_chunks = []
    fwd_times = [[] for _ in range(n_fwd)]
    chunk_ms = []

    for chunk_id in range(n_chunks):
        current_latent = latents_chunk[chunk_id]
        kwargs = {
            "context": [context[0]],
            "seq_len": max_seq_len,
            "y": [condition_chunk[chunk_id]],
            "dit_cond_dict": {
                "c2ws_plucker_emb": plucker_chunk[chunk_id].chunk(1, dim=0)},
            "kv_cache": self_kv_cache,
            "crossattn_cache": cross_kv_cache,
            "current_start": chunk_id * args.chunk_size * frame_seqlen,
            "max_attention_size": kv_size,
            "frame_seqlen": frame_seqlen,
        }

        def timed(fn, idx):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            e0.record()
            r = fn()
            e1.record()
            torch.cuda.synchronize()
            ms = e0.elapsed_time(e1)
            fwd_times[idx].append(ms)
            return r, ms

        whole0 = torch.cuda.Event(enable_timing=True)
        whole1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        whole0.record()
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad(), no_sync_model():
            for ti in range(len(timesteps)):
                def do_denoise():
                    return pipe.model(
                        x=[current_latent.to(dev, dtype=pdt)],
                        t=torch.stack([timesteps[ti]]).to(dev),
                        cross_attn_first_call=not pipe._cross_attn_initialized,
                        **kwargs)[0]
                noise_pred, _ = timed(do_denoise, ti)
                pipe._cross_attn_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=noise_pred, xt=current_latent,
                    timestep=timesteps[ti], scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    current_latent = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=seed_g,
                                        device=x0.device, dtype=x0.dtype),
                        timesteps[ti + 1])
                else:
                    break

            def do_kv():
                return pipe.model(
                    x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                    cross_attn_first_call=False, **kwargs)
            _, _ = timed(do_kv, len(timesteps))
        whole1.record()
        torch.cuda.synchronize()
        whole_ms = whole0.elapsed_time(whole1)

        pred_chunks.append(x0)
        cms = sum(t[-1] for t in fwd_times)
        chunk_ms.append(cms)
        per = " ".join(f"f{i}={fwd_times[i][-1]:.0f}" for i in range(n_fwd))
        log(f"[chunk {chunk_id}] sum={cms:.0f}ms whole={whole_ms:.0f}ms  {per}")

    if chunk_ms:
        import statistics
        avg = sum(chunk_ms) / len(chunk_ms)
        steady = chunk_ms[args.discard:] if len(chunk_ms) > args.discard else chunk_ms
        med = statistics.median(steady)
        p90 = sorted(steady)[min(len(steady) - 1, int(0.9 * len(steady)))]
        log(f"[probe] DiT avg = {avg:.0f} ms/chunk ({1000.0/avg:.2f} chunks/s)")
        log(f"[probe] STEADY_MEDIAN_MS={med:.1f}  STEADY_P90_MS={p90:.1f}  "
            f"STEADY_N={len(steady)}")
        for i in range(n_fwd):
            arr = fwd_times[i]
            log(f"[probe]   forward[{i}] avg = {sum(arr)/len(arr):.0f} ms")
    log(f"[probe] peak allocated = {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    log(f"[probe] peak reserved  = {torch.cuda.max_memory_reserved()/2**20:.0f} MiB")

    # ---- VAE streaming decode: correctness + perf ----
    if args.stream_decode:
        all_latent = torch.cat(pred_chunks, dim=1)
        log(f"[probe] === VAE decode: full vs streaming "
            f"({all_latent.shape[1]} latent frames) ===")
        with torch.no_grad():
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            full_rgb = pipe.vae.decode([all_latent])[0]
            e1.record()
            torch.cuda.synchronize()
            full_ms = e0.elapsed_time(e1)

            pipe.vae.stream_decode_start()
            stream_parts = []
            step_ms = []
            for i, c in enumerate(pred_chunks):
                s0 = torch.cuda.Event(enable_timing=True)
                s1 = torch.cuda.Event(enable_timing=True)
                s0.record()
                rgb_i = pipe.vae.stream_decode_step(c)
                s1.record()
                torch.cuda.synchronize()
                step_ms.append(s0.elapsed_time(s1))
                stream_parts.append(rgb_i)
                log(f"[decode step {i}] {step_ms[-1]:.0f} ms  rgb={tuple(rgb_i.shape)}")
            pipe.vae.stream_decode_end()
            stream_rgb = torch.cat(stream_parts, dim=1)

        log(f"[probe] full decode      = {full_ms:.0f} ms  rgb={tuple(full_rgb.shape)}")
        log(f"[probe] stream decode    = {sum(step_ms):.0f} ms total, "
            f"{sum(step_ms)/len(step_ms):.0f} ms/chunk")
        if full_rgb.shape == stream_rgb.shape:
            maxdiff = (full_rgb - stream_rgb).abs().max().item()
            cos = cos_sim(full_rgb, stream_rgb).item()
            log(f"[probe] correctness: max|full-stream| = {maxdiff:.6e}  cosine = {cos:.8f}")
        else:
            log(f"[probe] shape mismatch: full {tuple(full_rgb.shape)} vs "
                f"stream {tuple(stream_rgb.shape)}")

    log("[probe] DONE")


if __name__ == "__main__":
    main()
