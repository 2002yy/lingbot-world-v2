#!/usr/bin/env python
"""Real interactive loop: live SE(3) controller -> per-chunk plucker -> DiT -> streaming decode.

Unlike run_ctl.py (which feeds a pre-baked trajectory), this samples the camera
controller at the START of every chunk, so a change of input is reflected in the
very next generated chunk. Measures per-chunk DiT + decode wall time, which is
the dominant term of input->first-visible-response latency.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python interactive_loop.py --base examples/00
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
from wan.utils.utils import save_video
from wan.utils.cam_utils import (
    compute_relative_poses, get_Ks_transformed, get_plucker_embeddings,
)
from cam_controller import CameraController

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

# scripted "keyboard session": (n_chunks, fwd, strafe, yaw, pitch)
SESSION = [
    (3, 0.0, 0.0, 0.0, 0.0),
    (6, 0.6, 0.0, 0.0, 0.0),     # W
    (6, 0.4, 0.0, -0.6, 0.0),    # W + turn right
    (5, 0.5, 0.6, 0.0, 0.0),     # W + strafe right
    (5, 0.0, 0.0, -0.8, 0.0),    # turn only (floor kicks in)
    (4, 0.0, 0.0, 0.0, 0.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/00")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--out_dir", default="output/interactive")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    max_area = W * H
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    print("[il] pipe built", flush=True)
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
        gc.collect()

    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    vae_stride, patch = pipe.vae_stride, pipe.patch_size

    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(max_area * aspect) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(math.sqrt(max_area / aspect) // vae_stride[2] // patch[2] * patch[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    lat_f = sum(n for n, *_ in SESSION)
    F = (lat_f - 1) * 4 + 1
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * args.local_attn_size
    model_args = pipe.model.config
    local_heads = model_args.num_heads // pipe.sp_size
    head_dim = model_args.dim // model_args.num_heads

    print(f"[il] lat_f={lat_f} F={F} lat={lat_h}x{lat_w}", flush=True)
    pipe.prewarm(img_pil, max_area=max_area, frame_num=F, chunk_size=1)

    Ks = torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float()
    Ks = get_Ks_transformed(Ks, 480, 832, h, w, h, w)[0].to(dev)

    context = pipe._t5_cache[key]
    y = pipe.vae.encode([torch.concat([
        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                        mode='bicubic').transpose(0, 1),
        torch.zeros(3, F - 1, h, w)], dim=1).to(dev)])[0]
    # first-frame mask: 4 channels, all ones at latent 0, zeros elsewhere
    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    y = torch.concat([msk, y])

    timesteps = pipe.scheduler.timesteps[[int(x) for x in args.ts.split(",")]]
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=model_args.num_layers,
        shape=[1, kv_size, local_heads, head_dim], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=model_args.num_layers,
        shape=[1, 512, model_args.num_heads, head_dim], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False

    seed_g = torch.Generator(device=dev)
    seed_g.manual_seed(args.seed)
    noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32,
                        generator=seed_g, device=dev)
    cond_chunk = y.split(1, dim=1)
    lat_chunk = noise.split(1, dim=1)

    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    # in-distribution reference normalizer for live translation
    ref_norm = 1.0

    pipe.vae.stream_decode_start()
    parts, dit_ms, dec_ms, log_rows = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    cid = 0
    for n, fwd, strafe, yaw, pitch in SESSION:
        ctl.set_input(fwd, strafe, yaw, pitch)
        for _ in range(n):
            ctl.step()
            rel = np.eye(4) if cid == 0 else (
                np.linalg.inv(ctl.poses[cid - 1]) @ ctl.poses[cid])
            rel_t = torch.from_numpy(rel).float()
            rel_t[:3, 3] /= ref_norm
            c2ws = rel_t[None].to(dev)
            plk = get_plucker_embeddings(c2ws, Ks[None], h, w)
            plk = rearrange(plk, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                            c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(plk, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)

            cur = lat_chunk[cid]
            kwargs = {
                "context": [context[0]], "seq_len": max_seq_len,
                "y": [cond_chunk[cid]],
                "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                "kv_cache": self_kv, "crossattn_cache": cross_kv,
                "current_start": cid * frame_seqlen,
                "max_attention_size": kv_size, "frame_seqlen": frame_seqlen,
            }
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            e0.record()
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
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
            dit = e0.elapsed_time(e1)

            d0 = torch.cuda.Event(enable_timing=True)
            d1 = torch.cuda.Event(enable_timing=True)
            d0.record()
            with torch.no_grad():
                parts.append(pipe.vae.stream_decode_step(x0))
            d1.record()
            torch.cuda.synchronize()
            dec = d0.elapsed_time(d1)
            dit_ms.append(dit)
            dec_ms.append(dec)
            log_rows.append((cid, fwd, strafe, yaw, pitch, dit, dec))
            print(f"[il] chunk {cid:3d} fwd={fwd:+.1f} st={strafe:+.1f} "
                  f"yaw={yaw:+.1f} | dit={dit:.0f}ms dec={dec:.0f}ms "
                  f"visible={dit+dec:.0f}ms", flush=True)
            cid += 1
    pipe.vae.stream_decode_end()

    rgb = torch.cat(parts, dim=1)
    mp4 = f"{args.out_dir}/interactive.mp4"
    save_video(tensor=rgb[None], save_file=mp4, fps=cfg.sample_fps,
               nrow=1, normalize=True, value_range=(-1, 1))

    # steady-state stats (exclude chunk 0 warmup)
    d = np.array(dit_ms[1:])
    c = np.array(dec_ms[1:])
    tot = d + c
    peak = torch.cuda.max_memory_allocated() / 2**20
    print("\n[il] ===== latency (steady, excl chunk0) =====", flush=True)
    print(f"[il] dit   mean={d.mean():.0f}ms  p50={np.median(d):.0f}  p90={np.percentile(d,90):.0f}")
    print(f"[il] dec   mean={c.mean():.0f}ms  p50={np.median(c):.0f}  p90={np.percentile(c,90):.0f}")
    print(f"[il] visible (dit+dec) mean={tot.mean():.0f}ms  p50={np.median(tot):.0f}  "
          f"p90={np.percentile(tot,90):.0f}  -> {1000/tot.mean():.2f} world-fps")
    print(f"[il] peak={peak:.0f}MiB  video={mp4}")
    with open(f"{args.out_dir}/latency.txt", "w") as fh:
        fh.write("chunk fwd strafe yaw pitch dit_ms dec_ms visible_ms\n")
        for r in log_rows:
            fh.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]:.1f} {r[6]:.1f} "
                     f"{r[5]+r[6]:.1f}\n")


if __name__ == "__main__":
    main()
