#!/usr/bin/env python
"""§32 DiT timeline + condition lifetime + mid-flight perturbation probe.

Part A: time the chunk pipeline (preprocess / each denoise step / KV update / decode).
Part C: run chunk 0 with camera condition A, switch it to B at step k, and see
        whether the final latent lands closer to the A-baseline or the B-baseline.

  effective_control_age = how old the camera condition actually is in the output.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python midflight_probe.py --base examples/04
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yaw_deg", type=float, default=25.0)
    ap.add_argument("--out_dir", default="output/midflight")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[mf] pipe built", flush=True)
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
    F = 5
    lat_f = 2
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * pipe.local_attn_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads

    t_pre0 = time.perf_counter()
    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    t_prewarm = time.perf_counter() - t_pre0
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)

    t0 = time.perf_counter()
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
    t_cond = time.perf_counter() - t0

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False

    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, lat_f, lat_h, lat_w, generator=g, device=dev)
    cur0 = noise.split(1, dim=1)[0]

    # ---- two camera conditions ----
    def plucker_for(angle_deg):
        th = math.radians(angle_deg)
        R = np.array([[math.cos(th), 0, math.sin(th)],
                      [0, 1, 0],
                      [-math.sin(th), 0, math.cos(th)]])
        rel = torch.eye(4)[None].to(dev)
        rel[0, :3, :3] = torch.from_numpy(R).float().to(dev)
        # NOTE: do NOT pass through compute_relative_poses(framewise=True) here --
        # with a single frame it forces relative_poses[0] = identity and wipes
        # out the rotation. The raw rotation IS the framewise relative pose.
        p = get_plucker_embeddings(rel, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                         h=lat_h, w=lat_w).to(pdt)

    P_A = plucker_for(0.0)
    P_B = plucker_for(args.yaw_deg)

    def run_chunk(swap_step, record=False):
        """swap_step: steps < swap_step use A, >= use B. 0 => all B, 3 => all A."""
        g.manual_seed(args.seed)          # identical noise for every run
        cur = cur0.clone()
        pipe._cross_attn_initialized = False
        step_ms = []
        for ti in range(len(timesteps)):
            plk = P_A if ti < swap_step else P_B
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[0]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": 0, "max_attention_size": kv_size,
                  "frame_seqlen": frame_seqlen}
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
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
            e1.record(); torch.cuda.synchronize()
            step_ms.append(e0.elapsed_time(e1))
        return x0, step_ms

    # ---- Part A: timeline ----
    print("\n[mf] ===== Part A: DiT timeline (chunk 0) =====")
    x0_A, ms_A = run_chunk(3)
    print(f"[mf] prewarm={t_prewarm*1000:.0f}ms  cond(VAE encode+plucker)={t_cond*1000:.0f}ms")
    for i, m in enumerate(ms_A):
        print(f"[mf] denoise step {i} (t={int(timesteps[i])}): {m:.0f}ms")
    print(f"[mf] DiT total (3 steps) = {sum(ms_A):.0f}ms")

    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
        pipe.model(x=[x0_A], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                   cross_attn_first_call=False,
                   context=[pipe._t5_cache[key][0]], seq_len=max_seq_len,
                   y=[y.split(1, dim=1)[0]],
                   dit_cond_dict={"c2ws_plucker_emb": P_A.chunk(1, dim=0)},
                   kv_cache=self_kv, crossattn_cache=cross_kv, current_start=0,
                   max_attention_size=kv_size, frame_seqlen=frame_seqlen)
    torch.cuda.synchronize()
    print(f"[mf] KV-update forward = {(time.perf_counter()-t0)*1000:.0f}ms")

    # steady-state re-measure (first call carries autotune/cache overhead)
    _, ms_A2 = run_chunk(3)
    print(f"[mf] steady denoise steps = " +
          ", ".join(f"{m:.0f}ms" for m in ms_A2) +
          f"  total={sum(ms_A2):.0f}ms")

    t0 = time.perf_counter()
    pipe.vae.stream_decode_start()
    with torch.no_grad():
        dec_A = pipe.vae.stream_decode_step(x0_A)
    pipe.vae.stream_decode_end()
    torch.cuda.synchronize()
    print(f"[mf] VAE decode (1 latent) = {(time.perf_counter()-t0)*1000:.0f}ms")

    # ---- Part C: perturbation probe ----
    print("\n[mf] ===== Part C: mid-flight perturbation (A -> B at step k) =====")
    x0_B, ms_B = run_chunk(0)
    d_AB = float((x0_A - x0_B).abs().mean())
    print(f"[mf] baseline latent distance |A-B| = {d_AB:.4f}")
    rows = []
    for swap in (3, 2, 1, 0):
        x0, _ = run_chunk(swap)
        dA = float((x0 - x0_A).abs().mean())
        dB = float((x0 - x0_B).abs().mean())
        eff = dA / max(dA + dB, 1e-9)      # 0 = pure A, 1 = pure B
        t_switch = sum(ms_A[:swap]) if swap < len(ms_A) else sum(ms_A)
        rows.append(dict(swap=swap, dA=dA, dB=dB, effect=eff, t_ms=t_switch))
        print(f"[mf] switch at step {swap} (t~{t_switch:.0f}ms): "
              f"dA={dA:.4f} dB={dB:.4f}  effect={eff:.3f} "
              f"({'pure A' if swap>=3 else 'pure B' if swap==0 else 'mixed'})")

    print("\n[mf] ===== admission curve =====")
    print(f"{'switch_t_ms':>12s} {'effect(0=A,1=B)':>16s}")
    for r in rows:
        print(f"{r['t_ms']:12.0f} {r['effect']:16.3f}")
    print("\n[mf] NOTE: effect~0 means the output ignored the late B switch "
          "(condition effectively frozen); effect~1 means B fully took over.")


if __name__ == "__main__":
    main()
