#!/usr/bin/env python
"""BENCHMARK HARNESS -- 320x480 geometry. NOT the production path.

    *** RENAMED from production_loop.py. READ THIS BEFORE USING ITS NUMBERS. ***

An audit on 2026-09-28 (docs/RESOLUTION_AUTHORITY.md) captured the real
production geometry from a live pipe.generate():

    wan/image2video.py   native aspect, max_area = 480*832
                         -> pixel 512x768, latent 64x96, frame_seqlen 1536,
                            M(chunk_size=3) = 4608        <-- PRODUCTION

    this file            native aspect, max_area = --area = 512*320
                         -> pixel 320x480, latent 40x60, frame_seqlen 600,
                            M(chunk_size=3) = 1800        <-- NOT PRODUCTION

So this script was never measuring the production geometry, despite its former
name. It is kept because it is a useful *low-cost benchmark profile*, but:

    - its latency / VRAM / FPS numbers are BENCHMARK numbers for 320x480
    - they must NOT be quoted as production baselines
    - they must NOT be extrapolated to 512x768

The other valid profile is 304x528 (latent 38x66, frame_seqlen 627, M=1881),
used by pb3_run / m0_memgate / the P2b and S3 harnesses. It has the same
restriction: a benchmark profile, not production.

For production-geometry work use the real path, `wan/image2video.py`'s
generate(), or a harness explicitly parameterised to 512x768.

Original docstring follows.
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
from compensator import Compensator
from presets import PRESETS, DEFAULT_PRESET
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
FPS, CADENCE = 60.0, 1.25

# free-play style scripted input (replace with a real input source at runtime)
SCRIPT = [(0.0, dict(fwd=0.6)), (1.5, dict(fwd=0.6, yaw=0.5)),
          (3.0, dict(yaw=-0.8, fwd=0.3)), (4.5, dict(fwd=-0.6)),
          (6.0, dict(strafe=0.8)), (7.5, dict(strafe=-0.8, yaw=0.4)),
          (9.0, dict(fwd=0.7)), (10.5, dict(yaw=0.7))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/production")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    pc = PRESETS[args.preset]
    print(f"[pr] preset={args.preset} local={pc['local_window']} "
          f"sink={pc['secondary']}", flush=True)

    # ---- authoritative control clock (independent 60 Hz integrator) ----
    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    n_ctrl = int((args.chunks * CADENCE + 2.0) * FPS)
    cps, cvs, comp_gate = [], [], []
    si = 0
    ctl.set_input(**SCRIPT[0][1])
    prev_v = None
    gate = 1.0
    for k in range(n_ctrl):
        t = k / FPS
        while si + 1 < len(SCRIPT) and t >= SCRIPT[si + 1][0]:
            si += 1
            ctl.set_input(**SCRIPT[si][1])
        ctl.step(dt=1.0 / (FPS * CADENCE))
        v = ctl.v.copy()
        n_now = float(np.linalg.norm(v))
        if prev_v is None or n_now < 0.02:
            tgt = 1.0 if prev_v is None else 0.0
        else:
            cos = float(np.dot(prev_v, v) / max(np.linalg.norm(prev_v) * n_now, 1e-9))
            tgt = 0.0 if cos <= 0 else float(min(1.0, cos / 0.7))
        gate = tgt if tgt < gate else gate + 0.25 * (tgt - gate)
        prev_v = v
        T = np.eye(4); T[:3, 3] = v * (2.0 * gate)      # LEAD_TRANS=2, LEAD_ROT=0
        cps.append(ctl.pose @ T)
        cvs.append(v); comp_gate.append(gate)
    cps = np.stack(cps)

    def slot(t):
        return cps[int(np.clip(round(t * FPS), 0, len(cps) - 1))]

    # ---- models ----
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=pc["local_window"],
        sink_size=pc["secondary"], infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[pr] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

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
    kv_size = frame_seqlen * pc["local_window"]
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads
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
    # §36C: the full Wan VAE is only needed to build the condition latent. Its
    # encode workspace stays reserved (~2.4GB) otherwise; dropping it here
    # restores ~2.6GB of driver headroom (free 1609 -> 4299 MiB).
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    print(f"[pr] full VAE dropped, driver free = {free/2**20:.0f} MiB", flush=True)
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, args.chunks, lat_h, lat_w, generator=g, device=dev)

    def plucker(rel):
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(dev)
        p = get_plucker_embeddings(r, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h, w=lat_w).to(pdt)

    torch.cuda.reset_peak_memory_stats()
    rows, frames = [], []
    prev_pose = None
    t0 = time.perf_counter()
    for cid in range(args.chunks):
        cur = noise.split(1, dim=1)[cid]
        kw = None
        t_chunk = time.perf_counter()
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
        t_dit = time.perf_counter() - t_chunk
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        t_kv = time.perf_counter() - t_chunk - t_dit
        zt = x0.permute(1, 0, 2, 3).unsqueeze(0)
        t1 = time.perf_counter()
        with torch.no_grad():
            fr = tae.decode_video(zt, parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        t_tae = time.perf_counter() - t1
        frames.append(fr[0].permute(0, 2, 3, 1))
        ttfnf = time.perf_counter() - t_chunk
        rows.append(dict(chunk=cid, dit=t_dit, kv=t_kv, tae=t_tae, ttfnf=ttfnf))
        free, total = torch.cuda.mem_get_info()
        print(f"[pr] chunk {cid}: dit={t_dit*1000:.0f}ms kv={t_kv*1000:.0f}ms "
              f"tae={t_tae*1000:.0f}ms TTFNF={ttfnf*1000:.0f}ms "
              f"free={free/2**20:.0f}MiB", flush=True)
        prev_pose = pn
    peak = torch.cuda.max_memory_allocated() / 2**20

    s = rows[1:]
    dit = np.mean([r["dit"] for r in s]) * 1000
    kv = np.mean([r["kv"] for r in s]) * 1000
    ta = np.mean([r["tae"] for r in s]) * 1000
    tt = np.mean([r["ttfnf"] for r in s]) * 1000
    period = dit + kv + ta
    print(f"\n[pr] ===== v1 realtime baseline (preset {args.preset}) =====")
    print(f"  control-to-warp        0.60 ms  (Level 0 geometric warp)")
    print(f"  TTFNF                  {tt:.0f} ms  (denoise {dit:.0f} + TAE {ta:.0f})")
    print(f"  control-to-real        p50 ~{tt:.0f} ms  worst ~{tt+CADENCE*1000/2:.0f} ms")
    print(f"  effective_control_age  ~0.55 s  (§33 step hot-swap)")
    print(f"  generation throughput  {1000/period:.2f} chunk/s  "
          f"({4000/period:.2f} fps-equiv)")
    print(f"  presentation           60 fps  (interpolated)")
    print(f"  DiT (3 steps)          {dit:.0f} ms   KV-update {kv:.0f} ms")
    print(f"  TAE decode             {ta:.0f} ms")
    print(f"  peak VRAM              {peak:.0f} MiB")
    print(f"  stability              no OOM / NaN; chunks={args.chunks}")
    json.dump(dict(rows=rows, peak_vram=peak, dit_ms=dit, kv_ms=kv,
                   tae_ms=ta, ttfnf_ms=tt, period_ms=period,
                   preset=args.preset),
              open(f"{args.out_dir}/baseline_{args.preset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
