#!/usr/bin/env python
"""§37: free-play long runs (A 2min / B 5min / C 10min soak) on the frozen
production pipeline, with per-chunk instrumentation for system-health.

Per chunk it logs:
    driver free / reserved / allocated
    DiT / KV / TAE latency, TTFNF
    effective_control_age
    adaptive display state shares
    anomaly counters (NaN/Inf, latency spikes, VRAM growth, cache drift)

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python soak_loop.py --profile B --minutes 5
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

# profile -> (mean hold seconds, menu of inputs)
PROFILES = {
    "A": (3.0, [dict(fwd=0.5), dict(fwd=0.5, yaw=0.3), dict(yaw=0.4),
                dict(fwd=0.4, yaw=-0.3), dict(strafe=0.4), dict(fwd=0.6)]),
    "B": (0.9, [dict(yaw=0.9), dict(yaw=-0.9), dict(strafe=0.9),
                dict(strafe=-0.9), dict(fwd=0.8), dict(fwd=-0.8),
                dict(fwd=0.7, yaw=-0.7), dict(fwd=0.7, yaw=0.7),
                dict(fwd=-0.7, yaw=0.7), dict()]),
    "C": (2.0, [dict(fwd=0.5), dict(yaw=0.5), dict(yaw=-0.5), dict(strafe=0.5),
                dict(fwd=-0.5), dict(fwd=0.6, yaw=-0.4), dict(),
                dict(fwd=0.4, yaw=0.4)]),
}


def build_timeline(profile, minutes, seed):
    rng = np.random.default_rng(seed)
    hold, menu = PROFILES[profile]
    total = minutes * 60.0
    tl, t, i = [], 0.0, 0
    while t < total + 5.0:
        inp = menu[rng.integers(0, len(menu))]
        tl.append((t, inp))
        t += hold * (0.5 + rng.random())
        i += 1
    return tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", default="A", choices=list(PROFILES))
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/soak")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.profile}_{int(args.minutes)}min_s{args.seed}"
    timeline = build_timeline(args.profile, args.minutes, args.seed)
    print(f"[sk] profile {args.profile}, {args.minutes} min, "
          f"{len(timeline)} input segments", flush=True)

    # authoritative 60 Hz control clock
    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    n_ctrl = int((args.minutes * 60.0 + 8.0) * FPS)
    cps, cvs, gates = [], [], []
    si, prev_v, gate = 0, None, 1.0
    ctl.set_input(**timeline[0][1])
    for k in range(n_ctrl):
        t = k / FPS
        while si + 1 < len(timeline) and t >= timeline[si + 1][0]:
            si += 1
            ctl.set_input(**timeline[si][1])
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
        T = np.eye(4); T[:3, 3] = v * (2.0 * gate)
        cps.append(ctl.pose @ T); cvs.append(v); gates.append(gate)
    cps = np.stack(cps)

    def slot(t):
        return cps[int(np.clip(round(t * FPS), 0, len(cps) - 1))]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[sk] pipe + TAE built", flush=True)
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
    F = 81
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * 6
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
    pipe.vae = None                      # §36C fix
    gc.collect(); torch.cuda.empty_cache()
    free0, _ = torch.cuda.mem_get_info()
    print(f"[sk] VAE dropped, free={free0/2**20:.0f}MiB", flush=True)

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise_buf = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)

    def plucker(rel):
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(dev)
        p = get_plucker_embeddings(r, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h, w=lat_w).to(pdt)

    rows, anomalies = [], []
    saved_frames = []          # per chunk: first display frame of the 4 decoded
    saved_poses = []           # per chunk: the conditioning pose used
    prev_pose = None
    t0 = time.perf_counter()
    cid = 0
    deadline = args.minutes * 60.0
    dit_hist = []
    while (time.perf_counter() - t0) < deadline:
        g.manual_seed(args.seed + cid)
        cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
        tc = time.perf_counter()
        kw = None
        for ti in range(len(timesteps)):
            now = time.perf_counter() - t0
            pn = slot(now)
            rel = np.eye(4) if prev_pose is None else np.linalg.inv(prev_pose) @ pn
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[min(cid, F // 4 - 1)]],
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
        t_kv = time.perf_counter() - tc - t_dit
        zt = x0.permute(1, 0, 2, 3).unsqueeze(0)
        t1 = time.perf_counter()
        with torch.no_grad():
            fr = tae.decode_video(zt, parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        t_tae = time.perf_counter() - t1
        ttfnf = time.perf_counter() - tc
        free, _ = torch.cuda.mem_get_info()
        a = torch.cuda.memory_allocated() / 2**20
        r = torch.cuda.memory_reserved() / 2**20
        cond_age = (time.perf_counter() - t0) - now
        nan = int(torch.isnan(x0).any() or torch.isinf(x0).any())
        dit_hist.append(t_dit)
        med = float(np.median(dit_hist[-50:]))
        spike = int(t_dit > 2.0 * med) if len(dit_hist) > 10 else 0
        if nan or spike:
            anomalies.append(dict(chunk=cid, nan=nan, spike=spike, dit=t_dit))
        rows.append(dict(chunk=cid, t=time.perf_counter() - t0, dit=t_dit, kv=t_kv,
                         tae=t_tae, ttfnf=ttfnf, free=free / 2**20, alloc=a,
                         reserved=r, cond_age=cond_age, nan=nan, spike=spike))
        if cid % 10 == 0 or cid < 3:
            print(f"[sk] c{cid:4d} t={rows[-1]['t']:6.1f}s dit={t_dit*1000:5.0f} "
                  f"kv={t_kv*1000:4.0f} tae={t_tae*1000:4.0f} "
                  f"TTFNF={ttfnf*1000:5.0f} free={free/2**20:5.0f} "
                  f"age={cond_age*1000:5.0f}ms", flush=True)
        prev_pose = pn
        # stash the first decoded frame of this chunk for the offline display pass
        try:
            f0 = (fr[0][0].permute(1, 2, 0).float().cpu().numpy() * 255.0
                  ).clip(0, 255).astype(np.uint8)
            saved_frames.append(f0)
            saved_poses.append(pn)
        except Exception:
            pass
        cid += 1

    print(f"\n[sk] ===== §37 profile {args.profile} ({args.minutes} min) =====")
    s = rows[3:]
    def m(k):
        return float(np.mean([r[k] for r in s]))
    print(f"  chunks completed      {len(rows)}")
    print(f"  wall time             {rows[-1]['t']:.0f}s")
    print(f"  DiT / KV / TAE        {m('dit')*1000:.0f} / {m('kv')*1000:.0f} / "
          f"{m('tae')*1000:.0f} ms")
    print(f"  TTFNF                 {m('ttfnf')*1000:.0f} ms")
    print(f"  effective_control_age {m('cond_age')*1000:.0f} ms")
    print(f"  throughput            {1000/(m('dit')+m('kv')+m('tae')):.2f} chunk/s")
    print(f"  driver free           start={rows[0]['free']:.0f} end={rows[-1]['free']:.0f} "
          f"min={min(r['free'] for r in s):.0f} MiB")
    print(f"  reserved              start={rows[0]['reserved']:.0f} "
          f"end={rows[-1]['reserved']:.0f} max={max(r['reserved'] for r in s):.0f} MiB")
    print(f"  allocated             end={rows[-1]['alloc']:.0f} MiB")
    print(f"  anomalies             NaN={sum(r['nan'] for r in rows)} "
          f"latency_spikes={sum(r['spike'] for r in rows)}")
    # drift: first 20% vs last 20%
    k = max(5, len(s) // 5)
    d0 = np.mean([r["dit"] for r in s[:k]])
    d1 = np.mean([r["dit"] for r in s[-k:]])
    print(f"  DiT drift (first vs last 20%)  {d0*1000:.0f} -> {d1*1000:.0f} ms "
          f"({100*(d1/d0-1):+.1f}%)")
    f0 = np.mean([r["free"] for r in s[:k]])
    f1 = np.mean([r["free"] for r in s[-k:]])
    print(f"  free drift                     {f0:.0f} -> {f1:.0f} MiB "
          f"({100*(f1/f0-1):+.1f}%)")
    json.dump(dict(rows=rows, anomalies=anomalies, profile=args.profile,
                   minutes=args.minutes),
              open(f"{args.out_dir}/{tag}.json", "w"), indent=1)
    # save the frames + control timeline for the offline display pass
    if saved_frames:
        np.savez_compressed(f"{args.out_dir}/{tag}_frames.npz",
                            frames=np.stack(saved_frames),
                            poses=np.stack(saved_poses),
                            cps=cps, cvs=cvs, gates=np.array(gates))
        print(f"[sk] saved {len(saved_frames)} frames + timeline", flush=True)


if __name__ == "__main__":
    main()
