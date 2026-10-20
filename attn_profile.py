#!/usr/bin/env python
"""S3-A step 1: where does the DiT time actually go, and what SHAPES are the
attention calls?

THE QUESTION THE USER ACTUALLY WANTS ANSWERED
---------------------------------------------
DiT is ~750 ms/chunk. If attention is only ~50 ms of that, SageAttention is a
minor optimisation no matter how good the kernel is. If it is 150-250 ms, this
line can move the whole thing a tier.

So before touching any backend we instrument the REAL model and report:

  * every distinct attention shape: (Lq, Lkv, H, D, dtype), call count,
    cumulative GPU ms, and share of total DiT time
  * whether q/k/v arrive already contiguous in the layout SageAttention wants
    (NHD). If a transpose().contiguous() were needed, the production-shape
    advantage is only ~0.155 ms per call (0.629 -> 0.474) and one extra
    materialise can eat most of it -- so this must be measured, not assumed.
  * self-attention vs cross-attention split (they have very different shapes)

TIMING METHOD
-------------
Per-call CUDA events rather than torch.cuda.synchronize() around each call, so
the measurement does not serialise the pipeline and inflate the numbers.

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  source cuda_env.sh   # not needed for profiling, but harmless
  python attn_profile.py --scene 04 --seed 42 --chunks 8
"""
import argparse
import collections
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

import wan
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

# ---------------------------------------------------------------- instrumentation
STATS = collections.defaultdict(lambda: dict(calls=0, ms=0.0, contig=[0, 0, 0]))
KIND = {}


def describe(t, name):
    return (name, tuple(t.shape), str(t.dtype).replace("torch.", ""),
            bool(t.is_contiguous()))


def instrument():
    orig_flash = mf.flash_attention
    orig_attn = mf.attention

    def wrap(fn, tag):
        def inner(q, k, v, *a, **kw):
            # record shape key. q,k,v are [B, L, H, D] (NHD) at both call sites.
            try:
                key = (tag, int(q.shape[1]), int(k.shape[1]),
                       int(q.shape[2]), int(q.shape[3]),
                       str(q.dtype).replace("torch.", ""))
            except Exception:
                key = (tag, -1, -1, -1, -1, "?")
            s = STATS[key]
            s["calls"] += 1
            s["contig"][0] += int(bool(q.is_contiguous()))
            s["contig"][1] += int(bool(k.is_contiguous()))
            s["contig"][2] += int(bool(v.is_contiguous()))
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = fn(q, k, v, *a, **kw)
            ev1.record()
            PENDING.append((key, ev0, ev1))
            return out
        return inner

    mf.flash_attention = wrap(orig_flash, "flash_attn")
    mf.attention = wrap(orig_attn, "attention")
    return orig_flash, orig_attn


PENDING = []


def drain():
    torch.cuda.synchronize()
    for key, ev0, ev1 in PENDING:
        STATS[key]["ms"] += ev0.elapsed_time(ev1)
    PENDING.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/attnprof")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    frames_n = (args.frames - 1) // 4 * 4 + 1
    n_lat = (frames_n - 1) // 4 + 1
    n_test = min(args.chunks, n_lat)

    d = f"examples/ap_{scene}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    img = (torch.nn.functional.interpolate(
        torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None].float(),
        size=(int(np.sqrt(W * H * (480 / 832)) // 8 * 8),
              int(np.sqrt(W * H / (480 / 832)) // 8 * 8)),
        mode='bicubic').squeeze(0) / 255.0 - 0.5) / 0.5
    h, w = img.shape[1:]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = int(math.ceil(fsl / pipe.sp_size)) * pipe.sp_size
    kv_size = fsl * args.local_attn_size
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        img[None].transpose(0, 1).to(dev),
        torch.zeros(3, frames_n - 1, h, w, device=dev)], dim=1)])[0]
    msk = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y]).detach()
    del img
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()

    p = np.load(f"examples/{scene}/poses.npy")
    traj = np.tile(p, (frames_n // len(p) + 1, 1, 1))[:frames_n]
    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers,
        shape=[1, kv_size, ma.num_heads // pipe.sp_size, ma.dim // ma.num_heads],
        dtype=dtype, device=dev, python_metadata=pipe._py_cache_meta)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers,
        shape=[1, 512, ma.num_heads, ma.dim // ma.num_heads],
        dtype=dtype, device=dev, python_metadata=pipe._py_cache_meta)

    def reset():
        for c in self_kv:
            c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False
            c["k"].zero_(); c["v"].zero_()

    orig_flash, orig_attn = instrument()
    print(f"[ap] instrumented. fsl={fsl} kv_size={kv_size} "
          f"lat={lat_h}x{lat_w} layers={ma.num_layers} heads={ma.num_heads} "
          f"dim={ma.dim}", flush=True)

    # warmup (also drains first-call compilation)
    for warm in range(2):
        reset()
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        for cid in range(2):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            pp = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                           c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * fsl,
                  "max_attention_size": kv_size, "frame_seqlen": fsl}
            for ti in range(len(timesteps)):
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
                            x0, torch.randn(x0.shape, generator=g,
                                            device=x0.device, dtype=x0.dtype),
                            timesteps[ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
        drain()
    STATS.clear()
    print("[ap] warmup done, stats cleared. profiling...", flush=True)

    # ---- timed run ----
    reset()
    pipe._cross_attn_initialized = False
    g = torch.Generator(device=dev); g.manual_seed(sd)
    chunk_ms = []
    for cid in range(n_test):
        cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
        pp = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
        pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                       c1=int(h // lat_h), c2=int(w // lat_w))[None]
        plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=1,
                        h=lat_h, w=lat_w).to(pdt)
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
              "y": [y.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
              "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
              "kv_cache": self_kv, "crossattn_cache": cross_kv,
              "current_start": cid * fsl,
              "max_attention_size": kv_size, "frame_seqlen": fsl}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for ti in range(len(timesteps)):
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
                        x0, torch.randn(x0.shape, generator=g,
                                        device=x0.device, dtype=x0.dtype),
                        timesteps[ti + 1])
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        drain()
        torch.cuda.synchronize()
        chunk_ms.append((time.perf_counter() - t0) * 1000)

    total_attn = sum(s["ms"] for s in STATS.values())
    med_chunk = statistics.median(chunk_ms)
    print(f"\n[ap] chunks profiled: {n_test}, median chunk {med_chunk:.1f} ms")
    print(f"[ap] cumulative attention GPU time: {total_attn:.1f} ms over "
          f"{n_test} chunks = {total_attn/n_test:.1f} ms/chunk")
    print(f"[ap] attention share of chunk: {100*total_attn/n_test/med_chunk:.1f}%")

    rows = sorted(STATS.items(), key=lambda kv: -kv[1]["ms"])
    print("\n{:<12} {:>6} {:>7} {:>5} {:>4} {:>8} {:>7} {:>11} {:>10}".format(
        "kind", "Lq", "Lkv", "H", "D", "dtype", "calls", "cum_ms", "ms/chunk"))
    print("-" * 82)
    for (tag, Lq, Lkv, Hd, Dd, dt), s in rows:
        print("{:<12} {:>6} {:>7} {:>5} {:>4} {:>8} {:>7} {:>11.1f} {:>10.2f}"
              .format(tag, Lq, Lkv, Hd, Dd, dt, s["calls"], s["ms"],
                      s["ms"] / n_test), flush=True)

    print("\n[ap] contiguity (how many of the calls got contiguous q/k/v):")
    for (tag, Lq, Lkv, Hd, Dd, dt), s in rows:
        n = s["calls"]
        print(f"  {tag:12s} Lq={Lq:<5} Lkv={Lkv:<6} H={Hd:<3} "
              f"q={100*s['contig'][0]/n:.0f}% k={100*s['contig'][1]/n:.0f}% "
              f"v={100*s['contig'][2]/n:.0f}%")

    json.dump(dict(chunk_ms=chunk_ms, median_chunk_ms=med_chunk,
                   total_attn_ms=total_attn, n_chunks=n_test,
                   attn_ms_per_chunk=total_attn / n_test,
                   attn_share_pct=100 * total_attn / n_test / med_chunk,
                   shapes=[dict(kind=k[0], Lq=k[1], Lkv=k[2], H=k[3], D=k[4],
                                dtype=k[5], calls=v["calls"], ms=v["ms"],
                                contiguous_q=v["contig"][0],
                                contiguous_k=v["contig"][1],
                                contiguous_v=v["contig"][2])
                           for k, v in rows],
                   fsl=fsl, kv_size=kv_size, layers=ma.num_layers,
                   heads=ma.num_heads, dim=ma.dim),
              open(f"{args.out_dir}/attnprof.json", "w"), indent=1, default=str)
    print(f"\n[ap] wrote {args.out_dir}/attnprof.json")


if __name__ == "__main__":
    main()
