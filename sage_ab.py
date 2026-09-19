#!/usr/bin/env python
"""S3-A: single-variable A/B -- SageAttention 2.2 vs flash-attn 2, BOTH EAGER.

    A = current attention (flash-attn 2) + eager
    B = SageAttention 2.2 + eager

Nothing else changes: same seed, same control trajectory, same cache layout.
The backend is toggled at RUNTIME inside one process (sage_backend.set_enabled)
so the two arms cannot be confounded by process-level clock/warmup differences
-- the mistake that inflated an earlier measurement in this project.

We record, per arm:
  * per-chunk latency (median / p10 / p90)
  * peak VRAM allocated/reserved
  * the FULL latent trajectory, so we can measure how quantisation error
    accumulates over a causal rollout rather than judging a single cosine

Divergence is reported as a per-chunk curve: max|Δ| and cosine between the two
arms' latents, plus the FIRST chunk where they stop agreeing to within a
threshold. A world model is recurrent, so a per-step error that is tiny but
growing is a different risk from one that is tiny and flat.

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python sage_ab.py --scene 04 --seed 42 --chunks 32 --reps 1
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

import wan
import wan.modules.model_fast as mf
import wan.modules.sage_backend as sb
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def gpu_telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=power.draw,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=10).decode().strip()
        v = [x.strip() for x in out.split(",")]
        return dict(power_w=float(v[0]), sm_clk_mhz=float(v[1]), temp_c=float(v[2]))
    except Exception as e:
        return dict(err=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--min_kv", type=int, default=0)
    ap.add_argument("--skip_cross", type=int, default=0)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/sage_ab")
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
    print(f"[sa] pipe built; py_cache_meta={pipe._py_cache_meta}", flush=True)
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

    d = f"examples/sg_{scene}"
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

    def run(use_sage):
        sb.set_enabled(use_sage)
        sb.set_min_kv(args.min_kv)
        sb.set_skip_cross(bool(args.skip_cross))
        reset()
        sb.stats(reset=True)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        lat, outs = [], []
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
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            outs.append(x0.detach().float().cpu())
        return lat, outs, sb.stats()

    # warmup both arms once (kernel autotune, sage module load)
    run(False); run(True)

    order = []
    for r in range(args.reps):
        order += [False, True] if r % 2 == 0 else [True, False]
    res = {0: dict(lat=[], outs=None, st=[]), 1: dict(lat=[], outs=None, st=[])}
    for use_sage in order:
        tag = "SAGE" if use_sage else "FA2 "
        print(f"\n[sa] --- run {'B Sage' if use_sage else 'A FA2'} ---", flush=True)
        pre = gpu_telemetry()
        torch.cuda.reset_peak_memory_stats()
        lat, outs, st = run(use_sage)
        alloc = torch.cuda.max_memory_allocated() / 2**20
        resv = torch.cuda.max_memory_reserved() / 2**20
        post = gpu_telemetry()
        res[int(use_sage)]["lat"] += lat
        res[int(use_sage)]["st"].append(st)
        if res[int(use_sage)]["outs"] is None:
            res[int(use_sage)]["outs"] = outs
        print(f"[sa] {tag} median {statistics.median(lat)*1000:7.1f} ms  "
              f"peak {alloc:.0f}/{resv:.0f} MiB  "
              f"sage_taken={st['taken']} fell_through={st['fell_through']} "
              f"temp {post.get('temp_c','?')}C clk {post.get('sm_clk_mhz','?')}MHz",
              flush=True)

    a, b = res[0], res[1]
    ma_, mb = statistics.median(a["lat"]), statistics.median(b["lat"])
    print(f"\n[sa] ===== S3-A single-variable A/B (eager only) =====")
    print(f"  A  FA2  : median {ma_*1000:7.1f} ms  (n={len(a['lat'])})")
    print(f"  B  Sage : median {mb*1000:7.1f} ms  (n={len(b['lat'])})")
    print(f"  Δ = {(mb-ma_)*1000:+.1f} ms = {100*(mb-ma_)/ma_:+.2f}%")
    print(f"  sage dispatch: {b['st'][0]}")

    # ---- rollout divergence curve ----
    div = []
    if a["outs"] is not None and b["outs"] is not None:
        for i, (oa, ob) in enumerate(zip(a["outs"], b["outs"])):
            dd = (ob - oa).abs()
            cos = torch.nn.functional.cosine_similarity(
                ob.reshape(-1), oa.reshape(-1), dim=0).item()
            div.append(dict(chunk=i, max_abs=dd.max().item(),
                            mean_abs=dd.mean().item(), cosine=cos))
        print(f"\n[sa] ===== causal-rollout divergence (latent, per chunk) =====")
        print("{:>6} {:>12} {:>12} {:>10}".format("chunk", "max|d|", "mean|d|", "cosine"))
        for r in div:
            print("{:>6} {:>12.4e} {:>12.4e} {:>10.6f}".format(
                r["chunk"], r["max_abs"], r["mean_abs"], r["cosine"]))
        first = next((r for r in div if r["cosine"] < 0.999), None)
        print(f"\n[sa] first chunk with cosine < 0.999: "
              f"{first['chunk'] if first else 'never'}")
        c1 = div[0]["cosine"]; cl = div[-1]["cosine"]
        print(f"[sa] cosine chunk0 {c1:.6f} -> chunk{len(div)-1} {cl:.6f} "
              f"({'STABLE/IMPROVING' if cl >= c1 else 'DEGRADING'})")

    json.dump(dict(head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(
                        os.path.abspath(__file__))).decode().strip(),
                   min_kv=args.min_kv, skip_cross=args.skip_cross,
                   n_chunks=n_test, reps=args.reps,
                   fa2_ms=[x * 1000 for x in a["lat"]],
                   sage_ms=[x * 1000 for x in b["lat"]],
                   fa2_median_ms=ma_ * 1000, sage_median_ms=mb * 1000,
                   delta_pct=100 * (mb - ma_) / ma_,
                   sage_stats=b["st"], divergence=div),
              open(f"{args.out_dir}/sage_ab.json", "w"), indent=1, default=str)
    print(f"\n[sa] wrote {args.out_dir}/sage_ab.json")


if __name__ == "__main__":
    main()
