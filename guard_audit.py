#!/usr/bin/env python
"""P2: guard / recompilation audit for the causal-fast DiT under torch.compile.

WHY THIS EXISTS
---------------
`compile_ab2.py` showed mode="default" ~-9% and mode="reduce-overhead" ~-7%,
but those medians CONTAIN recompilation spikes (one chunk took 47.7 s under
reduce-overhead). A median that includes recompiles is a lower bound, and we
cannot call the compile路径 "validated" without knowing the steady-state
recompile count.

The suspicion is specific: `current_start` is a plain Python int that changes
every chunk (`chunk_id * frame_seqlen`). Dynamo specializes on Python ints, so
each new value is a new specialization -> a recompile per chunk. That matches
the `recompiles=257` seen in the earlier §7.3 experiment.

WHAT THIS MEASURES (per mode)
------------------------------
  * per-chunk latency (to expose recompile spikes directly)
  * steady-state recompiles (after the first warm chunks)
  * graph breaks
  * whether CUDA Graphs were actually captured/replayed (vs skipped)
  * peak allocated / reserved VRAM (cold and warm)

IT ALSO RUNS LONG ENOUGH TO CROSS THE LOCAL-WINDOW ROLLOVER
------------------------------------------------------------
local_attn_size=6 with one frame_seqlen of new tokens per chunk means the cache
is full at chunk 6 and EVICTION begins there. Cache position logic takes a
different branch from that point on, so any guard on the index state would only
fire after chunk 6. A 6-chunk benchmark cannot see that.

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCH_LOGS=recompiles,graph_breaks,perf_hints \
  python guard_audit.py --scene 04 --seed 42 --chunks 16 --mode default
"""
import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--out_dir", default="output/guards")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    dirty = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=repo).decode().strip().splitlines()
    print(f"[ga] HEAD={head[:12]} dirty={len(dirty)} files", flush=True)

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    print(f"[ga] py_cache_meta={getattr(pipe, '_py_cache_meta', None)}", flush=True)
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

    d = f"examples/ga_{scene}"
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

    def reset_caches():
        for c in self_kv:
            if torch.is_tensor(c["global_end_index"]):
                c["global_end_index"].zero_(); c["local_end_index"].zero_()
            else:
                c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False if not torch.is_tensor(c["is_init"]) else c["is_init"].zero_()
            c["k"].zero_(); c["v"].zero_()

    model = pipe.model
    if args.mode != "eager":
        torch._dynamo.reset()
        model = torch.compile(pipe.model, mode=args.mode, fullgraph=False)
    print(f"[ga] mode={args.mode}, {n_test} chunks, "
          f"window rollover expected at chunk {args.local_attn_size}", flush=True)

    def run(collect_per_chunk=False):
        reset_caches()
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        per = []
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
                  # NOTE: a fresh Python int every chunk. This is the suspected
                  # specialization trigger.
                  "current_start": cid * fsl,
                  "max_attention_size": kv_size, "frame_seqlen": fsl}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for ti in range(len(timesteps)):
                with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                    npred = model(x=[cur.to(dev)],
                                  t=torch.stack([timesteps[ti]]).to(dev),
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
                model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                      cross_attn_first_call=False, **kw)
            torch.cuda.synchronize()
            per.append(time.perf_counter() - t0)
        return per

    # --- cold pass: this is where compilation happens ---
    torch.cuda.reset_peak_memory_stats()
    cold = run()
    cold_peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    cold_peak_resv = torch.cuda.max_memory_reserved() / 2**20
    print(f"[ga] cold per-chunk ms: {[round(x*1000,1) for x in cold]}", flush=True)

    # --- warm pass: steady state, same caches/model ---
    torch.cuda.reset_peak_memory_stats()
    warm = run()
    warm_peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    warm_peak_resv = torch.cuda.max_memory_reserved() / 2**20
    print(f"[ga] warm per-chunk ms: {[round(x*1000,1) for x in warm]}", flush=True)

    warm_s = sorted(warm)
    print(f"[ga] warm median {statistics.median(warm)*1000:.1f} ms  "
          f"min {warm_s[0]*1000:.1f}  max {warm_s[-1]*1000:.1f}", flush=True)
    print(f"[ga] VRAM cold peak alloc {cold_peak_alloc:.0f} MiB / resv {cold_peak_resv:.0f} MiB",
          flush=True)
    print(f"[ga] VRAM warm peak alloc {warm_peak_alloc:.0f} MiB / resv {warm_peak_resv:.0f} MiB",
          flush=True)

    # --- dynamo counters ---
    counters = {}
    try:
        counters = {str(k): int(v) for k, v in torch._dynamo.utils.counters.items()
                    if isinstance(v, (int, float))}
    except Exception:
        pass
    recompiles = 0
    try:
        recompiles = int(torch._dynamo.utils.counters.get("stats", {}).get("calls_captured", 0))
    except Exception:
        pass
    print(f"[ga] dynamo counters: {json.dumps(counters, default=str)[:600]}", flush=True)

    json.dump(dict(mode=args.mode, head=head, dirty=dirty, n_chunks=n_test,
                   local_attn_size=args.local_attn_size, fsl=fsl,
                   cold_ms=[x*1000 for x in cold], warm_ms=[x*1000 for x in warm],
                   warm_median_ms=statistics.median(warm)*1000,
                   cold_peak_alloc_mib=cold_peak_alloc,
                   cold_peak_resv_mib=cold_peak_resv,
                   warm_peak_alloc_mib=warm_peak_alloc,
                   warm_peak_resv_mib=warm_peak_resv,
                   counters=counters),
              open(f"{args.out_dir}/guard_{args.mode}.json", "w"), indent=1,
              default=str)


if __name__ == "__main__":
    main()
