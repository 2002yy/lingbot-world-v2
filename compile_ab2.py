#!/usr/bin/env python
"""compile A/B round 2: does a NEWER compile mode help where §7.3 found none?

§7.3 tested `max-autotune-no-cudagraphs` on the DiT (2-step, 21 chunks) and saw
no gain (881.5 vs eager 913.6 ms, recompiles=257). The new external repo
lingbot-world-v2-realtime reports "eager -> single compiled graph" = 1.95 -> 1.68
s/chunk on the SAME base commit (1895d30), i.e. a real gain.

That is not a contradiction, it is a DIFFERENT configuration. What we never
tested:
    * mode="reduce-overhead"          (uses CUDA Graphs internally)
    * mode="default"                  (plain inductor)
    * capturing the whole denoise loop as one graph

This script runs a strictly paired A/B at our CURRENT production config
(3-step [0,250,750], local6/sink1) across:
    eager | default | reduce-overhead
and reports median latency + recompile counts.

Graph-capture obstacles found by reading the denoise loop (recorded here because
they define what a full-loop capture would require):
    1. torch.randn(..., generator=seed_g) INSIDE the loop: the RNG state differs
       per call, so a capture must pre-generate the noise or handle RNG
       correctly.
    2. the self/cross KV caches are MUTATED IN PLACE across chunks, while
       reduce-overhead/CUDA-Graph requires static inputs -> the cache would need
       refactoring into explicit in/out tensors.
    3. cross_attn_first_call toggles, giving two distinct graph shapes.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python compile_ab2.py --scene 04 --seed 42 --chunks 12
"""
import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import shutil
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
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--modes", default="eager,default,reduce-overhead")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/compile2")
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__))
                                       ).decode().strip()
    except Exception:
        head = "unknown"
    print(f"[c2] HEAD = {head}", flush=True)

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    modes = args.modes.split(",")

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[c2] pipe + TAE built", flush=True)
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

    d = f"examples/c2_{scene}"
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
    kv_size = fsl * 6
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    import torchvision.transforms.functional as TF
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
    print(f"[c2] latent {lat_h}x{lat_w}, testing {n_test} chunks, "
          f"modes {modes}", flush=True)

    p = np.load(f"examples/{scene}/poses.npy")
    traj = np.tile(p, (frames_n // len(p) + 1, 1, 1))[:frames_n]
    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    # Caches are allocated ONCE and reused across every chunk, matching the
    # production pattern (one allocation per generate(), then many chunks).
    # This matters for reduce-overhead/CUDA Graphs: graph replay writes to the
    # SAME buffer addresses, so freshly-allocated caches per call would make the
    # captured graph point at stale memory ("Expected curr_block->next ==
    # nullptr"). Reusing the buffers also means resetting the position state is
    # now a plain Python assignment (the host-sync refactor made it an int/bool).
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers,
        shape=[1, kv_size, ma.num_heads // pipe.sp_size, ma.dim // ma.num_heads],
        dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers,
        shape=[1, 512, ma.num_heads, ma.dim // ma.num_heads],
        dtype=dtype, device=dev)

    def reset_caches():
        for c in self_kv:
            c["global_end_index"] = 0
            c["local_end_index"] = 0
            c["k"].zero_()
            c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False
            c["k"].zero_()
            c["v"].zero_()

    def run_chunks(model):
        reset_caches()
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        lat = []
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
            t0 = time.perf_counter()
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
            lat.append(time.perf_counter() - t0)
        gc.collect(); torch.cuda.empty_cache()
        return lat

    results = {}
    for mode in modes:
        print(f"\n[c2] ===== mode={mode} =====", flush=True)
        model = pipe.model
        if mode != "eager":
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            try:
                model = torch.compile(pipe.model, mode=mode, fullgraph=False)
            except Exception as e:
                print(f"[c2]   compile FAILED: {type(e).__name__}: {e}", flush=True)
                results[mode] = dict(failed=str(e))
                continue
        try:
            # warmup (one chunk) then timed reps
            run_chunks(model)
            lat = []
            for _ in range(args.reps):
                lat += run_chunks(model)
            lat = sorted(lat)
            med = statistics.median(lat)
            results[mode] = dict(n=len(lat), median_ms=med * 1000,
                                 mean_ms=statistics.mean(lat) * 1000,
                                 min_ms=lat[0] * 1000, max_ms=lat[-1] * 1000)
            print(f"[c2]   {mode:18s} median {med*1000:7.1f} ms  "
                  f"(min {lat[0]*1000:.1f} / max {lat[-1]*1000:.1f})  n={len(lat)}",
                  flush=True)
        except Exception as e:
            print(f"[c2]   RUN FAILED: {type(e).__name__}: {e}", flush=True)
            results[mode] = dict(failed=f"{type(e).__name__}: {e}")

    print("\n[c2] ===== paired A/B (median per chunk, 3-step, "
          f"{n_test} chunks) =====")
    base = results.get("eager", {}).get("median_ms")
    for m in modes:
        r = results.get(m, {})
        if "median_ms" in r:
            d_ = f"{r['median_ms']-base:+.1f} ms ({100*(r['median_ms']-base)/base:+.1f}%)" \
                if base else "n/a"
            print(f"  {m:18s} {r['median_ms']:8.1f} ms   vs eager {d_}")
        else:
            print(f"  {m:18s} FAILED: {r.get('failed','?')}")

    json.dump(dict(results=results, head=head, n_chunks=n_test,
                   frames=frames_n, steps="[0,250,750]"),
              open(f"{args.out_dir}/compile2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
