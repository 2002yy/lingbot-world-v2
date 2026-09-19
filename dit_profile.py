#!/usr/bin/env python
"""Fresh DiT profile for THIS machine: where do the ~94% non-attention ms go?

WHY
---
S3 established that attention is only ~50 ms of a ~800 ms chunk (~6%). Every
ordering of future work depends on knowing the other ~94%, and the numbers
cannot be imported from the 5090 reference because its KV window (18) and
attention share are structurally different.

TWO ORTHOGONAL CUTS
-------------------
1. STRUCTURAL (module-level CUDA events, no per-call sync): attribute time to
   whichever module actually ran -- Linear, WanLayerNorm, self/cross attention,
   GELU. This answers "which component".
2. OPERATION TYPE (torch.profiler kernel table): group kernels into GEMM /
   elementwise / cast / reduction / attention / memory / other. This answers
   "what kind of work", which is what decides whether the next lever is FP8,
   fusion, or a cache.

The interesting hypothesis to test: each CausalWanAttentionBlock contains FOUR
camera-injection Linears (cam_injector_layer1/2, cam_scale_layer, cam_shift_layer),
and their input `c2ws_plucker_emb` is CONSTANT within a chunk while the block
runs 4 times per chunk. That is 4 x 30 x 4 = 480 Linear calls per chunk
recomputing the same thing.

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python dit_profile.py --scene 04 --seed 42 --chunks 6
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

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

MOD_MS = collections.defaultdict(float)
MOD_N = collections.defaultdict(int)
PENDING = []


def instrument_modules(model):
    """Wrap every nn.Module forward in CUDA events, keyed by class name."""
    def mk(cls):
        def pre(mod, args):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            mod._ev0 = ev
        def post(mod, args, out):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            PENDING.append((cls, mod._ev0, ev))
        return pre, post

    seen = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) > 0:
            continue                     # leaves only
        cls = type(mod).__name__
        if cls not in seen:
            seen[cls] = mk(cls)
        pre, post = seen[cls]
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
    return sorted(seen)


def drain(prefix=""):
    torch.cuda.synchronize()
    for cls, e0, e1 in PENDING:
        MOD_MS[prefix + cls] += e0.elapsed_time(e1)
        MOD_N[prefix + cls] += 1
    PENDING.clear()


def classify(name):
    n = name.lower()
    if any(k in n for k in ("gemm", "cutlass", "cublas", "sgemm", "gemv",
                            "matmul", "addmm", "nvjet")):
        return "GEMM"
    if any(k in n for k in ("flash", "sage", "fmha", "attention")):
        return "attention"
    if any(k in n for k in ("softmax", "reduce", "sum", "welford", "var_mean")):
        return "reduction/softmax"
    if any(k in n for k in ("cast", "to_copy", "copy", "memcpy", "memset",
                            "elementwise")):
        return "elementwise/cast/mem"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/ditprof")
    ap.add_argument("--mode", default="repro",
                    help="perf mode; repro = fa2+eager, the reproducibility baseline")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = args.mode
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
    print(f"[dp] LINGBOT_MODE={args.mode} perf_mode={pipe.perf_mode}", flush=True)
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

    d = f"examples/dp_{scene}"
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

    leaves = instrument_modules(pipe.model)
    print(f"[dp] instrumented leaf module classes: {leaves}", flush=True)

    def run_chunk(cid, profiler=None):
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
                                        device=dev, dtype=x0.dtype),
                        timesteps[ti + 1])
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)

    # warmup
    g = torch.Generator(device=dev); g.manual_seed(sd)
    reset(); pipe._cross_attn_initialized = False
    run_chunk(0); drain(); MOD_MS.clear(); MOD_N.clear()
    print("[dp] warmup done, counters cleared", flush=True)

    # ---------------- structural cut ----------------
    reset(); pipe._cross_attn_initialized = False
    g = torch.Generator(device=dev); g.manual_seed(sd)
    chunk_ms = []
    for cid in range(n_test):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_chunk(cid)
        torch.cuda.synchronize()
        chunk_ms.append((time.perf_counter() - t0) * 1000)
    drain()
    med = statistics.median(chunk_ms)
    mod_total = sum(MOD_MS.values())
    print(f"\n[dp] chunks={n_test} median chunk {med:.1f} ms")
    print(f"[dp] instrumented leaf-module GPU time {mod_total:.1f} ms "
          f"= {mod_total/n_test:.1f} ms/chunk ({100*mod_total/n_test/med:.1f}%)")
    print("\n[dp] ===== STRUCTURAL cut (leaf modules, ms/chunk) =====")
    print("   {:<28} {:>10} {:>9} {:>8}".format("class", "ms/chunk", "calls/ch", "%"))
    rows = sorted(MOD_MS.items(), key=lambda kv: -kv[1])
    struct = []
    for cls, ms in rows:
        n = MOD_N[cls]
        print("   {:<28} {:>10.2f} {:>9.0f} {:>7.1f}%".format(
            cls, ms / n_test, n / n_test, 100 * ms / n_test / med))
        struct.append(dict(cls=cls, ms_per_chunk=ms / n_test,
                           calls_per_chunk=n / n_test,
                           pct=100 * ms / n_test / med))

    # ---------------- op-type cut ----------------
    print("\n[dp] running torch.profiler for the op-type cut ...", flush=True)
    reset(); pipe._cross_attn_initialized = False
    g = torch.Generator(device=dev); g.manual_seed(sd)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for cid in range(n_test):
            run_chunk(cid)
    torch.cuda.synchronize()
    buckets = collections.defaultdict(float)
    kern = collections.defaultdict(float)
    for ev in prof.key_averages():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        t = ev.self_device_time_total
        buckets[classify(ev.key)] += t
        kern[ev.key] += t
    tot = sum(buckets.values())
    print(f"\n[dp] ===== OP-TYPE cut (CUDA kernels, {n_test} chunks) =====")
    print("   {:<22} {:>12} {:>9}".format("bucket", "ms/chunk", "% of kernels"))
    op = []
    for b, t in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print("   {:<22} {:>12.2f} {:>8.1f}%".format(b, t / n_test / 1000,
                                                     100 * t / tot))
        op.append(dict(bucket=b, ms_per_chunk=t / n_test / 1000,
                       pct=100 * t / tot))
    print(f"\n   total kernel time {tot/n_test/1000:.1f} ms/chunk "
          f"vs wall {med:.1f} ms/chunk -> GPU busy "
          f"{100*(tot/n_test/1000)/med:.0f}%")

    print("\n[dp] ===== top 25 kernels =====")
    print("   {:<60} {:>10} {:>7}".format("kernel", "ms/ch", "%"))
    for k, t in sorted(kern.items(), key=lambda kv: -kv[1])[:25]:
        print("   {:<60} {:>10.2f} {:>6.1f}%".format(
            k[:60], t / n_test / 1000, 100 * t / tot))

    json.dump(dict(mode=args.mode, scene=scene, seed=sd, n_chunks=n_test,
                   median_chunk_ms=med, structural=struct, op_type=op,
                   top_kernels=[dict(k=k, ms_per_chunk=t / n_test / 1000,
                                     pct=100 * t / tot)
                                for k, t in sorted(kern.items(),
                                                   key=lambda kv: -kv[1])[:60]],
                   kernel_total_ms_per_chunk=tot / n_test / 1000,
                   gpu_busy_pct=100 * (tot / n_test / 1000) / med),
              open(f"{args.out_dir}/ditprof.json", "w"), indent=1, default=str)
    print(f"\n[dp] wrote {args.out_dir}/ditprof.json")


if __name__ == "__main__":
    main()
