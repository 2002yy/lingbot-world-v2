#!/usr/bin/env python
"""P2b-0: production-faithful attribution of the 333.19 ms FFN-down budget.

The question is NOT "optimise FFN-down" but "why is it slow", and the first thing
to establish is whether the 333.19 ms is even the matrix multiply.

The discipline from RoPE applies here in full. This probe mirrors production
rather than an idealised version:

    shape            captured from a real ffn.2 call, not assumed
    dtype            captured
    device           captured
    stride           captured (a transposed or non-contiguous input selects a
                     different kernel, and that is exactly what we are hunting)
    contiguous       captured
    autocast state   captured (bf16 autocast changes the accumulate path)
    accumulate dtype captured
    stream           the default compute stream, as in production
    warmup           present before every timed region
    call frequency   120 per chunk for ffn.2 (30 blocks x 4 forwards), and the
                     REAL shape mix is M=1881 throughout a chunk_size=3 chunk,
                     not a lone M=3762

CUPTI is unavailable on this box (CUPTI_ERROR_INVALID_DEVICE, 0 CUDA events),
and ncu/nsys/nvprof are absent, so kernel names, tile shapes and occupancy
cannot be read directly. The substitutes used here are:

  1. capture the exact production input and time that GEMM verbatim
  2. compare 120 x isolated against the in-model attribution -> how much of the
     333.19 ms is NOT the GEMM
  3. measure the empirical hardware ceiling with large square GEMMs -> is the
     achieved throughput near the limit, or is there headroom
"""
import argparse
import gc
import hashlib
import json
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


def bench(fn, n=100, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n          # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/p2b0")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd, CS = args.scene, args.seed, args.chunk_size

    print("=" * 78)
    print("  P2b-0 production-faithful attribution of FFN-down")
    print("=" * 78, flush=True)

    # ---------------------------------------------------------- ceiling first
    print("\n=== A) empirical hardware ceiling (square bf16 GEMMs) ===")
    ceilings = {}
    for n in (1024, 2048, 4096, 8192):
        a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")

        def f(a=a, b=b):
            return a @ b
        t = bench(f, n=30, warm=5)
        tf = 2 * n ** 3 / (t * 1e-3) / 1e12
        ceilings[n] = dict(ms=t, tflops=tf)
        print(f"  {n:5d}^3  {t:8.3f} ms   {tf:7.1f} TFLOP/s")

    def f32c(a=a, b=b):
        return a.float() @ b.float()
    a8 = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    b8 = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    t = bench(lambda: a8.float() @ b8.float(), n=10, warm=3)
    print(f"  4096^3 fp32-cast {t:8.3f} ms   "
          f"{2*4096**3/(t*1e-3)/1e12:7.1f} TFLOP/s (for reference)")
    del a8, b8
    gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------ real model
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    head = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()

    # ---- capture every ffn.2 call: shape, dtype, stride, contiguity, autocast
    CAP = {"on": False, "calls": [], "times": {}}
    blocks = pipe.model.blocks

    def wrap(bi, lin):
        orig = lin.forward

        def fwd(x, *a, **kw):
            if CAP["on"]:
                CAP["calls"].append(dict(
                    block=bi,
                    shape=list(x.shape),
                    dtype=str(x.dtype),
                    stride=list(x.stride()),
                    contiguous=x.is_contiguous(),
                    device=str(x.device),
                    autocast=str(torch.is_autocast_enabled()),
                    ac_dtype=str(torch.get_autocast_gpu_dtype()),
                    w_shape=list(lin.weight.shape),
                    w_dtype=str(lin.weight.dtype),
                    b=(lin.bias is not None),
                ))
            return orig(x, *a, **kw)
        return fwd

    for i, blk in enumerate(blocks):
        blk.ffn[2].forward = wrap(i, blk.ffn[2])

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lat_needed = args.chunks * CS
    frames_n = (lat_needed - 1) * 4 + 1
    frames_n = ((frames_n - 1) // 4) * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))
    n_test = min(args.chunks, lat_f // CS)

    d = f"examples/p2b0_{scene}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    th = int(np.sqrt(W * H * (480 / 832)) // 8 * 8)
    tw = int(np.sqrt(W * H / (480 / 832)) // 8 * 8)
    img = (torch.nn.functional.interpolate(
        torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None].float(),
        size=(th, tw), mode='bicubic').squeeze(0) / 255.0 - 0.5) / 0.5
    h, w = img.shape[1:]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = CS * fsl
    kv_size = fsl * args.local_attn_size
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=CS)
    mf.bump_cam_epoch()
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
        np.linspace(0, frames_n - 1, lat_f)).to(dev)
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
    for c in self_kv:
        c["global_end_index"] = 0; c["local_end_index"] = 0
    for c in cross_kv:
        c["is_init"] = False

    def reset():
        for c in self_kv:
            c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False
            c["k"].zero_(); c["v"].zero_()

    def run(capture_all=False, time_ffn2=False):
        reset()
        pipe._cross_attn_initialized = False
        gg = torch.Generator(device=dev); gg.manual_seed(sd)
        chunk_ms = []
        for cid in range(n_test):
            c0 = cid * CS
            cur = torch.randn(16, CS, lat_h, lat_w, generator=gg, device=dev)
            pp = get_plucker_embeddings(rel_all[c0:c0 + CS], Ks[None], h, w)
            pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                           c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=CS,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(CS, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * CS * fsl,
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
                            x0, torch.randn(x0.shape, generator=gg,
                                            device=dev, dtype=x0.dtype),
                            timesteps[ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            torch.cuda.synchronize()
            chunk_ms.append((time.perf_counter() - t0) * 1000)
        return chunk_ms

    # warmup (also primes everything)
    run()

    print("\n=== B) production capture of ffn.2 ===")
    CAP["on"] = True
    CAP["calls"] = []
    chunk_ms = run()
    CAP["on"] = False
    calls = CAP["calls"]
    per_chunk = len(calls) / n_test
    print(f"  ffn.2 calls captured : {len(calls)} over {n_test} chunks "
          f"= {per_chunk:.1f} per chunk")
    shapes = {}
    for c in calls:
        k = (tuple(c["shape"]), c["dtype"], c["contiguous"], tuple(c["stride"]),
             c["autocast"], c["ac_dtype"], c["w_dtype"], c["b"])
        shapes[k] = shapes.get(k, 0) + 1
    print(f"  distinct call signatures: {len(shapes)}")
    for k, cnt in sorted(shapes.items(), key=lambda kv: -kv[1]):
        sh, dt, cont, st, ac, acd, wdt, hasb = k
        print(f"    x{cnt:<5} shape={sh} dtype={dt} contiguous={cont}")
        print(f"           stride={st}")
        print(f"           autocast_enabled={ac} autocast_dtype={acd}")
        print(f"           weight_dtype={wdt} bias={hasb}")
    print(f"  chunk time (median)   : {statistics.median(chunk_ms):.1f} ms")

    # ------------------------------------ isolated GEMM on the captured input
    print("\n=== C) isolated GEMM, verbatim production tensors ===")
    lin = blocks[0].ffn[2]
    print(f"  module          : {type(lin).__name__}")
    print(f"  weight shape    : {list(lin.weight.shape)} "
          f"dtype={lin.weight.dtype} type={type(lin.weight).__name__}")
    print(f"  in_features     : {getattr(lin, 'in_features', '?')}")
    print(f"  out_features    : {getattr(lin, 'out_features', '?')}")
    K_in = int(getattr(lin, "in_features", 8960))
    N_out = int(getattr(lin, "out_features", 1536))
    print(f"  -> this is the {'down' if K_in > N_out else 'up'} projection "
          f"(K={K_in} N={N_out})")
    results = {}
    for M in (627, 1881, 3762):
        xin = torch.randn(M, K_in, dtype=torch.float32, device=dev)
        xb = xin.to(torch.bfloat16)

        def f_mod(xb=xb):
            return lin(xb)
        t = bench(f_mod, n=60)
        fl = 2 * M * K_in * N_out
        results[M] = dict(ms=t, tflops=fl / (t * 1e-3) / 1e12, K=K_in, N=N_out)
        print(f"  M={M:<5} {t:7.3f} ms/call  {fl/(t*1e-3)/1e12:6.1f} TFLOP/s"
              f"   x120 = {t*120:7.1f} ms/chunk")
        del xin, xb
        gc.collect(); torch.cuda.empty_cache()

    print("\n=== D) the key arithmetic ===")
    t1881 = results[1881]["ms"]
    print(f"  isolated ffn.2 at production M=1881 x 120 calls")
    print(f"    = {t1881*120:.1f} ms/chunk")
    print(f"  in-model attribution (P1c/P1b): 333.19 ms/chunk")
    print(f"  unexplained by the GEMM itself : {333.19 - t1881*120:.1f} ms/chunk"
          f"  ({(333.19 - t1881*120)/333.19*100:.0f}% of the bucket)")
    print()
    best = max(ceilings.values(), key=lambda v: v["tflops"])["tflops"]
    print(f"  best square-GEMM throughput    : {best:.1f} TFLOP/s")
    print(f"  ffn.2 achieved at M=1881       : {results[1881]['tflops']:.1f} TFLOP/s"
          f"   ({results[1881]['tflops']/best*100:.0f}% of the empirical ceiling)")

    out = dict(head=head, n_test=n_test, per_chunk_calls=per_chunk,
               signature_count=len(shapes),
               signatures=[dict(shape=list(k[0]), dtype=k[1], contiguous=k[2],
                                stride=list(k[3]), autocast=k[4],
                                autocast_dtype=k[5], weight_dtype=k[6],
                                bias=k[7], count=v)
                           for k, v in shapes.items()],
               chunk_ms_median=statistics.median(chunk_ms),
               chunk_ms=chunk_ms, isolated=results, ceilings=ceilings,
               attributed_bucket_ms=333.19)
    with open(os.path.join(args.out_dir, "p2b0.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[P2b-0] wrote {args.out_dir}/p2b0.json")


if __name__ == "__main__":
    main()
