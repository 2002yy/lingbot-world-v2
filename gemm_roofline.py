#!/usr/bin/env python
"""P1-Pre: real-shape GEMM roofline on THIS machine.

WHY NOT "percent of theoretical peak"
-------------------------------------
Laptop power limits, boost state, torch/cuBLAS version, matrix shape and dtype
all push a theoretical peak percentage away from being actionable. What actually
decides whether FP8 is worth the numerical risk is the measured speed ratio
between configs at the REAL shapes, plus the Amdahl projection onto a chunk.

M IS MEASURED, NOT GUESSED
--------------------------
The Linear input is flattened to [M, K] x [K, N] with M = batch x tokens.
Whether these GEMMs are compute-bound depends on M, so M is captured from a real
forward pass rather than assumed.

THREE CONFIGS PER SHAPE
-----------------------
    A  bf16 F.linear                       (what production does today)
    B  torchao Float8WeightOnlyConfig      (what LINGBOT_FP8 applies today)
    C  torchao Float8DynamicActivation...  (rowwise FP8 *compute*)

B matters: the source says weight-only frees VRAM but still runs a bf16 GEMM, so
its compute gain should be ~0. Measuring it turns that source reading into a
fact about this GPU. If B is not faster than A, then the 1.30 GiB we free is a
pure memory win and the compute lever is entirely C.

DECISION MATRIX (applied to the FFN shapes, which are 47.9% of a chunk)
    C/A >= 1.30x -> go to P1 integration
    1.15-1.30x   -> worth it, but estimate end-to-end first
    1.05-1.15x   -> quantisation/cast overhead may eat it
    <= 1.05x     -> stop P1, work on the FFN kernel instead

Run:
  LINGBOT_FP8=0 python gemm_roofline.py --scene 04 --chunks 1
"""
import argparse
import collections
import gc
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
CAPTURED = []
CAM_MODULES = {"cam_injector_layer1", "cam_injector_layer2",
               "cam_scale_layer", "cam_shift_layer"}


def role_for(path):
    p = re.sub(r"\.\d+\.", ".*.", path)
    p = re.sub(r"\.\d+$", ".*", p)
    tail = p.split(".")[-1]
    if tail in CAM_MODULES:
        return "cam." + tail
    if "self_attn" in p:
        return "self_attn." + tail
    if "cross_attn" in p:
        return "cross_attn." + tail
    if ".ffn." in p:
        return "ffn." + tail
    return None


def capture_shapes(model):
    def hook(mod, args):
        x = args[0] if args else None
        if torch.is_tensor(x) and x.dim() >= 2:
            M = int(np.prod(x.shape[:-1]))
            CAPTURED.append(dict(role=mod._role, M=M,
                                 K=int(x.shape[-1]),
                                 N=int(mod.out_features),
                                 bt=tuple(x.shape)))
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            r = role_for(name)
            if r is None:
                continue
            mod._role = r
            mod.register_forward_pre_hook(hook)
            n += 1
    return n


def bench(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def tflops(M, K, N, ms):
    return (2.0 * M * K * N) / (ms * 1e-3) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/gemm_roofline")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    # Keep the production FP8 setting: weight-only FP8 changes the weight FORMAT,
    # not any tensor SHAPE, so the captured shapes are identical and we avoid the
    # extra ~2.6 GiB that plain bf16 weights need (which OOM'd on this 8 GB card
    # when an external process held ~4 GiB).
    os.environ["LINGBOT_FP8"] = "1"
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
    print(f"[gr] torch {torch.__version__} cap {torch.cuda.get_device_capability()} "
          f"param_dtype {pdt}", flush=True)
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

    d = f"examples/gr_{scene}"
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

    n_lin = capture_shapes(pipe.model)
    print(f"[gr] hooked {n_lin} Linear modules", flush=True)

    g = torch.Generator(device=dev); g.manual_seed(sd)
    reset(); pipe._cross_attn_initialized = False
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

    # dedupe captured shapes by (role, M, K, N)
    uniq = collections.OrderedDict()
    for c in CAPTURED:
        k = (c["role"], c["M"], c["K"], c["N"])
        uniq[k] = uniq.get(k, 0) + 1
    print(f"\n[gr] ===== captured Linear shapes ({len(uniq)} unique) =====")
    print("   {:<30} {:>7} {:>7} {:>7} {:>7}".format(
        "role", "M", "K", "N", "calls"))
    for (role, M, K, N), n in sorted(uniq.items(), key=lambda kv: -kv[0][0].count("ffn")):
        print("   {:<30} {:>7} {:>7} {:>7} {:>7}".format(role, M, K, N, n))

    # ---------------- roofline microbench ----------------
    print("\n[gr] ===== roofline: A=bf16  B=FP8 weight-only  C=FP8 rowwise compute =====",
          flush=True)
    try:
        import torchao
        from torchao.quantization import (
            Float8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig,
            quantize_)
        print(f"[gr] torchao {getattr(torchao, '__version__', '?')}", flush=True)
    except Exception as e:
        print(f"[gr] torchao unavailable: {e}")
        return

    targets = [
        ("ffn.0", 8960), ("ffn.2", 1536),
        ("self_attn.q", 1536), ("self_attn.o", 1536),
        ("cross_attn.q", 1536), ("cam.cam_injector_layer1", 1536),
    ]
    seen = set()
    plan = []
    for role, N in targets:
        for (r2, M, K, N2), n in uniq.items():
            if r2 == role and (role, M, K, N2) not in seen:
                seen.add((role, M, K, N2))
                plan.append((role, M, K, N2))

    results = []
    print("\n   {:<26} {:>6} {:>6} {:>6} {:>9} {:>9} {:>9} {:>8} {:>8}".format(
        "role", "M", "K", "N", "A bf16", "B WO", "C rowwise", "C/A", "A TFLOPs"))
    print("   " + "-" * 104)
    for role, M, K, N in plan:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        lin = nn.Linear(K, N, bias=False).to(dev, torch.bfloat16)
        with torch.no_grad():
            lin.weight.normal_(0, 0.02)

        def a_fn():
            return lin(x)

        # B: weight-only
        try:
            import copy
            lin_b = copy.deepcopy(lin)
            quantize_(lin_b, Float8WeightOnlyConfig())
            def b_fn():
                return lin_b(x)
        except Exception as e:
            b_fn = None
            print(f"   B unavailable for {role}: {type(e).__name__}: {str(e)[:50]}")

        # C: rowwise FP8 compute
        try:
            import copy
            lin_c = copy.deepcopy(lin)
            quantize_(lin_c, Float8DynamicActivationFloat8WeightConfig())
            def c_fn():
                return lin_c(x)
        except Exception as e:
            c_fn = None
            print(f"   C unavailable for {role}: {type(e).__name__}: {str(e)[:50]}")

        ta = bench(a_fn)
        tb = bench(b_fn) if b_fn else float("nan")
        tc = bench(c_fn) if c_fn else float("nan")
        ratio = (ta / tc) if c_fn else float("nan")
        print("   {:<26} {:>6} {:>6} {:>6} {:>9.4f} {:>9.4f} {:>9.4f} "
              "{:>7.2f}x {:>8.1f}".format(
                  role, M, K, N, ta, tb, tc, ratio, tflops(M, K, N, ta)),
              flush=True)
        results.append(dict(role=role, M=M, K=K, N=N,
                            a_bf16_ms=ta, b_wo_ms=tb, c_rowwise_ms=tc,
                            c_over_a=ratio,
                            a_tflops=tflops(M, K, N, ta),
                            c_tflops=tflops(M, K, N, tc) if c_fn else None))

    # Amdahl projections using the measured FFN traffic and the chunk split
    ffn = [r for r in results if r["role"] in ("ffn.0", "ffn.2")
           and not math.isnan(r["c_over_a"])]
    if ffn:
        ratio = statistics.mean(r["c_over_a"] for r in ffn)
        print(f"\n[gr] mean FP8 rowwise speedup on FFN shapes: {ratio:.3f}x")
        for share, label in ((0.479, "FFN only (measured 47.9%)"),
                             (0.787, "all Linear (measured 78.7%)")):
            gain = share * (1 - 1 / ratio)
            print(f"[gr] Amdahl if {label}: -{100*gain:.1f}% of chunk time")

    json.dump(dict(shapes=[dict(role=r, M=M, K=K, N=N, calls=n)
                           for (r, M, K, N), n in uniq.items()],
                   roofline=results),
              open(f"{args.out_dir}/gemm_roofline.json", "w"), indent=1,
              default=str)
    print(f"\n[gr] wrote {args.out_dir}/gemm_roofline.json")


if __name__ == "__main__":
    main()
