#!/usr/bin/env python
"""DiT profile v2: split the 642 ms/chunk of Linear by MODULE ROLE.

v1 answered "which class" (Linear = 75.5% of chunk time, 1486 calls/chunk).
That is not actionable on its own. This splits it by role so we can see where
the GEMM time actually sits:

    self_attn.q / .k / .v / .o
    cross_attn.q / .k / .v / .o
    ffn.0 / ffn.2
    cam_injector_layer1 / cam_injector_layer2 / cam_scale_layer / cam_shift_layer

TWO HYPOTHESES UNDER TEST
-------------------------
1. **Weight-only FP8.** lingbot_fp8.py applies torchao's
   `Float8WeightOnlyConfig`, i.e. weights are stored FP8 and dequantised for the
   GEMM. That is a VRAM win (~1.30 GiB) with NO compute win. The 5090 reference
   used rowwise FP8 *compute* (turbo 1.68 -> 1.47 s/chunk, -13%). If this profile
   shows bf16-class GEMM throughput here, that lever is unexploited.

2. **Redundant camera-injection GEMMs.** Each CausalWanAttentionBlock has four
   camera Linears whose input `c2ws_plucker_emb` is CONSTANT within a chunk,
   while the block forward runs 4x per chunk (3 denoise steps + 1 KV update).
   If that is a large share of Linear time, caching it removes the work with no
   numerical change at all -- same input, same weights -> identical output, so
   it can stay in `repro` mode.

Reports per-role ms/chunk, calls/chunk, and ms/call (mean), plus the redundancy
factor for the camera modules (how many of the calls are provably repeat work).
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
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

MOD_MS = collections.defaultdict(float)
MOD_N = collections.defaultdict(int)
PENDING = []
ROLE_OF = {}
CAM_MODULES = {"cam_injector_layer1", "cam_injector_layer2",
               "cam_scale_layer", "cam_shift_layer"}


def role_for(path):
    """Collapse a module path into a comparable role name."""
    p = re.sub(r"\.\d+\.", ".*.", path)      # blocks.12. -> blocks.*.
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
    if "time_embedding" in p or "time_projection" in p:
        return "time_emb." + tail
    if "text_embedding" in p:
        return "text_emb." + tail
    if tail == "head":
        return "head"
    return p


def instrument(model):
    def pre(mod, args):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        mod._ev0 = ev
    def post(mod, args, out):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        PENDING.append((mod._role, mod._ev0, ev))

    roles = set()
    for name, mod in model.named_modules():
        if len(list(mod.children())) > 0:
            continue
        mod._role = role_for(name)
        roles.add(mod._role)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
    return sorted(roles)


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
    ap.add_argument("--mode", default="repro")
    ap.add_argument("--fp8", type=int, default=1)
    ap.add_argument("--out_dir", default="output/ditprof2")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = args.mode
    if args.fp8:
        os.environ["LINGBOT_FP8"] = "1"
    else:
        os.environ.pop("LINGBOT_FP8", None)

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
    print(f"[d2] LINGBOT_FP8={args.fp8} LINGBOT_MODE={args.mode} "
          f"layers={pipe.model.config.num_layers}", flush=True)
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

    d = f"examples/d2_{scene}"
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

    roles = instrument(pipe.model)
    print(f"[d2] roles: {roles}", flush=True)

    def drain():
        torch.cuda.synchronize()
        for r, e0, e1 in PENDING:
            MOD_MS[r] += e0.elapsed_time(e1)
            MOD_N[r] += 1
        PENDING.clear()

    g = torch.Generator(device=dev); g.manual_seed(sd)

    def run_chunk(cid):
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

    reset(); pipe._cross_attn_initialized = False
    run_chunk(0); drain(); MOD_MS.clear(); MOD_N.clear()
    print("[d2] warmup done", flush=True)

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
    tot = sum(MOD_MS.values())
    print(f"\n[d2] fp8_weight_only={bool(args.fp8)} chunks={n_test} "
          f"median chunk {med:.1f} ms")
    print(f"[d2] instrumented total {tot/n_test:.1f} ms/chunk "
          f"({100*tot/n_test/med:.1f}% of chunk)")
    print("\n[d2] ===== by ROLE (ms/chunk) =====")
    print("   {:<30} {:>10} {:>10} {:>10} {:>8}".format(
        "role", "ms/chunk", "calls/ch", "ms/call", "%chunk"))
    rows = sorted(MOD_MS.items(), key=lambda kv: -kv[1])
    out = []
    for r, ms in rows:
        n = MOD_N[r]
        print("   {:<30} {:>10.2f} {:>10.0f} {:>10.4f} {:>7.1f}%".format(
            r, ms / n_test, n / n_test, (ms / n) if n else 0,
            100 * ms / n_test / med))
        out.append(dict(role=r, ms_per_chunk=ms / n_test,
                        calls_per_chunk=n / n_test,
                        ms_per_call=(ms / n) if n else 0,
                        pct=100 * ms / n_test / med))

    lin = [o for o in out if o["role"].startswith(
        ("self_attn", "cross_attn", "ffn", "cam", "head", "time_emb"))]
    lin_ms = sum(o["ms_per_chunk"] for o in lin)
    cam_ms = sum(o["ms_per_chunk"] for o in lin if o["role"].startswith("cam."))
    print(f"\n[d2] Linear-ish total {lin_ms:.1f} ms/chunk")
    print(f"[d2] camera-injection share {cam_ms:.1f} ms/chunk "
          f"({100*cam_ms/med:.1f}% of chunk, {100*cam_ms/lin_ms:.1f}% of Linear)")
    n_fwd = len(timesteps) + 1
    print(f"[d2] block forwards per chunk = {n_fwd} (3 denoise + 1 KV update); "
          f"camera input is CONSTANT within a chunk, so up to "
          f"{100*(n_fwd-1)/n_fwd:.0f}% of camera GEMMs are repeat work "
          f"=> potential saving ~{cam_ms*(n_fwd-1)/n_fwd:.1f} ms/chunk "
          f"({100*cam_ms*(n_fwd-1)/n_fwd/med:.1f}% of chunk)")

    json.dump(dict(fp8_weight_only=bool(args.fp8), mode=args.mode, scene=scene,
                   seed=sd, n_chunks=n_test, median_chunk_ms=med,
                   instrumented_ms_per_chunk=tot / n_test,
                   roles=out, linear_ms_per_chunk=lin_ms,
                   cam_ms_per_chunk=cam_ms,
                   cam_redundant_ms_per_chunk=cam_ms * (n_fwd - 1) / n_fwd),
              open(f"{args.out_dir}/ditprof2.json", "w"), indent=1, default=str)
    print(f"\n[d2] wrote {args.out_dir}/ditprof2.json")


if __name__ == "__main__":
    main()
