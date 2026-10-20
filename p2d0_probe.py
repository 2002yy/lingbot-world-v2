#!/usr/bin/env python
"""P2d-0: coarse attribution of the non-attention residual, then one probe.

WHY A COARSE SPLIT FIRST
------------------------
P1c put "non-attention residual" at 19.2% of a chunk (~289 ms at M=1881), but
that bucket is defined by SUBTRACTION (chunk minus leaf modules minus attention
core), so it mixes genuinely fusible elementwise work with layout/cast, RoPE and
plain launch/Python gaps. Feeding 19.2% straight into an Amdahl estimate would
repeat exactly the mistake P2a just punished: a big pool is not a realisable
pool.

So this script does the smallest thing that answers one question:

    which sub-bucket is largest, and does it form a clean elementwise chain?

METHOD
------
torch.profiler is not usable here (cupti fails on WSL with
CUPTI_ERROR_INVALID_DEVICE), so the split is built by difference from CUDA
events, three levels:

    block-level   hook CausalWanAttentionBlock.forward directly (it is not a
                  leaf, so the generic leaf hooking in earlier scripts skipped it)
    leaf-level    all nn.Module leaves as before
    rope-level    monkeypatch causal_rope_apply (a module-level function)

    block_total - leaves_total - attention_core = non-leaf remainder
    of which causal_rope_apply is measured directly,
    leaving {modulation, residual adds, cam elementwise, casts, Python gaps}

Only the kernel-shape of the answer matters here, not a perfect accounting.

REPORTED PER SUB-BUCKET, with a note on which is a candidate for fusion.
No fusion is attempted yet.
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
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

MOD_MS = collections.defaultdict(float)
MOD_N = collections.defaultdict(int)
BLOCK_MS = [0.0]
BLOCK_N = [0]
PENDING = []
CAM_MODULES = {"cam_injector_layer1", "cam_injector_layer2",
               "cam_scale_layer", "cam_shift_layer"}


def role_for(path):
    p = re.sub(r"blocks\.\d+\.", "blocks.*.", path)
    parts = p.split(".")
    tail = ".".join(parts[-2:]) if len(parts) >= 2 and parts[-1].isdigit() \
        else parts[-1]
    if tail in CAM_MODULES:
        return "cam"
    if "self_attn" in p:
        return "self_attn"
    if "cross_attn" in p:
        return "cross_attn"
    if ".ffn." in p:
        return "ffn." + tail
    if "time_embedding" in p or "time_projection" in p:
        return "time_emb"
    if "text_embedding" in p:
        return "text_emb"
    return p


def instrument(model):
    # leaf modules
    n_leaf = 0
    for name, mod in model.named_modules():
        if len(list(mod.children())) > 0:
            continue
        mod._role = role_for(name)
        mod._is_leaf_hook = True

        def pre(m, args, _r=mod._role):
            e = torch.cuda.Event(enable_timing=True); e.record(); m._ev0 = e

        def post(m, args, out, _r=mod._role):
            e = torch.cuda.Event(enable_timing=True); e.record()
            PENDING.append(("L", _r, m._ev0, e))

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        n_leaf += 1

    # blocks (non-leaf, must be hooked explicitly)
    n_blk = 0
    for mod in model.blocks:
        def pre(m, args):
            e = torch.cuda.Event(enable_timing=True); e.record(); m._ev0 = e

        def post(m, args, out):
            e = torch.cuda.Event(enable_timing=True); e.record()
            PENDING.append(("B", "block", m._ev0, e))

        # pre_hook must run BEFORE the children's pre-hooks -> register first.
        # torch runs pre-hooks in registration order and children after the
        # parent's, so registering on the parent after its children still
        # produces a correct enclosing interval because the parent's own
        # pre-hook fires before descending into children.
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        n_blk += 1
    return n_leaf, n_blk


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
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/p2d0")
    args = ap.parse_args()

    os.environ["LINGBOT_FP8"] = "1"
    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    CS = args.chunk_size

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    print(f"[p2d0] layer={pipe.model.config.num_layers} ffn_dim="
          f"{pipe.model.config.ffn_dim}", flush=True)

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    frames_n = (args.frames - 1) // 4 * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))
    n_test = min(args.chunks, lat_f // CS)

    d = f"examples/p2d0_{scene}"
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
    # prewarm runs a dummy forward at current_start 0 which would
    # otherwise populate the camera cache and poison chunk 0 of the
    # real loop below; the harness does not go through generate().
    from wan.modules.model_fast import bump_cam_epoch
    bump_cam_epoch()
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

    def reset():
        for c in self_kv:
            c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False
            c["k"].zero_(); c["v"].zero_()

    # ---- instrument ----
    n_leaf, n_blk = instrument(pipe.model)
    orig_attn = mf.attention
    orig_flash = mf.flash_attention
    orig_rope = mf.causal_rope_apply

    def wrap_attn(fn, tag):
        def inner(q, k, v, *a, **kw):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); out = fn(q, k, v, *a, **kw); e1.record()
            PENDING.append(("A", tag, e0, e1))
            return out
        return inner

    def wrap_rope(fn):
        def inner(*a, **kw):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); out = fn(*a, **kw); e1.record()
            PENDING.append(("R", "rope", e0, e1))
            return out
        return inner

    mf.attention = wrap_attn(orig_attn, "attn")
    mf.flash_attention = wrap_attn(orig_flash, "flash")
    mf.causal_rope_apply = wrap_rope(orig_rope)
    print(f"[p2d0] hooked {n_leaf} leaves + {n_blk} blocks + attn/rope fns",
          flush=True)

    def drain():
        torch.cuda.synchronize()
        for kind, tag, e0, e1 in PENDING:
            ms = e0.elapsed_time(e1)
            if kind == "L":
                MOD_MS[tag] += ms; MOD_N[tag] += 1
            elif kind == "B":
                BLOCK_MS[0] += ms; BLOCK_N[0] += 1
            elif kind == "A":
                MOD_MS["ZZ." + tag] += ms; MOD_N["ZZ." + tag] += 1
            else:
                MOD_MS["ZZ.rope"] += ms; MOD_N["ZZ.rope"] += 1
        PENDING.clear()

    gg = torch.Generator(device=dev); gg.manual_seed(sd)

    def run_chunk(cid):
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

    reset(); pipe._cross_attn_initialized = False
    run_chunk(0); drain()
    MOD_MS.clear(); MOD_N.clear(); BLOCK_MS[0] = 0.0; BLOCK_N[0] = 0
    print("[p2d0] warmup done", flush=True)

    reset(); pipe._cross_attn_initialized = False
    gg = torch.Generator(device=dev); gg.manual_seed(sd)
    ch_ms = []
    for cid in range(n_test):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_chunk(cid)
        torch.cuda.synchronize()
        ch_ms.append((time.perf_counter() - t0) * 1000)
    drain()

    med = statistics.median(ch_ms)
    blk = BLOCK_MS[0] / n_test
    blkn = BLOCK_N[0] / n_test
    leaf = sum(v for k, v in MOD_MS.items() if not k.startswith("ZZ.")) / n_test
    attn = MOD_MS.get("ZZ.attn", 0) / n_test + MOD_MS.get("ZZ.flash", 0) / n_test
    rope = MOD_MS.get("ZZ.rope", 0) / n_test
    nonleaf = blk - leaf - attn

    print(f"\n[p2d0] median chunk {med:.1f} ms   (n={n_test})")
    print(f"[p2d0] block total        {blk:8.1f} ms/chunk  "
          f"({blkn:.0f} block-forwards/chunk)  {100*blk/med:5.1f}%")
    print(f"[p2d0]   leaf modules     {leaf:8.1f} ms         {100*leaf/med:5.1f}%")
    print(f"[p2d0]   attention core   {attn:8.1f} ms         {100*attn/med:5.1f}%")
    print(f"[p2d0]   RoPE             {rope:8.1f} ms         {100*rope/med:5.1f}%")
    print(f"[p2d0]   NON-LEAF remainder (modulation + residual adds +")
    print(f"[p2d0]     cam elementwise + casts + Python gaps)")
    print(f"[p2d0]                      {nonleaf:8.1f} ms         "
          f"{100*nonleaf/med:5.1f}%")
    print(f"[p2d0] chunk minus block   {med-blk:8.1f} ms         "
          f"{100*(med-blk)/med:5.1f}%   (T5 embed / head / launch gaps)")

    print(f"\n[p2d0] ===== leaf roles (top 12) =====")
    for k, v in sorted([(k, v) for k, v in MOD_MS.items()
                        if not k.startswith("ZZ.")],
                       key=lambda kv: -kv[1])[:12]:
        print(f"   {k:<20} {v/n_test:8.2f} ms  {100*v/n_test/med:5.1f}%  "
              f"calls/ch {MOD_N[k]/n_test:.0f}")

    # roofline note for the non-leaf remainder
    print(f"\n[p2d0] ===== the question =====")
    print(f"   non-leaf remainder = {nonleaf:.1f} ms/chunk "
          f"({100*nonleaf/med:.1f}% of chunk)")
    if rope > 0:
        print(f"   of which RoPE (measured separately) = {rope:.1f} ms "
              f"({100*rope/nonleaf:.0f}% of the remainder)")
    rest = nonleaf - rope
    print(f"   remaining for modulation / residual adds / cam elementwise / "
          f"casts / gaps = {rest:.1f} ms ({100*rest/med:.1f}% of chunk)")
    print(f"\n   A single fusion probe must clear ~1% of the chunk to be worth")
    print(f"   expanding: 1% = {0.01*med:.1f} ms, 2% = {0.02*med:.1f} ms, "
          f"3% = {0.03*med:.1f} ms")

    json.dump(dict(n_chunks=n_test, chunk_size=CS, max_seq_len=max_seq_len,
                   median_chunk_ms=med, block_total_ms=blk,
                   block_forwards_per_chunk=blkn, leaf_ms=leaf,
                   attention_core_ms=attn, rope_ms=rope,
                   nonleaf_remainder_ms=nonleaf,
                   nonleaf_minus_rope_ms=rest,
                   leaf_roles={k: dict(ms_per_chunk=v / n_test,
                                       pct=100 * v / n_test / med,
                                       calls_per_chunk=MOD_N[k] / n_test)
                               for k, v in MOD_MS.items()},
                   one_pct_ms=0.01 * med, two_pct_ms=0.02 * med,
                   three_pct_ms=0.03 * med),
              open(f"{args.out_dir}/p2d0.json", "w"), indent=1, default=str)
    print(f"\n[p2d0] wrote {args.out_dir}/p2d0.json")


if __name__ == "__main__":
    main()
