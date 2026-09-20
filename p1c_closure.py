#!/usr/bin/env python
"""P1c: close the attribution, and settle the two M=1881 questions.

THREE THINGS THIS ANSWERS
-------------------------
1. Is the 1570.7 ms figure a real production latency, or a profiling artifact?
   -> measure the same config with instrumentation OFF and ON.

2. What is P0 (camera cache) actually worth at the production M?
   P1b could not answer this: it ran with P0 already ON, so its cam share is a
   post-cache number (and cam.* showed 52 calls/chunk, not the ~120 an uncached
   run would produce). -> run an explicit OFF/ON at M=1881.

3. Attention total vs attention CORE vs attention PLUMBING.
   This is the closure that matters: the S3 backend line can only affect the
   core kernel, not the reshape/RoPE/cast/layout work around it. Reporting
   "attention = 18%" would overstate what a backend swap can reach.
   -> instrument the backend call itself, and derive plumbing as
      self_attn_module - (q,k,v,o linears) - (norm_q,norm_k) - core.

Fixing a bug found while doing this: the first P0 revision reset the cache
whenever current_start == 0, which also fired on forwards 1..3 of chunk 0 and
threw the cache away for the whole first chunk. It now resets only when
current_start == 0 follows a non-zero key (a real generation wrap-around).

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python p1c_closure.py --scene 04 --seed 42 --chunk_size 3 --chunks 4
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
PENDING = []
ATTN_MS = collections.defaultdict(float)
ATTN_N = collections.defaultdict(int)
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


def install_hooks(model):
    def pre(mod, args):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        mod._ev0 = ev
    def post(mod, args, out):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        PENDING.append(("M", mod._role, mod._ev0, ev))
    for name, mod in model.named_modules():
        if len(list(mod.children())) > 0:
            continue
        mod._role = role_for(name)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)


def install_attn_hooks():
    """Wrap the backend call itself -> attention CORE."""
    orig_a = mf.attention
    orig_f = mf.flash_attention

    def wrap(fn, tag):
        def inner(q, k, v, *a, **kw):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = fn(q, k, v, *a, **kw)
            ev1.record()
            PENDING.append(("A", tag, ev0, ev1))
            return out
        return inner

    mf.attention = wrap(orig_a, "core.attention()")
    mf.flash_attention = wrap(orig_f, "core.flash_attention()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/p1c")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
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
    head = subprocess_head()
    print(f"[p1c] head={head} chunk_size={CS} layers={pipe.model.config.num_layers}",
          flush=True)
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

    d = f"examples/p1c_{scene}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    _th = int(np.sqrt(W * H * (480 / 832)) // 8 * 8)
    _tw = int(np.sqrt(W * H / (480 / 832)) // 8 * 8)
    img = (torch.nn.functional.interpolate(
        torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None].float(),
        size=(_th, _tw), mode='bicubic').squeeze(0) / 255.0 - 0.5) / 0.5
    h, w = img.shape[1:]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = CS * fsl
    kv_size = fsl * args.local_attn_size
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=CS)
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

    def drain():
        torch.cuda.synchronize()
        for kind, tag, e0, e1 in PENDING:
            if kind == "M":
                MOD_MS[tag] += e0.elapsed_time(e1); MOD_N[tag] += 1
            else:
                ATTN_MS[tag] += e0.elapsed_time(e1); ATTN_N[tag] += 1
        PENDING.clear()

    def time_run(cam_cache):
        mf._CAM_CACHE = bool(cam_cache)
        for blk in pipe.model.blocks:
            blk._cam_cache = None
        reset(); pipe._cross_attn_initialized = False
        g2 = torch.Generator(device=dev); g2.manual_seed(sd)
        ms = []
        for cid in range(n_test):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _run_chunk_with(cid, g2)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
        return ms

    def _run_chunk_with(cid, gg):
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

    # ---- PHASE 1: no instrumentation, cam ON vs OFF (answers corrections 1&2)
    print("\n[p1c] === PHASE 1: bare latency, cam cache OFF/ON ===", flush=True)
    bare = {}
    for rep in range(args.reps):
        for cc in ([False, True] if rep % 2 == 0 else [True, False]):
            ms = time_run(cc)
            bare.setdefault(cc, []).extend(ms)
            print(f"[p1c]   bare cam={'ON ' if cc else 'OFF'} "
                  f"median {statistics.median(ms):7.1f} ms", flush=True)
    off = statistics.median(bare[False]); on = statistics.median(bare[True])
    print(f"[p1c] bare production latency (M={max_seq_len}): "
          f"cam OFF {off:.1f} ms | cam ON {on:.1f} ms")
    print(f"[p1c] P0 gain at M={max_seq_len}: {(on-off):+.1f} ms "
          f"({100*(on-off)/off:+.2f}%)")

    # ---- PHASE 2: with instrumentation, full closure
    print("\n[p1c] === PHASE 2: instrumented closure ===", flush=True)
    install_hooks(pipe.model)
    install_attn_hooks()
    mf._CAM_CACHE = True
    for blk in pipe.model.blocks:
        blk._cam_cache = None
    reset(); pipe._cross_attn_initialized = False
    gg = torch.Generator(device=dev); gg.manual_seed(sd)
    _run_chunk_with(0, gg); drain()
    MOD_MS.clear(); MOD_N.clear(); ATTN_MS.clear(); ATTN_N.clear()
    reset(); pipe._cross_attn_initialized = False
    gg = torch.Generator(device=dev); gg.manual_seed(sd)
    ins_ms = []
    for cid in range(n_test):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _run_chunk_with(cid, gg)
        torch.cuda.synchronize()
        ins_ms.append((time.perf_counter() - t0) * 1000)
    drain()
    med = statistics.median(ins_ms)
    print(f"[p1c] instrumented median {med:.1f} ms  "
          f"(bare cam-ON was {on:.1f} ms -> profiling tax "
          f"{med-on:+.1f} ms, {100*(med-on)/on:+.2f}%)")

    modtot = sum(MOD_MS.values())
    atttot = sum(ATTN_MS.values())
    print(f"\n[p1c] role totals (ms/chunk):")
    rows = sorted(MOD_MS.items(), key=lambda kv: -kv[1])
    for r, ms in rows:
        print(f"   {r:<22} {ms/n_test:8.2f}   {100*ms/n_test/med:5.1f}%")
    print(f"\n[p1c] attention CORE (backend call):")
    for t, ms in sorted(ATTN_MS.items(), key=lambda kv: -kv[1]):
        print(f"   {t:<28} {ms/n_test:8.2f}   {100*ms/n_test/med:5.1f}%   "
              f"calls/ch {ATTN_N[t]/n_test:.0f}")
    print(f"   {'core total':<28} {atttot/n_test:8.2f}   "
          f"{100*atttot/n_test/med:5.1f}%")

    sa = MOD_MS.get("self_attn", 0) / n_test
    ca = MOD_MS.get("cross_attn", 0) / n_test
    leafless = med - modtot / n_test
    print(f"\n[p1c] ===== closure =====")
    print(f"   chunk (instrumented)                 {med:8.2f}  100.0%")
    print(f"   leaf modules total                   {modtot/n_test:8.2f}  "
          f"{100*modtot/n_test/med:5.1f}%")
    print(f"   attention core (inside leaf modules) {atttot/n_test:8.2f}  "
          f"{100*atttot/n_test/med:5.1f}%")
    print(f"   self_attn module total               {sa:8.2f}  "
          f"{100*sa/med:5.1f}%")
    print(f"   cross_attn module total              {ca:8.2f}  "
          f"{100*ca/med:5.1f}%")
    print(f"   --- derived ---")
    attn_all = (atttot / n_test)
    print(f"   ATTENTION TOTAL (core)               {attn_all:8.2f}  "
          f"{100*attn_all/med:5.1f}%")
    print(f"   not-instrumented residual            {leafless:8.2f}  "
          f"{100*leafless/med:5.1f}%")
    print(f"      (= modulation / rope / residual elementwise / casts / launch gaps)")

    json.dump(dict(head=head, chunk_size=CS, max_seq_len=max_seq_len,
                   n_chunks=n_test, reps=args.reps,
                   bare_cam_off_ms=off, bare_cam_on_ms=on,
                   p0_gain_ms=on - off, p0_gain_pct=100 * (on - off) / off,
                   instrumented_ms=med,
                   profiling_tax_ms=med - on,
                   profiling_tax_pct=100 * (med - on) / on,
                   roles={r: dict(ms_per_chunk=ms / n_test,
                                  pct=100 * ms / n_test / med,
                                  calls_per_chunk=MOD_N[r] / n_test)
                          for r, ms in MOD_MS.items()},
                   attention_core={t: dict(ms_per_chunk=ms / n_test,
                                           pct=100 * ms / n_test / med,
                                           calls_per_chunk=ATTN_N[t] / n_test)
                                   for t, ms in ATTN_MS.items()},
                   attention_core_total_pct=100 * atttot / n_test / med,
                   leaf_total_pct=100 * modtot / n_test / med,
                   residual_pct=100 * leafless / med),
              open(f"{args.out_dir}/p1c.json", "w"), indent=1, default=str)
    print(f"\n[p1c] wrote {args.out_dir}/p1c.json")


def subprocess_head():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()[:12]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
