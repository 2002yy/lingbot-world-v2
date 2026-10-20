#!/usr/bin/env python
"""Same-process paired benchmark + fail_on_recompile gate.

WHY THIS EXISTS
---------------
We have two promising numbers (host-sync removal ~-10%, compile default ~-9%)
but they came from DIFFERENT harnesses, so chaining them is invalid. This runs
every configuration in ONE process, on ONE model instance, with ONE set of
conditioning tensors, interleaved (Latin-square-ish) to average out thermal and
clock drift on a laptop.

CONFIGURATIONS
--------------
  A  original semantics      : cache metadata as CUDA tensors, eager
  B  sync-refactor           : cache metadata as Python int/bool, eager
  C  sync-refactor + compile : Python int/bool, torch.compile(mode="default")

A and B differ ONLY in what `_initialize_*_cache` writes and how the attention
reads it; `python_metadata` is a flag on the same code, so no git stash is
needed. NOTE: A is "original *semantics*" -- the refactor also collapsed some
redundant `.item()` reads into one, so A understates the original sync count.
That makes the A->B delta CONSERVATIVE, which is the safe direction.

`--fail-on-recompile` enables torch.compiler.set_stance AFTER warmup, so the
gate only covers steady state. Cold specialization is known and bounded (14
recompiles: dtype ping-pong + index-value guards + rollover).

TELEMETRY
---------
Every rep records GPU power / SM clock / temperature so a drifting laptop GPU
cannot silently masquerade as a speedup. VRAM peak allocated/reserved too.

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python paired_ab.py --scene 04 --seed 42 --chunks 12 --reps 3 \
      --configs A,B,C --order latin
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
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def gpu_telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=power.draw,clocks.sm,temperature.gpu,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], timeout=10).decode().strip()
        vals = [v.strip() for v in out.split(",")]
        return dict(power_w=float(vals[0]), sm_clk_mhz=float(vals[1]),
                    temp_c=float(vals[2]), mem_free_mib=float(vals[3]),
                    util_pct=float(vals[4]))
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
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--configs", default="A,B,C")
    ap.add_argument("--order", default="latin", choices=["latin", "abba", "seq"])
    ap.add_argument("--fail_on_recompile", action="store_true")
    ap.add_argument("--fail_chunks", type=int, default=16)
    ap.add_argument("--warm_chunks", type=int, default=None,
                    help="warmup chunks before enabling fail_on_recompile")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--out_dir", default="output/paired")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    dirty = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=repo).decode().strip().splitlines()
    print(f"[pa] HEAD={head[:12]} dirty={len(dirty)} files", flush=True)
    if dirty:
        for f in dirty:
            print(f"[pa]   M {f}", flush=True)

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
    print(f"[pa] pipe._py_cache_meta={pipe._py_cache_meta}", flush=True)
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

    d = f"examples/pa_{scene}"
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

    def make_caches(py_meta):
        sk = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers,
            shape=[1, kv_size, ma.num_heads // pipe.sp_size, ma.dim // ma.num_heads],
            dtype=dtype, device=dev, python_metadata=py_meta)
        ck = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers,
            shape=[1, 512, ma.num_heads, ma.dim // ma.num_heads],
            dtype=dtype, device=dev, python_metadata=py_meta)
        return sk, ck

    def reset(self_kv, cross_kv):
        for c in self_kv:
            if torch.is_tensor(c["global_end_index"]):
                c["global_end_index"].zero_(); c["local_end_index"].zero_()
            else:
                c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            if torch.is_tensor(c["is_init"]):
                c["is_init"].zero_()
            else:
                c["is_init"] = False
            c["k"].zero_(); c["v"].zero_()

    # One compiled instance per distinct mode, built lazily.
    compiled_cache = {}

    def get_model(cfg_id):
        mode = CONFIGS[cfg_id]["mode"]
        if mode == "eager":
            return pipe.model
        if mode not in compiled_cache:
            torch._dynamo.reset()
            compiled_cache[mode] = torch.compile(pipe.model, mode=mode, fullgraph=False)
        return compiled_cache[mode]

    CONFIGS = {
        "A": dict(py_meta=False, mode="eager", label="orig-semantics eager"),
        "B": dict(py_meta=True, mode="eager", label="sync-refactor eager"),
        "C": dict(py_meta=True, mode="default", label="sync-refactor compile-default"),
        "D": dict(py_meta=True, mode="reduce-overhead", label="sync-refactor reduce-overhead"),
    }
    cfg_ids = [c for c in args.configs.split(",") if c in CONFIGS]

    # Caches are allocated per config (metadata type differs) but ONCE per config
    # so CUDA Graph address stability is not confounded.
    caches = {c: make_caches(CONFIGS[c]["py_meta"]) for c in cfg_ids}

    def run_one(cfg_id, nchunks):
        sk, ck = caches[cfg_id]
        reset(sk, ck)
        model = get_model(cfg_id)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        per, outs = [], []
        for cid in range(nchunks):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            pp = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                           c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": sk, "crossattn_cache": ck,
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
            outs.append(x0.detach().float().cpu())
        return per, outs

    # --- build the interleaved order ---
    if args.order == "seq":
        order = cfg_ids * args.reps
    elif args.order == "abba":
        order = []
        fwd = list(cfg_ids)
        for r in range(args.reps):
            order += fwd if r % 2 == 0 else list(reversed(fwd))
    else:  # latin-ish: rotate
        order = []
        for r in range(args.reps):
            k = r % len(cfg_ids)
            order += cfg_ids[k:] + cfg_ids[:k]
    print(f"[pa] chunks/rep={n_test} reps={args.reps} order={order}", flush=True)

    results = {c: dict(lat=[], hashes=[], vram=[], tele=[]) for c in cfg_ids}
    total = len(order)
    for i, cid in enumerate(order):
        print(f"[pa] --- rep {i+1}/{total}: {cid} ({CONFIGS[cid]['label']}) ---",
              flush=True)
        t_pre = gpu_telemetry()
        torch.cuda.reset_peak_memory_stats()
        per, outs = run_one(cid, n_test)
        alloc = torch.cuda.max_memory_allocated() / 2**20
        resv = torch.cuda.max_memory_reserved() / 2**20
        t_post = gpu_telemetry()
        cat = torch.cat([o.reshape(-1) for o in outs])
        hsh = hashlib.sha256(cat.numpy().tobytes()).hexdigest()[:16]
        results[cid]["lat"] += per
        results[cid]["hashes"].append(hsh)
        results[cid]["vram"].append(dict(alloc=alloc, resv=resv))
        results[cid]["tele"].append(dict(pre=t_pre, post=t_post))
        print(f"[pa]   median {statistics.median(per)*1000:.1f} ms  "
              f"peak {alloc:.0f}/{resv:.0f} MiB  hash {hsh}  "
              f"temp {t_post.get('temp_c','?')}C pwr {t_post.get('power_w','?')}W "
              f"clk {t_post.get('sm_clk_mhz','?')}MHz", flush=True)

    # --- summary ---
    print(f"\n[pa] ===== same-process paired summary ({n_test} chunks/rep, "
          f"{args.reps} reps) =====")
    summary = {}
    for c in cfg_ids:
        lat = sorted(results[c]["lat"])
        n = len(lat)
        med = statistics.median(lat)
        p10 = lat[max(0, int(0.10 * (n - 1)))]
        p90 = lat[min(n - 1, int(0.90 * (n - 1)))]
        mean = statistics.mean(lat)
        cv = (statistics.pstdev(lat) / mean * 100) if mean else 0
        allocs = [v["alloc"] for v in results[c]["vram"]]
        resvs = [v["resv"] for v in results[c]["vram"]]
        temps = [t["post"].get("temp_c") for t in results[c]["tele"]
                 if "temp_c" in t["post"]]
        hs = set(results[c]["hashes"])
        summary[c] = dict(label=CONFIGS[c]["label"], n=n, median_ms=med * 1000,
                          mean_ms=mean * 1000, p10_ms=p10 * 1000,
                          p90_ms=p90 * 1000, cv_pct=cv, min_ms=lat[0] * 1000,
                          max_ms=lat[-1] * 1000, alloc_mib=max(allocs),
                          resv_mib=max(resvs), hashes=sorted(hs),
                          temps=temps)
        print(f"  {c}  {CONFIGS[c]['label']:34s} "
              f"median {med*1000:7.1f}  p10 {p10*1000:7.1f}  p90 {p90*1000:7.1f}  "
              f"CV {cv:4.1f}%  peak {max(allocs):.0f}/{max(resvs):.0f} MiB")
        print(f"       hashset {sorted(hs)}")

    if len(cfg_ids) > 1:
        base = cfg_ids[0]
        print(f"\n[pa] paired deltas vs {base} ({CONFIGS[base]['label']}):")
        for c in cfg_ids[1:]:
            d_ = summary[c]["median_ms"] - summary[base]["median_ms"]
            pct = 100 * d_ / summary[base]["median_ms"]
            dv = summary[c]["alloc_mib"] - summary[base]["alloc_mib"]
            print(f"  {c}: {d_:+8.1f} ms ({pct:+5.1f}%)   "
                  f"peak VRAM {dv:+6.0f} MiB")

    # --- fail_on_recompile gate (WARM ONLY) ---
    if args.fail_on_recompile:
        gate_cfg = cfg_ids[-1]
        warm_n = args.warm_chunks or (args.local_attn_size * 2 + 4)
        print(f"\n[pa] ===== fail_on_recompile gate =====")
        print(f"[pa] config {gate_cfg}; warming {warm_n} chunks to establish "
              f"all specializations (incl. rollover at chunk {args.local_attn_size})")
        run_one(gate_cfg, warm_n)
        print(f"[pa] warmup done; enabling fail_on_recompile for "
              f"{args.fail_chunks} more chunks", flush=True)
        try:
            with torch.compiler.set_stance("fail_on_recompile"):
                per, _ = run_one(gate_cfg, args.fail_chunks)
            print(f"[pa] GATE PASS: {args.fail_chunks} steady-state chunks, "
                  f"0 recompiles, median {statistics.median(per)*1000:.1f} ms")
            gate = dict(pass_=True, chunks=args.fail_chunks,
                        median_ms=statistics.median(per) * 1000)
        except Exception as e:
            print(f"[pa] GATE FAIL: {type(e).__name__}: {str(e)[:400]}")
            gate = dict(pass_=False, error=f"{type(e).__name__}: {str(e)[:400]}")
        summary["fail_on_recompile"] = gate

    json.dump(dict(head=head, dirty=dirty, order=order, n_chunks=n_test,
                   reps=args.reps, summary=summary,
                   raw={c: dict(lat=results[c]["lat"],
                                hashes=results[c]["hashes"],
                                vram=results[c]["vram"])
                        for c in cfg_ids}),
              open(f"{args.out_dir}/paired.json", "w"), indent=1, default=str)
    print(f"\n[pa] wrote {args.out_dir}/paired.json", flush=True)


if __name__ == "__main__":
    main()
