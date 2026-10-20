#!/usr/bin/env python
"""S3-B: 2x2 -- Hybrid backend x torch.compile.

    A = FA2    + eager       (historical production baseline)
    B = Hybrid + eager       (visual gate already PASSED)
    C = FA2    + compile
    D = Hybrid + compile     (candidate production config)

WHAT WE ARE ACTUALLY ASKING
---------------------------
Not "is compile faster" but **does the Hybrid backend's gain survive
compilation**. Four quantities fall out:

    backend effect (eager)    = B - A
    backend effect (compile)  = D - C
    compile effect (FA2)      = C - A
    compile effect (Hybrid)   = D - B
    interaction               = (D - C) - (B - A)

  interaction ~ 0  -> gains are independent, stacking is safe
  interaction > 0  -> compile eats part of the backend gain
  interaction ~ 0 but compile effect ~ 0 on Hybrid -> just ship Hybrid eager

SCOPE DISCIPLINE (learned the hard way)
---------------------------------------
* `torch.compile` here means **ordinary Inductor optimisation** with
  mode="default". It does NOT mean CUDA Graph capture.
* `reduce-overhead` / CUDA Graph is a SEPARATE, already-closed line: CUDAGraph
  Trees skips capture because the KV cache is a mutated eager input
  ("skipping cudagraphs due to mutated inputs" at crossattn_cache["k"].copy_(k)),
  and forcing it hits `_cuda_setCheckpointPoolState` ->
  "Expected curr_block->next == nullptr". Do not conflate the two.
* Nothing synchronises inside the attention hot path -- an earlier revision did
  and cost ~5040 device syncs per run, falsely making the Sage arm +1.0% slower.

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python compile_2x2.py --scene 04 --seed 42 --chunks 21 --reps 2
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
import wan.modules.sage_backend as sb
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

# config id -> (backend, compiled)
CFG = {
    "A": ("fa2", False),
    "B": ("hybrid", False),
    "C": ("fa2", True),
    "D": ("hybrid", True),
}
ORDER = ["A", "B", "C", "D"]


def tele():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=10).decode().strip()
        v = [x.strip() for x in o.split(",")]
        return dict(power_w=float(v[0]), sm_clk=float(v[1]), temp_c=float(v[2]))
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
    ap.add_argument("--chunks", type=int, default=21)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mode", default="default",
                    help="torch.compile mode; deliberately NOT reduce-overhead")
    ap.add_argument("--sage_min_kv", type=int, default=2508)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/compile2x2")
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
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    print(f"[2x2] head={head[:12]} compile_mode={args.mode} "
          f"(NOT reduce-overhead: CUDA Graph is a closed, separate line)",
          flush=True)
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

    d = f"examples/cx_{scene}"
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

    compiled = {}

    def get_model(cid):
        backend, do_compile = CFG[cid]
        if not do_compile:
            return pipe.model
        if cid not in compiled:
            print(f"[2x2] compiling for config {cid} (mode={args.mode})", flush=True)
            compiled[cid] = torch.compile(pipe.model, mode=args.mode,
                                          fullgraph=False)
        return compiled[cid]

    def run(cid):
        backend, _ = CFG[cid]
        sb.set_backend(backend)
        sb.set_sage_min_kv(args.sage_min_kv)
        model = get_model(cid)
        reset()
        sb.stats(reset=True)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        lat, outs = [], []
        for ch in range(n_test):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            pp = get_plucker_embeddings(rel_all[ch:ch + 1], Ks[None], h, w)
            pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                           c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[min(ch, frames_n // 4 - 1)]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": ch * fsl,
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
            lat.append(time.perf_counter() - t0)
            outs.append(x0.detach().float().cpu())
        return lat, outs, sb.stats()

    ids = [c for c in ORDER]
    for c in ids:                      # warmup: also triggers compilation
        print(f"[2x2] warmup {c} ({CFG[c][0]}{'+compile' if CFG[c][1] else ''})",
              flush=True)
        run(c)

    order = []
    for r in range(args.reps):
        seq = ids if r % 2 == 0 else list(reversed(ids))
        order += seq
    print(f"[2x2] order={order} chunks={n_test}", flush=True)

    res = {c: dict(lat=[], outs=None, st=[]) for c in ids}
    for cid in order:
        pre = tele()
        torch.cuda.reset_peak_memory_stats()
        lat, outs, st = run(cid)
        alloc = torch.cuda.max_memory_allocated() / 2**20
        resv = torch.cuda.max_memory_reserved() / 2**20
        post = tele()
        res[cid]["lat"] += lat
        res[cid]["st"].append(st)
        if res[cid]["outs"] is None:
            res[cid]["outs"] = outs
        tag = f"{CFG[cid][0]:6s}+{'compile' if CFG[cid][1] else 'eager  '}"
        print(f"[2x2] {cid} ({tag}) median {statistics.median(lat)*1000:7.1f} ms "
              f"peak {alloc:.0f}/{resv:.0f} MiB "
              f"temp {post.get('temp_c','?')}C clk {post.get('sm_clk','?')}MHz",
              flush=True)

    print(f"\n[2x2] ===== Hybrid x torch.compile =====")
    med = {c: statistics.median(res[c]["lat"]) * 1000 for c in ids}
    for c in ids:
        tag = f"{CFG[c][0]:6s} + {'compile' if CFG[c][1] else 'eager'}"
        print(f"  {c}  {tag:20s} {med[c]:7.1f} ms")
    print()
    be = med["B"] - med["A"]
    be_c = med["D"] - med["C"]
    ce = med["C"] - med["A"]
    ce_h = med["D"] - med["B"]
    inter = be_c - be
    print(f"  backend effect (eager)   B-A = {be:+7.1f} ms ({100*be/med['A']:+5.2f}%)")
    print(f"  backend effect (compile) D-C = {be_c:+7.1f} ms ({100*be_c/med['C']:+5.2f}%)")
    print(f"  compile effect (FA2)     C-A = {ce:+7.1f} ms ({100*ce/med['A']:+5.2f}%)")
    print(f"  compile effect (Hybrid)  D-B = {ce_h:+7.1f} ms ({100*ce_h/med['B']:+5.2f}%)")
    print(f"  INTERACTION (D-C)-(B-A)      = {inter:+7.1f} ms")
    print(f"  best config: {min(ids, key=lambda c: med[c])} "
          f"({CFG[min(ids, key=lambda c: med[c])][0]}"
          f"{'+compile' if CFG[min(ids, key=lambda c: med[c])][1] else '+eager'})")

    # divergence: D vs B is the COMPILE increment on the already-validated
    # Hybrid config, which is the one that matters for the final decision.
    print(f"\n[2x2] ===== latent divergence vs A (fa2+eager) =====")
    ref = res["A"]["outs"]
    div = {}
    for c in ["B", "C", "D"]:
        rows = []
        for i, (oa, ob) in enumerate(zip(ref, res[c]["outs"])):
            dd = (ob - oa).abs()
            rows.append(dict(chunk=i, max_abs=dd.max().item(),
                             mean_abs=dd.mean().item(),
                             cosine=torch.nn.functional.cosine_similarity(
                                 ob.reshape(-1), oa.reshape(-1), dim=0).item()))
        div[c] = rows
        print(f"  {c}: cos {rows[0]['cosine']:.6f} -> {rows[-1]['cosine']:.6f}  "
              f"(mean|d| {rows[-1]['mean_abs']:.3e})")
    # D vs B specifically
    rows = []
    for i, (oa, ob) in enumerate(zip(res["B"]["outs"], res["D"]["outs"])):
        dd = (ob - oa).abs()
        rows.append(dict(chunk=i, max_abs=dd.max().item(),
                         mean_abs=dd.mean().item(),
                         cosine=torch.nn.functional.cosine_similarity(
                             ob.reshape(-1), oa.reshape(-1), dim=0).item()))
    div["D_vs_B"] = rows
    print(f"  D vs B (compile increment on Hybrid): "
          f"cos {rows[0]['cosine']:.6f} -> {rows[-1]['cosine']:.6f}")

    json.dump(dict(head=head, scene=scene, seed=sd, n_chunks=n_test,
                   reps=args.reps, compile_mode=args.mode,
                   sage_min_kv=args.sage_min_kv,
                   cfg={c: dict(backend=CFG[c][0], compiled=CFG[c][1]) for c in ids},
                   order=order, median_ms=med,
                   effects=dict(backend_eager=be, backend_compile=be_c,
                                compile_fa2=ce, compile_hybrid=ce_h,
                                interaction=inter),
                   raw={c: dict(ms=[x * 1000 for x in res[c]["lat"]],
                                stats=res[c]["st"]) for c in ids},
                   divergence=div),
              open(f"{args.out_dir}/compile2x2.json", "w"), indent=1, default=str)
    print(f"\n[2x2] wrote {args.out_dir}/compile2x2.json")


if __name__ == "__main__":
    main()
