#!/usr/bin/env python
"""S3-A.1: eager-only backend triage.  Four arms, one process.

    A  fa2     flash-attn 2 everywhere      (historical production baseline)
    B  sdpa    torch SDPA everywhere
    C  sage    SageAttention 2.2 everywhere
    D  hybrid  long-window self -> Sage, cross/short self -> SDPA

WHY THIS ORDER
--------------
The per-shape microbench (output/attnprof/sage_vs_fa2.json) shows flash-attn 2
is the SLOWEST option at every shape this model uses -- SDPA beats it by 22-53%,
and Sage wins only where the KV window is long. If plain SDPA already captures
most of the chunk-level gain, then Sage's extra dependency and its quantisation
cost may not be worth it; if hybrid clearly beats both, Sage's value is proven.
So we triage all four cheaply BEFORE spending budgets on 64-chunk QA.

DISCIPLINE
----------
  * same process, Latin-square-ish interleaving, so clock/thermal drift cannot
    masquerade as a backend difference (this project has been bitten by that)
  * NOTHING synchronises inside the attention hot path -- an earlier revision
    did and cost ~5040 device syncs per run (Sage falsely measured +1.0%)
  * same seed / control trajectory / cache layout across arms
  * per-arm latency, VRAM, backend call counts, and the full latent trajectory
    so divergence can be compared between arms

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python backend_triage.py --scene 04 --seed 42 --chunks 21 --reps 2
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
ARMS = "A,B,C,D"
BACKEND = {"A": "fa2", "B": "sdpa", "C": "sage", "D": "hybrid"}


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
    ap.add_argument("--sage_min_kv", type=int, default=2508)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/triage")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    report = {}

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    print(f"[bt] head={subprocess.check_output(['git','rev-parse','HEAD'],cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()[:12]}"
          f" py_cache_meta={pipe._py_cache_meta}", flush=True)
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

    d = f"examples/bt_{scene}"
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

    def run(arm):
        sb.set_backend(BACKEND[arm])
        sb.set_sage_min_kv(args.sage_min_kv)
        reset()
        sb.stats(reset=True)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        lat, outs = [], []
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
                            x0, torch.randn(x0.shape, generator=g,
                                            device=x0.device, dtype=x0.dtype),
                            timesteps[ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            outs.append(x0.detach().float().cpu())
        return lat, outs, sb.stats()

    arms = [a for a in ARMS.split(",") if a]
    # warm every arm once so kernel autotune / sage import are not in the timing
    for a in arms:
        run(a)
    print("[bt] warmup done for all arms", flush=True)

    order = []
    for r in range(args.reps):
        seq = arms if r % 2 == 0 else list(reversed(arms))
        order += seq
    print(f"[bt] arms={arms} order={order} chunks={n_test}", flush=True)

    res = {a: dict(lat=[], outs=None, st=[]) for a in arms}
    for arm in order:
        pre = tele()
        torch.cuda.reset_peak_memory_stats()
        lat, outs, st = run(arm)
        alloc = torch.cuda.max_memory_allocated() / 2**20
        resv = torch.cuda.max_memory_reserved() / 2**20
        post = tele()
        res[arm]["lat"] += lat
        res[arm]["st"].append(st)
        if res[arm]["outs"] is None:
            res[arm]["outs"] = outs
        print(f"[bt] {arm} ({BACKEND[arm]:6s}) median "
              f"{statistics.median(lat)*1000:7.1f} ms  peak {alloc:.0f}/{resv:.0f} "
              f"MiB  counts={ {k:v for k,v in st.items() if k.startswith('n_')} } "
              f"fell={st['fell_through']}  temp {post.get('temp_c','?')}C "
              f"clk {post.get('sm_clk','?')}MHz", flush=True)

    print(f"\n[bt] ===== S3-A.1 eager backend triage =====")
    base = statistics.median(res[arms[0]]["lat"])
    summary = {}
    for a in arms:
        lat = sorted(res[a]["lat"])
        med = statistics.median(lat)
        p10 = lat[max(0, int(0.10 * (len(lat) - 1)))]
        p90 = lat[min(len(lat) - 1, int(0.90 * (len(lat) - 1)))]
        summary[a] = dict(backend=BACKEND[a], median_ms=med * 1000,
                          p10_ms=p10 * 1000, p90_ms=p90 * 1000,
                          delta_ms=(med - base) * 1000,
                          delta_pct=100 * (med - base) / base)
        print(f"  {a} {BACKEND[a]:6s} median {med*1000:7.1f}  "
              f"p10 {p10*1000:7.1f}  p90 {p90*1000:7.1f}  "
              f"Δ {(med-base)*1000:+7.1f} ms ({100*(med-base)/base:+5.2f}%)")

    # divergence of each arm vs arm A (reference)
    ref = res[arms[0]]["outs"]
    print(f"\n[bt] ===== latent divergence vs {arms[0]} ({BACKEND[arms[0]]}) =====")
    div = {}
    for a in arms[1:]:
        outs = res[a]["outs"]
        if outs is None or ref is None:
            continue
        rows = []
        for i, (oa, ob) in enumerate(zip(ref, outs)):
            dd = (ob - oa).abs()
            rows.append(dict(chunk=i, max_abs=dd.max().item(),
                             mean_abs=dd.mean().item(),
                             cosine=torch.nn.functional.cosine_similarity(
                                 ob.reshape(-1), oa.reshape(-1), dim=0).item()))
        div[a] = rows
        c0, cl = rows[0]["cosine"], rows[-1]["cosine"]
        print(f"  {a} {BACKEND[a]:6s} cos {c0:.6f} (chunk0) -> {cl:.6f} "
              f"(chunk{rows[-1]['chunk']})  "
              f"{'DEGRADING' if cl < c0 - 1e-6 else 'stable/improving'}")

    print("\n[bt] per-chunk cosine vs reference:")
    hdr = "chunk " + " ".join(f"{a:>12}" for a in arms[1:])
    print(hdr)
    for i in range(n_test):
        row = f"{i:>5} "
        for a in arms[1:]:
            if a in div:
                row += f"{div[a][i]['cosine']:>12.6f}"
        print(row)

    report = dict(head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip(),
        arms=arms, backends={a: BACKEND[a] for a in arms},
        order=order, n_chunks=n_test, reps=args.reps,
        sage_min_kv=args.sage_min_kv,
        fa2_ms=[x * 1000 for x in res["A"]["lat"]] if "A" in res else [],
        summary=summary,
        raw={a: dict(ms=[x * 1000 for x in res[a]["lat"]],
                     stats=res[a]["st"]) for a in arms},
        divergence=div)
    json.dump(report, open(f"{args.out_dir}/triage.json", "w"), indent=1,
              default=str)
    print(f"\n[bt] wrote {args.out_dir}/triage.json")


if __name__ == "__main__":
    main()
