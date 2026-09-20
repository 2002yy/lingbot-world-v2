#!/usr/bin/env python
"""P1b: production-shape DiT attribution (chunk_size=3, M=1881).

WHY THIS EXISTS -- A METHODOLOGICAL CORRECTION
----------------------------------------------
Every harness in this project so far built its denoise loop with ONE latent
frame per forward, giving M = 627. Production `generate()` defaults to
`chunk_size = 3`, so it feeds THREE latent frames per forward and
M = 3 x 627 = 1881, with max_seq_len = 1881.

That matters because M is exactly the variable that decides FP8 viability, and
because FFN grows faster with M than attention does. So the earlier split
(FFN 47.9% / attention 9.2%) is an M=627 result and must not be quoted as the
production breakdown.

This script redoes the attribution at M=1881 and, per the review, additionally:

  * splits FFN-up (ffn.0, K=1536 N=8960) from FFN-down (ffn.2, K=8960 N=1536),
    because FP8 measured at 1.47x on up and 0.69x on down -- recording them as
    one "ffn.*" number would hide the single most decision-relevant fact;
  * samples the ACTUAL runtime M of every Linear instead of deriving it from
    chunk_size x 627, so a second harness/production mismatch cannot hide;
  * reports two denominators: % of total chunk and % of Linear-ish time.

M derivation, for the record: WanModelFast.forward rearranges each plucker
tensor with

    1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)

so cam tokens = f x 19 x 33 = f x 627, and f is the plucker's frame count. With
three latent frames per forward that is 1881.

Run:
  LINGBOT_FP8=1 LINGBOT_CAM_CACHE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python dit_profile3.py --scene 04 --seed 42 --chunk_size 3 --chunks 4
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

MOD_MS = collections.defaultdict(float)
MOD_N = collections.defaultdict(int)
MOD_SHAPES = collections.defaultdict(collections.Counter)
PENDING = []
CAM_MODULES = {"cam_injector_layer1", "cam_injector_layer2",
               "cam_scale_layer", "cam_shift_layer"}


def role_for(path):
    """Collapse a module path to a role, PRESERVING ffn.0 vs ffn.2."""
    p = re.sub(r"blocks\.\d+\.", "blocks.*.", path)
    parts = p.split(".")
    # keep a trailing ".0"/".2"/".1" so ffn.0 (up) and ffn.2 (down) stay apart
    if len(parts) >= 2 and parts[-1].isdigit():
        tail = ".".join(parts[-2:])
    else:
        tail = parts[-1]
    if tail in CAM_MODULES:
        return "cam." + tail
    if "self_attn" in p:
        return "self_attn." + tail
    if "cross_attn" in p:
        return "cross_attn." + tail
    if ".ffn." in p:
        return "ffn." + tail          # ffn.0 (up) vs ffn.2 (down) kept apart
    if "time_embedding" in p or "time_projection" in p:
        return "time_emb." + tail
    if "text_embedding" in p:
        return "text_emb." + tail
    if "norm" in tail:
        return tail
    return p


def instrument(model):
    def pre(mod, args):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        mod._ev0 = ev
        x = args[0] if args else None
        if torch.is_tensor(x) and x.dim() >= 2:
            MOD_SHAPES[mod._role][(int(np.prod(x.shape[:-1])), int(x.shape[-1]))] += 1

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
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=3,
                    help="latent frames per forward; production default is 3")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--mode", default="repro")
    ap.add_argument("--fp8", type=int, default=1)
    ap.add_argument("--out_dir", default="output/ditprof3")
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
    CS = args.chunk_size

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    print(f"[d3] fp8={args.fp8} mode={args.mode} chunk_size={CS} "
          f"layers={pipe.model.config.num_layers}", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    frames_n = (args.frames - 1) // 4 * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))        # production trims to a multiple
    n_test = min(args.chunks, lat_f // CS)
    print(f"[d3] lat_f={lat_f} -> {lat_f // CS} chunks available, "
          f"profiling {n_test}", flush=True)

    d = f"examples/d3_{scene}"
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
    max_seq_len = CS * fsl                   # production: chunk_size * frame_seqlen
    kv_size = fsl * args.local_attn_size
    print(f"[d3] fsl={fsl} max_seq_len={max_seq_len} kv_size={kv_size} "
          f"(expect Linear M = {max_seq_len})", flush=True)
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n,
                 chunk_size=CS)
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

    roles = instrument(pipe.model)
    print(f"[d3] roles: {roles}", flush=True)

    def drain():
        torch.cuda.synchronize()
        for r, e0, e1 in PENDING:
            MOD_MS[r] += e0.elapsed_time(e1)
            MOD_N[r] += 1
        PENDING.clear()

    g = torch.Generator(device=dev); g.manual_seed(sd)

    def run_chunk(cid):
        c0 = cid * CS
        cur = torch.randn(16, CS, lat_h, lat_w, generator=g, device=dev)
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
                        x0, torch.randn(x0.shape, generator=g,
                                        device=dev, dtype=x0.dtype),
                        timesteps[ti + 1])
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)

    reset(); pipe._cross_attn_initialized = False
    run_chunk(0); drain(); MOD_MS.clear(); MOD_N.clear(); MOD_SHAPES.clear()
    print("[d3] warmup done", flush=True)

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
    print(f"\n[d3] ===== chunk_size={CS}  M={max_seq_len} =====")
    print(f"[d3] median chunk {med:.1f} ms  (n={n_test})")
    print(f"[d3] instrumented leaf-module time {tot/n_test:.1f} ms/chunk "
          f"= {100*tot/n_test/med:.1f}% of chunk")

    print("\n[d3] ===== ACTUAL runtime M per role (sampled, not derived) =====")
    bad = 0
    for r in sorted(MOD_SHAPES):
        for (M, K), n in MOD_SHAPES[r].most_common(3):
            flag = ""
            if r.startswith(("ffn.", "self_attn", "cross_attn", "cam.")) \
                    and K == pipe.model.config.dim and M != max_seq_len \
                    and M != 512:
                flag = "  <-- unexpected"
                bad += 1
            print(f"   {r:<28} M={M:<6} K={K:<6} n={n}{flag}")
    print(f"[d3] roles whose M differs from expected: {bad}")

    LIN_PREFIX = ("ffn.", "self_attn.", "cross_attn.", "cam.", "head",
                  "time_emb.", "text_emb.", "patch_embedding")
    rows = sorted(MOD_MS.items(), key=lambda kv: -kv[1])
    lin_total = sum(ms for r, ms in rows if r.startswith(LIN_PREFIX))
    print(f"\n[d3] ===== attribution (Linear-ish total {lin_total/n_test:.1f} "
          f"ms/chunk) =====")
    print("   {:<28} {:>10} {:>9} {:>9} {:>9}".format(
        "role", "ms/chunk", "calls/ch", "%chunk", "%Linear"))
    out = []
    for r, ms in rows:
        per = ms / n_test
        pc = 100 * per / med
        pl = 100 * per / lin_total if r.startswith(LIN_PREFIX) else float("nan")
        print("   {:<28} {:>10.2f} {:>9.0f} {:>8.1f}% {:>8.1f}%".format(
            r, per, MOD_N[r] / n_test, pc, pl))
        out.append(dict(role=r, ms_per_chunk=per, calls_per_chunk=MOD_N[r] / n_test,
                        pct_chunk=pc, pct_linear=pl))

    def get(r):
        return next((o for o in out if o["role"] == r), None)
    up, dn = get("ffn.0"), get("ffn.2")
    attn_share = 0.092   # measured at M=627; not re-measured here
    print("\n[d3] ===== FFN split (the decision-relevant pair) =====")
    if up and dn:
        print(f"   ffn.0 (up,  K=1536 N=8960): {up['ms_per_chunk']:7.2f} ms/chunk"
              f"  {up['pct_chunk']:5.1f}% chunk  {up['pct_linear']:5.1f}% Linear")
        print(f"   ffn.2 (down,K=8960 N=1536): {dn['ms_per_chunk']:7.2f} ms/chunk"
              f"  {dn['pct_chunk']:5.1f}% chunk  {dn['pct_linear']:5.1f}% Linear")
        ffn_sum = up["pct_chunk"] + dn["pct_chunk"]
        print(f"   FFN pair total: {ffn_sum:.1f}% of chunk")
        # Amdahl for selective FFN-up FP8 at the production M
        ratio_up_m1881 = 1.47      # measured: 1.6260 -> 1.1058 ms
        if ratio_up_m1881 > 1:
            gain = up["pct_chunk"] / 100 * (1 - 1 / ratio_up_m1881)
            print(f"   P2a selective FFN-up FP8 (measured 1.47x at M=1881): "
                  f"Amdahl = -{100*gain:.1f}% of chunk")

    json.dump(dict(chunk_size=CS, max_seq_len=max_seq_len, fp8=args.fp8,
                   mode=args.mode, scene=scene, seed=sd, n_chunks=n_test,
                   median_chunk_ms=med, instrumented_ms_per_chunk=tot / n_test,
                   linear_ms_per_chunk=lin_total / n_test,
                   roles=out,
                   actual_shapes={r: [dict(M=M, K=K, n=n)
                                      for (M, K), n in MOD_SHAPES[r].items()]
                                  for r in MOD_SHAPES}),
              open(f"{args.out_dir}/ditprof3.json", "w"), indent=1, default=str)
    print(f"\n[d3] wrote {args.out_dir}/ditprof3.json")


if __name__ == "__main__":
    main()
