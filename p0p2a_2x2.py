#!/usr/bin/env python
"""P0 x P2a: 2x2 factorial, and the first formal definition of the fast config.

WHY
---
P2a's -7.07% was measured with the camera cache already ON: p2a_ab.py never
touched mf._CAM_CACHE and LINGBOT_CAM_CACHE defaults to 1, and indeed its OFF
arm read 1532.3 ms against P0's ON baseline of 1524.5 ms. So P2a stacks on top of
P0 rather than being cancelled by it -- but the two have never been measured in
one harness, and adding 6.45 + 7.07 would be wrong (percentages compound).

ARMS
    T00  cam OFF  ffn0 weight-only   original production compute path
    T10  cam ON   ffn0 weight-only   P0 alone
    T01  cam OFF  ffn0 rowwise FP8   P2a alone
    T11  cam ON   ffn0 rowwise FP8   the fast candidate

    interaction  I = T11 - T10 - T01 + T00    (ms)
      I ~ 0  the two are independent and stack
      I < 0  positive synergy
      I > 0  they partly eat each other

Bare timing only (no instrumentation), single model, single warmup, interleaved
ordering, so clock state cannot masquerade as a difference.

The four arms also settle the fast configuration definitionally: fast = cam cache
ON + FFN0 rowwise FP8, over repro = cam cache ON only (the cache is bit-exact and
so is part of repro).

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python p0p2a_2x2.py --scene 04 --seed 42 --chunk_size 3 --chunks 4 --reps 2
"""
import argparse
import copy
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
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
ARMS = ["T00", "T10", "T01", "T11"]
MEAN = {"T00": (False, False), "T10": (True, False),
        "T01": (False, True), "T11": (True, True)}


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
    ap.add_argument("--out_dir", default="output/p0p2a")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "1"          # weight-only filter skips ffn.0
    os.environ["LINGBOT_FFN0_FP8_DEFER"] = "1"    # harness owns the two variants

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
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()[:12]
    print(f"[2x2] head={head} chunk_size={CS} mode=repro", flush=True)

    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig,
        quantize_)
    wo, rw = [], []
    for blk in pipe.model.blocks:
        up = blk.ffn[0]
        w = copy.deepcopy(up); quantize_(w, Float8WeightOnlyConfig())
        r = copy.deepcopy(up)
        quantize_(r, Float8DynamicActivationFloat8WeightConfig())
        wo.append(w); rw.append(r)
    gc.collect(); torch.cuda.empty_cache()
    print(f"[2x2] built {len(wo)} weight-only and {len(rw)} rowwise ffn.0 variants",
          flush=True)

    def set_arm(cam_on, ffn_on):
        mf._CAM_CACHE = bool(cam_on)
        mods = rw if ffn_on else wo
        with torch.no_grad():
            for blk, m in zip(pipe.model.blocks, mods):
                blk.ffn[0] = m
                blk._cam_cache = None

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

    d = f"examples/p0p2a_{scene}"
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
    print(f"[2x2] M={max_seq_len}", flush=True)
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

    def run(arm):
        cam_on, ffn_on = MEAN[arm]
        set_arm(cam_on, ffn_on)
        reset()
        pipe._cross_attn_initialized = False
        gg = torch.Generator(device=dev); gg.manual_seed(sd)
        ms = []
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
            ms.append((time.perf_counter() - t0) * 1000)
        return ms

    for a in ARMS:                       # warm every arm once
        run(a)
    print("[2x2] warmup done for all arms", flush=True)

    order = []
    for r in range(args.reps):
        order += ARMS if r % 2 == 0 else list(reversed(ARMS))
    res = {a: [] for a in ARMS}
    for a in order:
        ms = run(a)
        res[a].extend(ms)
        print(f"[2x2] {a} cam={'ON ' if MEAN[a][0] else 'OFF'} "
              f"ffn0={'rowwise' if MEAN[a][1] else 'weight-only'} "
              f"median {statistics.median(ms):7.1f} ms", flush=True)

    T = {a: statistics.median(res[a]) for a in ARMS}
    print(f"\n[2x2] ===== cam cache  x  FFN0 rowwise FP8 (M={max_seq_len}) =====")
    print("   {:<6} {:<10} {:<14} {:>10}".format("arm", "cam", "ffn0", "ms"))
    for a in ARMS:
        print("   {:<6} {:<10} {:<14} {:>10.1f}".format(
            a, "ON" if MEAN[a][0] else "OFF",
            "rowwise" if MEAN[a][1] else "weight-only", T[a]))

    I = T["T11"] - T["T10"] - T["T01"] + T["T00"]
    base = T["T00"]
    print(f"\n   P0 alone        (T10-T00) = {T['T10']-T['T00']:+7.1f} ms "
          f"({100*(T['T10']-T['T00'])/base:+.2f}%)")
    print(f"   P2a alone       (T01-T00) = {T['T01']-T['T00']:+7.1f} ms "
          f"({100*(T['T01']-T['T00'])/base:+.2f}%)")
    print(f"   combined        (T11-T00) = {T['T11']-T['T00']:+7.1f} ms "
          f"({100*(T['T11']-T['T00'])/base:+.2f}%)")
    print(f"   P2a on top of P0 (T11-T10) = {T['T11']-T['T10']:+7.1f} ms "
          f"({100*(T['T11']-T['T10'])/T['T10']:+.2f}%)")
    print(f"\n   interaction  I = T11-T10-T01+T00 = {I:+7.1f} ms "
          f"({100*I/base:+.2f}% of baseline)")
    if abs(I) < 0.02 * base:
        print("   -> I ~ 0: the two levers are INDEPENDENT and stack")
    elif I < 0:
        print("   -> I < 0: positive synergy (combined beats the sum)")
    else:
        print("   -> I > 0: the two partly eat each other")
    naive = 100 * ((T['T10'] - T['T00']) + (T['T01'] - T['T00'])) / base
    print(f"\n   (additive-in-percent would have said {naive:+.2f}%; "
          f"compound truth is {100*(T['T11']-T['T00'])/base:+.2f}%)")

    json.dump(dict(head=head, chunk_size=CS, max_seq_len=max_seq_len,
                   n_chunks=n_test, reps=args.reps, order=order,
                   arms={a: dict(cam=MEAN[a][0], ffn0_rowwise=MEAN[a][1],
                                 median_ms=T[a], runs_ms=res[a]) for a in ARMS},
                   p0_ms=T["T10"] - T["T00"], p2a_ms=T["T01"] - T["T00"],
                   combined_ms=T["T11"] - T["T00"],
                   p2a_on_top_of_p0_ms=T["T11"] - T["T10"],
                   interaction_ms=I,
                   interaction_pct=100 * I / base),
              open(f"{args.out_dir}/p0p2a_2x2.json", "w"), indent=1, default=str)
    print(f"\n[2x2] wrote {args.out_dir}/p0p2a_2x2.json")


if __name__ == "__main__":
    main()
