#!/usr/bin/env python
"""P2a-0: selective FFN-up rowwise FP8 -- does the projected -12.6% materialise?

THE BASELINE MATTERS
--------------------
The production path runs weight-only FP8, not bf16. Measured at M=1881:

    ffn.0 up   (K=1536 N=8960)   bf16 1.6260 | weight-only 2.7110 | rowwise 1.1058
    ffn.2 down (K=8960 N=1536)   bf16 1.6896 | weight-only 2.7811 | rowwise 2.4603

and P1b independently measured ffn.0 at 326.29 ms/chunk over 120 calls, i.e.
2.719 ms/call, which matches the weight-only figure. So the real lever on the
up-projection is 2.7110 -> 1.1058 = 1.6052 ms/call, and at 120 calls/chunk that
projects to ~193 ms/chunk, about -12.6% of the 1524.5 ms production baseline.

An earlier estimate used bf16 (1.6260) as the baseline and so projected only
-6.6%. Comparing against bf16 rather than against what production actually runs
understated the lever by about 2x.

Only ffn.0 is converted. Rowwise FP8 LOSES on the down-projection (narrow
N=1536 cannot amortise the quantisation), so ffn.2 keeps its current path.

METHOD
------
Single model, runtime toggle: the original and quantised ffn[0] modules are both
kept and swapped into blk.ffn[0], so both arms share one model instance, one
warmup, and interleaved ordering -- a two-process comparison would confound the
result with clock state (this project has already been bitten by that).

Bare run: no instrumentation, so the numbers are production latency.

Decision gate:
    >= 10%   strong PASS, go to correctness
    7-10%    PASS, still clearly worth it
    5-7%     borderline, short gate may continue
    < 5%     integration overhead ate the microbench gain; stop and go to P2d

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python p2a_ab.py --scene 04 --seed 42 --chunk_size 3 --chunks 4 --reps 2
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
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


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
    ap.add_argument("--out_dir", default="output/p2a")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = "1"
    # The weight-only pass must SKIP ffn.0 so it stays bf16 for our rowwise
    # conversion, and the constructor must NOT convert it either -- otherwise
    # there is no bf16 variant left to swap back to. FFN0_FP8 makes the filter
    # skip it; FFN0_FP8_DEFER stops the ctor from converting it. (A first attempt
    # set FFN0_FP8=0, which let the weight-only pass quantise ffn.0 and then
    # crashed trying to quantise an already-quantised tensor.)
    os.environ["LINGBOT_FFN0_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8_DEFER"] = "1"

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
    print(f"[p2a] head={head} chunk_size={CS} LINGBOT_MODE=repro", flush=True)

    # ---- build both ffn[0] variants ----------------------------------------
    # torchao's quantize_ mutates the module IN PLACE (the module object is the
    # same before and after; its weight becomes a Float8Tensor). So we cannot
    # "keep the original and swap it back". Instead we build both variants
    # explicitly and keep them:
    #
    #   OFF = weight-only FP8  -> what production actually runs today
    #   ON  = rowwise FP8      -> the P2a candidate
    #
    # The ctor's weight-only pass skipped ffn.0 (LINGBOT_FFN0_FP8=1), so the
    # weights are still bf16 at this point and both quantisations see the
    # original weights rather than a quantised tensor.
    import copy
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig,
        quantize_)
    orig = [blk.ffn[0] for blk in pipe.model.blocks]
    print(f"[p2a] found {len(orig)} FFN up-projections, weights are "
          f"{orig[0].weight.dtype}", flush=True)

    wo, rw = [], []
    for blk in pipe.model.blocks:
        up = blk.ffn[0]
        w = copy.deepcopy(up)                      # -> weight-only (production)
        quantize_(w, Float8WeightOnlyConfig())
        r = copy.deepcopy(up)                      # -> rowwise (candidate)
        quantize_(r, Float8DynamicActivationFloat8WeightConfig())
        wo.append(w)
        rw.append(r)
    del orig
    gc.collect(); torch.cuda.empty_cache()
    print("[p2a] built weight-only (OFF) and rowwise (ON) variants", flush=True)

    def set_ffn0(on):
        with torch.no_grad():
            mods = rw if on else wo
            for blk, m in zip(pipe.model.blocks, mods):
                blk.ffn[0] = m

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

    d = f"examples/p2a_{scene}"
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
    print(f"[p2a] M={max_seq_len} fsl={fsl}", flush=True)
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

    def run(use_fp8):
        set_ffn0(use_fp8)
        for blk in pipe.model.blocks:
            blk._cam_cache = None
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

    run(False); run(True)             # warmup both
    print("[p2a] warmup done", flush=True)

    order = []
    for r in range(args.reps):
        order += [False, True] if r % 2 == 0 else [True, False]
    res = {0: [], 1: []}
    for on in order:
        ms = run(on)
        res[int(on)].extend(ms)
        print(f"[p2a] rowwise {'ON ' if on else 'OFF'} median "
              f"{statistics.median(ms):7.1f} ms", flush=True)

    off = statistics.median(res[0]); on = statistics.median(res[1])
    pct = 100 * (on - off) / off
    print(f"\n[p2a] ===== selective FFN-up rowwise FP8 (M={max_seq_len}) =====")
    print(f"  OFF (current weight-only path) : {off:8.1f} ms")
    print(f"  ON  (rowwise FP8 on ffn.0)     : {on:8.1f} ms")
    print(f"  delta                          : {on-off:+8.1f} ms ({pct:+.2f}%)")
    if pct <= -10:
        verdict = "STRONG PASS -> go to correctness gates"
    elif pct <= -7:
        verdict = "PASS -> clearly worth it"
    elif pct <= -5:
        verdict = "BORDERLINE -> short gate may continue"
    else:
        verdict = "FAIL -> integration overhead ate the microbench gain; go to P2d"
    print(f"  verdict: {verdict}")

    proj = 1.6052 * 120
    print(f"\n  microbench projection: 1.6052 ms/call x 120 = {proj:.1f} ms/chunk "
          f"= {100*proj/off:.2f}% of the {off:.0f} ms baseline")

    json.dump(dict(head=head, chunk_size=CS, max_seq_len=max_seq_len,
                   n_chunks=n_test, reps=args.reps,
                   off_median_ms=off, on_median_ms=on,
                   delta_ms=on - off, delta_pct=pct, verdict=verdict,
                   projected_ms_per_chunk=proj,
                   projected_pct=100 * proj / off,
                   off_ms=res[0], on_ms=res[1]),
              open(f"{args.out_dir}/p2a_ab.json", "w"), indent=1, default=str)
    print(f"\n[p2a] wrote {args.out_dir}/p2a_ab.json")


if __name__ == "__main__":
    main()
