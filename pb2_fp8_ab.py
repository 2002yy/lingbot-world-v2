#!/usr/bin/env python
"""P2b-2: end-to-end verdict on the FP8 weight-only path.

P2b-0c ranks "store the weights in bf16 instead of FP8" as worth 413.2 ms/chunk
(26.5% of a 1560 ms chunk) for 1327 MB of VRAM, with bit-equal outputs.

By the discipline established after the RoPE incident, a microbenchmark ranks and
does NOT declare the win. This measures the real thing: the same 21-chunk rollout
run once with LINGBOT_FP8=1 and once with LINGBOT_FP8=0, reporting chunk time and
peak VRAM. LINGBOT_FP8 is read at import time, so the two arms cannot share a
process; the effect being tested is far larger than the ~0.3% process-to-process
drift, so separate runs are acceptable here (unlike the 1% question in P2d-1b1).

Also reports whether the two configs produce the same latent hashes. FP8 here is
a WEIGHT STORAGE format, not a compute format: the shipping path dequantises back
to bf16 before the GEMM, so the arithmetic should be identical and the rollout
should be bit-identical. If it is, this change needs no correctness gate at all.

Usage:
    python pb2_fp8_ab.py --fp8 1 --out_dir output/p2b2/fp8_1
    python pb2_fp8_ab.py --fp8 0 --out_dir output/p2b2/fp8_0
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import statistics
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
    ap.add_argument("--fp8", type=int, required=True, choices=[0, 1])
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=21)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    # set BEFORE importing anything that reads them (lingbot_fp8 reads at import)
    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = str(args.fp8)
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd, CS = args.scene, args.seed, args.chunk_size

    torch.cuda.reset_peak_memory_stats()
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    load_peak = torch.cuda.max_memory_allocated() / 2**20

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lat_needed = args.chunks * CS
    frames_n = (lat_needed - 1) * 4 + 1
    frames_n = ((frames_n - 1) // 4) * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))
    n_test = min(args.chunks, lat_f // CS)

    d = f"examples/p2b2_{scene}"
    os.makedirs(d, exist_ok=True)
    for f in ("intrinsics.npy", "image.jpg"):
        shutil.copy(f"examples/{scene}/{f}", f"{d}/{f}")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    th = int(np.sqrt(W * H * (480 / 832)) // 8 * 8)
    tw = int(np.sqrt(W * H / (480 / 832)) // 8 * 8)
    img = (torch.nn.functional.interpolate(
        torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None].float(),
        size=(th, tw), mode='bicubic').squeeze(0) / 255.0 - 0.5) / 0.5
    hh, ww = img.shape[1:]
    lat_h, lat_w = hh // vae_stride[1], ww // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = CS * fsl
    kv_size = fsl * args.local_attn_size
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=CS)
    mf.bump_cam_epoch()
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, hh, ww, hh, ww)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        img[None].transpose(0, 1).to(dev),
        torch.zeros(3, frames_n - 1, hh, ww, device=dev)], dim=1)])[0]
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

    def run():
        reset()
        pipe._cross_attn_initialized = False
        gg = torch.Generator(device=dev); gg.manual_seed(sd)
        ms, lats = [], []
        for cid in range(n_test):
            c0 = cid * CS
            cur = torch.randn(16, CS, lat_h, lat_w, generator=gg, device=dev)
            pp = get_plucker_embeddings(rel_all[c0:c0 + CS], Ks[None], hh, ww)
            pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                           c1=int(hh // lat_h), c2=int(ww // lat_w))[None]
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
            lats.append(hashlib.sha256(
                x0.detach().float().cpu().numpy().tobytes()).hexdigest()[:16])
        return ms, lats

    run()                                    # warmup
    run()                                    # warmup 2
    ms, lats = run()
    peak = torch.cuda.max_memory_allocated() / 2**20

    print("=" * 78)
    print(f"  LINGBOT_FP8={args.fp8}")
    print("=" * 78)
    print(f"  chunks={n_test}  median chunk = {statistics.median(ms):7.1f} ms")
    print(f"  VRAM after load          : {load_peak:7.1f} MiB")
    print(f"  VRAM peak (allocated)    : {peak:7.1f} MiB")
    print(f"  latent hash[0]           : {lats[0]}")

    out = dict(fp8=args.fp8, chunks=n_test, median_ms=statistics.median(ms),
               ms=ms, load_peak_mib=load_peak, peak_mib=peak, hashes=lats)
    with open(os.path.join(args.out_dir, "pb2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {args.out_dir}/pb2.json")


if __name__ == "__main__":
    main()
