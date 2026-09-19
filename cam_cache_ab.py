#!/usr/bin/env python
"""P0 verification: camera-injection cache -- is it bit-exact, and what is it worth?

CLAIM UNDER TEST
----------------
Each CausalWanAttentionBlock has four camera Linears whose input
`c2ws_plucker_emb` is constant within a chunk, while the block forward runs 4x
per chunk (3 denoise steps + 1 KV update). So ~75% of that work is provably
recomputation, and caching it should change NOTHING numerically -- same input,
same weights, identical output.

If both halves hold (bit-exact + real gain) this is the best kind of win: it can
live in `repro` mode, unlike the attention line which cost reproducibility for
-7%.

METHOD
------
Same process, interleaved OFF/ON/ON/OFF for clock fairness, same seed and
control trajectory. Compare the full latent trajectory bit-for-bit and the
per-chunk timings. Runs in `repro` mode (fa2 + eager) so the measurement is not
confounded by any other perf lever.

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python cam_cache_ab.py --scene 04 --seed 42 --chunks 8 --reps 2
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
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--mode", default="repro")
    ap.add_argument("--out_dir", default="output/cam_cache")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = args.mode
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
    print(f"[cc] head={head[:12]} mode={args.mode} "
          f"_CAM_CACHE default={mf._CAM_CACHE}", flush=True)

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

    d = f"examples/cc_{scene}"
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

    def run(use_cache):
        mf._CAM_CACHE = bool(use_cache)
        # drop any cache left from the other arm so the first chunk recomputes
        for blk in pipe.model.blocks:
            blk._cam_cache = None
        reset()
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        lat, hashes = [], []
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
                                            device=dev, dtype=x0.dtype),
                            timesteps[ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            hashes.append(hashlib.sha256(
                x0.detach().float().cpu().numpy().tobytes()).hexdigest()[:16])
        return lat, hashes

    run(False); run(True)                  # warmup both arms
    print("[cc] warmup done", flush=True)

    order = []
    for r in range(args.reps):
        order += [False, True] if r % 2 == 0 else [True, False]
    res = {0: dict(lat=[], hashes=None), 1: dict(lat=[], hashes=None)}
    for uc in order:
        ms, hs = run(uc)
        tag = "ON " if uc else "OFF"
        res[int(uc)]["lat"] += ms
        if res[int(uc)]["hashes"] is None:
            res[int(uc)]["hashes"] = hs
        print(f"[cc] cache {tag} median {statistics.median(ms)*1000:7.1f} ms",
              flush=True)

    off = statistics.median(res[0]["lat"])
    on = statistics.median(res[1]["lat"])
    print(f"\n[cc] ===== camera-injection cache A/B ({n_test} chunks, "
          f"{args.reps} reps, mode={args.mode}) =====")
    print(f"  OFF : {off:7.1f} ms")
    print(f"  ON  : {on:7.1f} ms")
    print(f"  Δ   : {(on-off)*1000:+7.1f} ms "
          f"({100*(on-off)/off:+.2f}%)")

    same = res[0]["hashes"] == res[1]["hashes"]
    n_diff = sum(1 for a, b in zip(res[0]["hashes"], res[1]["hashes"]) if a != b)
    print(f"\n  bit-exact: {'YES' if same else 'NO'}  "
          f"({n_diff}/{n_test} chunks differ)")
    print(f"  OFF hashes: {res[0]['hashes'][:6]} ...")
    print(f"  ON  hashes: {res[1]['hashes'][:6]} ...")

    json.dump(dict(head=head, mode=args.mode, scene=scene, seed=sd,
                   n_chunks=n_test, reps=args.reps,
                   off_median_ms=off * 1000, on_median_ms=on * 1000,
                   delta_ms=(on - off) * 1000,
                   delta_pct=100 * (on - off) / off,
                   bit_exact=same, n_chunks_differing=n_diff,
                   off_hashes=res[0]["hashes"], on_hashes=res[1]["hashes"]),
              open(f"{args.out_dir}/cam_cache_ab.json", "w"), indent=1,
              default=str)
    print(f"\n[cc] wrote {args.out_dir}/cam_cache_ab.json")


if __name__ == "__main__":
    main()
