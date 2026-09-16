#!/usr/bin/env python
"""§42E resource probe: how many persistent generative entities fit in 8 GB?

The §42D reading was VRAM peak 3307 (3 obj) -> 5413 MiB (5 obj), +64%. But
`max_memory_allocated` is a PROCESS-WIDE peak, and the render phase shares one
KV cache and one model regardless of object count -- the anchors themselves are
tiny [16,1,dh,dw] latent patches. So the growth is most likely SETUP (one
canonicalisation pass per object), not the render loop.

This probe measures the two phases SEPARATELY as N grows, which is what decides
the real object capacity of the card:

    per N:  setup peak / render peak / reserved / chunk latency / host RSS

SOFT STOP at 7.3 GiB: record "approaching capacity" instead of crashing.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python resource_scaling.py --scene 04 --seed 42 --scales 5,7,10
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time

import cv2
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
SKY_BB = (0.05, 0.02, 0.35, 0.16)
SOFT_STOP_MIB = 7.3 * 1024

ALL_SLOTS = [
    ("A1", (0.28, 0.56, 0.52, 0.90), ("crack", 8, 3, 3)),
    ("A2", (0.04, 0.12, 0.20, 0.42), ("crack", 9, 3, 4)),
    ("B1", (0.68, 0.42, 0.88, 0.72), ("crack", 4, 2, 11)),
    ("B2", (0.36, 0.13, 0.52, 0.37), ("crack", 5, 2, 12)),
    ("C1", (0.80, 0.10, 0.94, 0.40), ("hole", 0, 0, 0)),
    ("C2", (0.62, 0.66, 0.76, 0.92), ("hole", 0, 0, 0)),
    ("D1", (0.06, 0.66, 0.22, 0.94), ("crack", 14, 1, 21)),
    ("D2", (0.86, 0.56, 1.00, 0.82), ("crack", 15, 1, 22)),
    ("E1", (0.42, 0.70, 0.56, 0.94), ("hole", 0, 0, 0)),
    ("E2", (0.16, 0.40, 0.28, 0.62), ("hole", 0, 0, 0)),
]
ORDER = ["A1", "B1", "C1", "D1", "E1", "A2", "C2", "B2", "D2", "E2"]


def make_correction(spec, frame, bb):
    kind, n_lines, th, seed = spec
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H),
                      int(bb[2] * W), int(bb[3] * H)]
    out = frame.copy()
    if kind == "hole":
        bx0, by0, bx1, by1 = [int(SKY_BB[0] * W), int(SKY_BB[1] * H),
                              int(SKY_BB[2] * W), int(SKY_BB[3] * H)]
        hole = cv2.resize(frame[by0:by1, bx0:bx1], (x1 - x0, y1 - y0))
        rim = max(2, (x1 - x0) // 12)
        out[y0 + rim:y1 - rim, x0 + rim:x1 - rim] = \
            hole[rim:hole.shape[0] - rim, rim:hole.shape[1] - rim]
        return out
    rng = np.random.RandomState(seed)
    seg = out[y0:y1, x0:x1]
    hh, ww = seg.shape[:2]
    for _ in range(n_lines):
        pts = [(rng.randint(0, max(1, ww)), rng.randint(0, max(1, hh)))]
        for _ in range(4):
            pts.append((int(np.clip(pts[-1][0] + rng.randint(-ww // 5, ww // 5),
                                    0, ww - 1)),
                        int(np.clip(pts[-1][1] + rng.randint(0, hh // 3),
                                    0, hh - 1))))
        cv2.polylines(out[y0:y1, x0:x1], [np.array(pts, np.int32)], False,
                      (12, 10, 10), th)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scales", default="5,7,10")
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/resource_scale")
    args = ap.parse_args()

    try:
        import psutil
    except Exception:
        psutil = None

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    scales = [int(x) for x in args.scales.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[rs] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    from eval_two_layer import load_models
    load_models()

    # must be 1 (mod 4) for the mask/conditioning layout
    frames_n = (args.frames - 1) // 4 * 4 + 1
    n_lat = (frames_n - 1) // 4 + 1
    d = f"examples/rs_{scene}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch_sz[1] * patch_sz[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch_sz[2] * patch_sz[2])
    h = lat_h * vae_stride[1]; w = lat_w * vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = int(math.ceil(fsl / pipe.sp_size)) * pipe.sp_size
    kv_size = fsl * 6
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
    traj = np.load(f"examples/{scene}/poses.npy")
    if len(traj) < frames_n:
        reps = frames_n // len(traj) + 1
        traj = np.tile(traj, (reps, 1, 1))[:frames_n]
    traj = traj[:frames_n]
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    print(f"[rs] {frames_n} frames -> {n_lat} chunks; scales {scales}; "
          f"latent {lat_h}x{lat_w}", flush=True)

    def build_y(first):
        with torch.no_grad():
            z = pipe.vae.encode([torch.concat([
                first, torch.zeros(3, frames_n - 1, h, w)], dim=1).to(dev)])[0]
        m = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
        m[:, 1:] = 0
        m = torch.concat([torch.repeat_interleave(m[:, 0:1], repeats=4, dim=1),
                          m[:, 1:]], dim=1)
        m = m.view(1, m.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        return torch.concat([m, z]).detach()

    y = build_y(torch.nn.functional.interpolate(
        img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1))
    ref_img = np.array(img_pil.resize((w, h), Image.BICUBIC))
    ymap = {}
    for nm, bb, spec in ALL_SLOTS:
        corr = make_correction(spec, ref_img, bb)
        ymap[nm] = build_y(TF.to_tensor(Image.fromarray(corr)).sub_(0.5)
                           .div_(0.5).unsqueeze(0).transpose(0, 1))
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def gen(y_cond, max_chunks=None):
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers,
            shape=[1, kv_size, ma.num_heads // pipe.sp_size, ma.dim // ma.num_heads],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers,
            shape=[1, 512, ma.num_heads, ma.dim // ma.num_heads],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        dts, decs = [], []
        N = n_lat if max_chunks is None else min(n_lat, max_chunks)
        for cid in range(N):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            p = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                          c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y_cond.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * fsl,
                  "max_attention_size": kv_size, "frame_seqlen": fsl}
            t0 = time.perf_counter()
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
            dts.append(time.perf_counter() - t0)
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            t1 = time.perf_counter()
            with torch.no_grad():
                tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                 parallel=False, show_progress_bar=False)
            decs.append(time.perf_counter() - t1)
        del self_kv, cross_kv
        gc.collect(); torch.cuda.empty_cache()
        return dts, decs

    results = []
    stopped = None
    for N in scales:
        names = ORDER[:N]
        print(f"\n[rs] ===== N={N}: {names} =====", flush=True)
        # ---- render phase (shared model + KV, so near N-independent) ----
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        dts, decs = gen(y)
        render_s = time.perf_counter() - t0
        render_peak = torch.cuda.max_memory_allocated() / 2**20
        render_resv = torch.cuda.max_memory_reserved() / 2**20
        # ---- setup phase: one canonicalisation pass per object ----
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        for nm in names:
            gen(ymap[nm], max_chunks=args.canon_chunks)
        setup_s = time.perf_counter() - t1
        setup_peak = torch.cuda.max_memory_allocated() / 2**20
        rss = psutil.Process().memory_info().rss / 2**20 if psutil else float("nan")
        free_mib = torch.cuda.mem_get_info()[0] / 2**20
        obj = dict(n=N, names=names,
                   render_peak_mib=float(render_peak),
                   render_reserved_mib=float(render_resv),
                   setup_peak_mib=float(setup_peak),
                   driver_free_mib=float(free_mib),
                   host_rss_mib=float(rss),
                   dit_latency_ms=float(np.mean(dts)) * 1000.0,
                   decode_latency_ms=float(np.mean(decs)) * 1000.0,
                   chunk_latency_ms=float(np.mean(dts) + np.mean(decs)) * 1000.0,
                   render_s=float(render_s), setup_s=float(setup_s))
        results.append(obj)
        print(f"[rs]   render peak {render_peak:.0f} MiB | setup peak "
              f"{setup_peak:.0f} MiB | driver free {free_mib:.0f} MiB | "
              f"chunk {obj['chunk_latency_ms']:.0f} ms", flush=True)
        if max(render_peak, setup_peak) >= SOFT_STOP_MIB:
            stopped = N
            print(f"[rs]   SOFT STOP at N={N}", flush=True)
            break

    print("\n[rs] ===== §42E resource scaling curve =====")
    print(f"  {'N':>3s} {'render peak':>12s} {'setup peak':>11s} "
          f"{'driver free':>12s} {'host RSS':>9s} {'chunk ms':>9s} "
          f"{'DiT ms':>7s} {'dec ms':>7s}")
    for r in results:
        print(f"  {r['n']:3d} {r['render_peak_mib']:12.0f} "
              f"{r['setup_peak_mib']:11.0f} {r['driver_free_mib']:12.0f} "
              f"{r['host_rss_mib']:9.0f} {r['chunk_latency_ms']:9.0f} "
              f"{r['dit_latency_ms']:7.0f} {r['decode_latency_ms']:7.0f}")
    print(f"\n  resource breakpoint: "
          f"{('N=' + str(stopped) + ' (soft stop)') if stopped else 'not reached'}")
    if len(results) >= 2:
        dr = results[-1]["render_peak_mib"] - results[0]["render_peak_mib"]
        ds = results[-1]["setup_peak_mib"] - results[0]["setup_peak_mib"]
        dn = results[-1]["n"] - results[0]["n"]
        print(f"\n  growth over {dn} extra objects:")
        print(f"    render phase: {dr:+.0f} MiB  ({dr/max(dn,1):+.0f} MiB/object)")
        print(f"    setup phase : {ds:+.0f} MiB  ({ds/max(dn,1):+.0f} MiB/object)")
        print("  -> whichever phase dominates decides the real object capacity")
    json.dump(results, open(f"{args.out_dir}/resource_curve.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
