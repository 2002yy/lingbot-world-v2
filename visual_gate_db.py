#!/usr/bin/env python
"""S3-B.1: the COMPILE increment visual gate, on the already-validated Hybrid.

    B = Hybrid + eager      (already passed the backend visual gate, LPIPS 0.248 vs FA2)
    D = Hybrid + compile    (the performance champion, 786.6 ms)

WHY D-vs-B AND NOT D-vs-A
--------------------------
A -> D moves TWO variables at once (attention backend AND compiler). Using D-vs-A
as the correctness evidence for D would silently fold the backend's already-
accepted drift into the compiler's. What we actually need to know is narrow:

    given that Hybrid eager is visually acceptable, how much EXTRA drift does
    torch.compile introduce on top of it?

That is exactly D vs B. The latent cosine for this pair is 0.999995 -> 0.889001
at chunk 20, which is NOT small, so this gate decides whether D can be promoted.

Metrics are the same as the backend gate: PSNR / SSIM (own implementation) /
LPIPS / mean-p95-max |dRGB|, on DECODED RGB frames, segmented 0-3, 7-12, 13-20
because the divergence is known to be non-stationary.

Note "compile" means mode="default" (Inductor). NOT reduce-overhead, NOT CUDA
Graph -- that is a separate, closed line.
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
import torchvision
from PIL import Image

import wan
import wan.modules.sage_backend as sb
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taehv import TAEHV  # noqa: E402
from visual_gate import psnr, ssim, rgb_stats  # noqa: E402


def _squeeze(fr):
    """TAEHV returns [N,T,C,H,W]; one latent frame decodes to one RGB frame."""
    while fr.dim() > 3:
        fr = fr[0]
    return fr


PROMPT = "A first-person view of a natural landscape with smooth camera motion."
BACKEND = "hybrid"
ARMS = ["B", "D"]          # B=eager, D=compile
COMPILED = {"B": False, "D": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=21)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--sage_min_kv", type=int, default=2508)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/vg_db")
    ap.add_argument("--no_cache", action="store_true")
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
    print(f"[db] head={head[:12]} B=hybrid+eager D=hybrid+compile(mode={args.mode})",
          flush=True)
    sb.set_backend(BACKEND)
    sb.set_sage_min_kv(args.sage_min_kv)

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    frames_n = (args.frames - 1) // 4 * 4 + 1
    n_lat = (frames_n - 1) // 4 + 1
    n_test = min(args.chunks, n_lat)

    d = f"examples/db_{scene}"
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

    def model_for(arm):
        if not COMPILED[arm]:
            return pipe.model
        if arm not in compiled:
            print(f"[db] compiling {arm} (mode={args.mode})", flush=True)
            compiled[arm] = torch.compile(pipe.model, mode=args.mode,
                                          fullgraph=False)
        return compiled[arm]

    def run(arm, decode=True):
        model = model_for(arm)
        reset()
        sb.stats(reset=True)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        lat_ms, frames = [], []
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
            lat_ms.append(time.perf_counter() - t0)
            if decode:
                with torch.no_grad():
                    fr = tae.decode_video(
                        x0.to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                        parallel=False, show_progress_bar=False)
                frames.append(_squeeze(fr).float().clamp(0, 1).cpu())
        return lat_ms, frames

    cache = {a: os.path.join(args.out_dir, f"frames_{a}.pt") for a in ARMS}
    res = {a: dict(ms=[], frames=None) for a in ARMS}
    if all(os.path.exists(cache[a]) for a in ARMS) and not args.no_cache:
        print("[db] using cached frames", flush=True)
        for a in ARMS:
            blob = torch.load(cache[a], map_location="cpu", weights_only=False)
            res[a]["frames"] = blob["frames"]
            res[a]["ms"] = blob["ms"]
    else:
        run("B", decode=False); run("D", decode=False)   # warmup + compile
        order = []
        for r in range(args.reps):
            order += ARMS if r % 2 == 0 else list(reversed(ARMS))
        print(f"[db] order={order} chunks={n_test}", flush=True)
        for arm in order:
            ms, frames = run(arm)
            res[arm]["ms"] += ms
            if res[arm]["frames"] is None:
                res[arm]["frames"] = frames
            print(f"[db] {arm} ({'compile' if COMPILED[arm] else 'eager  '}) "
                  f"median {statistics.median(ms)*1000:7.1f} ms", flush=True)
        for a in ARMS:
            torch.save(dict(frames=res[a]["frames"], ms=res[a]["ms"]), cache[a])

    print(f"\n[db] B (hybrid eager)   median "
          f"{statistics.median(res['B']['ms'])*1000:7.1f} ms")
    print(f"[db] D (hybrid compile) median "
          f"{statistics.median(res['D']['ms'])*1000:7.1f} ms")
    print(f"[db] compile increment = "
          f"{(statistics.median(res['D']['ms'])-statistics.median(res['B']['ms']))*1000:+.1f} ms")

    try:
        import lpips as lpips_mod
        _lp = lpips_mod.LPIPS(net="alex").to(dev).eval()
    except Exception as e:
        print("  LPIPS unavailable:", e)
        _lp = None

    def lpips_fn(a, b):
        if _lp is None:
            return float("nan")
        return _lp(a.to(dev) * 2 - 1, b.to(dev) * 2 - 1).mean().item()

    ref, tgt = res["B"]["frames"], res["D"]["frames"]
    rows = []
    for i in range(n_test):
        a, b = ref[i], tgt[i]
        m = dict(chunk=i, psnr=psnr(a, b), ssim=ssim(a[None], b[None]),
                 lpips=lpips_fn(a[None], b[None]))
        m.update({f"rgb_{k}": v for k, v in rgb_stats(a[None], b[None]).items()})
        rows.append(m)

    SEGS = [("0-3", 0, 3), ("7-12", 7, 12), ("13-20", 13, n_test - 1)]
    print(f"\n[db] ===== D (compile) vs B (eager), decoded frames =====")
    print("   {:>7} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
        "segment", "PSNR", "SSIM", "LPIPS", "|d|mean", "|d|p95", "|d|max"))
    for name, lo, hi in SEGS:
        seg = rows[lo:hi + 1]
        if not seg:
            continue
        print("   {:>7} {:>8.2f} {:>8.4f} {:>8.4f} {:>10.5f} {:>10.5f} {:>10.5f}"
              .format(name,
                      statistics.mean(x["psnr"] for x in seg),
                      statistics.mean(x["ssim"] for x in seg),
                      statistics.mean(x["lpips"] for x in seg),
                      statistics.mean(x["rgb_mean"] for x in seg),
                      statistics.mean(x["rgb_p95"] for x in seg),
                      statistics.mean(x["rgb_max"] for x in seg)))
    print("   per-chunk LPIPS: " + " ".join(f"{x['lpips']:.4f}" for x in rows))
    print("   per-chunk SSIM:  " + " ".join(f"{x['ssim']:.4f}" for x in rows))

    # saturation check: rate of LPIPS growth per segment
    def rate(lo, hi):
        if hi <= lo:
            return float("nan")
        return (rows[hi]["lpips"] - rows[lo]["lpips"]) / (hi - lo)
    print(f"\n   LPIPS growth rate: 0-7 {rate(0,7):+.4f}/chunk  "
          f"7-13 {rate(7,13):+.4f}/chunk  13-20 {rate(13,20):+.4f}/chunk")

    vis = os.path.join(args.out_dir, "vis")
    os.makedirs(vis, exist_ok=True)
    for i in [0, 8, 11, 16, n_test - 1]:
        a, b = ref[i], tgt[i]
        dm = (b - a).abs()
        dm = (dm / max(dm.max().item(), 1e-6)).clamp(0, 1)
        row = torch.cat([a, b, dm.expand(3, -1, -1)], dim=2)
        torchvision.utils.save_image(row, f"{vis}/chunk{i:02d}_B_D_DIFF.png")
    print(f"\n[db] wrote {vis}/chunk*_B_D_DIFF.png  (B | D | |D-B|)")

    json.dump(dict(head=head, scene=scene, seed=sd, n_chunks=n_test,
                   reps=args.reps, compile_mode=args.mode,
                   backend=BACKEND, sage_min_kv=args.sage_min_kv,
                   b_median_ms=statistics.median(res["B"]["ms"]) * 1000,
                   d_median_ms=statistics.median(res["D"]["ms"]) * 1000,
                   segments={n: [lo, hi] for n, lo, hi in SEGS},
                   per_chunk=rows),
              open(f"{args.out_dir}/vg_db.json", "w"), indent=1, default=str)
    print(f"[db] wrote {args.out_dir}/vg_db.json")


if __name__ == "__main__":
    main()
