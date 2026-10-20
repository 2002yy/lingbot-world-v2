#!/usr/bin/env python
"""S3-A.2: visual correctness gate for the attention-backend change.

ARMS (all eager, same seed, same control trajectory)
----------------------------------------------------
    A1  fa2     reference
    A2  fa2     SAME backend repeated -- the control experiment
    B   sdpa    zero-dependency alternative
    D   hybrid  measured per-shape winner (the performance champion)

WHY A2 EXISTS
-------------
The earlier triage showed SDPA (no quantisation at all) diverges from FA2 by
almost exactly as much as SageAttention does (cosine 0.9435 vs 0.9445 at chunk
20). That is strong evidence the divergence is NOT a Sage quantisation artefact.

But it is not yet proof of "recurrent amplification". If two runs of the SAME
backend already drift apart, then the system has its own nondeterminism/sensitivity
and the backend comparison is measuring that instead. Only the A1-vs-A2 repeat
distinguishes those two explanations, and it costs almost nothing.

WHAT IS MEASURED
----------------
Per chunk, B-vs-A1 and D-vs-A1, on DECODED RGB frames (not latents):
    PSNR, SSIM (own implementation -- skimage is not installed), LPIPS,
    mean/p95/max |dRGB|

Reported per segment, because the latent divergence is known to be
non-stationary (steepest around chunks 7-11, saturating after):
    0-3     initial numerical perturbation
    7-12    strongest amplification
    13-20   does it saturate visually too?

The decisive question is NOT bit-exactness. It is:
    1. does Hybrid cost visibly more than SDPA? (2.36 pp of latency is only
       worth it if the answer is no)
    2. does visual divergence saturate in the back half?
    3. any structural / identity / camera change?
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
import torch.nn.functional as F
import torchvision
from PIL import Image

import wan
import wan.modules.sage_backend as sb
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
ARMS = ["A1", "A2", "B", "D"]
BACKEND = {"A1": "fa2", "A2": "fa2", "B": "sdpa", "D": "hybrid"}


# ------------------------------------------------------------------ SSIM
def _gauss_win(size=11, sigma=1.5, device="cuda", dtype=torch.float32):
    c = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    return g[:, None] @ g[None, :]


def ssim(a, b, win_size=11):
    """a,b: [N,3,H,W] in [0,1]. Returns mean SSIM over the batch."""
    dev = a.device
    w = _gauss_win(win_size, 1.5, dev, torch.float32)[None, None]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    a = a.float(); b = b.float()
    pad = win_size // 2
    mu_a = F.conv2d(a, w.expand(3, 1, win_size, win_size), padding=pad, groups=3)
    mu_b = F.conv2d(b, w.expand(3, 1, win_size, win_size), padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, w.expand(3, 1, win_size, win_size), padding=pad, groups=3) - mu_a2
    sb_ = F.conv2d(b * b, w.expand(3, 1, win_size, win_size), padding=pad, groups=3) - mu_b2
    sab = F.conv2d(a * b, w.expand(3, 1, win_size, win_size), padding=pad, groups=3) - mu_ab
    s = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb_ + C2))
    return s.mean().item()


def psnr(a, b, eps=1e-10):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return 10 * math.log10(1.0 / max(mse, eps))


def rgb_stats(a, b):
    d = (a.float() - b.float()).abs()
    per_px = d.flatten(1)  # [N, -1]
    return dict(mean=d.mean().item(),
                p95=torch.quantile(per_px, 0.95, dim=1).mean().item(),
                max=per_px.max(dim=1).values.mean().item())


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
    ap.add_argument("--sage_min_kv", type=int, default=2508)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/visual_gate")
    ap.add_argument("--no_cache", action="store_true",
                    help="ignore cached frames_*.pt and re-run the rollout")
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
    print(f"[vg] head={head[:12]}", flush=True)
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

    d = f"examples/vg_{scene}"
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

    def run(arm, decode=True):
        sb.set_backend(BACKEND[arm])
        sb.set_sage_min_kv(args.sage_min_kv)
        reset()
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        lat_ms, lats, frames = [], [], []
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
            lat_ms.append(time.perf_counter() - t0)
            lats.append(x0.detach().float().cpu())
            if decode:
                with torch.no_grad():
                    fr = tae.decode_video(
                        x0.to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                        parallel=False, show_progress_bar=False)
                fr = fr[0] if fr.dim() == 5 else fr        # [T,C,H,W]
                frames.append(fr.float().clamp(0, 1).cpu())
        return lat_ms, lats, frames

    order = []
    for a in ARMS:
        order.append(a)
    for a in reversed(ARMS):
        order.append(a)

    # Cache decoded frames per arm so iterating on METRICS never requires
    # re-running the rollout. A full 4-arm interleaved rollout is ~6 minutes.
    cache = {a: os.path.join(args.out_dir, f"frames_{a}.pt") for a in ARMS}
    have_cache = all(os.path.exists(cache[a]) for a in ARMS) and not args.no_cache

    res = {a: dict(ms=[], lats=None, frames=None) for a in ARMS}
    if have_cache:
        print("[vg] using cached frames from", args.out_dir, flush=True)
        for a in ARMS:
            blob = torch.load(cache[a], map_location="cpu", weights_only=False)
            res[a]["frames"] = blob["frames"]
            res[a]["ms"] = blob["ms"]
            print(f"[vg] {a} ({BACKEND[a]:6s}) cached, median "
                  f"{statistics.median(blob['ms'])*1000:7.1f} ms", flush=True)
    else:
        run("A1", decode=False); run("B", decode=False); run("D", decode=False)
        print("[vg] warmup done; running interleaved arms", flush=True)
        for arm in order:
            t0 = time.perf_counter()
            ms, lats, frames = run(arm)
            res[arm]["ms"] += ms
            if res[arm]["frames"] is None:
                res[arm]["lats"] = lats
                res[arm]["frames"] = frames
            print(f"[vg] {arm} ({BACKEND[arm]:6s}) median "
                  f"{statistics.median(ms)*1000:7.1f} ms  "
                  f"({time.perf_counter()-t0:.0f}s wall)", flush=True)
        for a in ARMS:
            torch.save(dict(frames=res[a]["frames"], ms=res[a]["ms"]), cache[a])
        print(f"[vg] cached frames to {args.out_dir}/frames_*.pt", flush=True)

    def _squeeze(fr):
        """TAEHV returns [N,T,C,H,W]; one latent frame decodes to one RGB frame.
        Normalise to [C,H,W] so the metric code and the cache stay consistent."""
        while fr.dim() > 3:
            fr = fr[0]
        return fr

    ref = res["A1"]
    ref_fr = [_squeeze(f) for f in ref["frames"]]
    for a in ARMS:
        res[a]["frames"] = [_squeeze(f) for f in res[a]["frames"]]
    assert ref_fr is not None and len(ref_fr) == n_test

    print(f"\n[vg] ===== decoded-frame metrics vs A1 (FA2) =====")
    try:
        import lpips as lpips_mod
        _lpips = lpips_mod.LPIPS(net="alex").to(dev).eval()
    except Exception as e:
        print("  LPIPS unavailable:", e)
        _lpips = None

    def lpips_fn(a, b):
        if _lpips is None:
            return float("nan")
        return _lpips(a.to(dev) * 2 - 1, b.to(dev) * 2 - 1).mean().item()

    rows = {a: [] for a in ARMS if a != "A1"}
    for i in range(n_test):
        fa = ref_fr[i]                       # [C,H,W]
        for a in rows:
            fb = res[a]["frames"][i]
            m = dict(chunk=i,
                     psnr=psnr(fa, fb),
                     ssim=ssim(fa[None], fb[None]),
                     lpips=lpips_fn(fa[None], fb[None]))
            m.update({f"rgb_{k}": v for k, v in rgb_stats(fa[None], fb[None]).items()})
            rows[a].append(m)

    SEGS = [("0-3", 0, 3), ("7-12", 7, 12), ("13-20", 13, n_test - 1)]
    for a in rows:
        print(f"\n  --- {a} ({BACKEND[a]}) vs A1 ---")
        print("   {:>7} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
            "segment", "PSNR", "SSIM", "LPIPS", "|d|mean", "|d|p95", "|d|max"))
        for name, lo, hi in SEGS:
            seg = rows[a][lo:hi + 1]
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
        print("   per-chunk LPIPS: " +
              " ".join(f"{x['lpips']:.4f}" for x in rows[a]))

    # ---- A1 vs A2 control ----
    if res["A2"]["frames"] is not None:
        ctl = []
        for i in range(n_test):
            fa, fb = ref_fr[i], res["A2"]["frames"][i]
            ctl.append(dict(chunk=i, psnr=psnr(fa, fb), ssim=ssim(fa[None], fb[None]),
                            lpips=lpips_fn(fa[None], fb[None]),
                            **{f"rgb_{k}": v for k, v in
                               rgb_stats(fa[None], fb[None]).items()}))
        print(f"\n  --- CONTROL A2 (FA2 repeat) vs A1 ---")
        print("   PSNR mean {:.2f}  SSIM {:.4f}  LPIPS {:.4f}  "
              "|d|mean {:.5f}".format(
                  statistics.mean(x["psnr"] for x in ctl),
                  statistics.mean(x["ssim"] for x in ctl),
                  statistics.mean(x["lpips"] for x in ctl),
                  statistics.mean(x["rgb_mean"] for x in ctl)))
        print("   per-chunk LPIPS: " + " ".join(f"{x['lpips']:.4f}" for x in ctl))
        rows["A2"] = ctl

    # ---- visualizations ----
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for i in [0, 8, 11, 16, n_test - 1]:
        strip = []
        fa = ref_fr[i]
        for a in ["A1", "B", "D"]:
            strip.append(res[a]["frames"][i])
        dmap = (res["D"]["frames"][i] - fa).abs()
        dmap = (dmap / max(dmap.max().item(), 1e-6)).clamp(0, 1)
        strip.append(dmap.expand(3, -1, -1))
        row = torch.cat(strip, dim=2)
        torchvision.utils.save_image(row, f"{vis_dir}/chunk{i:02d}_A1_SDPA_Hybrid_DIFF.png")
    print(f"\n[vg] wrote visual diffs to {vis_dir}")

    json.dump(dict(head=head, scene=scene, seed=sd, n_chunks=n_test,
                   sage_min_kv=args.sage_min_kv,
                   arms=ARMS, backends=BACKEND, order=order,
                   median_ms={a: statistics.median(res[a]["ms"]) * 1000
                              for a in ARMS},
                   per_chunk=rows,
                   segments={n: [lo, hi] for n, lo, hi in SEGS}),
              open(f"{args.out_dir}/visual_gate.json", "w"), indent=1, default=str)
    print(f"[vg] wrote {args.out_dir}/visual_gate.json")


if __name__ == "__main__":
    main()
