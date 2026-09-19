#!/usr/bin/env python
"""S3-C: 65-chunk long-horizon gate for the final candidate D.

    A = FA2    + eager      (original production path -- the reference)
    D = Hybrid + compile    (final candidate)

WHY A REFERENCE IS STILL NEEDED
-------------------------------
The user asked not to run the four arms again, and that is respected -- the
2x2 already established interaction ~ 0. But a drift CURVE is meaningless
without a reference: "LPIPS at chunk 48" is D-vs-something. So this runs the
minimum pair, A and D, interleaved for clock fairness. Two arms, not four.

    65 chunks = frames 257  (n_lat = (257-1)/4 + 1)

CONFIGURATION IS THE PRODUCTION CONFIG, UNCHANGED
-------------------------------------------------
    LINGBOT_ATTN_BACKEND=hybrid   long-KV self -> Sage, cross/short self -> SDPA
    torch.compile(mode="default")
    CUDA Graph OFF, reduce-overhead OFF

WHAT IS MEASURED
----------------
1. Drift curve, segmented 0-7 / 7-13 / 13-20 / 20-32 / 32-48 / 48-64, to test
   whether the observed "grow -> decelerate -> stabilise" shape holds.
2. Whether the divergence is TEXTURE or WORLD STATE. Raw LPIPS cannot tell those
   apart, so alongside it we measure:
     * edge-map SSIM   -- geometry / silhouette preservation
     * DINOv2 feature cosine -- semantic content preservation
   If pixel metrics degrade while edge-SSIM and DINO stay high, the world state
   is intact and only texture detail moved.

NOTE ON CAMERA
--------------
The camera trajectory is an INPUT to this model (we supply c2w and the plucker
embeddings), it is not generated. So a "camera trajectory drift" between A and
D cannot occur by construction; what could differ is how faithfully each renders
the commanded view. Edge-SSIM and DINO are the proxies for that.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taehv import TAEHV  # noqa: E402
from visual_gate import psnr, ssim, rgb_stats  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
ALL_CFG = {"A": ("fa2", False), "B": ("hybrid", False),
           "C": ("fa2", True), "D": ("hybrid", True)}
SEGS = [("0-7", 0, 7), ("7-13", 7, 13), ("13-20", 13, 20),
        ("20-32", 20, 32), ("32-48", 32, 48), ("48-64", 48, 64)]


def _squeeze(fr):
    while fr.dim() > 3:
        fr = fr[0]
    return fr


def edge_map(x):
    """Sobel magnitude, per channel then mean. x: [N,3,H,W] in [0,1]."""
    g = x.float().mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    e = (gx * gx + gy * gy).sqrt()
    return e / e.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)


def edge_ssim(a, b):
    return ssim(edge_map(a).repeat(1, 3, 1, 1), edge_map(b).repeat(1, 3, 1, 1))


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
    ap.add_argument("--frames", type=int, default=257)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--sage_min_kv", type=int, default=2508)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/long_gate")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--arms", default="A,D",
                    help="comma-separated arm ids; the first is the reference")
    args = ap.parse_args()

    global ARMS, CFG
    ARMS = [a for a in args.arms.split(",") if a in ALL_CFG]
    if len(ARMS) < 2:
        raise SystemExit(f"need at least 2 arms from {sorted(ALL_CFG)}")
    CFG = {a: ALL_CFG[a] for a in ARMS}
    print(f"[lg] arms={ARMS} configs={CFG}", flush=True)

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
    sb.set_backend("hybrid")
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
    print(f"[lg] head={head[:12]} frames={frames_n} -> {n_lat} chunks "
          f"(A=fa2+eager, D=hybrid+compile mode={args.mode})", flush=True)

    d = f"examples/lg_{scene}"
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
        backend, do_c = ALL_CFG[arm]
        sb.set_backend(backend)
        sb.set_sage_min_kv(args.sage_min_kv)
        if not do_c:
            return pipe.model
        if arm not in compiled:
            print(f"[lg] compiling {arm} (mode={args.mode})", flush=True)
            compiled[arm] = torch.compile(pipe.model, mode=args.mode,
                                          fullgraph=False)
        return compiled[arm]

    def run(arm, decode=True, nchunks=None):
        n = nchunks or n_lat
        model = model_for(arm)
        reset()
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        lat_ms, frames = [], []
        for ch in range(n):
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
    # Load each arm's cache INDEPENDENTLY so a previously-run reference (e.g. A
    # from the D comparison) can be reused and only the new arm re-rolled.
    todo = []
    for a in ARMS:
        if os.path.exists(cache[a]) and not args.no_cache:
            blob = torch.load(cache[a], map_location="cpu", weights_only=False)
            res[a]["frames"] = blob["frames"]
            res[a]["ms"] = blob["ms"]
            print(f"[lg] {a}: loaded {len(blob['frames'])} cached frames", flush=True)
        else:
            todo.append(a)
    if todo:
        print(f"[lg] warmup (short, 3 chunks) for {todo}", flush=True)
        for a in todo:
            run(a, decode=False, nchunks=3)
        order = []
        for r in range(args.reps):
            seq = todo if r % 2 == 0 else list(reversed(todo))
            order += seq
        print(f"[lg] order={order} chunks={n_lat}", flush=True)
        first = {a: True for a in todo}
        for arm in order:
            t0 = time.perf_counter()
            ms, frames = run(arm, decode=first[arm])
            res[arm]["ms"] += ms
            if first[arm]:
                res[arm]["frames"] = frames
                first[arm] = False
            print(f"[lg] {arm} ({ALL_CFG[arm][0]}"
                  f"{'+compile' if ALL_CFG[arm][1] else '+eager'}) "
                  f"median {statistics.median(ms)*1000:7.1f} ms  "
                  f"({time.perf_counter()-t0:.0f}s wall)", flush=True)
        for a in todo:
            torch.save(dict(frames=res[a]["frames"], ms=res[a]["ms"]), cache[a])
        print(f"[lg] cached frames -> {args.out_dir}/frames_*.pt", flush=True)

    ref_id, cand_id = ARMS[0], ARMS[-1]
    for a in ARMS:
        b_, c_ = ALL_CFG[a]
        print(f"[lg] {a} ({b_}{'+compile' if c_ else '+eager'}) median "
              f"{statistics.median(res[a]['ms'])*1000:7.1f} ms  "
              f"n={len(res[a]['ms'])}")
    dv = (statistics.median(res[cand_id]["ms"])
          - statistics.median(res[ref_id]["ms"]))
    print(f"[lg] total gain ({cand_id} vs {ref_id}) = {dv*1000:+.1f} ms "
          f"({100*dv/statistics.median(res[ref_id]['ms']):+.2f}%)")

    # ---- metrics: pixel, edge, semantic ----
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

    dino = None
    try:
        from transformers import AutoModel, AutoImageProcessor
        name = "facebook/dinov2-base"
        dino = (AutoImageProcessor.from_pretrained(name),
                AutoModel.from_pretrained(name).to(dev).eval())
        print("[lg] DINOv2 loaded for semantic comparison", flush=True)
    except Exception as e:
        print(f"[lg] DINOv2 unavailable ({type(e).__name__}); "
              f"semantic cosine will be skipped", flush=True)

    @torch.no_grad()
    def dino_feat(x):
        """x: [3,H,W] in [0,1] -> normalized CLS feature."""
        proc, mod = dino
        inp = proc(images=(x.permute(1, 2, 0).numpy() * 255).astype("uint8"),
                   return_tensors="pt").to(dev)
        out = mod(**inp).last_hidden_state[:, 0]
        return F.normalize(out.float(), dim=-1)

    fa, fd = res[ref_id]["frames"], res[cand_id]["frames"]
    n = min(len(fa), len(fd), n_lat)
    print(f"\n[lg] comparing {n} chunks ({cand_id} vs {ref_id})", flush=True)
    rows = []
    for i in range(n):
        a, b = fa[i], fd[i]
        m = dict(chunk=i, psnr=psnr(a, b), ssim=ssim(a[None], b[None]),
                 edge_ssim=edge_ssim(a[None], b[None]),
                 lpips=lpips_fn(a[None], b[None]))
        m.update({f"rgb_{k}": v for k, v in rgb_stats(a[None], b[None]).items()})
        if dino is not None:
            m["dino_cos"] = F.cosine_similarity(
                dino_feat(a), dino_feat(b), dim=-1).item()
        rows.append(m)

    print(f"\n[lg] ===== {cand_id} vs {ref_id}, segmented =====")
    print("   {:>7} {:>7} {:>7} {:>8} {:>9} {:>9} {:>9}".format(
        "segment", "PSNR", "SSIM", "LPIPS", "edgeSSIM", "dinoCos", "|d|mean"))
    for name, lo, hi in SEGS:
        seg = rows[lo:min(hi + 1, n)]
        if not seg:
            continue
        dc = (statistics.mean(x["dino_cos"] for x in seg)
              if "dino_cos" in seg[0] else float("nan"))
        print("   {:>7} {:>7.2f} {:>7.4f} {:>8.4f} {:>9.4f} {:>9.4f} {:>9.5f}"
              .format(name,
                      statistics.mean(x["psnr"] for x in seg),
                      statistics.mean(x["ssim"] for x in seg),
                      statistics.mean(x["lpips"] for x in seg),
                      statistics.mean(x["edge_ssim"] for x in seg),
                      dc,
                      statistics.mean(x["rgb_mean"] for x in seg)))
    print("\n   per-chunk LPIPS    : " + " ".join(f"{x['lpips']:.4f}" for x in rows))
    print("   per-chunk edgeSSIM : " + " ".join(f"{x['edge_ssim']:.4f}" for x in rows))
    if "dino_cos" in rows[0]:
        print("   per-chunk dinoCos  : " + " ".join(f"{x['dino_cos']:.4f}" for x in rows))

    print("\n   LPIPS growth rate by segment:")
    for name, lo, hi in SEGS:
        hi2 = min(hi, n - 1)
        if hi2 <= lo:
            continue
        print(f"     {name:>7}: {((rows[hi2]['lpips']-rows[lo]['lpips'])/(hi2-lo)):+.4f}/chunk")

    vis = os.path.join(args.out_dir, "vis")
    os.makedirs(vis, exist_ok=True)
    for i in [20, 32, 48, 64]:
        if i >= n:
            continue
        a, b = fa[i], fd[i]
        dm = (b - a).abs()
        dm = (dm / max(dm.max().item(), 1e-6)).clamp(0, 1)
        torch.cat([a, b, dm.expand(3, -1, -1)], dim=2)
        torchvision.utils.save_image(
            torch.cat([a, b, dm.expand(3, -1, -1)], dim=2),
            f"{vis}/chunk{i:02d}_A_D_DIFF.png")
    print(f"\n[lg] wrote {vis}/chunk*_A_D_DIFF.png")

    json.dump(dict(head=head, scene=scene, seed=sd, n_chunks=n,
                   reps=args.reps, compile_mode=args.mode,
                   sage_min_kv=args.sage_min_kv,
                   ref_arm=ref_id, cand_arm=cand_id,
                   ref_median_ms=statistics.median(res[ref_id]["ms"]) * 1000,
                   cand_median_ms=statistics.median(res[cand_id]["ms"]) * 1000,
                   segments={k: [lo, hi] for k, lo, hi in SEGS},
                   per_chunk=rows),
              open(f"{args.out_dir}/long_gate.json", "w"), indent=1, default=str)
    print(f"[lg] wrote {args.out_dir}/long_gate.json")


if __name__ == "__main__":
    main()
