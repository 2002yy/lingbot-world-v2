#!/usr/bin/env python
"""p2a_gate: one harness for the 21-chunk pre-screen and the 65-chunk gate.

WHY ONE SCRIPT FOR BOTH LENGTHS
-------------------------------
If 21ch passes and 65ch fails, we must be certain the difference comes from
rollout length and not from configuration drift between two scripts. So both
lengths run from this file, with identical seed / conditioning / camera
trajectory / backend / weight-only policy / P0 / P2a / decoder / metric
implementation / frame sampling. The ONLY value that changes is --chunks.

ARMS
    repro  cam_cache=1, ffn0_rowwise=0   -> the bit-exact production config
    fast   cam_cache=1, ffn0_rowwise=1   -> the P2a candidate
    repro2 cam_cache=1, ffn0_rowwise=0   -> same as repro; used as a
                                            deterministic control

The control matters now in a way it did not before: repro/repro must still be
bit-identical after P0, M=1881 and this harness were introduced. If it is, then a
repro-vs-fast divergence is cleanly attributable to P2a's numerical path. If the
control itself drifts, stop and fix the harness before reading any visual number.

The effective configuration is printed and written to config.json, so a result
file always says what was compared.

SEGMENTS
    0-3 / 7-12 / 13-20 short, plus 20-32 / 32-48 / 48-64 when chunks allow.
    Per-segment means AND the deltas between them, because the dangerous shape
    for a recurrent world model is not "differs slightly from the start" but
    "nearly identical -> small mid-run difference -> late blow-up". A late-segment
    acceleration is a yellow flag even if the overall mean looks fine.

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python p2a_gate.py --lhs repro --rhs repro2 --chunks 21 \
      --out-dir output/p2a_gate/21ch --control
  python p2a_gate.py --lhs repro --rhs fast  --chunks 21 \
      --out-dir output/p2a_gate/21ch
  python p2a_gate.py --lhs repro --rhs fast  --chunks 65 \
      --out-dir output/p2a_gate/65ch
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
import torch.nn.functional as F
import torchvision
from PIL import Image

import wan
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

# frozen arm definitions -- the experiment variable is WHICH ARM, nothing else
ARMS = {
    "repro":  dict(cam_cache=1, ffn0_rowwise=0),
    "repro2": dict(cam_cache=1, ffn0_rowwise=0),
    "fast":   dict(cam_cache=1, ffn0_rowwise=1),
}


def _squeeze(fr):
    while fr.dim() > 3:
        fr = fr[0]
    return fr


def edge_map(x):
    g = x.float().mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    e = (gx * gx + gy * gy).sqrt()
    return e / e.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)


def ssim(a, b, win_size=11):
    dev = a.device
    c = torch.arange(win_size, device=dev, dtype=torch.float32) - (win_size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    a = a.float(); b = b.float()
    pad = win_size // 2
    w3 = w.expand(3, 1, win_size, win_size)
    mu_a = F.conv2d(a, w3, padding=pad, groups=3)
    mu_b = F.conv2d(b, w3, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, w3, padding=pad, groups=3) - mu_a2
    sb_ = F.conv2d(b * b, w3, padding=pad, groups=3) - mu_b2
    sab = F.conv2d(a * b, w3, padding=pad, groups=3) - mu_ab
    s = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb_ + C2))
    return s.mean().item()


def psnr(a, b, eps=1e-10):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return 10 * math.log10(1.0 / max(mse, eps))


def rgb_stats(a, b):
    d = (a.float() - b.float()).abs().flatten(1)
    return dict(mean=d.mean().item(),
                p95=torch.quantile(d, 0.95, dim=1).mean().item(),
                max=d.max(dim=1).values.mean().item())


def segments_for(n):
    segs = [("0-3", 0, min(3, n - 1)), ("7-12", 7, min(12, n - 1)),
            ("13-20", 13, min(20, n - 1))]
    if n > 20:
        segs.append(("20-32", 20, min(32, n - 1)))
    if n > 32:
        segs.append(("32-48", 32, min(48, n - 1)))
    if n > 48:
        segs.append(("48-64", 48, n - 1))
    return [s for s in segs if s[1] <= s[2]]


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
    ap.add_argument("--lhs", default="repro")
    ap.add_argument("--rhs", default="fast")
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="run lhs twice as the determinism control")
    args = ap.parse_args()

    for a in (args.lhs, args.rhs):
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; expected one of {sorted(ARMS)}")

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8_DEFER"] = "1"

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    CS = args.chunk_size

    print("=" * 72)
    print(f"  LEFT  {args.lhs}: cam_cache={ARMS[args.lhs]['cam_cache']} "
          f"ffn0_rowwise={ARMS[args.lhs]['ffn0_rowwise']}")
    print(f"  RIGHT {args.rhs}: cam_cache={ARMS[args.rhs]['cam_cache']} "
          f"ffn0_rowwise={ARMS[args.rhs]['ffn0_rowwise']}")
    print(f"  chunks={args.chunks}  chunk_size={CS}  seed={sd}  scene={scene}"
          f"  reps={args.reps}")
    print("=" * 72, flush=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()

    need_rowwise = any(ARMS[a]["ffn0_rowwise"] for a in (args.lhs, args.rhs))
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig,
        quantize_)
    wo, rw = [], []
    for blk in pipe.model.blocks:
        up = blk.ffn[0]
        w = copy.deepcopy(up); quantize_(w, Float8WeightOnlyConfig())
        wo.append(w)
        if need_rowwise:
            r = copy.deepcopy(up)
            quantize_(r, Float8DynamicActivationFloat8WeightConfig())
            rw.append(r)
    gc.collect(); torch.cuda.empty_cache()

    def set_arm(name):
        spec = ARMS[name]
        mf._CAM_CACHE = bool(spec["cam_cache"])
        mods = rw if spec["ffn0_rowwise"] else wo
        with torch.no_grad():
            for blk, m in zip(pipe.model.blocks, mods):
                blk.ffn[0] = m
                blk._cam_cache = None

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    frames_n = (args.frames if hasattr(args, "frames") else 0) or 0
    # frames must cover chunks * chunk_size latent frames, plus the 4:1 tail
    lat_needed = args.chunks * CS
    frames_n = (lat_needed - 1) * 4 + 1
    frames_n = ((frames_n - 1) // 4) * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))
    n_test = min(args.chunks, lat_f // CS)
    print(f"[gate] frames={frames_n} lat_f={lat_f} -> n_test={n_test}", flush=True)

    d = f"examples/gate_{scene}"
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

    def run(arm, decode=True):
        set_arm(arm)
        reset()
        pipe._cross_attn_initialized = False
        gg = torch.Generator(device=dev); gg.manual_seed(sd)
        ms, lats, frames = [], [], []
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
            lats.append(hashlib.sha256(
                x0.detach().float().cpu().numpy().tobytes()).hexdigest()[:16])
            if decode:
                with torch.no_grad():
                    fr = tae.decode_video(
                        x0.to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                        parallel=False, show_progress_bar=False)
                frames.append(_squeeze(fr).float().clamp(0, 1).cpu())
        return ms, lats, frames

    cache = {a: os.path.join(args.out_dir, f"frames_{a}.pt")
             for a in (args.lhs, args.rhs)}
    res = {}
    todo = []
    for a in (args.lhs, args.rhs):
        if os.path.exists(cache[a]) and not args.no_cache:
            blob = torch.load(cache[a], map_location="cpu", weights_only=False)
            res[a] = dict(ms=blob["ms"], hashes=blob["hashes"], frames=blob["frames"])
            print(f"[gate] {a}: loaded {len(blob['frames'])} cached frames",
                  flush=True)
        elif a not in todo:
            todo.append(a)
    if todo:
        for a in todo:
            run(a, decode=False)                 # warmup
        order = []
        for r in range(args.reps):
            seq = todo if r % 2 == 0 else list(reversed(todo))
            order += seq
        print(f"[gate] order={order}", flush=True)
        first = {a: True for a in todo}
        for arm in order:
            t0 = time.perf_counter()
            ms, lats, frames = run(arm, decode=first[arm])
            if arm not in res:
                res[arm] = dict(ms=[], hashes=[], frames=None)
            res[arm]["ms"] += ms
            if first[arm]:
                res[arm]["hashes"] = lats
                res[arm]["frames"] = frames
                first[arm] = False
            print(f"[gate] {arm} median {statistics.median(ms):7.1f} ms "
                  f"({time.perf_counter()-t0:.0f}s wall)", flush=True)
        for a in todo:
            torch.save(dict(ms=res[a]["ms"], hashes=res[a]["hashes"],
                            frames=res[a]["frames"]), cache[a])

    L, R = args.lhs, args.rhs
    print(f"\n[gate] {L} median {statistics.median(res[L]['ms']):7.1f} ms | "
          f"{R} median {statistics.median(res[R]['ms']):7.1f} ms")
    same = res[L]["hashes"] == res[R]["hashes"]
    ndiff = sum(1 for a, b in zip(res[L]["hashes"], res[R]["hashes"]) if a != b)
    print(f"[gate] latent hash: bit-identical={same}  "
          f"({ndiff}/{n_test} chunks differ)")

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
        nm = "facebook/dinov2-base"
        dino = (AutoImageProcessor.from_pretrained(nm),
                AutoModel.from_pretrained(nm).to(dev).eval())
    except Exception as e:
        print(f"[gate] DINOv2 unavailable: {type(e).__name__}")

    @torch.no_grad()
    def dino_feat(x):
        proc, mod = dino
        inp = proc(images=(x.permute(1, 2, 0).numpy() * 255).astype("uint8"),
                   return_tensors="pt").to(dev)
        return F.normalize(mod(**inp).last_hidden_state[:, 0].float(), dim=-1)

    FA, FB = res[L]["frames"], res[R]["frames"]
    n = min(len(FA), len(FB))
    rows = []
    for i in range(n):
        a, b = FA[i], FB[i]
        m = dict(chunk=i, psnr=psnr(a, b), ssim=ssim(a[None], b[None]),
                 edge_ssim=ssim(edge_map(a[None]).repeat(1, 3, 1, 1),
                                edge_map(b[None]).repeat(1, 3, 1, 1)),
                 lpips=lpips_fn(a[None], b[None]))
        m.update({f"rgb_{k}": v for k, v in rgb_stats(a[None], b[None]).items()})
        if dino is not None:
            m["dino_cos"] = F.cosine_similarity(
                dino_feat(a), dino_feat(b), dim=-1).item()
        rows.append(m)

    segs = segments_for(n)
    print(f"\n[gate] ===== {R} vs {L}  (n={n}) =====")
    print("   {:<7} {:>8} {:>8} {:>8} {:>9} {:>9} {:>9}".format(
        "segment", "PSNR", "SSIM", "LPIPS", "edgeSSIM", "dinoCos", "|d|mean"))
    seg_means = {}
    for name, lo, hi in segs:
        seg = rows[lo:hi + 1]
        m = dict(psnr=statistics.mean(x["psnr"] for x in seg),
                 ssim=statistics.mean(x["ssim"] for x in seg),
                 lpips=statistics.mean(x["lpips"] for x in seg),
                 edge_ssim=statistics.mean(x["edge_ssim"] for x in seg),
                 rgb_mean=statistics.mean(x["rgb_mean"] for x in seg))
        if "dino_cos" in seg[0]:
            m["dino_cos"] = statistics.mean(x["dino_cos"] for x in seg)
        seg_means[name] = m
        print("   {:<7} {:>8.2f} {:>8.4f} {:>8.4f} {:>9.4f} {:>9.4f} {:>9.5f}"
              .format(name, m["psnr"], m["ssim"], m["lpips"], m["edge_ssim"],
                      m.get("dino_cos", float("nan")), m["rgb_mean"]))
    print("   per-chunk LPIPS   : " + " ".join(f"{x['lpips']:.4f}" for x in rows))
    print("   per-chunk SSIM    : " + " ".join(f"{x['ssim']:.4f}" for x in rows))
    print("   per-chunk edgeSSIM: " + " ".join(f"{x['edge_ssim']:.4f}" for x in rows))
    if "dino_cos" in rows[0]:
        print("   per-chunk dinoCos : " + " ".join(f"{x['dino_cos']:.4f}" for x in rows))

    def rate(lo, hi):
        return ((rows[hi]["lpips"] - rows[lo]["lpips"]) / (hi - lo)) if hi > lo else float("nan")
    print("\n   LPIPS growth rate (per chunk):")
    for nm, lo, hi in segs:
        if hi > lo and hi < n:
            print(f"     {nm:<7} {rate(lo, hi):+.4f}")
    print("\n   segment deltas (early -> mid -> late):")
    ks = [s[0] for s in segs]
    for i in range(1, len(ks)):
        dm = seg_means[ks[i]]["lpips"] - seg_means[ks[i - 1]]["lpips"]
        ds = seg_means[ks[i]]["dino_cos"] - seg_means[ks[i - 1]].get("dino_cos", float("nan"))
        print(f"     {ks[i-1]}->{ks[i]}: LPIPS {dm:+.4f}   dinoCos {ds:+.4f}")

    out = dict(head=head, lhs=L, rhs=R, lhs_spec=ARMS[L], rhs_spec=ARMS[R],
               note="repro2 is the same definition as repro; used as the "
                    "determinism control",
               scene=scene, seed=sd, chunks=n_test, chunk_size=CS, reps=args.reps,
               max_seq_len=max_seq_len,
               lhs_median_ms=statistics.median(res[L]["ms"]),
               rhs_median_ms=statistics.median(res[R]["ms"]),
               bit_identical=same, n_chunks_differing=ndiff,
               segments=seg_means, per_chunk=rows)
    json.dump(out, open(os.path.join(args.out_dir, "metrics.json"), "w"),
              indent=1, default=str)
    cfgpub = dict(head=head, scene=scene, seed=sd, chunks=n_test,
                  chunk_size=CS, area=args.area,
                  local_attn_size=args.local_attn_size,
                  arms=ARMS, lhs=L, rhs=R, max_seq_len=max_seq_len,
                  frames=frames_n)
    cpath = os.path.join(os.path.dirname(args.out_dir), "config.json")
    if not os.path.exists(cpath):
        json.dump(cfgpub, open(cpath, "w"), indent=1, default=str)
        print(f"[gate] wrote shared {cpath}")
    print(f"[gate] wrote {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
