#!/usr/bin/env python
"""§Quantum-1A: DiT step-count ablation, 3 / 2 / 1 authoritative steps.

THE BOUNDARY THIS RESPECTS. "step0 preview already resembles the final frame" is
evidence about IMAGE-level progressive refinement. It is NOT evidence that a coarser
latent can serve as authoritative state, because the authoritative path also writes
that state into the KV cache, and errors there can accumulate across chunks into
long-horizon trajectory drift. S3-C is the precedent: short-horizon visual similarity
did not imply long-horizon safety.

So the headline question is not single-frame quality:

    after dropping a step, does the KV/state update carry the error into the future?

Only the DiT step count changes. Preview is OFF, so its 26.7 ms cannot contaminate the
authoritative comparison. Scheduler, KV implementation, decode and the rest are
untouched, so a result is attributable.

Layer 1  per-chunk visual quality against the 3-step baseline
Layer 2  KV/state divergence, and whether it grows monotonically with chunk index
Layer 3  long-horizon trajectory drift (10 -> 32 -> 65), gated by a stop rule
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

import wan
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
CTRL = [{"forward": 1.0}, {"yaw": 1.0}, {"forward": 1.0, "right": 1.0},
        {"right": -1.0}, {"pitch": -1.0}, {"forward": 0.7, "yaw": 0.6},
        {"strafe": 0.9}, {"forward": 0.4, "pitch": -0.7}]

# step-count arms as index subsets of scheduler.timesteps (values 999 -> 0)
ARMS = {"A3": [0, 250, 750], "B2": [0, 750], "C1": [0]}


def edge_map(x):
    g = x.mean(0, keepdim=True).unsqueeze(0)
    k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                     device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, k, padding=1)
    gy = F.conv2d(g, k.transpose(-1, -2), padding=1)
    e = (gx * gx + gy * gy).sqrt()
    return (e / e.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6))[0]


import torch.nn.functional as F  # noqa: E402


def ssim(a, b, win=11):
    c = torch.arange(win, dtype=torch.float32, device=a.device) - (win - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    a, b = a.unsqueeze(0).float(), b.unsqueeze(0).float()
    p = win // 2
    w3 = w.expand(3, 1, win, win)
    ma = F.conv2d(a, w3, padding=p, groups=3)
    mb = F.conv2d(b, w3, padding=p, groups=3)
    va = F.conv2d(a * a, w3, padding=p, groups=3) - ma * ma
    vb = F.conv2d(b * b, w3, padding=p, groups=3) - mb * mb
    vab = F.conv2d(a * b, w3, padding=p, groups=3) - ma * mb
    return (((2 * ma * mb + C1) * (2 * vab + C2)) /
            ((ma * ma + mb * mb + C1) * (va + vb + C2))).mean().item()


def to_img(fr):
    f = fr[0] if isinstance(fr, (list, tuple)) else fr
    while f.dim() > 3:
        if f.dim() == 4 and f.shape[0] in (1, 3):
            f = f[:, f.shape[1] // 2]
        elif f.dim() == 4:
            f = f[f.shape[0] // 2]
        else:
            f = f[0]
    if f.dim() == 3 and f.shape[0] not in (1, 3):
        f = f.permute(2, 0, 1)
    if f.dim() == 3 and f.shape[0] == 1:
        f = f.repeat(3, 1, 1)
    return f.float().clamp(0, 1)


def kv_summary(self_kv):
    """Compact, comparable summary of the whole KV cache."""
    tot = 0.0; sq = 0.0; flat = []
    for c in self_kv:
        k = c["k"][:, :int(c["local_end_index"])]
        if k.numel() == 0:
            continue
        tot += float(k.abs().sum())
        sq += float((k.float() ** 2).sum())
        flat.append(k.float().flatten())
    v = torch.cat(flat) if flat else torch.zeros(1)
    return dict(abs_sum=tot, l2=float(sq ** 0.5), n=int(v.numel()),
                vec=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--weight", default="bf16", choices=["bf16", "fp8_lowmem"])
    ap.add_argument("--pixel", default="304x528")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--local_window", type=int, default=6)
    ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_WEIGHT_MODE"] = args.weight
    os.environ["LINGBOT_FP8"] = "0" if args.weight == "bf16" else "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"
    os.environ.setdefault("LINGBOT_STREAM_ENCODE", "1")

    W, H = (int(x) for x in args.pixel.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 92)
    print(f"  Quantum-1A  DiT step ablation   chunks={args.chunks}  "
          f"weight={args.weight}  preview=OFF")
    print(f"  arms: " + "  ".join(f"{k}={v}" for k, v in ARMS.items()))
    print("=" * 92, flush=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_window,
        sink_size=args.sink, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    ALL_TS = pipe.scheduler.timesteps

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    img_pil = img_pil.resize((W, H), Image.BICUBIC)
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch[1] * patch[2])
    F = (args.chunks - 1) * 4 + 1
    kv_size = fsl * args.local_window
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads
    print(f"  geometry {h}x{w} latent {lat_h}x{lat_w} fsl {fsl}  "
          f"P(pixel frames)={F}", flush=True)

    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    mf.bump_cam_epoch()
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe._condition_latent(img, F, h, w)
    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y])
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()

    def plucker(rel):
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(dev)
        p = get_plucker_embeddings(r, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h,
                         w=lat_w).to(pdt)

    def reset_and_build():
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype,
            device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        return self_kv, cross_kv

    def run_arm(arm):
        ts_idx = ARMS[arm]
        timesteps = ALL_TS[ts_idx]
        self_kv, cross_kv = reset_and_build()
        mf.bump_cam_epoch()
        g = torch.Generator(device=dev); g.manual_seed(args.seed)
        noise = torch.randn(16, args.chunks, lat_h, lat_w, generator=g,
                            device=dev)
        rows = []
        prev_pose = np.eye(4)
        for cid in range(args.chunks):
            ctrl = CTRL[cid % len(CTRL)]
            from control_reduce import reduce_controls
            from interactive_runtime import CameraState
            cam = reduce_controls(CameraState(pose=prev_pose, v=np.zeros(3)),
                                  [ctrl])
            chunk_pose = cam.pose
            rel0 = np.linalg.inv(prev_pose) @ chunk_pose
            plk = plucker(rel0)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": fsl,
                  "y": [y.split(1, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * fsl,
                  "max_attention_size": kv_size, "frame_seqlen": fsl}
            cur = noise.split(1, dim=1)[cid]
            torch.cuda.synchronize(); t0 = __import__("time").perf_counter()
            for ti in range(len(timesteps)):
                with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                    npred = pipe.model(
                        x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                        cross_attn_first_call=(ti == 0 and cid == 0), **kw)[0]
                    x0 = pipe._convert_flow_pred_to_x0(
                        flow_pred=npred, xt=cur, timestep=timesteps[ti],
                        scheduler=pipe.scheduler)
                    if ti < len(timesteps) - 1:
                        cur = pipe.scheduler.add_noise(
                            x0, torch.randn(x0.shape, generator=g, device=dev,
                                            dtype=x0.dtype), timesteps[ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            torch.cuda.synchronize()
            ms = (__import__("time").perf_counter() - t0) * 1000
            kv = kv_summary(self_kv)
            rows.append(dict(chunk=cid, ms=ms, kv_abs=kv["abs_sum"],
                             kv_l2=kv["l2"], kv_n=kv["n"],
                             latent=x0.detach().float().cpu().clone()))
            prev_pose = chunk_pose
            print(f"  [{arm}] chunk {cid}: {ms:7.1f} ms  KV L2 {kv['l2']:.2f}",
                  flush=True)
        return rows

    res = {}
    for arm in ARMS:
        print()
        res[arm] = run_arm(arm)

    # ---------------------------------------------------------- layer 1 + 2
    print()
    print("=" * 92)
    print("  LAYER 1  per-chunk visual quality vs the 3-step baseline")
    print("=" * 92)
    base = res["A3"]
    for arm in ("B2", "C1"):
        ss, es = [], []
        for i in range(args.chunks):
            a = to_img(tae.decode_video(
                base[i]["latent"].to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                parallel=False, show_progress_bar=False))
            b = to_img(tae.decode_video(
                res[arm][i]["latent"].to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                parallel=False, show_progress_bar=False))
            ss.append(ssim(a, b))
            es.append(ssim(edge_map(a).repeat(3, 1, 1),
                           edge_map(b).repeat(3, 1, 1)))
        res[arm + "_ssim"] = ss
        res[arm + "_edge"] = es
        print(f"  {arm}: ssim p50 {statistics.median(ss):.4f} "
              f"min {min(ss):.4f}  |  edgeSSIM p50 {statistics.median(es):.4f} "
              f"min {min(es):.4f}")

    print()
    print("=" * 92)
    print("  LAYER 2  KV/state divergence vs the 3-step baseline")
    print("=" * 92)
    print(f"  {'chunk':>6} {'|B2-A| rel L2':>15} {'cos':>8} {'norm ratio':>11} "
          f"{'|C1-A| rel L2':>15} {'cos':>8} {'norm ratio':>11}")
    print("  " + "-" * 80)
    for i in range(args.chunks):
        line = f"  {i:>6}"
        for arm in ("B2", "C1"):
            va = res["A3"][i]
            vb = res[arm][i]
            a = torch.tensor([va["kv_abs"]])
            b = torch.tensor([vb["kv_abs"]])
            rel = abs(vb["kv_abs"] - va["kv_abs"]) / max(abs(va["kv_abs"]), 1e-9)
            nr = vb["kv_l2"] / max(va["kv_l2"], 1e-9)
            line += f" {rel:>15.4f} {'-':>8} {nr:>11.4f}"
        print(line)
    print()
    print("  (KV vectors are not retained across arms in this pass to keep memory")
    print("   bounded; the abs-sum relative deviation and the L2 norm ratio are)")
    print("  latency:")
    for arm in ARMS:
        ms = [r["ms"] for r in res[arm][1:]]
        print(f"    {arm}: p50 {statistics.median(ms):7.1f} ms  "
              f"(n={len(ms)} warm chunks)")

    out = dict(chunks=args.chunks, arms={k: v for k, v in ARMS.items()},
               ms={a: [r["ms"] for r in res[a]] for a in ARMS},
               kv_abs={a: [r["kv_abs"] for r in res[a]] for a in ARMS},
               kv_l2={a: [r["kv_l2"] for r in res[a]] for a in ARMS},
               ssim={a: res.get(a + "_ssim") for a in ("B2", "C1")},
               edge={a: res.get(a + "_edge") for a in ("B2", "C1")})
    with open(f"{args.out_dir}/quantum1a_{args.chunks}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out_dir}/quantum1a_{args.chunks}.json")


if __name__ == "__main__":
    main()
