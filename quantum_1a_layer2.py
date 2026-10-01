#!/usr/bin/env python
"""§Quantum-1A layer 2, done properly: KV/state DIRECTION divergence, not just norm.

The first pass measured the KV L2 norm and found it almost unchanged across step
counts (22368.7 / 22370.9 / 22374.1). That is not evidence of agreement: two vectors
can share a magnitude and point in different directions. The norm alone cannot
distinguish "the KV is nearly the same" from "the KV is completely different but
happens to have a similar magnitude".

So the three arms are run in LOCKSTEP -- all three advance one chunk, then the KV
vectors are compared immediately and released -- which keeps memory bounded and gives
a per-chunk cosine of exactly the quantity the stop rule needs.

    stop rule: if the 2-step arm shows clear state divergence by chunk 10, do not run
    65; fail it.
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time

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
ARMS = {"A3": [0, 250, 750], "B2": [0, 750], "C1": [0]}


def kv_stats(self_kv):
    """Per-layer accumulation of (dot, norm_a2, norm_b2, diff2, n) with no large
    temporary: a full KV vector is 173M elements (347 MB in fp32) and materialising
    several of them OOMs.
    """
    stats = []
    for c in self_kv:
        n = int(c["local_end_index"])
        if n > 0:
            stats.append(c["k"][:, :n].detach())
    return stats


def kv_compare(a_stats, b_stats):
    """Cosine and relative L2 between two KV states, accumulated per layer."""
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    d2 = 0.0
    n = 0
    for ka, kb in zip(a_stats, b_stats):
        m = min(ka.shape[1], kb.shape[1])
        x = ka[:, :m].float().reshape(-1)
        y = kb[:, :m].float().reshape(-1)
        dot += float(torch.dot(x, y))
        na2 += float(torch.dot(x, x))
        nb2 += float(torch.dot(y, y))
        d2 += float(((x - y) ** 2).sum())
        n += int(x.numel())
    cos = dot / max((na2 ** 0.5) * (nb2 ** 0.5), 1e-12)
    rel = (d2 ** 0.5) / max(na2 ** 0.5, 1e-12)
    return cos, rel, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
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
    cfg = WAN_CONFIGS["i2v-1.3B"]
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 92)
    print(f"  Quantum-1A LAYER 2  KV direction divergence, lockstep, "
          f"chunks={args.chunks}")
    print("=" * 92, flush=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_window,
        sink_size=args.sink, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
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

    from control_reduce import reduce_controls
    from interactive_runtime import CameraState

    # one persistent runner per arm, all in lockstep
    runners = {}
    for arm, idx in ARMS.items():
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype,
            device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        g = torch.Generator(device=dev); g.manual_seed(args.seed)
        runners[arm] = dict(
            ts=ALL_TS[idx], self_kv=self_kv, cross_kv=cross_kv,
            noise=torch.randn(16, args.chunks, lat_h, lat_w, generator=g,
                              device=dev),
            g=g, prev_pose=np.eye(4), iter=0)
    mf.bump_cam_epoch()

    rows = []
    for cid in range(args.chunks):
        ctrl = CTRL[cid % len(CTRL)]
        kvs = {}
        for arm in ARMS:
            r = runners[arm]
            cam = reduce_controls(
                CameraState(pose=r["prev_pose"], v=np.zeros(3)), [ctrl])
            chunk_pose = cam.pose
            rel0 = np.linalg.inv(r["prev_pose"]) @ chunk_pose
            plk = plucker(rel0)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": fsl,
                  "y": [y.split(1, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": r["self_kv"], "crossattn_cache": r["cross_kv"],
                  "current_start": cid * fsl,
                  "max_attention_size": kv_size, "frame_seqlen": fsl}
            cur = r["noise"].split(1, dim=1)[cid]
            for ti in range(len(r["ts"])):
                with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                    npred = pipe.model(
                        x=[cur.to(dev)],
                        t=torch.stack([r["ts"][ti]]).to(dev),
                        cross_attn_first_call=(ti == 0 and cid == 0), **kw)[0]
                    x0 = pipe._convert_flow_pred_to_x0(
                        flow_pred=npred, xt=cur, timestep=r["ts"][ti],
                        scheduler=pipe.scheduler)
                    if ti < len(r["ts"]) - 1:
                        cur = pipe.scheduler.add_noise(
                            x0, torch.randn(x0.shape, generator=r["g"],
                                            device=dev, dtype=x0.dtype),
                            r["ts"][ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0],
                           t=torch.stack([r["ts"][-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            r["prev_pose"] = chunk_pose
            r["iter"] += 1
            kvs[arm] = kv_stats(r["self_kv"])

        a = kvs["A3"]
        row = dict(chunk=cid, dim=0)
        for arm in ("B2", "C1"):
            cos, rel, n = kv_compare(a, kvs[arm])
            row[f"{arm}_cos"] = cos
            row[f"{arm}_relL2"] = rel
            row["dim"] = n
        rows.append(row)
        print(f"  chunk {cid:>2}  dim {row['dim']:>7}  "
              f"B2 cos {row['B2_cos']:.6f} relL2 {row['B2_relL2']:.5f}   "
              f"C1 cos {row['C1_cos']:.6f} relL2 {row['C1_relL2']:.5f}",
              flush=True)
        del kvs
        torch.cuda.empty_cache()

    print()
    print("=" * 92)
    print("  KV DIRECTION DIVERGENCE vs the 3-step baseline")
    print("=" * 92)
    for arm in ("B2", "C1"):
        cs = [r[f"{arm}_cos"] for r in rows]
        rs = [r[f"{arm}_relL2"] for r in rows]
        first, last = cs[0], cs[-1]
        trend = cs[-1] - cs[0]
        print(f"  {arm}: cos first {first:.6f} -> last {last:.6f}  "
              f"(trend {trend:+.6f})")
        print(f"      relL2 first {rs[0]:.5f} -> last {rs[-1]:.5f}  "
              f"(trend {rs[-1]-rs[0]:+.5f})")
    print()
    for arm in ("B2", "C1"):
        cs = [r[f"{arm}_cos"] for r in rows]
        deg = cs[0] - cs[-1]
        if deg < 1e-4:
            v = "STABLE (KV direction essentially unchanged)"
        elif deg < 1e-2:
            v = "MILD DRIFT"
        else:
            v = "DIVERGENT"
        print(f"  {arm}: {v}   (cosine loss over {args.chunks} chunks "
              f"{deg:.2e})")

    with open(f"{args.out_dir}/quantum1a_layer2_{args.chunks}.json", "w") as f:
        json.dump(dict(rows=rows, chunks=args.chunks, arms=ARMS), f, indent=2)
    print(f"\n  wrote {args.out_dir}/quantum1a_layer2_{args.chunks}.json")


if __name__ == "__main__":
    main()
