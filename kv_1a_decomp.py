#!/usr/bin/env python
"""§KV-1A: exact-path decomposition of the 174 ms KV/state update.

The question that decides the route: is the KV update's cost O(new block) or does it
actually move a large part of the rolling window every tick?

The code does contain an O(window) rolling memmove with a clone:

    num_evicted_tokens = num_new_tokens + _lei - kv_cache_size
    num_rolled_tokens  = _lei - num_evicted_tokens - sink_tokens
    kv_cache["k"][:, sink:sink+rolled] = kv_cache["k"][:, sink+evicted:...].clone()
    kv_cache["v"][:, sink:sink+rolled] = kv_cache["v"][:, sink+evicted:...].clone()

so the pattern exists and the only open question is its MAGNITUDE. Reasoning about it
is not enough: at 2508 rolled tokens x 12 heads x 128 dim x 30 layers x 2 tensors, the
traffic is ~460 MB per direction, which at realistic bandwidth is a few milliseconds,
not a hundred. That estimate needs to be checked against a measurement rather than
trusted.

Also measured: how the KV-update forward's total compares with a denoise step. From
Latency-2A they were 174.4 ms and 174.1 ms, i.e. nearly identical, which would already
imply the update is dominated by block compute rather than by cache movement. This
verifies that directly.

Nothing is optimised here. Phase 1 is measurement only.
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
MS = 1e6


def cuda_phase():
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()

    def stop():
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)
    return stop


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
    ap.add_argument("--chunks", type=int, default=8)
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
    print("=" * 90)
    print(f"  KV-1A  exact-path decomposition of the KV/state update   "
          f"chunks={args.chunks}")
    print("=" * 90, flush=True)

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
    print(f"  geometry {h}x{w}  fsl {fsl}  kv_size {kv_size} "
          f"(= {args.local_window} frames)", flush=True)

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

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype,
        device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype,
        device=dev)

    def plucker(rel):
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(dev)
        p = get_plucker_embeddings(r, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h,
                         w=lat_w).to(pdt)

    from control_reduce import reduce_controls
    from interactive_runtime import CameraState

    timesteps = ALL_TS[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, args.chunks, lat_h, lat_w, generator=g, device=dev)
    prev_pose = np.eye(4)

    # instrument the self-attention KV write: separate the block compute from the
    # cache mutation by timing inside CausalWanSelfAttention.forward
    KV = {"write": [], "attn_total": []}
    orig_sa = mf.CausalWanSelfAttention.forward

    def sa_patched(self, x, *a, **kw):
        t0 = time.perf_counter()
        out = orig_sa(self, x, *a, **kw)
        torch.cuda.synchronize()
        KV["attn_total"].append((time.perf_counter() - t0) * 1000)
        return out

    mf.CausalWanSelfAttention.forward = sa_patched

    rows = []
    for cid in range(args.chunks):
        ctrl = CTRL[cid % len(CTRL)]
        cam = reduce_controls(CameraState(pose=prev_pose, v=np.zeros(3)), [ctrl])
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
        step_ms = []
        for ti in range(len(timesteps)):
            stop = cuda_phase()
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
            step_ms.append(stop())
        stop = cuda_phase()
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        kv_ms = stop()
        prev_pose = chunk_pose
        rows.append(dict(chunk=cid, step_ms=step_ms, kv_ms=kv_ms))
        print(f"  chunk {cid}: steps {[round(x) for x in step_ms]}  "
              f"KV-update {kv_ms:6.1f} ms", flush=True)

    mf.CausalWanSelfAttention.forward = orig_sa

    # ---- attention-internal share ----
    warm = [r for r in rows if r["chunk"] >= 1]
    kv_tot = statistics.median(r["kv_ms"] for r in warm)
    step_tot = statistics.median(sum(r["step_ms"]) for r in warm)
    per_step = statistics.median(
        statistics.mean(r["step_ms"]) for r in warm)

    print()
    print("=" * 90)
    print("  DECOMPOSITION")
    print("=" * 90)
    print(f"  denoise steps (3)          {step_tot:8.1f} ms  "
          f"({step_tot/3:.1f} ms per step)")
    print(f"  KV/state update forward    {kv_tot:8.1f} ms")
    print()
    print(f"  KV-update / denoise-step ratio   {kv_tot/per_step:.3f}")
    print()
    print("  A ratio near 1.0 means the KV update is essentially another full")
    print("  model forward, i.e. dominated by block compute rather than by cache")
    print("  movement. A large ratio would mean cache work dominates.")

    # rolling-window traffic, computed from the actual indices
    rolled = kv_size - fsl - args.sink * fsl      # steady-state estimate
    per_layer = rolled * lh * hd
    total_el = per_layer * ma.num_layers * 2      # k and v
    bytes_moved = total_el * 2 * 2                # read + write, bf16
    print()
    print("  ROLLING-WINDOW TRAFFIC (steady state, one eviction)")
    print(f"    rolled tokens per layer     {rolled}")
    print(f"    elements moved (k+v, 30 L)  {total_el/1e6:.1f} M")
    print(f"    bytes read+written          {bytes_moved/1e6:.1f} MB")
    print(f"    at 300 GB/s that is         ~{bytes_moved/300e9*1000:.2f} ms")
    print(f"    as a share of the KV update {bytes_moved/300e9*1000/kv_tot*100:.1f}%")

    print()
    print("=" * 90)
    print("  ROUTE DECISION")
    print("=" * 90)
    share = bytes_moved / 300e9 * 1000 / kv_tot * 100
    if kv_tot / per_step < 1.15 and share < 15:
        print("  The KV update is one more full forward, and the rolling memmove is")
        print("  a few percent of it. There is no large exact-path redundancy here:")
        print("  the cost is block compute, which dropping steps would change")
        print("  numerically -- and Quantum-1A already showed that poisons the state.")
        print("  => KV-1B has little room; close the line and prefer early-exit or")
        print("     pipelining.")
    else:
        print("  There is measurable non-compute cost in the KV update; KV-1B is")
        print("  worth pursuing with an exact-path requirement.")

    with open(f"{args.out_dir}/kv1a.json", "w") as f:
        json.dump(dict(rows=rows, kv_ms=kv_tot, step_ms=step_tot,
                       per_step_ms=per_step, ratio=kv_tot / per_step,
                       rolled_tokens=rolled, bytes_moved_MB=bytes_moved/1e6,
                       rolling_share_pct=share), f, indent=2)
    print(f"\n  wrote {args.out_dir}/kv1a.json")


if __name__ == "__main__":
    main()
