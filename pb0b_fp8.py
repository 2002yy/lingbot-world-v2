#!/usr/bin/env python
"""P2b-0b: is the FP8 weight-only path itself the loss?

P2b-0 established:
  * the 333.19 ms bucket is 99% the GEMM (only 2.6 ms of surrounding ops)
  * exactly ONE call signature: (1,1881,8960) bf16, contiguous, standard stride,
    bf16 autocast -- so layout/stride/transpose are NOT the problem
  * the module's actual weight is a Float8Tensor
  * measured through the real module: 2.755 ms/call   -> 330.5 ms/chunk
  * but P1-Pre measured the same shape in plain bf16 at 1.6896 ms/call
    -> a 1.07 ms/call = ~128 ms/chunk (8%) gap

If that gap is real, the shipping FP8 weight-only config is paying ~8% of the
chunk in latency to save ~1.30 GiB of VRAM. That is a trade-off worth pricing
explicitly rather than assuming.

This probe captures a REAL ffn.2 input and times, on that exact tensor:

    the quantized module as shipped
    a bf16 dequantised copy of the same weight (same tensor, same layout)
    F.linear on bf16
    rowwise FP8, for completeness

Also establishes the shape's realistic ceiling: the square-GEMM number (37.8
TFLOP/s) is M=N=K and may be unreachable for K=8960, N=1536, so a wide-M variant
of the same shape is measured to separate "this shape" from "this kernel".
"""
import argparse
import copy
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


def bench(fn, n=60, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/p2b0b")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd, CS = args.scene, args.seed, args.chunk_size

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype

    # ---- capture one real ffn.2 input -------------------------------------
    HOLD = {}

    def grab(lin, inp, out):
        if "x" not in HOLD:
            HOLD["x"] = inp[0].detach().clone()
            HOLD["mod"] = lin

    h = pipe.model.blocks[0].ffn[2].register_forward_hook(grab)

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lat_needed = max(CS, 1 * CS)
    frames_n = (lat_needed - 1) * 4 + 1
    frames_n = ((frames_n - 1) // 4) * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))
    n_test = 1

    d = f"examples/p2b0_{scene}"
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

    gg = torch.Generator(device=dev); gg.manual_seed(sd)
    cur = torch.randn(16, CS, lat_h, lat_w, generator=gg, device=dev)
    pp = get_plucker_embeddings(rel_all[0:CS], Ks[None], hh, ww)
    pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                   c1=int(hh // lat_h), c2=int(ww // lat_w))[None]
    plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=CS,
                    h=lat_h, w=lat_w).to(pdt)
    with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
        pipe.model(x=[cur.to(dev)], t=torch.stack([timesteps[0]]).to(dev),
                   context=[pipe._t5_cache[key][0]], seq_len=max_seq_len,
                   y=[y.split(CS, dim=1)[0]],
                   dit_cond_dict={"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                   kv_cache=self_kv, crossattn_cache=cross_kv,
                   current_start=0, max_attention_size=kv_size,
                   frame_seqlen=fsl, cross_attn_first_call=True)
    h.remove()
    torch.cuda.synchronize()

    if "x" not in HOLD:
        raise SystemExit("failed to capture a real ffn.2 input")
    x_real = HOLD["x"]
    mod = HOLD["mod"]
    print("=" * 78)
    print("  P2b-0b: price the FP8 weight-only path")
    print("=" * 78)
    print(f"  captured input : shape={list(x_real.shape)} dtype={x_real.dtype} "
          f"contiguous={x_real.is_contiguous()} stride={list(x_real.stride())}")
    print(f"  module         : {type(mod).__name__} weight={type(mod.weight).__name__}")
    M = int(np.prod(x_real.shape[:-1]))
    K_in = int(getattr(mod, "in_features"))
    N_out = int(getattr(mod, "out_features"))
    fl = 2 * M * K_in * N_out
    print(f"  M={M} K={K_in} N={N_out}  FLOPs={fl/1e9:.2f} G")
    print()

    # ---- de-quantise the shipping weight to plain bf16 ---------------------
    wq = mod.weight
    try:
        w_bf16 = wq.dequantize().to(torch.bfloat16)
        dq_ok = True
    except Exception as e:
        print("  dequantize() unavailable:", e)
        dq_ok = False
        w_bf16 = None
    plain = None
    if dq_ok:
        plain = torch.nn.Linear(K_in, N_out, bias=mod.bias is not None).to(dev)
        with torch.no_grad():
            plain.weight.copy_(w_bf16)
            if mod.bias is not None:
                plain.bias.copy_(mod.bias.to(torch.bfloat16))
        plain = plain.to(torch.bfloat16)
    print("=== A/B on the captured tensor ===")

    def timed(name, fn, check_vs=None):
        t = bench(fn, n=60)
        msg = f"  {name:<34} {t:7.4f} ms  {fl/(t*1e-3)/1e12:6.1f} TFLOP/s"
        if check_vs is not None:
            a = fn(); b = check_vs()
            d = (a.float() - b.float()).abs().max().item()
            same = torch.equal(a, b)
            msg += f"   bit-equal={same} max|d|={d:.3e}"
        print(msg, flush=True)
        return t

    x2 = x_real[0] if x_real.dim() == 3 else x_real   # [M, K]
    print(f"  input for bench: {list(x2.shape)} {x2.dtype} "
          f"contig={x2.is_contiguous()}")

    t_q = timed("shipping FP8 weight-only module",
                lambda: mod(x_real) if x_real.dim() == 3 else mod(x2))
    if plain is not None:
        t_p = timed("dequantised bf16 Linear",
                    lambda: plain(x2), check_vs=lambda: mod(x2))
        t_f = timed("F.linear bf16 (same weight)",
                    lambda: torch.nn.functional.linear(x2, plain.weight,
                                                       plain.bias))
        print()
        print(f"  => FP8 overhead vs bf16: {t_q-t_p:+.4f} ms/call "
              f"({(t_q-t_p)/t_p*100:+.1f}%)")
        print(f"  => per chunk (120 calls): {(t_q-t_p)*120:+.1f} ms")

    print()
    print("=== shape ceiling: does the square number apply to K=8960 N=1536? ===")
    wt = plain.weight if plain is not None else w_bf16
    for MM in (1881, 3762, 7524, 15048):
        xx = torch.randn(MM, K_in, dtype=torch.bfloat16, device=dev)
        t = bench(lambda xx=xx, wt=wt: torch.nn.functional.linear(xx, wt), n=30)
        f_ = 2 * MM * K_in * N_out
        print(f"  M={MM:<6} {t:7.3f} ms  {f_/(t*1e-3)/1e12:6.1f} TFLOP/s")
        del xx
        gc.collect(); torch.cuda.empty_cache()

    print()
    print("=== for reference: square bf16 at the same sizes ===")
    for n in (1536, 8960):
        a = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
        b = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
        t = bench(lambda a=a, b=b: a @ b, n=30)
        print(f"  {n}^3     {t:7.3f} ms  {2*n**3/(t*1e-3)/1e12:6.1f} TFLOP/s")
        del a, b
        gc.collect(); torch.cuda.empty_cache()

    out = dict(M=M, K=K_in, N=N_out, t_quant_ms=t_q,
               t_bf16_ms=(t_p if plain is not None else None),
               fp8_overhead_ms=(t_q - t_p) if plain is not None else None,
               fp8_overhead_per_chunk_ms=((t_q - t_p) * 120)
               if plain is not None else None)
    with open(os.path.join(args.out_dir, "p2b0b.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[P2b-0b] wrote {args.out_dir}/p2b0b.json")


if __name__ == "__main__":
    main()
