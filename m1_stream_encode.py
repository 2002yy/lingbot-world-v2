#!/usr/bin/env python
"""M1-0: prove a chunked causal encode is equivalent to the whole-clip encode.

Why this is even possible: the encoder is ALREADY chunked internally.

    vae2_1.py:515  def encode(self, x, scale):
                       t = x.shape[2]
                       iter_ = 1 + (t - 1) // 4          # 4-frame blocks
                       for i in range(iter_):
                           if i == 0: out  = self.encoder(x[:, :, :1], fc, fi)
                           else:      out_ = self.encoder(x[:,:,1+4*(i-1):1+4*i], fc, fi)
                                      out = torch.cat([out, out_], 2)

So the OOM at 777 frames is NOT the encoder's internals. It is the CALLER
materialising the whole zero-padded input at once:

    torch.concat([img, torch.zeros(3, F-1, h, w)], dim=1)   # ~1.5 GB
    plus the zeros tensor itself                             # ~1.5 GB

The fix is therefore to feed the blocks -- 1 frame, then 4 frames at a time --
building each block from the image or from freshly allocated zeros, so the full
sequence never exists.

M1-0 is semantic proof only, no performance work: compare the streamed result
against the existing whole-tensor path on shape, dtype, hash, max/mean abs diff,
and the per-latent-timestep diff distribution.

State discipline, borrowed from the vLLM-Omni audit rather than from memory:
the streamed encoder owns a FRESH `clear_cache()` at entry, so neither a prior
whole-clip encode nor prewarm can contaminate it. Prepare state and committed
state are separate objects; correctness does not depend on remembering to bump an
epoch.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS

MB = 2 ** 20


def encode_whole(vae_model, x_full, scale):
    """The existing path, verbatim."""
    return vae_model.encode(x_full, scale)


def encode_streamed(vae_model, img, t_total, h, w, scale, z_dim,
                    block=4, zero_alloc="zeros"):
    """Mirror the internal blocking, but never materialise the full input.

    The reference loop is:
        i = 0        -> frames [0:1]
        i >= 1       -> frames [1 + 4*(i-1) : 1 + 4*i]   (the last may be short)

    Each block here is built on the fly: block 0 from the image, every later
    block from freshly allocated zeros, so peak input memory is one block.
    """
    dev = img.device
    dt = img.dtype
    vae_model.clear_cache()                       # own state, cannot be poisoned

    C = img.shape[0]
    outs = []
    # Block 0 is a single frame, mirroring the library's own special first call.
    # Later calls take `block` frames. At block == 4 the boundaries coincide with
    # the library's internal 4-frame loop and the result is bit-exact; any other
    # value changes where the causal padding falls, so it is a different (not
    # merely differently-batched) computation and must be checked, not assumed.
    vae_model._enc_conv_idx = [0]
    outs.append(vae_model.encoder(img[:, None].unsqueeze(0),
                                  feat_cache=vae_model._enc_feat_map,
                                  feat_idx=vae_model._enc_conv_idx))
    s = 1
    while s < t_total:
        e = min(s + block, t_total)
        n = e - s
        vae_model._enc_conv_idx = [0]
        if zero_alloc == "zeros":
            blk = torch.zeros(C, n, h, w, device=dev, dtype=dt)
        else:
            blk = torch.zeros(C, 1, h, w, device=dev,
                              dtype=dt).expand(C, n, h, w)
        outs.append(vae_model.encoder(blk.unsqueeze(0),
                                      feat_cache=vae_model._enc_feat_map,
                                      feat_idx=vae_model._enc_conv_idx))
        del blk
        s = e
    out = torch.cat(outs, 2)
    del outs
    mu, log_var = vae_model.conv1(out).chunk(2, dim=1)
    if isinstance(scale[0], torch.Tensor):
        mu = (mu - scale[0].view(1, z_dim, 1, 1, 1)) * \
             scale[1].view(1, z_dim, 1, 1, 1)
    else:
        mu = (mu - scale[0]) * scale[1]
    vae_model.clear_cache()
    return mu


def h(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes())\
        .hexdigest()[:16]


def peak_reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="bf16", choices=["bf16", "fp8_lowmem", "fp8"])
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--frames", type=int, default=777,
                    help="777 = the whole-clip build for 65 chunks at chunk_size 3")
    ap.add_argument("--block", type=int, default=4,
                    help="temporal frames per encoder call. 4 mirrors the "
                         "library's own internal blocking and is the only value "
                         "expected to be bit-exact")
    ap.add_argument("--mode", default="both", choices=["both", "streamed"],
                    help="'streamed' skips the whole-clip reference, which is "
                         "needed when the reference cannot fit (bf16 at 777)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    if args.weight == "bf16":
        os.environ["LINGBOT_WEIGHT_MODE"] = "bf16"
        os.environ["LINGBOT_FP8"] = "0"
    else:
        os.environ["LINGBOT_WEIGHT_MODE"] = "fp8_lowmem"
        os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 90)
    print(f"  M1-0 chunked-vs-whole condition encode equivalence   "
          f"weight={args.weight}  frames={args.frames}")
    print("=" * 90, flush=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev = pipe.device
    vae = pipe.vae
    vm = vae.model
    scale = vae.scale
    z_dim = vm.z_dim

    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    # Geometry must match the production harness (pb3_run / m0), which RESIZES to
    # the 480/832 training aspect and then divides by the stride. Deriving the
    # aspect from the native image instead gives 320x480 / lat 40x60, and that is
    # exactly where the earlier mysterious "1800 tokens" came from:
    # 3 * (40/2) * (60/2) = 1800. Forcing the training aspect gives 304x528 /
    # lat 38x66 -> 3*19*33 = 1881.
    th = int(np.sqrt(W * H * (480 / 832)) // 8 * 8)
    tw = int(np.sqrt(W * H / (480 / 832)) // 8 * 8)
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    img = torch.nn.functional.interpolate(img[None], size=(th, tw),
                                          mode="bicubic").squeeze(0)
    hh, ww = img.shape[1:]
    lat_h, lat_w = hh // vae_stride[1], ww // vae_stride[2]
    F = args.frames
    lat_f = 1 + (F - 1) // 4
    print(f"  image -> {hh}x{ww} (latent {lat_h}x{lat_w})   frames={F} -> "
          f"latent frames={lat_f}")
    print(f"  the whole-clip input would be {3*F*hh*ww*2/MB:.0f} MiB of bf16 on "
          f"its own, plus the zeros it is concatenated from", flush=True)

    x_full = torch.concat([
        img[None].cpu().transpose(0, 1),
        torch.zeros(3, F - 1, hh, ww)], dim=1).to(dev).unsqueeze(0)
    print(f"  built x_full {list(x_full.shape)} "
          f"{x_full.element_size()*x_full.nelement()/MB:.0f} MiB", flush=True)

    # ---------------------------------------------------------------- reference
    y_ref = None
    ref_peak_alloc = ref_peak_res = float("nan")
    t_ref = float("nan")
    if args.mode == "both":
        peak_reset()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=vae.dtype):
            y_ref = encode_whole(vm, x_full, scale)
        torch.cuda.synchronize()
        t_ref = time.perf_counter() - t0
        ref_peak_alloc = torch.cuda.max_memory_allocated() / MB
        ref_peak_res = torch.cuda.max_memory_reserved() / MB
        y_ref = y_ref.float().clone()
        print(f"\n  [whole]    {t_ref:6.2f} s  peak_alloc {ref_peak_alloc:7.0f}  "
              f"peak_reserved {ref_peak_res:7.0f} MiB", flush=True)
    else:
        print(f"\n  [whole]    skipped (mode=streamed; the reference cannot fit "
              f"at this weight/frame combination)", flush=True)
    del x_full
    peak_reset()

    # ---------------------------------------------------------------- streamed
    # img is already [C, hh, ww] at the right geometry; encode_streamed adds the
    # batch and time dims itself.
    img_use = img
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=vae.dtype):
        y_str = encode_streamed(vm, img_use, F, hh, ww, scale, z_dim,
                                block=args.block)
    torch.cuda.synchronize()
    t_str = time.perf_counter() - t0
    str_peak_alloc = torch.cuda.max_memory_allocated() / MB
    str_peak_res = torch.cuda.max_memory_reserved() / MB
    y_str = y_str.float().clone()
    print(f"  [streamed] {t_str:6.2f} s  peak_alloc {str_peak_alloc:7.0f}  "
          f"peak_reserved {str_peak_res:7.0f} MiB  block={args.block}",
          flush=True)
    gc.collect(); torch.cuda.empty_cache()

    # ---------------------------------------------------------------- compare
    if y_ref is None:
        print()
        print("=" * 90)
        print("  MEMORY / TIME (streamed only; no reference to compare)")
        print("=" * 90)
        print(f"  streamed   peak_alloc {str_peak_alloc:7.0f}  "
              f"peak_reserved {str_peak_res:7.0f} MiB   {t_str:6.2f} s")
        with open(f"{args.out_dir}/m1_0.json", "w") as f:
            json.dump(dict(frames=F, lat_f=lat_f, block=args.block,
                           mode=args.mode,
                           streamed=dict(peak_reserved_mib=str_peak_res,
                                         peak_alloc_mib=str_peak_alloc,
                                         s=t_str)), f, indent=2)
        print(f"\n[m1-0] wrote {args.out_dir}/m1_0.json")
        return

    print()
    print("=" * 90)
    print("  EQUIVALENCE")
    print("=" * 90)
    same_shape = tuple(y_ref.shape) == tuple(y_str.shape)
    same_dtype = y_ref.dtype == y_str.dtype
    print(f"  shape       ref {list(y_ref.shape)}   str {list(y_str.shape)}   "
          f"equal={same_shape}")
    print(f"  dtype       ref {y_ref.dtype}   str {y_str.dtype}   "
          f"equal={same_dtype}")
    if not same_shape:
        raise SystemExit("shape mismatch -- cannot compare")
    d = (y_ref - y_str).abs()
    mx, mn = d.max().item(), d.mean().item()
    beq = torch.equal(y_ref, y_str)
    print(f"  hash        ref {h(y_ref)}   str {h(y_str)}   "
          f"bit-identical={beq}")
    print(f"  max_abs_diff  {mx:.6e}")
    print(f"  mean_abs_diff {mn:.6e}")
    print(f"  reference scale: |y_ref| max {y_ref.abs().max().item():.4f}  "
          f"mean {y_ref.abs().mean().item():.4f}")
    if not beq and mx > 0:
        rel = mx / max(y_ref.abs().max().item(), 1e-9)
        print(f"  relative max diff {rel:.3e}")
    print()
    print("  per-latent-timestep max|diff| (first 8, last 4):")
    per = d.amax(dim=(0, 1, 3, 4))
    for i in list(range(min(8, per.shape[0]))):
        print(f"    t={i:>3}: {per[i].item():.3e}")
    print(f"    ...")
    for i in range(max(0, per.shape[0] - 4), per.shape[0]):
        print(f"    t={i:>3}: {per[i].item():.3e}")
    print(f"  timesteps with any diff: {(per > 0).sum().item()}/{per.shape[0]}")

    print()
    print("=" * 90)
    print("  MEMORY / TIME")
    print("=" * 90)
    print(f"  whole      peak_reserved {ref_peak_res:7.0f} MiB   {t_ref:6.2f} s")
    print(f"  streamed   peak_reserved {str_peak_res:7.0f} MiB   {t_str:6.2f} s")
    print(f"  saving     {ref_peak_res - str_peak_res:+7.0f} MiB   "
          f"time {t_str - t_ref:+.2f} s "
          f"({(t_str - t_ref)/max(t_ref,1e-9)*100:+.1f}%)")

    verdict = "EXACT" if beq else ("CLOSE" if mn < 1e-6 else "DIFFERENT")
    print()
    print(f"  VERDICT: {verdict}")

    with open(f"{args.out_dir}/m1_0.json", "w") as f:
        json.dump(dict(frames=F, lat_f=lat_f, bit_identical=beq,
                       max_abs_diff=mx, mean_abs_diff=mn,
                       per_timestep_max=[float(v) for v in per.tolist()],
                       whole=dict(peak_reserved_mib=ref_peak_res,
                                  peak_alloc_mib=ref_peak_alloc, s=t_ref),
                       streamed=dict(peak_reserved_mib=str_peak_res,
                                     peak_alloc_mib=str_peak_alloc, s=t_str),
                       verdict=verdict), f, indent=2)
    print(f"\n[m1-0] wrote {args.out_dir}/m1_0.json")


if __name__ == "__main__":
    main()
