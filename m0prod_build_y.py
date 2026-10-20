#!/usr/bin/env python
"""M0-prod step 1: build the production-geometry (512x768) condition latent.

Production geometry, captured from a live generate() in the M1-1.5 audit:

    pixel 512x768   latent 64x96   frame_seqlen 1536   M(cs=3) 4608

The whole-clip encode for 65 chunks (777 frames) already OOMs at this size, so
this uses the M1 streamed encoder, which is bit-identical to the library's own
internal blocking and never materialises the full padded input.
"""
import argparse
import gc
import hashlib
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_stream_encode import encode_streamed, encode_whole  # noqa: E402

MB = 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--chunks", type=int, default=65)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--max_area", type=int, default=480 * 832)
    ap.add_argument("--weight", default="fp8_lowmem",
                    choices=["bf16", "fp8_lowmem", "fp8"])
    ap.add_argument("--out", required=True)
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

    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=8, sink_size=2,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev = pipe.device
    vae = pipe.vae
    vm = vae.model
    z_dim = vm.z_dim
    vae_stride, patch = pipe.vae_stride, pipe.patch_size

    # ---- production geometry: native aspect + max_area ----
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    nat_w, nat_h = img_pil.size
    aspect = nat_h / nat_w
    lat_h = round(np.sqrt(args.max_area * aspect) // vae_stride[1] //
                  patch[1] * patch[1])
    lat_w = round(np.sqrt(args.max_area / aspect) // vae_stride[2] //
                  patch[2] * patch[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    fsl = (lat_h * lat_w) // (patch[1] * patch[2])
    print("=" * 84)
    print("  M0-prod condition latent build")
    print("=" * 84)
    print(f"  native image {nat_w}x{nat_h}  aspect {aspect:.6f}  "
          f"max_area {args.max_area}")
    print(f"  -> pixel {h}x{w}   latent {lat_h}x{lat_w}   frame_seqlen {fsl}   "
          f"M(cs={args.chunk_size}) = {fsl*args.chunk_size}")
    print(f"  chunks {args.chunks}  -> latent frames {args.chunks*args.chunk_size}"
          f"  -> pixel frames F", flush=True)

    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    img = torch.nn.functional.interpolate(img[None], size=(h, w),
                                          mode="bicubic").squeeze(0)
    F = (args.chunks * args.chunk_size - 1) * 4 + 1
    print(f"  resized image {list(img.shape)}   F = {F}", flush=True)

    # ---- streamed encode (the whole-clip form OOMs at this size) ----
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=vae.dtype):
        mu = encode_streamed(vm, img, F, h, w, vae.scale, z_dim, block=4)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / MB
    peak_res = torch.cuda.max_memory_reserved() / MB
    print(f"  streamed encode: {dt:.2f} s  peak_alloc {peak:.0f}  "
          f"peak_reserved {peak_res:.0f} MiB", flush=True)

    # ---- the mask, exactly as the pipeline builds it ----
    mu = mu.float()
    lat_f = mu.shape[2]
    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, mu.squeeze(0)])
    print(f"  y = {list(y.shape)}  dtype={y.dtype}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(y.cpu(), args.out)
    print(f"  saved {args.out}  "
          f"({os.path.getsize(args.out)/MB:.1f} MiB)")

    with open(args.out + ".json", "w") as f:
        import json
        json.dump(dict(base=args.base, native=[nat_w, nat_h], aspect=aspect,
                       max_area=args.max_area, pixel=[h, w],
                       latent=[lat_h, lat_w], frame_seqlen=fsl,
                       chunks=args.chunks, chunk_size=args.chunk_size, F=F,
                       lat_f=lat_f, y_shape=list(y.shape),
                       encode_s=dt, encode_peak_alloc_mib=peak,
                       encode_peak_reserved_mib=peak_res,
                       weight=args.weight), f, indent=2)
    print("  wrote sidecar json")


if __name__ == "__main__":
    main()
