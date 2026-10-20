#!/usr/bin/env python
"""M0-prod: resolution feasibility frontier for the condition encode.

The 512x768 streamed encode OOMs even at 7 chunks, so the bottleneck is the
per-block activation (512x768 has 2.45x the pixels of 304x528), not the number of
chunks accumulated. That reframes the question:

    is the production default geometry (512x768, native aspect, max_area=480*832)
    even feasible on an 8 GB card?

This measures the streamed-encode peak across resolutions and frame counts and
reports the frontier. Each point is a fresh process (the allocator state must not
leak between points), so the script is driven per-point by the shell wrapper.

It reports the peak even on OOM by catching the failure and reading the
allocator, since a failed point's peak is exactly what defines the frontier.
"""
import argparse
import gc
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
from m1_stream_encode import encode_streamed  # noqa: E402

MB = 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--max_area", type=int, required=True)
    ap.add_argument("--aspect", default="native", choices=["native", "train"])
    ap.add_argument("--weight", default="fp8_lowmem",
                    choices=["bf16", "fp8_lowmem", "fp8"])
    ap.add_argument("--local_window", type=int, default=8)
    ap.add_argument("--sink", type=int, default=2)
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
        convert_model_dtype=False, local_attn_size=args.local_window,
        sink_size=args.sink, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    dev = pipe.device
    vae = pipe.vae
    vm = vae.model
    z_dim = vm.z_dim
    vae_stride, patch = pipe.vae_stride, pipe.patch_size

    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    nat_w, nat_h = img_pil.size
    aspect = (nat_h / nat_w) if args.aspect == "native" else (480 / 832)
    lat_h = round(np.sqrt(args.max_area * aspect) // vae_stride[1] //
                  patch[1] * patch[1])
    lat_w = round(np.sqrt(args.max_area / aspect) // vae_stride[2] //
                  patch[2] * patch[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    fsl = (lat_h * lat_w) // (patch[1] * patch[2])
    F = args.frames

    resident = torch.cuda.memory_allocated() / MB
    print(f"  area={args.max_area} aspect={args.aspect} F={F} weight={args.weight}"
          f"  -> pixel {h}x{w} latent {lat_h}x{lat_w} fsl {fsl}  "
          f"resident_after_load {resident:.0f} MiB", flush=True)

    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    img = torch.nn.functional.interpolate(img[None], size=(h, w),
                                          mode="bicubic").squeeze(0)
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    ok, err = True, None
    try:
        with torch.amp.autocast("cuda", dtype=vae.dtype):
            mu = encode_streamed(vm, img, F, h, w, vae.scale, z_dim, block=4)
        torch.cuda.synchronize()
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    pa = torch.cuda.max_memory_allocated() / MB
    pr = torch.cuda.max_memory_reserved() / MB

    print(f"  RESULT ok={ok}  peak_alloc {pa:.0f}  peak_reserved {pr:.0f} MiB  "
          f"{dt:.2f} s" + (f"  err={err}" if err else ""))
    if ok:
        print(f"  mu {list(mu.shape)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  driver free after {free/MB:.0f} / {total/MB:.0f} MiB")
    rec = dict(area=args.max_area, aspect=args.aspect, F=F, h=h, w=w,
               lat_h=lat_h, lat_w=lat_w, fsl=fsl, weight=args.weight, ok=ok,
               peak_alloc=pa, peak_reserved=pr, s=dt, resident=resident)
    print(f"  JSON {rec}")


if __name__ == "__main__":
    main()
