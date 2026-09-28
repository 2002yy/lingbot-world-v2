#!/usr/bin/env python
"""M1-1.5: Resolution Authority Audit -- capture REAL execution evidence.

Three geometries are in play and they disagree:

  wan/image2video.py  (the real production generate path)
      aspect = native image h/w, max_area = 480*832 by default
  production_loop.py
      same formula, but max_area = --area = 512*320
  pb3_run.py and the M0/M1/P2b harnesses
      force the 480/832 training aspect and area 512*320

Formula inference is not evidence, so this drives the real `pipe.generate()` and
captures the actual tensors at every stage:

    input image HxW
    condition pixel tensor HxW
    VAE latent HxW
    patchified token count
    grid_sizes
    DiT x.shape
    TAE decode output HxW
    final display HxW

and reports which of the three candidate geometries the real path selects.
"""
import argparse
import gc
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS

MB = 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--weight", default="fp8_lowmem",
                    choices=["bf16", "fp8_lowmem", "fp8"])
    ap.add_argument("--max_area", type=int, default=480 * 832)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--frames", type=int, default=13)
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

    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 96)
    print("  M1-1.5 RESOLUTION AUTHORITY AUDIT")
    print("=" * 96)

    im = Image.open(f"{args.base}/image.jpg")
    nat_w, nat_h = im.size
    nat_aspect = nat_h / nat_w
    print(f"  native image              {nat_w}x{nat_h} (WxH)   "
          f"aspect h/w = {nat_aspect:.6f}")

    intr = np.load(f"{args.base}/intrinsics.npy")
    print(f"  intrinsics                fx={intr.flat[0]:.2f} fy={intr.flat[1]:.2f} "
          f"cx={intr.flat[2]:.2f} cy={intr.flat[3]:.2f}")
    print(f"  intrinsics are documented for 480x832 (HxW) = 832x480 WxH")
    print()

    # --- the three candidate geometries, computed the way each code path does ---
    def geom(aspect, area, tag):
        lat_h = round(np.sqrt(area * aspect) // cfg.vae_stride[1] //
                      cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(np.sqrt(area / aspect) // cfg.vae_stride[2] //
                      cfg.patch_size[2] * cfg.patch_size[2])
        h = lat_h * cfg.vae_stride[1]
        w = lat_w * cfg.vae_stride[2]
        fsl = (lat_h * lat_w) // (cfg.patch_size[1] * cfg.patch_size[2])
        print(f"  {tag:<34} lat {lat_h}x{lat_w}  pixel {h}x{w}  "
              f"frame_seqlen {fsl}  M(cs={args.chunk_size}) {fsl*args.chunk_size}")
        return dict(tag=tag, lat_h=lat_h, lat_w=lat_w, h=h, w=w, fsl=fsl)

    print("  candidate geometries:")
    g_native_prod = geom(nat_aspect, 480 * 832,
                         "image2video.py (native, 480*832)")
    g_native_pl = geom(nat_aspect, 512 * 320,
                       "production_loop.py (native, 512*320)")
    g_train = geom(480 / 832, 512 * 320,
                   "pb3/M0/M1 harness (480/832, 512*320)")
    print()

    # ------------------------------------------------------------------ real run
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev = pipe.device

    CAP = {}

    def hook_mod(name):
        def f(mod, inp, out):
            if name not in CAP:
                try:
                    CAP[name] = (list(inp[0].shape) if torch.is_tensor(inp[0])
                                 else str(type(inp[0])),
                                 list(out.shape) if torch.is_tensor(out)
                                 else str(type(out)))
                except Exception as e:
                    CAP[name] = f"err {e}"
        return f

    hs = []
    hs.append(pipe.vae.model.encoder.register_forward_hook(
        hook_mod("vae.encoder")))
    hs.append(pipe.model.register_forward_hook(hook_mod("dit")))
    hs.append(pipe.vae.model.conv1.register_forward_hook(
        hook_mod("vae.conv1")))

    print("  driving the real pipe.generate() ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        out = pipe.generate(
            input_prompt="A first-person view of a natural landscape with "
                         "smooth camera motion.",
            img=Image.open(f"{args.base}/image.jpg").convert("RGB"),
            action_path=args.base,
            max_area=args.max_area,
            frame_num=args.frames,
            chunk_size=args.chunk_size,
            shift=5.0,
            seed=42,
            offload_model=False,
        )
        ran = True
        err = None
    except Exception as e:
        ran = False
        err = f"{type(e).__name__}: {e}"
        out = None
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / MB
    for h in hs:
        h.remove()

    print(f"  generate() ran={ran}  peak_alloc {peak:.0f} MiB")
    if err:
        print(f"  error: {err}")

    # the authoritative geometry, straight from the pipeline
    lat_h_a = getattr(pipe, "_audit_lat_h", None)
    print()
    print("=" * 96)
    print("  CAPTURED SHAPES (real execution)")
    print("=" * 96)
    for k in ("vae.encoder", "vae.conv1", "dit"):
        if k in CAP:
            print(f"  {k:<14} in={CAP[k][0]}  out={CAP[k][1]}")
    if out is not None:
        if torch.is_tensor(out):
            print(f"  generate() output          {list(out.shape)} {out.dtype}")
        else:
            print(f"  generate() output          {type(out)}")
            try:
                print(f"    len={len(out)}  first "
                      f"{list(out[0].shape) if torch.is_tensor(out[0]) else type(out[0])}")
            except Exception:
                pass

    # decode path
    if torch.is_tensor(out):
        print(f"  -> final tensor {list(out.shape)}")

    print()
    print("=" * 96)
    print("  VERDICT")
    print("=" * 96)
    matched = None
    dit_in = CAP.get("dit", (None, None))[0]
    if isinstance(dit_in, list) and len(dit_in) >= 3:
        # DiT x is [1, L, C]; L = chunk_size * frame_seqlen
        L = dit_in[1] if len(dit_in) > 1 else None
        if L:
            fsl_real = L // args.chunk_size
            for g in (g_native_prod, g_native_pl, g_train):
                if g["fsl"] == fsl_real:
                    matched = g
            print(f"  DiT x.shape {dit_in} -> frame_seqlen = {L}/{args.chunk_size}"
                  f" = {fsl_real}")
            print(f"  matches: {matched['tag'] if matched else 'NONE OF THE THREE'}")
    print()
    print(f"  image2video.py  lat {g_native_prod['lat_h']}x{g_native_prod['lat_w']}"
          f"  pixel {g_native_prod['h']}x{g_native_prod['w']}")
    print(f"  production_loop lat {g_native_pl['lat_h']}x{g_native_pl['lat_w']}"
          f"  pixel {g_native_pl['h']}x{g_native_pl['w']}")
    print(f"  my harness      lat {g_train['lat_h']}x{g_train['lat_w']}"
          f"  pixel {g_train['h']}x{g_train['w']}")

    with open(f"{args.out_dir}/m1_1_5.json", "w") as f:
        json.dump(dict(native=[nat_w, nat_h], aspect=nat_aspect,
                       candidates=dict(image2video=g_native_prod,
                                       production_loop=g_native_pl,
                                       harness=g_train),
                       captured=CAP, ran=ran, error=err,
                       peak_alloc_mib=peak), f, indent=2)
    print(f"\n[audit] wrote {args.out_dir}/m1_1_5.json")


if __name__ == "__main__":
    main()
