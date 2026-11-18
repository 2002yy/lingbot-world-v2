#!/usr/bin/env python
"""§35B-1 Temporal early-output audit for the VAE decoder.

Q: inside the 370ms decode, when is output frame 0 mathematically finished?
If the 4 frames are produced by a single final conv3d over a temporal block,
there is no early-output headroom (clean negative result). If the temporal dim
expands early and the remaining work is per-frame spatial, frame 0 could be
exported early.

Part 1/2: instrument every decoder submodule (input/output shapes + timing) to
find where the temporal dimension expands 1 -> 4 and how much time is spent
before vs after that point.

  python vae_temporal_audit.py
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-world-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast",
        assets_dir=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    print("[tv] pipe built", flush=True)

    vae = pipe.vae
    dec = vae.model.decoder
    conv_num = 0
    for m in dec.modules():
        from wan.modules.vae2_1 import CausalConv3d
        if isinstance(m, CausalConv3d):
            conv_num += 1
    print(f"[tv] decoder causal conv3d count = {conv_num}", flush=True)

    # ---- instrument ----
    events = []
    handles = []

    def make_hook(name, kind):
        def hook(mod, inp, out):
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            shp = None
            if isinstance(out, torch.Tensor):
                shp = tuple(out.shape)
            elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
                shp = tuple(out[0].shape)
            events.append(dict(name=name, kind=kind, shape=shp,
                               t=t1 - T0))
        return hook

    T0 = time.perf_counter()
    handles.append(dec.register_forward_hook(make_hook("DECODER", "Decoder3d")))
    for name, mod in dec.named_children():
        handles.append(mod.register_forward_hook(make_hook(name, type(mod).__name__)))
        for sub_name, sub in mod.named_children():
            handles.append(sub.register_forward_hook(
                make_hook(f"{name}.{sub_name}", type(sub).__name__)))
    print(f"[tv] registered {len(handles)} hooks", flush=True)

    # ---- build a real latent via a short denoise (reuse the pipe path) ----
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(pipe.device)
    h, w = img.shape[1:]
    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch[2] * patch[2])
    h2 = lat_h * vae_stride[1]
    w2 = lat_w * vae_stride[2]
    pipe.prewarm(img_pil, max_area=W * H, frame_num=5, chunk_size=1)
    z = torch.randn(1, 16, 1, lat_h, lat_w, device=pipe.device,
                    dtype=torch.float32)

    # warmup decode (autotune)
    vae.stream_decode_start()
    with torch.no_grad():
        _ = vae.stream_decode_step(z[0])
    vae.stream_decode_end()
    torch.cuda.empty_cache()

    events.clear()
    T0 = time.perf_counter()
    vae.stream_decode_start()
    with torch.no_grad():
        out = vae.stream_decode_step(z[0])
    torch.cuda.synchronize()
    total = time.perf_counter() - T0
    vae.stream_decode_end()
    print(f"[tv] decode out shape {tuple(out.shape)}  total={total*1000:.0f}ms",
          flush=True)
    for hd in handles:
        hd.remove()

    print(f"\n[tv] ===== decoder timeline (first occurrence per module) =====")
    print(f"{'t_ms':>8s} {'module':34s} {'kind':16s} {'shape':28s}")
    seen = set()
    for e in events:
        if e["name"] in seen:
            continue
        seen.add(e["name"])
        print(f"{e['t']*1000:8.1f} {e['name']:34s} {e['kind']:16s} "
              f"{str(e['shape']):28s}")

    # where does the temporal dim reach 4?
    print(f"\n[tv] ===== temporal expansion trace =====")
    for e in events:
        if e["shape"] and len(e["shape"]) == 5:
            print(f"  t={e['t']*1000:7.1f}ms {e['name']:34s} "
                  f"[b,c,t,h,w]={e['shape']}")


if __name__ == "__main__":
    main()
