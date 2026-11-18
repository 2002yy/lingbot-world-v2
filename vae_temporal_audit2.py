#!/usr/bin/env python
"""§35B-1 Temporal early-output audit (monkey-patch instrumentation).

Finds where the decoder's temporal dimension expands 1 -> 4 and how much of the
decode time is spent before vs after that point.

  python vae_temporal_audit2.py --base examples/04
"""
import argparse
import math
import os
import time
import types

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
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

    dec = pipe.vae.model.decoder
    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    lat_h, lat_w = 36, 66
    events = []
    T0 = [time.perf_counter()]

    def wrap(mod, name):
        orig = mod.forward

        def patched(self, *a, **kw):
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            torch.cuda.synchronize()
            shp = tuple(out.shape) if torch.is_tensor(out) else None
            events.append(dict(name=name, shape=shp,
                               t=time.perf_counter() - T0[0]))
            return out
        mod.forward = types.MethodType(patched, mod)

    wrap(dec, "Decoder3d")
    for n, m in dec.named_children():
        wrap(m, f"  {n}")
        for sn, sm in m.named_children():
            wrap(sm, f"    {n}.{sn}")

    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(pipe.device)
    pipe.prewarm(img_pil, max_area=W * H, frame_num=5, chunk_size=1)
    z = torch.randn(16, 1, lat_h, lat_w, device=pipe.device, dtype=torch.float32)

    # warmup
    vae = pipe.vae
    vae.stream_decode_start()
    with torch.no_grad():
        _ = vae.stream_decode_step(z)
    vae.stream_decode_end()
    torch.cuda.empty_cache()

    # first chunk -> 1 frame; then a second chunk -> 4 frames
    events.clear()
    vae.stream_decode_start()
    T0[0] = time.perf_counter()
    with torch.no_grad():
        o1 = vae.stream_decode_step(z)
    torch.cuda.synchronize()
    t1 = time.perf_counter() - T0[0]
    events.clear()
    T0[0] = time.perf_counter()
    with torch.no_grad():
        o2 = vae.stream_decode_step(z)
    torch.cuda.synchronize()
    t2 = time.perf_counter() - T0[0]
    vae.stream_decode_end()
    print(f"[tv] chunk1 out {tuple(o1.shape)} in {t1*1000:.0f}ms ; "
          f"chunk2 out {tuple(o2.shape)} in {t2*1000:.0f}ms", flush=True)

    print(f"\n[tv] ===== decoder timeline (steady chunk, 4 output frames) =====")
    print(f"{'t_ms':>8s} {'module':22s} {'[b,c,t,h,w]':>26s}")
    for e in events:
        print(f"{e['t']*1000:8.1f} {e['name']:22s} {str(e['shape']):>26s}")

    print(f"\n[tv] ===== temporal expansion =====")
    prev_t = None
    for e in events:
        if e["shape"] and len(e["shape"]) == 5:
            mark = ""
            if prev_t is not None and e["shape"][2] != prev_t:
                mark = f"   <== temporal {prev_t} -> {e['shape'][2]}"
            prev_t = e["shape"][2]
            print(f"  t={e['t']*1000:7.1f}ms {e['name']:22s} "
                  f"t={e['shape'][2]}{mark}")


if __name__ == "__main__":
    main()
