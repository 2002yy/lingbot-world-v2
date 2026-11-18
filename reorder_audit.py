#!/usr/bin/env python
"""§35A KV/Decode dependency reorder audit.

§34 proved the KV-update has ZERO effect on the current chunk's latent. So the
current frame should not have to wait for it. Proposed reorder:

    old:  denoise(688) -> KV-update(229) -> VAE decode(370)   TTFNF ~1287ms
    new:  denoise(688) -> VAE decode(370) -> KV-update(229)   TTFNF ~1058ms

Throughput is unchanged (688+370+229 = same total, no concurrency), but the
current frame becomes visible ~229ms earlier and the KV-update reads the camera
condition ~370ms later (fresher future cache).

Checks performed here:
  1. does the KV-update forward mutate x0 in place?
  2. is the decoded RGB identical between the two orders?
  3. is the written self_kv_cache identical between the two orders?
  4. is the NEXT chunk identical between the two orders?
  5. measured TTFNF for both orders.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python reorder_audit.py --base examples/04
"""
import argparse
import gc
import hashlib
import math
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
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
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[ro] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch[2] * patch[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = 9
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * pipe.local_attn_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads

    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                        mode='bicubic').transpose(0, 1),
        torch.zeros(3, F - 1, h, w)], dim=1).to(dev)])[0]
    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y])
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, 3, lat_h, lat_w, generator=g, device=dev)

    rel = torch.eye(4)[None].to(dev)
    p = get_plucker_embeddings(rel, Ks[None], h, w)
    p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                  c1=int(h // lat_h), c2=int(w // lat_w))[None]
    PLK = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                    h=lat_h, w=lat_w).to(pdt)

    def fresh():
        sk = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
        ck = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
        return sk, ck

    def denoise(sk, ck, cid):
        g.manual_seed(args.seed)
        pipe._cross_attn_initialized = False
        cur = noise.split(1, dim=1)[cid].clone()
        for ti in range(len(timesteps)):
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": PLK.chunk(1, dim=0)},
                  "kv_cache": sk, "crossattn_cache": ck,
                  "current_start": cid * frame_seqlen,
                  "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=not pipe._cross_attn_initialized, **kw)[0]
                pipe._cross_attn_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=g, device=x0.device,
                                        dtype=x0.dtype), timesteps[ti + 1])
        return x0

    def kv_update(sk, ck, x0, cid):
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
              "y": [y.split(1, dim=1)[cid]],
              "dit_cond_dict": {"c2ws_plucker_emb": PLK.chunk(1, dim=0)},
              "kv_cache": sk, "crossattn_cache": ck,
              "current_start": cid * frame_seqlen,
              "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)

    def cache_sig(sk):
        def tensors(o):
            if torch.is_tensor(o):
                return [o]
            if isinstance(o, dict):
                out = []
                for v in o.values():
                    out += tensors(v)
                return out
            if isinstance(o, (list, tuple)):
                out = []
                for v in o:
                    out += tensors(v)
                return out
            return []
        ts = [t for t in tensors(sk) if t is not None]
        return hashlib.sha256(
            b"".join(t.detach().float().cpu().numpy().tobytes() for t in ts)
        ).hexdigest()[:16]

    def run_order(order, cid=0, sk=None, ck=None):
        """order: 'kv_then_decode' or 'decode_then_kv'. Returns metrics dict."""
        g.manual_seed(args.seed)
        sk = sk if sk is not None else fresh()[0]
        ck = ck if ck is not None else fresh()[1]
        t0 = time.perf_counter()
        x0 = denoise(sk, ck, cid)
        torch.cuda.synchronize()
        t_denoise = time.perf_counter() - t0
        x0_snapshot = x0.clone()

        pipe.vae.stream_decode_start()
        t_d0 = time.perf_counter()
        if order == "kv_then_decode":
            kv_update(sk, ck, x0, cid)
            torch.cuda.synchronize()
            t_kv = time.perf_counter() - t_d0
            with torch.no_grad():
                rgb = pipe.vae.stream_decode_step(x0)
            torch.cuda.synchronize()
            t_dec = time.perf_counter() - t_d0 - t_kv
        else:
            with torch.no_grad():
                rgb = pipe.vae.stream_decode_step(x0)
            torch.cuda.synchronize()
            t_dec = time.perf_counter() - t_d0
            t_kv0 = time.perf_counter()
            kv_update(sk, ck, x0, cid)
            torch.cuda.synchronize()
            t_kv = time.perf_counter() - t_kv0
        pipe.vae.stream_decode_end()
        t_ttfnf = t_denoise + t_dec
        mutated = float((x0 - x0_snapshot).abs().max())
        return dict(x0=x0_snapshot, rgb=rgb, sk=sk, ck=ck,
                    t_denoise=t_denoise, t_kv=t_kv, t_dec=t_dec,
                    t_ttfnf=t_ttfnf, mutated=mutated, sig=cache_sig(sk))

    print("\n[ro] ===== dependency audit =====")
    _ = run_order("kv_then_decode")      # warmup (autotune / allocator)
    A = run_order("kv_then_decode")
    B = run_order("decode_then_kv")
    print(f"[ro] x0 mutated by KV-update in-place? max|dx0| = {A['mutated']:.3e} "
          f"(0 => pure dependency, no in-place write)")
    d_rgb = float((A["rgb"].float() - B["rgb"].float()).abs().max())
    print(f"[ro] decoded RGB  old vs new: max|diff| = {d_rgb:.3e}")
    print(f"[ro] self_kv_cache old={A['sig']}  new={B['sig']}  "
          f"{'IDENTICAL' if A['sig']==B['sig'] else 'DIFFERENT'}")
    print(f"[ro] crossattn_cache "
          f"{'identical' if cache_sig(A['ck'])==cache_sig(B['ck']) else 'DIFFERENT'}")

    # next chunk from each order's cache
    x0n_A = denoise(A["sk"], A["ck"], 1)
    x0n_B = denoise(B["sk"], B["ck"], 1)
    print(f"[ro] next-chunk latent old vs new: "
          f"max|diff| = {float((x0n_A - x0n_B).abs().max()):.3e}")

    print("\n[ro] ===== TTFNF (time to first neural frame) =====")
    print(f"{'order':16s} {'denoise':>9s} {'decode':>8s} {'KV':>7s} {'TTFNF':>9s}")
    for name, r in (("denoise->KV->dec", A), ("denoise->dec->KV", B)):
        print(f"{name:16s} {r['t_denoise']*1000:9.0f} {r['t_dec']*1000:8.0f} "
              f"{r['t_kv']*1000:7.0f} {r['t_ttfnf']*1000:9.0f}")
    gain = (A["t_ttfnf"] - B["t_ttfnf"]) * 1000
    print(f"\n[ro] TTFNF gain = {gain:.0f}ms "
          f"({100*gain/max(A['t_ttfnf']*1000,1e-9):.1f}%)")
    print(f"[ro] total chunk period: old={(A['t_denoise']+A['t_kv']+A['t_dec'])*1000:.0f}ms  "
          f"new={(B['t_denoise']+B['t_kv']+B['t_dec'])*1000:.0f}ms  (throughput unchanged)")


if __name__ == "__main__":
    main()
