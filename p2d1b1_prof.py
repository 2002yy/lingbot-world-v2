#!/usr/bin/env python
"""P2d-1b1 decisive check: did addcmul actually remove kernels, and did the
CUDA time move?

Wall-clock over a 21-chunk rollout has a ~1.3% drift floor (thermal), which
swamps any ~1% effect, so the timing A/B was inconclusive. This uses
torch.profiler instead:

  * kernel launch count per chunk -- if the fusion works, this must DROP
  * total CUDA kernel time per chunk -- the quantity that matters

If the kernel count drops but the time does not, the launch overhead was already
hidden behind asynchronous execution, and the fusion is genuinely neutral. That
is a real answer, not a failure to measure.
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys

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
ARMS = {"off": dict(mod=False, res=False, cam=False),
        "mod_res_cam": dict(mod=True, res=True, cam=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/p2d1b1_prof")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"

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

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lat_needed = args.chunks * CS
    frames_n = (lat_needed - 1) * 4 + 1
    frames_n = ((frames_n - 1) // 4) * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))
    n_test = min(args.chunks, lat_f // CS)

    d = f"examples/p2d1b1_{scene}"
    os.makedirs(d, exist_ok=True)
    import shutil
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    th = int(np.sqrt(W * H * (480 / 832)) // 8 * 8)
    tw = int(np.sqrt(W * H / (480 / 832)) // 8 * 8)
    img = (torch.nn.functional.interpolate(
        torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None].float(),
        size=(th, tw), mode='bicubic').squeeze(0) / 255.0 - 0.5) / 0.5
    h, w = img.shape[1:]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = CS * fsl
    kv_size = fsl * args.local_attn_size
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=CS)
    mf.bump_cam_epoch()
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        img[None].transpose(0, 1).to(dev),
        torch.zeros(3, frames_n - 1, h, w, device=dev)], dim=1)])[0]
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

    def reset():
        for c in self_kv:
            c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False
            c["k"].zero_(); c["v"].zero_()

    def run(arm, prof=None):
        spec = ARMS[arm]
        mf.set_fuse(mod=spec["mod"], res=spec["res"], cam=spec["cam"])
        for blk in pipe.model.blocks:
            blk._cam_cache = None
        reset()
        pipe._cross_attn_initialized = False
        gg = torch.Generator(device=dev); gg.manual_seed(sd)
        lats = []
        for cid in range(n_test):
            c0 = cid * CS
            cur = torch.randn(16, CS, lat_h, lat_w, generator=gg, device=dev)
            pp = get_plucker_embeddings(rel_all[c0:c0 + CS], Ks[None], h, w)
            pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                           c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=CS,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(CS, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * CS * fsl,
                  "max_attention_size": kv_size, "frame_seqlen": fsl}
            for ti in range(len(timesteps)):
                with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                    npred = pipe.model(
                        x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                        cross_attn_first_call=not pipe._cross_attn_initialized,
                        **kw)[0]
                    pipe._cross_attn_initialized = True
                    x0 = pipe._convert_flow_pred_to_x0(
                        flow_pred=npred, xt=cur, timestep=timesteps[ti],
                        scheduler=pipe.scheduler)
                    if ti < len(timesteps) - 1:
                        cur = pipe.scheduler.add_noise(
                            x0, torch.randn(x0.shape, generator=gg,
                                            device=dev, dtype=x0.dtype),
                            timesteps[ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            if prof is not None:
                prof.step()
            lats.append(hashlib.sha256(
                x0.detach().float().cpu().numpy().tobytes()).hexdigest()[:16])
        return lats

    from torch.profiler import profile, ProfilerActivity

    out = {}
    for arm in ("off", "mod_res_cam"):
        run(arm)                                   # warmup
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            lats = run(arm, prof=prof)
        ev = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
        nk = sum(e.count for e in ev)
        tt = sum(e.self_device_time_total for e in ev)   # microseconds
        # top kernels by self time
        top = sorted(ev, key=lambda e: -e.self_device_time_total)[:6]
        out[arm] = dict(n_kernels=nk, cuda_us=tt, hashes=lats,
                        top=[(e.key[:60], e.count,
                              round(e.self_device_time_total, 1)) for e in top])
        print(f"[prof] {arm:<12} kernels={nk:6d}  cuda_total={tt/1000:8.1f} ms  "
              f"per_chunk={tt/1000/n_test:7.2f} ms", flush=True)

    print()
    a, b = out["off"], out["mod_res_cam"]
    print(f"[prof] kernel count   {a['n_kernels']} -> {b['n_kernels']}  "
          f"({b['n_kernels']-a['n_kernels']:+d}, "
          f"{(b['n_kernels']-a['n_kernels'])/a['n_kernels']*100:+.2f}%)")
    print(f"[prof] cuda total     {a['cuda_us']/1000:.1f} -> {b['cuda_us']/1000:.1f} ms  "
          f"({(b['cuda_us']-a['cuda_us'])/a['cuda_us']*100:+.2f}%)")
    print(f"[prof] bit-identical  {a['hashes'] == b['hashes']}")
    print()
    print("[prof] top kernels, off:")
    for k in a["top"]:
        print(f"    {k[0]:<60} n={k[1]:5d} {k[2]/1000:8.2f} ms")
    print("[prof] top kernels, mod_res_cam:")
    for k in b["top"]:
        print(f"    {k[0]:<60} n={k[1]:5d} {k[2]/1000:8.2f} ms")

    with open(os.path.join(args.out_dir, "p2d1b1_prof.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[prof] wrote {args.out_dir}/p2d1b1_prof.json")


if __name__ == "__main__":
    main()
