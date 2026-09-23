#!/usr/bin/env python
"""P2d-1b1: does replacing `a + b*c` with addcmul survive the rollout bit-exactly?

The elementwise matrix (r3) and the corrected expression test (p2d1b1_expr) both
say the five chains are bit-exact in isolation. That is necessary but not
sufficient: this is a recurrent rollout, and S3-C already showed that a tiny
numerical change gets amplified into a different world within ~32 chunks. So the
question here is narrower and harder:

    is the latent hash STILL bit-identical after 21 chunks?

If yes, the fusion is repro-safe and we can measure what it buys. If no, the
chain is closed regardless of how good the isolated exactness looked.

Arms (one process, same weights, same seed, same conditioning, only the fuse
flags differ):

    off          mod=0 res=0 cam=0    the control
    mod          mod=1                model line 349 + 391
    mod_res      mod=1 res=1          + model lines 353 + 393
    mod_res_cam  mod=1 res=1 cam=1    + model line 382

`off` is also run as `off2` to confirm the harness itself is deterministic: if
off != off2 the harness is broken and no other number may be read.
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import statistics
import subprocess
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

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

ARMS = {
    "off":         dict(mod=False, res=False, cam=False),
    "off2":        dict(mod=False, res=False, cam=False),
    "mod":         dict(mod=True,  res=False, cam=False),
    "mod_res":     dict(mod=True,  res=True,  cam=False),
    "mod_res_cam": dict(mod=True,  res=True,  cam=True),
}


def _squeeze(fr):
    if isinstance(fr, (list, tuple)):
        fr = fr[0]
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=21)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--arms", default="off,off2,mod,mod_res,mod_res_cam")
    ap.add_argument("--out_dir", default="output/p2d1b1")
    args = ap.parse_args()

    # repro production config: shipping FP8, no P2a ffn.0 rowwise, cam cache on
    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_names:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; expected one of {sorted(ARMS)}")

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    CS = args.chunk_size

    print("=" * 72)
    for a in arm_names:
        print(f"  {a:<12} mod={int(ARMS[a]['mod'])} res={int(ARMS[a]['res'])} "
              f"cam={int(ARMS[a]['cam'])}")
    print(f"  chunks={args.chunks} chunk_size={CS} seed={sd} scene={scene} "
          f"reps={args.reps}")
    print("=" * 72, flush=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()

    def set_arm(name):
        spec = ARMS[name]
        mf.set_fuse(mod=spec["mod"], res=spec["res"], cam=spec["cam"])
        for blk in pipe.model.blocks:
            blk._cam_cache = None

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lat_needed = args.chunks * CS
    frames_n = (lat_needed - 1) * 4 + 1
    frames_n = ((frames_n - 1) // 4) * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))
    n_test = min(args.chunks, lat_f // CS)
    print(f"[1b1] frames={frames_n} lat_f={lat_f} -> n_test={n_test}", flush=True)

    d = f"examples/p2d1b1_{scene}"
    os.makedirs(d, exist_ok=True)
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

    def run(arm, decode=False):
        set_arm(arm)
        reset()
        pipe._cross_attn_initialized = False
        gg = torch.Generator(device=dev); gg.manual_seed(sd)
        ms, lats, frames = [], [], []
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
            torch.cuda.synchronize(); t0 = time.perf_counter()
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
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
            lats.append(hashlib.sha256(
                x0.detach().float().cpu().numpy().tobytes()).hexdigest()[:16])
            if decode:
                with torch.no_grad():
                    fr = tae.decode_video(
                        x0.to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                        parallel=False, show_progress_bar=False)
                frames.append(_squeeze(fr).float().clamp(0, 1).cpu())
        return ms, lats, frames

    # warmup every arm once so the first timed arm is not paying for lazy init
    for a in arm_names:
        run(a, decode=False)
    order = []
    for r in range(args.reps):
        order += arm_names if r % 2 == 0 else list(reversed(arm_names))
    print(f"[1b1] order={order}", flush=True)

    res = {}
    for arm in order:
        t0 = time.perf_counter()
        ms, lats, frames = run(arm, decode=(arm not in res))
        if arm not in res:
            res[arm] = dict(ms=[], hashes=lats, frames=frames)
        res[arm]["ms"] += ms
        print(f"[1b1] {arm:<12} median {statistics.median(ms):7.1f} ms "
              f"({time.perf_counter()-t0:.0f}s wall)", flush=True)

    print("\n" + "=" * 72)
    base = "off"
    bm = statistics.median(res[base]["ms"])
    print(f"  control {base}: median {bm:7.1f} ms, hash[0]={res[base]['hashes'][0]}")
    print("-" * 72)
    summary = {}
    for a in arm_names:
        med = statistics.median(res[a]["ms"])
        same = res[a]["hashes"] == res[base]["hashes"]
        ndiff = sum(1 for x, yv in zip(res[a]["hashes"], res[base]["hashes"])
                    if x != yv)
        first = next((i for i, (x, yv) in enumerate(
            zip(res[a]["hashes"], res[base]["hashes"])) if x != yv), None)
        pct = (med - bm) / bm * 100.0
        print(f"  {a:<12} median {med:7.1f} ms ({pct:+6.2f}%)  "
              f"bit-identical={str(same):<5}  "
              f"({ndiff}/{n_test} differ"
              + (f", first at chunk {first}" if first is not None else "") + ")")
        summary[a] = dict(median_ms=med, pct_vs_off=pct, bit_identical=same,
                          n_diff=ndiff, first_diff=first)
    print("=" * 72)

    out = dict(head=head, chunks=args.chunks, chunk_size=CS, seed=sd, scene=scene,
               arms=ARMS, arm_order=order, n_test=n_test,
               hashes={a: res[a]["hashes"] for a in arm_names},
               ms={a: res[a]["ms"] for a in arm_names}, summary=summary)
    with open(os.path.join(args.out_dir, "p2d1b1.json"), "w") as f:
        json.dump(out, f, indent=2)
    for a in arm_names:
        if res[a]["frames"]:
            torch.save(res[a]["frames"], os.path.join(args.out_dir, f"frames_{a}.pt"))
    print(f"[1b1] wrote {args.out_dir}/p2d1b1.json")


if __name__ == "__main__":
    main()
