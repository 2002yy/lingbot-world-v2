#!/usr/bin/env python
"""§38A: Long-Horizon World Consistency -- Loop Closure.

For each (scene, seed, trajectory type, horizon H) build an out-and-back
trajectory: move away for H chunks, then return exactly to the start pose.
Measure how well the RETURNED frame matches frame 0.

Metrics (frame0 vs returned frame):
    DINO similarity            semantic identity
    ORB nGood / medReproj      geometric recoverability
    aligned LPIPS              appearance after homography alignment
    residual flow              geometric mismatch after alignment

Output: loop_closure_error vs elapsed_chunks -> T90 / T75 / T50

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python loop_closure.py --scenes 04,01 --seeds 42 --types fwd,yaw,orbit
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from cam_controller import CameraController
from wan.utils.cam_utils import (interpolate_camera_poses,
                                 get_Ks_transformed)

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
HORIZONS = [1, 5, 10, 20, 40]

# type -> (outbound inputs, return inputs)
TYPES = {
    "fwd":   (dict(fwd=0.8), dict(fwd=-0.8)),
    "yaw":   (dict(yaw=0.9), dict(yaw=-0.9)),
    "orbit": (dict(fwd=0.7, strafe=0.6), dict(fwd=-0.7, strafe=-0.6)),
}


def build_traj(scene, typ, H):
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    out_in, ret_in = TYPES[typ]
    fr = []
    ctl.set_input(fwd=0.0)
    for _ in range(4):
        ctl.step(dt=0.25); fr.append(ctl.pose.copy())
    ctl.set_input(**out_in)
    for _ in range(H * 4):
        ctl.step(dt=0.25); fr.append(ctl.pose.copy())
    ctl.set_input(**ret_in)
    for _ in range(H * 4):
        ctl.step(dt=0.25); fr.append(ctl.pose.copy())
    ctl.set_input()
    for _ in range(4):
        ctl.step(dt=0.25); fr.append(ctl.pose.copy())
    traj = np.stack(fr)
    # the VAE needs a 4n+1 frame count
    n = (len(traj) - 1) // 4 * 4 + 1
    traj = traj[:n]
    return traj, 4 + H * 4 + H * 4 - 1   # return_idx = last return frame


def orb(f0, f1):
    o = cv2.ORB_create(2000)
    g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    k0, d0 = o.detectAndCompute(g0, None)
    k1, d1 = o.detectAndCompute(g1, None)
    if d0 is None or d1 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    m = bf.knnMatch(d0, d1, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return None
    p0 = np.float32([k0[a.queryIdx].pt for a in good])
    p1 = np.float32([k1[a.trainIdx].pt for a in good])
    Hm, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    if Hm is None:
        return None
    inl = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    err = np.linalg.norm(proj - p1, axis=1)[inl]
    return dict(nGood=len(good), inlier=float(inl.mean()),
                medReproj=float(np.median(err)), H=Hm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scenes", default="04,01")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--types", default="fwd,yaw,orbit")
    ap.add_argument("--horizons", default="1,5,10,20,40")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/loopclosure")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",")]
    scenes = args.scenes.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    types = args.types.split(",")

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[lc] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads

    from eval_two_layer import load_models, dino_feat, lpips_d
    load_models()

    rows = []
    cases = [(sc, sd) for sc in scenes for sd in seeds]
    for scene, sd in cases:
        for typ in types:
            for hz in horizons:
                traj, ret_idx = build_traj(scene, typ, hz)
                frames_n = len(traj)
                d = f"examples/lc_{scene}_{typ}_H{hz}"
                os.makedirs(d, exist_ok=True)
                np.save(f"{d}/poses.npy", traj)
                import shutil
                shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
                shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
                img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
                import torchvision.transforms.functional as TF
                img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
                h, w = img.shape[1:]
                aspect = h / w
                lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch[1] * patch[1])
                lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch[2] * patch[2])
                h = lat_h * vae_stride[1]; w = lat_w * vae_stride[2]
                frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
                max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
                kv_size = frame_seqlen * 6
                n_lat = (frames_n - 1) // 4 + 1
                pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
                Ks = get_Ks_transformed(
                    torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
                    480, 832, h, w, h, w)[0].to(dev)
                y = pipe.vae.encode([torch.concat([
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                                    mode='bicubic').transpose(0, 1),
                    torch.zeros(3, frames_n - 1, h, w)], dim=1).to(dev)])[0]
                msk = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
                msk[:, 1:] = 0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                                    msk[:, 1:]], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
                y = torch.concat([msk, y])
                self_kv = pipe._initialize_self_kv_cache(
                    num_layers=ma.num_layers, shape=[1, kv_size, lh, hd],
                    dtype=dtype, device=dev)
                cross_kv = pipe._initialize_crossattn_cache(
                    num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
                    dtype=dtype, device=dev)
                pipe._cross_attn_initialized = False
                timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
                g = torch.Generator(device=dev); g.manual_seed(sd)
                from wan.utils.cam_utils import compute_relative_poses, get_plucker_embeddings
                from einops import rearrange
                # per-latent relative poses (framewise), like the pipeline
                c2w = interpolate_camera_poses(
                    np.linspace(0, frames_n - 1, frames_n),
                    torch.from_numpy(traj[:, :3, :3]).float(),
                    torch.from_numpy(traj[:, :3, 3]).float(),
                    np.linspace(0, frames_n - 1, n_lat)).to(dev)
                rel_all = compute_relative_poses(c2w, framewise=True)
                Ks_t = None
                out_frames = []
                for cid in range(n_lat):
                    cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
                    rel = rel_all[cid:cid + 1]
                    p = get_plucker_embeddings(rel, Ks[None], h, w)
                    p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                                  c1=int(h // lat_h), c2=int(w // lat_w))[None]
                    plk = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                                    h=lat_h, w=lat_w).to(pdt)
                    kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                          "y": [y.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
                          "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                          "kv_cache": self_kv, "crossattn_cache": cross_kv,
                          "current_start": cid * frame_seqlen,
                          "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}
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
                                    x0, torch.randn(x0.shape, generator=g,
                                                    device=x0.device, dtype=x0.dtype),
                                    timesteps[ti + 1])
                    with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                        pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                                   cross_attn_first_call=False, **kw)
                    with torch.no_grad():
                        fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                              parallel=False, show_progress_bar=False)
                    out_frames.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                                       * 255.0).clip(0, 255).astype(np.uint8))
                    if cid % 10 == 0:
                        print(f"[lc] {scene}/{typ}/H{hz} chunk {cid}/{n_lat}",
                              flush=True)
                # first frame of chunk 0 is the input image
                f_first = np.array(Image.open(f"{d}/image.jpg").convert("RGB"))
                f_first = cv2.resize(f_first, (out_frames[0].shape[1],
                                               out_frames[0].shape[0]))
                # returned frame = the frame closest to ret_idx
                ridx = min(ret_idx // 4, len(out_frames) - 1)
                f_ret = out_frames[ridx]
                r = orb(f_first, f_ret)
                dino = float((dino_feat(f_first) * dino_feat(f_ret)).sum())
                if r:
                    warped = cv2.warpPerspective(
                        f_ret, np.linalg.inv(r["H"]),
                        (f_first.shape[1], f_first.shape[0]), flags=cv2.INTER_LINEAR)
                    al = lpips_d(f_first, warped)
                    nG, mr = r["nGood"], r["medReproj"]
                else:
                    al, nG, mr = float("nan"), 0, float("nan")
                rows.append(dict(scene=scene, seed=sd, type=typ, H=hz, chunks=2 * hz,
                                 dino=dino, lpips_aligned=al, nGood=nG,
                                 medReproj=mr))
                print(f"[lc] {scene}/{typ}/H{hz}: DINO={dino:.4f} "
                      f"LPIPS_al={al:.4f} nGood={nG} reproj={mr:.2f}", flush=True)
                del y, self_kv, cross_kv, out_frames
                gc.collect(); torch.cuda.empty_cache()

    print("\n[lc] ===== loop closure vs elapsed chunks =====")
    print(f"{'scene':6s} {'type':7s} {'H':>3s} {'chunks':>7s} {'DINO':>7s} "
          f"{'LPIPS_al':>9s} {'nGood':>6s} {'reproj':>7s}")
    for r in rows:
        print(f"{r['scene']:6s} {r['type']:7s} {r['H']:3d} {r['chunks']:7d} "
              f"{r['dino']:7.4f} {r['lpips_aligned']:9.4f} {r['nGood']:6d} "
              f"{r['medReproj']:7.2f}")
    json.dump(rows, open(f"{args.out_dir}/loop_closure.json", "w"), indent=1)


if __name__ == "__main__":
    main()
