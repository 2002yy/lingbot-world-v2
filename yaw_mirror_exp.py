#!/usr/bin/env python
"""Stage B/C: horizontal-mirror equivariance test for the yaw asymmetry.

For each scene (01, 04) x {original, horizontally mirrored} x {yaw +, yaw -}:
build the identical yaw-probe trajectory, run the model, and measure the signed
rotation response (mean dx of the optical flow).

Discriminator (the strong *label*):
  - strong label stays yaw- after mirroring  -> global directional prior (DiT-internal)
  - strong label flips to yaw+ after mirroring -> scene-content / geometry driven

Mirroring: image flipped horizontally, poses R' = M R M, t' = M t,
intrinsics cx' = (W-1) - cx, with M = diag(-1,1,1).

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python yaw_mirror_exp.py
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from cam_controller import CameraController

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
M = np.diag([-1.0, 1.0, 1.0])
YAW_RATE = 6.0
V_MAX = 0.6
SCENES = ["01", "04"]
SEEDS = [42, 123, 7, 2024]


def mirror_pose(P):
    R = P[:3, :3]
    out = np.eye(4)
    out[:3, :3] = M @ R @ M
    out[:3, 3] = M @ P[:3, 3]
    return out


def make_example(scene, mirrored, outdir):
    """Write an example dir with (optionally) mirrored image / poses / intrinsics."""
    src = f"examples/{scene}"
    os.makedirs(outdir, exist_ok=True)
    img = cv2.imread(f"{src}/image.jpg")
    P = np.load(f"{src}/poses.npy")
    K = np.load(f"{src}/intrinsics.npy").copy()
    if mirrored:
        img = cv2.flip(img, 1)
        P = np.stack([mirror_pose(p) for p in P])
        K[:, 2] = (832 - 1) - K[:, 2]        # cx mirrored at original width
    cv2.imwrite(f"{outdir}/image.jpg", img)
    np.save(f"{outdir}/poses.npy", P)
    np.save(f"{outdir}/intrinsics.npy", K)
    return outdir


def yaw_probe(base_pose, outdir, yaw, exdir):
    ctl = CameraController(base_pose[:3, :3], base_pose[:3, 3])
    ctl.cfg.yaw_rate_max = YAW_RATE
    ctl.cfg.v_max = V_MAX
    ctl.poses = []
    fr = []
    ctl.set_input(fwd=0.0)
    for _ in range(4):
        ctl.step(dt=0.25); fr.append(ctl.pose.copy())
    ctl.set_input(yaw=yaw)
    for _ in range(16):
        ctl.step(dt=0.25); fr.append(ctl.pose.copy())
    ctl.set_input()
    for _ in range(4):
        ctl.step(dt=0.25); fr.append(ctl.pose.copy())
    os.makedirs(outdir, exist_ok=True)
    np.save(f"{outdir}/poses.npy", np.stack(fr))
    shutil.copy(f"{exdir}/intrinsics.npy", f"{outdir}/intrinsics.npy")
    shutil.copy(f"{exdir}/image.jpg", f"{outdir}/image.jpg")
    return len(fr)


def signed_yaw_response(mp4, lo=4, hi=16):
    cap = cv2.VideoCapture(mp4)
    g = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    dxs = []
    for i in range(max(1, lo), min(hi, len(g))):
        fl = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dxs.append(float(fl[..., 0].mean()))
    return float(np.mean(dxs)) if dxs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--out_dir", default="output/yawmirror")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    timesteps = [int(x) for x in args.ts.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    print("[ym] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for scene in SCENES:
        for mirrored in (False, True):
            tag = f"{scene}{'M' if mirrored else ''}"
            ex = make_example(scene, mirrored, f"examples/ex_{tag}")
            base = np.load(f"{ex}/poses.npy")[0]
            for yaw in (+1.0, -1.0):
                ytag = "yawP" if yaw > 0 else "yawM"
                traj = f"examples/probe_{tag}_{ytag}"
                frames = yaw_probe(base, traj, yaw, ex)
                img_pil = Image.open(f"{traj}/image.jpg").convert("RGB")
                for seed in SEEDS:
                    name = f"{tag}_{ytag}_s{seed}"
                    mp4 = f"{args.out_dir}/{name}.mp4"
                    if os.path.exists(mp4):
                        resp = signed_yaw_response(mp4)
                        rows.append((name, resp))
                        continue
                    video = pipe.generate(PROMPT, img_pil, action_path=traj,
                                          chunk_size=1, max_area=W * H,
                                          frame_num=frames,
                                          timesteps_index=timesteps,
                                          shift=args.shift, seed=seed,
                                          offload_model=False)
                    save_video(tensor=video[None], save_file=mp4,
                               fps=cfg.sample_fps, nrow=1, normalize=True,
                               value_range=(-1, 1))
                    del video
                    gc.collect(); torch.cuda.empty_cache()
                    resp = signed_yaw_response(mp4)
                    rows.append((name, resp))
                    print(f"[ym] {name}: signed yaw response = {resp:+.4f}", flush=True)

    print("\n[ym] ===== mirror equivariance summary =====", flush=True)
    print(f"{'case':22s} {'mean dx':>9s}")
    for n, r in rows:
        print(f"{n:22s} {r:9.4f}")
    print("\n[ym] by scene:")
    for scene in SCENES:
        for mirrored in (False, True):
            tag = f"{scene}{'M' if mirrored else ''}"
            p = [r for n, r in rows if n.startswith(f"{tag}_yawP")]
            m = [r for n, r in rows if n.startswith(f"{tag}_yawM")]
            if p and m:
                print(f"  {tag:5s} yaw+={np.mean(p):+8.4f}  yaw-={np.mean(m):+8.4f}  "
                      f"|yaw-|/|yaw+|={abs(np.mean(m))/max(abs(np.mean(p)),1e-6):5.2f}  "
                      f"strong={'yaw-' if abs(np.mean(m))>abs(np.mean(p)) else 'yaw+'}")
    with open(f"{args.out_dir}/summary.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
