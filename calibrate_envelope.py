#!/usr/bin/env python
"""Four-action x three-speed safety-envelope calibration.

Builds the pipe once, sweeps {yaw, pitch, strafe, forward} x {small, med, large},
generates each clip, saves mp4 + contact sheet, and writes a summary table.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python calibrate_envelope.py --base examples/00
"""
import argparse
import copy
import gc
import hashlib
import math
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from cam_controller import CameraController, EnvelopeConfig

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

ACTIONS = {
    "yaw_l":   dict(yaw=+1.0),
    "yaw_r":   dict(yaw=-1.0),
    "pitch_u": dict(pitch=+1.0),
    "pitch_d": dict(pitch=-1.0),
    "strafe_l": dict(strafe=-1.0),
    "strafe_r": dict(strafe=+1.0),
    "fwd":     dict(fwd=+1.0),
    "bwd":     dict(fwd=-1.0),
}

# multipliers applied to the per-action rate limit
LEVELS = {"small": 0.30, "med": 0.60, "large": 1.00}
IDLE = 2
NACT = 8


def build_traj(base_poses, action, level, outdir):
    cfg = EnvelopeConfig()
    m = LEVELS[level]
    cfg.v_max = m
    cfg.yaw_rate_max = 6.0 * m
    cfg.pitch_rate_max = 2.0 * m
    ctl = CameraController(base_poses[:3, :3], base_poses[:3, 3], cfg)
    ctl.set_input(0, 0, 0, 0)
    for _ in range(IDLE):
        ctl.step()
    ctl.set_input(**ACTIONS[action])
    for _ in range(NACT):
        ctl.step()
    ctl.set_input(0, 0, 0, 0)
    for _ in range(IDLE):
        ctl.step()
    ctl.save(outdir, os.path.dirname(base_poses_path))
    return len(ctl.poses)


def flow_stats(video):
    """video: (C,F,H,W) in [-1,1]. Return (mean|flow| in action segment, motion ratio)."""
    try:
        import cv2
    except ImportError:
        return float("nan"), float("nan")
    arr = ((video.permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in arr]
    mags = []
    for i in range(1, len(grays)):
        fl = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None,
                                          0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.mean(np.linalg.norm(fl, axis=2))))
    mags = np.array(mags)
    act = mags[4 * IDLE:4 * (IDLE + NACT)]
    idle = np.concatenate([mags[:4 * IDLE], mags[4 * (IDLE + NACT):]])
    return float(act.mean()), float(idle.mean())


def main():
    global base_poses_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/00")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--out_dir", default="output/calib")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    base_poses_path = f"{args.base}/poses.npy"
    base = np.load(base_poses_path)[0]
    W, H = (int(x) for x in args.area.lower().split("x"))
    max_area = W * H
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.perf_counter()
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    print(f"[cal] pipe built in {time.perf_counter()-t0:.1f}s", flush=True)
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
        gc.collect()
    timesteps = [int(x) for x in args.ts.split(",")]

    runs = [(a, l) for a in ACTIONS for l in LEVELS]
    if args.only:
        want = set(args.only.split(","))
        runs = [r for r in runs if f"{r[0]}_{r[1]}" in want]

    rows = []
    for action, level in runs:
        name = f"{action}_{level}"
        mp4 = f"{args.out_dir}/{name}.mp4"
        traj = f"examples/calib_{name}"
        if os.path.exists(mp4):
            print(f"[cal] skip {name} (exists)", flush=True)
            continue
        build_traj(base, action, level, traj)
        frames = np.load(f"{traj}/poses.npy").shape[0]
        t0 = time.perf_counter()
        video = pipe.generate(PROMPT, img_pil, action_path=traj, chunk_size=1,
                              max_area=max_area, frame_num=frames,
                              timesteps_index=timesteps, shift=args.shift,
                              seed=args.seed, offload_model=False)
        dt = time.perf_counter() - t0
        save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        act_m, idle_m = flow_stats(video)
        rows.append((name, act_m, idle_m, dt))
        print(f"[cal] {name}: act|flow|={act_m:.3f} idle|flow|={idle_m:.3f} "
              f"({dt:.1f}s)", flush=True)
        del video
        gc.collect()
        torch.cuda.empty_cache()

    with open(f"{args.out_dir}/summary.txt", "w") as f:
        f.write("action\tact_flow\tidle_flow\tsec\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.1f}\n")
    print("[cal] done ->", f"{args.out_dir}/summary.txt", flush=True)


if __name__ == "__main__":
    main()
