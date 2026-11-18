#!/usr/bin/env python
"""Test whether rotation is controllable when combined with translation.

Hypothesis: pure-rotation trajectories are OOD (official translation is never 0),
so the model ignores rotation unless some translation is present.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python calib_rot_trans.py --base examples/00
"""
import argparse
import gc
import hashlib
import os
import time

import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from cam_controller import CameraController, EnvelopeConfig

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
IDLE, NACT = 2, 8

# name -> (yaw, pitch, fwd, strafe, yaw_rate, pitch_rate, v_max)
COMBOS = {
    "yawR_fwd03":   (1.0, 0.0, 0.5, 0.0, 6.0, 2.0, 0.6),
    "yawL_fwd03":   (-1.0, 0.0, 0.5, 0.0, 6.0, 2.0, 0.6),
    "pitchU_fwd03": (0.0, 1.0, 0.5, 0.0, 6.0, 2.0, 0.6),
    "yawR_only":    (1.0, 0.0, 0.0, 0.0, 6.0, 2.0, 0.6),
    "yawL_only":    (-1.0, 0.0, 0.0, 0.0, 6.0, 2.0, 0.6),
}


def build(name, base, outdir):
    yaw, pitch, fwd, strafe, yr, pr, vm = COMBOS[name]
    cfg = EnvelopeConfig()
    cfg.v_max = vm
    cfg.yaw_rate_max = yr
    cfg.pitch_rate_max = pr
    ctl = CameraController(base[:3, :3], base[:3, 3], cfg)
    ctl.set_input(0, 0, 0, 0)
    for _ in range(IDLE):
        ctl.step()
    ctl.set_input(fwd=fwd, strafe=strafe, yaw=yaw, pitch=pitch)
    for _ in range(NACT):
        ctl.step()
    ctl.set_input(0, 0, 0, 0)
    for _ in range(IDLE):
        ctl.step()
    ctl.save(outdir, os.path.dirname(base_path))


def main():
    global base_path
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
    ap.add_argument("--out_dir", default="output/rottrans")
    args = ap.parse_args()

    base_path = f"{args.base}/poses.npy"
    base = np.load(base_path)[0]
    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    print("[rt] pipe built", flush=True)
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
        gc.collect()
    timesteps = [int(x) for x in args.ts.split(",")]

    try:
        import cv2
    except ImportError:
        cv2 = None

    for name in COMBOS:
        traj = f"examples/rt_{name}"
        build(name, base, traj)
        frames = np.load(f"{traj}/poses.npy").shape[0]
        t0 = time.perf_counter()
        video = pipe.generate(PROMPT, img_pil, action_path=traj, chunk_size=1,
                              max_area=W * H, frame_num=frames,
                              timesteps_index=timesteps, shift=args.shift,
                              seed=args.seed, offload_model=False)
        mp4 = f"{args.out_dir}/{name}.mp4"
        save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        am = float("nan")
        if cv2 is not None:
            arr = ((video.permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            gr = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in arr]
            mags = [float(np.mean(np.linalg.norm(cv2.calcOpticalFlowFarneback(
                gr[i - 1], gr[i], None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2)))
                for i in range(1, len(gr))]
            am = float(np.mean(mags[4 * IDLE:4 * (IDLE + NACT)]))
        print(f"[rt] {name}: act|flow|={am:.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)
        del video
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
