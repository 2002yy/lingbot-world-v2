#!/usr/bin/env python
"""Run one camera-controller trajectory through the causal-fast model.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_ctl.py --demo walk_turn --base examples/00
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
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video

from cam_controller import run_demo

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--demo", default="walk_turn")
    ap.add_argument("--base", default="examples/00")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--chunk_size", type=int, default=1)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--out_dir", default="output/ctl")
    args = ap.parse_args()

    traj_dir = f"examples/ctl_{args.demo}"
    run_demo(args.demo, args.base, traj_dir)
    frames = np.load(f"{traj_dir}/poses.npy").shape[0]
    print(f"[ctl] {args.demo}: {frames} frames from {args.base}")

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
    print(f"[ctl] pipe built in {time.perf_counter()-t0:.1f}s")

    img_pil = Image.open(f"{traj_dir}/image.jpg").convert("RGB")
    key = hashlib.sha256(args.prompt.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([args.prompt], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
        gc.collect()

    timesteps = [int(x) for x in args.ts.split(",")]
    t0 = time.perf_counter()
    video = pipe.generate(
        args.prompt, img_pil, action_path=traj_dir,
        chunk_size=args.chunk_size, max_area=max_area, frame_num=frames,
        timesteps_index=timesteps, shift=args.shift, seed=args.seed,
        offload_model=False)
    print(f"[ctl] generated {tuple(video.shape)} in {time.perf_counter()-t0:.1f}s")

    save_file = f"{args.out_dir}/{args.demo}.mp4"
    save_video(tensor=video[None], save_file=save_file, fps=cfg.sample_fps,
               nrow=1, normalize=True, value_range=(-1, 1))
    print(f"[ctl] saved {save_file}")


if __name__ == "__main__":
    main()
