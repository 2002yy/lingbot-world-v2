#!/usr/bin/env python
"""Generate the motion-showcase clips (2 scenes x 2 seeds).

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_motions.py
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
from gen_motions import build as build_traj

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


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
    ap.add_argument("--scenes", default="04,01")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--out_dir", default="output/motions")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    timesteps = [int(x) for x in args.ts.split(",")]
    scenes = args.scenes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    print("[mo] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    for sc in scenes:
        traj = f"examples/mot_{sc}"
        frames, marks = build_traj(sc, traj)
        img_pil = Image.open(f"{traj}/image.jpg").convert("RGB")
        for sd in seeds:
            mp4 = f"{args.out_dir}/{sc}_s{sd}.mp4"
            if os.path.exists(mp4):
                continue
            t0 = time.perf_counter()
            video = pipe.generate(PROMPT, img_pil, action_path=traj,
                                  chunk_size=1, max_area=W * H,
                                  frame_num=frames, timesteps_index=timesteps,
                                  shift=args.shift, seed=sd, offload_model=False)
            save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                       nrow=1, normalize=True, value_range=(-1, 1))
            print(f"[mo] {sc} s{sd}: {tuple(video.shape)} in "
                  f"{time.perf_counter()-t0:.0f}s -> {mp4}", flush=True)
            del video
            gc.collect(); torch.cuda.empty_cache()

    print("[mo] done", flush=True)


if __name__ == "__main__":
    main()
