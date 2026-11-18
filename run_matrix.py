#!/usr/bin/env python
"""Run the 24-condition memory matrix (2 scenes x 2 configs x 2 intervals x 3 seeds).

Builds the pipe once per (local_window, sink) config and reuses it.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_matrix.py
"""
import argparse
import gc
import hashlib
import json
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
from gen_matrix import build as build_traj

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
CONFIGS = [(6, 1), (8, 2)]
SCENES = ["01", "04"]
INTERVALS = [6, 8]
SEEDS = [42, 123, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--out_dir", default="output/matrix")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    max_area = W * H
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    timesteps = [int(x) for x in args.ts.split(",")]
    want = set(args.only.split(",")) if args.only else None

    runs = [(sc, lw, ls, K, sd)
            for (lw, ls) in CONFIGS
            for sc in SCENES
            for K in INTERVALS
            for sd in SEEDS]

    # group by config so the pipe is built once per config
    for lw, ls in CONFIGS:
        todo = [r for r in runs if r[1] == lw and r[2] == ls]
        todo = [r for r in todo
                if not os.path.exists(f"{args.out_dir}/{r[0]}_lw{lw}s{ls}_K{r[3]}_s{r[4]}.mp4")]
        if want:
            todo = [r for r in todo
                    if f"{r[0]}_lw{lw}s{ls}_K{r[3]}_s{r[4]}" in want]
        if not todo:
            print(f"[mx] config {lw}/{ls}: nothing to do", flush=True)
            continue

        print(f"[mx] building pipe local={lw} sink={ls} ({len(todo)} runs)", flush=True)
        t0 = time.perf_counter()
        pipe = wan.WanI2VCausal(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
            convert_model_dtype=False, local_attn_size=lw,
            sink_size=ls, infer_mode="causal_fast", assets_dir=args.assets_dir)
        print(f"[mx] pipe built in {time.perf_counter()-t0:.1f}s", flush=True)
        key = hashlib.sha256(PROMPT.encode()).hexdigest()
        if key not in pipe._t5_cache:
            ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
            pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder = None
            gc.collect()
        gc.collect()
        torch.cuda.empty_cache()

        for sc, _, _, K, sd in todo:
            name = f"{sc}_lw{lw}s{ls}_K{K}_s{sd}"
            traj = f"examples/mx_{sc}_K{K}"
            marks = build_traj(sc, K, traj)
            frames = np.load(f"{traj}/poses.npy").shape[0]
            img_pil = Image.open(f"{traj}/image.jpg").convert("RGB")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            video = pipe.generate(PROMPT, img_pil, action_path=traj,
                                  chunk_size=1, max_area=max_area,
                                  frame_num=frames, timesteps_index=timesteps,
                                  shift=args.shift, seed=sd,
                                  offload_model=False)
            dt = time.perf_counter() - t0
            mp4 = f"{args.out_dir}/{name}.mp4"
            save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                       nrow=1, normalize=True, value_range=(-1, 1))
            peak = torch.cuda.max_memory_allocated() / 2**20
            with open(f"{args.out_dir}/{name}.json", "w") as fh:
                json.dump(dict(marks, scene=sc, local=lw, sink=ls, K=K,
                               seed=sd, frames=frames, sec=dt, peak_mib=peak,
                               mp4=mp4), fh, indent=1)
            print(f"[mx] {name}: {frames}f {dt:.0f}s peak={peak:.0f}MiB", flush=True)
            del video
            gc.collect()
            torch.cuda.empty_cache()

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    print("[mx] done", flush=True)


if __name__ == "__main__":
    main()
