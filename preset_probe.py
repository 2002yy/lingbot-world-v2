#!/usr/bin/env python
"""Short preset probe: pick default_6_1 vs temporal_stability_8_2 for a scene.

Runs one forward -> hold -> return trajectory under BOTH presets (2 seeds by
default), computes closure (ORB/RANSAC) + memory (DINO after homography
alignment), then applies the evidence gate in presets.py.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python preset_probe.py --scene 04
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from eval_two_layer import load_models, dino_feat, orb_closure
from presets import PRESETS, select_preset

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
K = 6   # probe interval in chunks


def build_probe(scene, outdir):
    p = np.load(f"examples/{scene}/poses.npy")
    fwd = p[:4 * K]
    hold_out = np.tile(p[4 * K - 1], (4 * K, 1, 1))
    bwd = p[4 * K - 2::-1]
    hold = np.tile(p[0], (4, 1, 1))
    traj = np.concatenate([fwd, hold_out, bwd, hold], 0)
    os.makedirs(outdir, exist_ok=True)
    np.save(f"{outdir}/poses.npy", traj)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{outdir}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{outdir}/image.jpg")
    return len(traj), 12 * K - 2


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def evaluate(mp4, return_idx):
    fr = read_frames(mp4)
    h, w = fr[0].shape[:2]
    cl = orb_closure(fr[0], fr[return_idx])
    if cl is None:
        return dict(ngood=0.0, medreproj=float("nan"), dino=1.0)
    warped = cv2.warpPerspective(fr[return_idx], np.linalg.inv(cl["H"]), (w, h),
                                 flags=cv2.INTER_LINEAR)
    d = 1.0 - float((dino_feat(cv2.cvtColor(fr[0], cv2.COLOR_BGR2RGB))
                     * dino_feat(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))).sum())
    return dict(ngood=float(cl["nGood"]), medreproj=float(cl["medReproj"]),
                dino=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--out_dir", default="output/preset_probe")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    max_area = W * H
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    timesteps = [int(x) for x in args.ts.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    traj = f"examples/probe_preset_{args.scene}"
    frames, return_idx = build_probe(args.scene, traj)
    print(f"[pp] scene {args.scene}: probe {frames} frames, return_idx={return_idx}",
          flush=True)

    load_models()
    per_preset = {}
    for pname, pc in PRESETS.items():
        lw, ls = pc["local_window"], pc["secondary"]
        print(f"[pp] building pipe for {pname} (local={lw} sink={ls})", flush=True)
        pipe = wan.WanI2VCausal(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
            convert_model_dtype=False, local_attn_size=lw,
            sink_size=ls, infer_mode="causal_fast", assets_dir=args.assets_dir)
        key = hashlib.sha256(PROMPT.encode()).hexdigest()
        if key not in pipe._t5_cache:
            ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
            pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder = None
        gc.collect(); torch.cuda.empty_cache()

        img_pil = Image.open(f"{traj}/image.jpg").convert("RGB")
        for seed in seeds:
            mp4 = f"{args.out_dir}/{args.scene}_{pname}_s{seed}.mp4"
            if not os.path.exists(mp4):
                video = pipe.generate(PROMPT, img_pil, action_path=traj,
                                      chunk_size=1, max_area=max_area,
                                      frame_num=frames, timesteps_index=timesteps,
                                      shift=args.shift, seed=seed,
                                      offload_model=False)
                save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                           nrow=1, normalize=True, value_range=(-1, 1))
                del video
                gc.collect(); torch.cuda.empty_cache()
            per_preset.setdefault(pname, {})[seed] = evaluate(mp4, return_idx)
            m = per_preset[pname][seed]
            print(f"[pp] {pname} s{seed}: nGood={m['ngood']:.0f} "
                  f"medReproj={m['medreproj']:.2f} DINO={m['dino']:.3f}", flush=True)
        del pipe
        gc.collect(); torch.cuda.empty_cache()

    a = per_preset["default_6_1"]
    b = per_preset["temporal_stability_8_2"]
    repeats = []
    for s in seeds:
        if s in a and s in b:
            repeats.append(dict(ngood_61=a[s]["ngood"], ngood_82=b[s]["ngood"],
                                dino_61=a[s]["dino"], dino_82=b[s]["dino"]))
    preset, detail = select_preset(repeats)
    print(f"\n[pp] scene {args.scene} -> PRESET = {preset}  "
          f"({detail['passed']}/{detail['needed']})", flush=True)
    for i, ok, why in detail["details"]:
        print(f"[pp]   repeat {i} (seed {seeds[i]}): {'PASS' if ok else 'fail'}  {why}")
    with open(f"{args.out_dir}/{args.scene}_verdict.json", "w") as fh:
        json.dump(dict(scene=args.scene, preset=preset, detail=detail,
                       per_preset=per_preset), fh, indent=1)


if __name__ == "__main__":
    main()
