#!/usr/bin/env python
"""lead_rot sweep for the compensator (overshoot diagnosis on the rotation channel)."""
import argparse
import gc
import hashlib
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from compensator import Compensator
from inertia_branch import MOTIONS, SETTLE, STEADY, POST, build_branch, flow_fields, chunk_flow, alpha
from gen_predictive import write_traj

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--motion", default="yaw")
    ap.add_argument("--leads", default="0.0,0.5,1.0")
    ap.add_argument("--out_dir", default="output/sweep_comp")
    args = ap.parse_args()

    W, H = 512, 320
    cfg = WAN_CONFIGS["i2v-1.3B"]
    os.makedirs(args.out_dir, exist_ok=True)
    timesteps = [0, 250, 750]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=os.path.expanduser(
            "~/ai/models/lingbot-world-v2-1.3b-causal-fast"),
        device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=True, convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast",
        assets_dir=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    print("[sw] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    leads = [float(x) for x in args.leads.split(",")]
    curves = {}
    for lead in leads:
        for branch in ("CONTINUE", "STOP", "REVERSE"):
            cs, fp, marks = build_branch(args.scene, args.motion, branch)
            comp = Compensator(lead_rot=lead, lead_trans=2.0)
            lats = comp.trajectory(cs)
            d = f"examples/sw_{args.scene}_{args.motion}_{branch}_r{lead}"
            N = write_traj(None, lats, d, f"examples/{args.scene}")
            mp4 = f"{args.out_dir}/{args.scene}_{args.motion}_{branch}_r{lead}_s{args.seed}.mp4"
            if not os.path.exists(mp4):
                img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
                t0 = time.perf_counter()
                video = pipe.generate(PROMPT, img_pil, action_path=d, chunk_size=1,
                                      max_area=W * H, frame_num=N,
                                      timesteps_index=timesteps, shift=5.0,
                                      seed=args.seed, offload_model=False)
                save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                           nrow=1, normalize=True, value_range=(-1, 1))
                del video
                gc.collect(); torch.cuda.empty_cache()
                print(f"[sw] r={lead} {branch} {time.perf_counter()-t0:.0f}s", flush=True)
    del pipe
    gc.collect(); torch.cuda.empty_cache()

    def read(mp4):
        cap = cv2.VideoCapture(mp4); out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
        cap.release()
        return out

    print(f"\n[sw] ===== lead_rot sweep, motion={args.motion} =====")
    for lead in leads:
        for branch in ("CONTINUE", "STOP", "REVERSE"):
            mp4 = f"{args.out_dir}/{args.scene}_{args.motion}_{branch}_r{lead}_s{args.seed}.mp4"
            if not os.path.exists(mp4):
                continue
            fl = flow_fields(read(mp4))
            refs = [chunk_flow(fl, c) for c in range(SETTLE, SETTLE + STEADY)]
            Fref = np.mean([r for r in refs if r is not None], axis=0)
            a = np.array([alpha(chunk_flow(fl, SETTLE + STEADY + k), Fref)
                          for k in range(POST)])
            curves[(lead, branch)] = a
            print(f"  r={lead:.1f} {branch:8s} " + " ".join(f"{v:+5.2f}" for v in a))
        cont, stop = curves.get((lead, "CONTINUE")), curves.get((lead, "STOP"))
        if cont is not None and stop is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                resid = np.where(np.abs(cont) > 0.05, stop / cont, np.nan)
            def fb(x, thr):
                i = np.where(x < thr)[0]
                return int(i[0]) if len(i) else -1
            print(f"    -> T50={fb(resid,0.5)} T20={fb(resid,0.2)} "
                  f"min_alpha={np.nanmin(stop):+.2f} (overshoot)")
        print()


if __name__ == "__main__":
    main()
