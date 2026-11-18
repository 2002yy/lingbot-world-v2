#!/usr/bin/env python
"""Gate 5/6 retest with the Camera Response Compensator.

Gate 5 (inertia, closed loop through the plant):
    for each motion x branch, compare conditioning A_current vs F_comp
    rotation    : T50 / Trev must stay healthy (~1 chunk)
    translation : steady T50 must shorten
    translation : reverse Trev must NOT get worse

Gate 6 (display): adaptive display over changepoint clips
    refresh jump, mean warp gain, reset/fail occupancy

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python gate_comp.py --scenes 04 --seeds 42
"""
import argparse
import gc
import hashlib
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from wan.utils.cam_utils import get_Ks_transformed
from compensator import Compensator
from inertia_branch import (MOTIONS, SETTLE, STEADY, POST, build_branch,
                            flow_fields, chunk_flow, alpha)
from gen_predictive import write_traj

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--scenes", default="04,01")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--lead_rot", type=float, default=1.0)
    ap.add_argument("--lead_trans", type=float, default=2.0)
    ap.add_argument("--out_dir", default="output/compensator")
    ap.add_argument("--motions", default=",".join(MOTIONS))
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
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[cp] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    def gen(trajdir, N, mp4):
        if os.path.exists(mp4):
            return
        img_pil = Image.open(f"{trajdir}/image.jpg").convert("RGB")
        t0 = time.perf_counter()
        video = pipe.generate(PROMPT, img_pil, action_path=trajdir, chunk_size=1,
                              max_area=W * H, frame_num=N,
                              timesteps_index=timesteps, shift=args.shift,
                              seed=sd, offload_model=False)
        save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        del video
        gc.collect(); torch.cuda.empty_cache()
        print(f"[cp] {os.path.basename(mp4)} {time.perf_counter()-t0:.0f}s", flush=True)

    def read(mp4):
        cap = cv2.VideoCapture(mp4); out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
        cap.release()
        return out

    # ---------------- Gate 5 ----------------
    print("[cp] ===== Gate 5: inertia with compensator =====", flush=True)
    for scene in scenes:
        for motion in args.motions.split(","):
            for branch in ("CONTINUE", "STOP", "REVERSE"):
                cs, fp, marks = build_branch(scene, motion, branch)
                comp = Compensator(args.lead_rot, args.lead_trans)
                lats = comp.trajectory(cs)
                d = f"examples/cp_{scene}_{motion}_{branch}"
                N = write_traj(None, lats, d, f"examples/{scene}")
                for sd in seeds:
                    gen(d, N, f"{args.out_dir}/{scene}_{motion}_{branch}_comp_s{sd}.mp4")

    print(f"\n[cp] ===== alpha_k with compensator =====")
    print(f"{'scene':6s} {'motion':10s} {'seed':>5s} {'br':8s} " +
          " ".join(f"c{c:<5d}" for c in range(POST)))
    res = {}
    for scene in scenes:
        for motion in args.motions.split(","):
            for sd in seeds:
                al = {}
                for branch in ("CONTINUE", "STOP", "REVERSE"):
                    mp4 = f"{args.out_dir}/{scene}_{motion}_{branch}_comp_s{sd}.mp4"
                    if not os.path.exists(mp4):
                        continue
                    fl = flow_fields(read(mp4))
                    refs = [chunk_flow(fl, c) for c in range(SETTLE, SETTLE + STEADY)]
                    Fref = np.mean([r for r in refs if r is not None], axis=0)
                    a = np.array([alpha(chunk_flow(fl, SETTLE + STEADY + k), Fref)
                                  for k in range(POST)])
                    al[branch] = a
                    print(f"{scene:6s} {motion:10s} {sd:5d} {branch:8s} " +
                          " ".join(f"{v:+5.2f}" for v in a))
                if "CONTINUE" in al and "STOP" in al:
                    cont, stop = al["CONTINUE"], al["STOP"]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        resid = np.where(np.abs(cont) > 0.05, stop / cont, np.nan)
                    def fb(x, thr):
                        i = np.where(x < thr)[0]
                        return int(i[0]) if len(i) else -1
                    res[(scene, motion, sd)] = dict(
                        T50=fb(resid, 0.5), T20=fb(resid, 0.2),
                        Trev=(fb(al["REVERSE"], 0.0) if "REVERSE" in al else -1))
                    print(f"{'':6s} {'':10s} {'':5s} {'  ->comp':8s} "
                          f"T50={res[(scene,motion,sd)]['T50']} "
                          f"T20={res[(scene,motion,sd)]['T20']} "
                          f"Trev={res[(scene,motion,sd)]['Trev']}")
    print()
    print("[cp] ===== per-motion compensator summary =====")
    for motion in MOTIONS:
        sub = [v for k, v in res.items() if k[1] == motion]
        if not sub:
            continue
        def m(k):
            v = [s[k] for s in sub if s[k] >= 0]
            return np.mean(v) if v else float("nan")
        print(f"  {motion:10s} T50={m('T50'):5.2f} T20={m('T20'):5.2f} "
              f"Trev={m('Trev'):5.2f}  (n={len(sub)})")
    json.dump({str(k): v for k, v in res.items()},
              open(f"{args.out_dir}/gate5_comp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
