#!/usr/bin/env python
"""Camera-response inertia: branch-from-same-state experiment.

The model is a stateful dynamic system, not an instantaneous camera follower.
Three branches share an identical prefix (same trajectory prefix AND same seed,
so the autoregressive prefix is bit-identical and the branch state is identical):

    CONTINUE : keep the motion
    STOP     : conditioning immediately zeroed
    REVERSE  : conditioning immediately reversed

Unified metric -- projection onto the pre-stop motion template:
    alpha_k = <F_k, F_ref> / |F_ref|^2      (F_ref = mean flow of the last
                                             steady prefix chunks)
    alpha ~ +1 still moving the old way, 0 gone, <0 truly reversed

Normalised against the CONTINUE branch:
    residual_k = alpha_k(STOP) / alpha_k(CONTINUE)

Non-parametric outputs: T50, T20, T0, T_reversal (no forced exponential fit).

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python inertia_branch.py --scenes 04 --seeds 42
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
from cam_controller import CameraController
from gen_predictive import write_traj

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

MOTIONS = {
    "yaw": dict(yaw=0.7),
    "fwd": dict(fwd=0.7),
    "strafe": dict(strafe=0.7),
    "walk_turn": dict(fwd=0.6, yaw=-0.5),
}
OPPOSITE = {
    "yaw": dict(yaw=-0.7),
    "fwd": dict(fwd=-0.7),
    "strafe": dict(strafe=-0.7),
    "walk_turn": dict(fwd=-0.6, yaw=0.5),
}
SETTLE, STEADY, POST = 2, 5, 8


def build_branch(scene, motion, branch):
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max = 6.0
    ctl.cfg.pitch_rate_max = 2.0
    ctl.cfg.v_max = 1.0
    ctl.poses = []
    cs, fp, marks = [], [], {}

    def run(n, inp):
        ctl.set_input(**inp)
        lo = len(fp)
        for _ in range(n):
            cs.append(ctl.clone())
            for _ in range(4):
                ctl.step(dt=0.25)
                fp.append(ctl.pose.copy())
        return lo, len(fp)

    ctl.set_input()
    run(SETTLE, dict())
    marks["steady"] = run(STEADY, MOTIONS[motion])
    if branch == "CONTINUE":
        marks["post"] = run(POST, MOTIONS[motion])
    elif branch == "STOP":
        marks["post"] = run(POST, dict())
    else:
        marks["post"] = run(POST, OPPOSITE[motion])
    return cs, np.stack(fp), marks


def flow_fields(frames):
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    out = [None]
    for i in range(1, len(g)):
        out.append(cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None,
                                                0.5, 3, 15, 3, 5, 1.2, 0))
    return out


def chunk_flow(fl, c):
    """Mean flow field over chunk c (frames 4c..4c+3); None only if empty."""
    vals = [fl[k] for k in range(4 * c, min(4 * c + 4, len(fl))) if fl[k] is not None]
    return np.mean(vals, axis=0) if vals else None


def alpha(F, Fref):
    if F is None or Fref is None:
        return float("nan")
    den = float(np.sum(Fref * Fref))
    if den < 1e-9:
        return float("nan")
    return float(np.sum(F * Fref) / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--scenes", default="04,01")
    ap.add_argument("--seeds", default="42,123,7")
    ap.add_argument("--out_dir", default="output/inertia")
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
    print("[in] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    n_chunks = SETTLE + STEADY + POST
    for scene in scenes:
        for motion in MOTIONS:
            for branch in ("CONTINUE", "STOP", "REVERSE"):
                cs, fp, marks = build_branch(scene, motion, branch)
                d = f"examples/in_{scene}_{motion}_{branch}"
                N = write_traj(None, [c.pose for c in cs], d, f"examples/{scene}")
                for sd in seeds:
                    mp4 = f"{args.out_dir}/{scene}_{motion}_{branch}_s{sd}.mp4"
                    if os.path.exists(mp4):
                        continue
                    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
                    t0 = time.perf_counter()
                    video = pipe.generate(PROMPT, img_pil, action_path=d,
                                          chunk_size=1, max_area=W * H,
                                          frame_num=N, timesteps_index=timesteps,
                                          shift=args.shift, seed=sd,
                                          offload_model=False)
                    save_video(tensor=video[None], save_file=mp4,
                               fps=cfg.sample_fps, nrow=1, normalize=True,
                               value_range=(-1, 1))
                    del video
                    gc.collect(); torch.cuda.empty_cache()
                    print(f"[in] {scene}/{motion}/{branch} s{sd} "
                          f"{time.perf_counter()-t0:.0f}s", flush=True)
    del pipe
    gc.collect(); torch.cuda.empty_cache()

    # ---------------- analysis ----------------
    def read(mp4):
        cap = cv2.VideoCapture(mp4); out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
        cap.release()
        return out

    print(f"\n[in] ===== alpha_k (projection onto pre-stop template) =====")
    print(f"{'scene':6s} {'motion':10s} {'seed':>5s} {'br':8s} " +
          " ".join(f"c{c:<5d}" for c in range(POST)))
    summary = {}
    for scene in scenes:
        for motion in MOTIONS:
            for sd in seeds:
                alphas = {}
                for branch in ("CONTINUE", "STOP", "REVERSE"):
                    mp4 = f"{args.out_dir}/{scene}_{motion}_{branch}_s{sd}.mp4"
                    if not os.path.exists(mp4):
                        continue
                    fl = flow_fields(read(mp4))
                    refs = [chunk_flow(fl, c) for c in range(SETTLE, SETTLE + STEADY)]
                    refs = [r for r in refs if r is not None]
                    Fref = np.mean(refs, axis=0)
                    al = [alpha(chunk_flow(fl, SETTLE + STEADY + k), Fref)
                          for k in range(POST)]
                    alphas[branch] = np.array(al)
                    print(f"{scene:6s} {motion:10s} {sd:5d} {branch:8s} " +
                          " ".join(f"{v:+5.2f}" for v in al))
                if "CONTINUE" in alphas:
                    cont = alphas["CONTINUE"]
                    for branch in ("STOP", "REVERSE"):
                        if branch not in alphas:
                            continue
                        b = alphas[branch]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            resid = np.where(np.abs(cont) > 0.05, b / cont, np.nan)
                        def first_below(x, thr):
                            idx = np.where(x < thr)[0]
                            return int(idx[0]) if len(idx) else -1
                        T50 = first_below(resid, 0.5)
                        T20 = first_below(resid, 0.2)
                        T0 = first_below(np.abs(resid), 0.05)
                        Trev = first_below(b, 0.0)
                        summary[(scene, motion, sd, branch)] = dict(
                            T50=T50, T20=T20, T0=T0, Trev=Trev,
                            resid=[float(x) for x in resid])
                        print(f"{'':6s} {'':10s} {'':5s} {'  ->'+branch:8s} "
                              f"T50={T50} T20={T20} T0={T0} Trev={Trev}")
    print()
    print("[in] ===== T50 / T20 / Trev by branch (chunks) =====")
    for branch in ("STOP", "REVERSE"):
        vals = {k: v for k, v in summary.items() if k[3] == branch}
        if not vals:
            continue
        t50 = [v["T50"] for v in vals.values() if v["T50"] >= 0]
        t20 = [v["T20"] for v in vals.values() if v["T20"] >= 0]
        trev = [v["Trev"] for v in vals.values() if v["Trev"] >= 0]
        print(f"  {branch:8s} T50 mean={np.mean(t50) if t50 else float('nan'):.2f} "
              f"({len(t50)}/{len(vals)} reached)  "
              f"T20 mean={np.mean(t20) if t20 else float('nan'):.2f} "
              f"({len(t20)}/{len(vals)})  "
              f"Trev mean={np.mean(trev) if trev else float('nan'):.2f} "
              f"({len(trev)}/{len(vals)})")
    json.dump({str(k): v for k, v in summary.items()},
              open(f"{args.out_dir}/inertia.json", "w"), indent=1)


if __name__ == "__main__":
    main()
