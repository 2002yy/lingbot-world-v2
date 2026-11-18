#!/usr/bin/env python
"""§26 CV replication gate: 2 scenes x 3 seeds x 2 kinds x {A_current, B_cv}
plus diagnostics (Oracle, CV@0.75L, CV@1.25L) and the E_scaled control.

Reports PAIRED deltas (CV - Current) per case, not just group means.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python replicate_cv.py
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
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses
from gen_predictive import build_base, predict, cv_gain, write_traj
from run_predictive import read_frames, gen_flow, analytic_flow, corr_at

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

SCENES = ["04", "01"]
SEEDS = [42, 123, 7]
KINDS = ["steady", "changepoint"]
MAIN = [("A_current", 1.0), ("B_cv", 1.0)]
DIAG = [("D_oracle", 1.0), ("B_cv", 0.75), ("B_cv", 1.25), ("E_scaled", 1.0)]


def build_all(scene, kind, specs):
    states, frame_poses, marks = build_base(scene, kind)
    np.save(f"examples/pred_{kind}_{scene}_intended.npy", frame_poses)
    g = cv_gain(states, 1.0)
    out = {}
    for mode, h in specs:
        lats = [predict(states, c, mode, h, gain=g) for c in range(len(states))]
        d = f"examples/pred_{kind}_{scene}_{mode}_h{h}"
        N = write_traj(None, lats, d, f"examples/{scene}")
        out[(mode, h)] = dict(dir=d, frames=N)
    return out, marks, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--out_dir", default="output/replicate")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
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
    print("[rp] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    results = {}
    for scene in scenes:
        for kind in KINDS:
            specs = MAIN + DIAG
            dirs, marks, gain = build_all(scene, kind, specs)
            print(f"[rp] {scene}/{kind}: E_scaled gain = {gain:.3f}", flush=True)
            for (mode, h) in specs:
                for sd in seeds:
                    if (mode, h) not in MAIN and sd != seeds[0]:
                        continue
                    d = dirs[(mode, h)]
                    mp4 = f"{args.out_dir}/{scene}_{kind}_{mode}_h{h}_s{sd}.mp4"
                    if not os.path.exists(mp4):
                        img_pil = Image.open(f"{d['dir']}/image.jpg").convert("RGB")
                        t0 = time.perf_counter()
                        video = pipe.generate(PROMPT, img_pil, action_path=d["dir"],
                                              chunk_size=1, max_area=W * H,
                                              frame_num=d["frames"],
                                              timesteps_index=timesteps,
                                              shift=args.shift, seed=sd,
                                              offload_model=False)
                        save_video(tensor=video[None], save_file=mp4,
                                   fps=cfg.sample_fps, nrow=1, normalize=True,
                                   value_range=(-1, 1))
                        del video
                        gc.collect(); torch.cuda.empty_cache()
                        print(f"[rp] {scene}/{kind}/{mode}@h{h} s{sd} "
                              f"{time.perf_counter()-t0:.0f}s", flush=True)
                    results[(scene, kind, mode, h, sd)] = mp4
    del pipe
    gc.collect(); torch.cuda.empty_cache()

    # ---- evaluate ----
    def metrics(mp4, intended):
        frames = read_frames(mp4)
        h, w = frames[0].shape[:2]
        Ks = get_Ks_transformed(
            torch.from_numpy(np.load(f"examples/{scene}/intrinsics.npy")).float(),
            480, 832, h, w, h, w)[0].numpy()
        P = interpolate_camera_poses(
            np.linspace(0, len(intended) - 1, len(intended)),
            torch.from_numpy(intended[:, :3, :3]).float(),
            torch.from_numpy(intended[:, :3, 3]).float(),
            np.linspace(0, len(intended) - 1, len(frames))).numpy()
        a = analytic_flow(P, Ks, h, w)
        g = gen_flow(frames)
        return dict(lag0=corr_at(a, g, 0), lagm4=corr_at(a, g, -4),
                    lagm8=corr_at(a, g, -8))

    rows = []
    for scene in scenes:
        for kind in KINDS:
            intended = np.load(f"examples/pred_{kind}_{scene}_intended.npy")
            for sd in seeds:
                m = {}
                for (mode, h) in MAIN:
                    mp4 = results.get((scene, kind, mode, h, sd))
                    if mp4:
                        m[mode] = metrics(mp4, intended)
                if "A_current" in m and "B_cv" in m:
                    rows.append(dict(scene=scene, kind=kind, seed=sd,
                                     a=m["A_current"]["lagm4"],
                                     b=m["B_cv"]["lagm4"],
                                     d=m["B_cv"]["lagm4"] - m["A_current"]["lagm4"]))

    print("\n[rp] ===== paired display-time match (lag -4) =====")
    print(f"{'scene':6s} {'kind':12s} {'seed':>5s} {'Current':>8s} {'CV':>7s} {'delta':>8s}")
    for r in rows:
        print(f"{r['scene']:6s} {r['kind']:12s} {r['seed']:5d} "
              f"{r['a']:8.3f} {r['b']:7.3f} {r['d']:+8.3f}")
    d = np.array([r["d"] for r in rows])
    print(f"\n  mean delta = {d.mean():+.4f}   improved {int((d>0).sum())}/{len(d)} cases")
    for kind in KINDS:
        dd = np.array([r["d"] for r in rows if r["kind"] == kind])
        print(f"  {kind:12s} mean delta = {dd.mean():+.4f}  "
              f"improved {int((dd>0).sum())}/{len(dd)}")
    for sc in scenes:
        dd = np.array([r["d"] for r in rows if r["scene"] == sc])
        print(f"  scene {sc}      mean delta = {dd.mean():+.4f}  "
              f"improved {int((dd>0).sum())}/{len(dd)}")
    with open(f"{args.out_dir}/paired.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
