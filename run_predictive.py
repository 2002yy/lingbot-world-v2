#!/usr/bin/env python
"""Run and evaluate predictive conditioning A/B/C/D.

Evaluation framing: a chunk generated at T0 is DISPLAYED at T0 + L. So its
motion should match the control at display time (4 frames later), not at T0.
We therefore measure corr(gen_flow, intended_flow shifted by +4 frames).

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_predictive.py --scene 04 --seed 42
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
from gen_predictive import MODES, build_base, predict

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


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


def gen_flow(frames):
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    out = []
    for i in range(1, len(g)):
        fl = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        out.append(float(np.mean(np.linalg.norm(fl, axis=2))))
    return np.array(out)


def analytic_flow(poses, Ks, h, w):
    fx, fy, cx, cy = [float(x) for x in Ks]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    xs = np.arange(0, w, 4)
    ys = np.arange(0, h, 4)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 0).astype(np.float64)
    out = []
    for k in range(1, len(poses)):
        R = poses[k][:3, :3] @ poses[k - 1][:3, :3].T
        H = K @ R @ Ki
        q = H @ pts
        q = q[:2] / q[2]
        out.append(float(np.mean(np.linalg.norm(q - pts[:2], axis=0))))
    return np.array(out)


def corr_at(a, g, lag):
    n = min(len(a), len(g))
    a, g = a[:n], g[:n]
    if lag >= 0:
        x, y = a[:n - lag], g[lag:]
    else:
        x, y = a[-lag:], g[:n + lag]
    if len(x) < 10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--kind", default="steady")
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--out_dir", default="output/predictive")
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    timesteps = [int(x) for x in args.ts.split(",")]
    states, frame_poses, marks = build_base(args.scene, args.kind)
    np.save(f"examples/pred_{args.kind}_{args.scene}_intended.npy", frame_poses)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[pr] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    results = {}
    for mode in args.modes.split(","):
        d = f"examples/pred_{args.kind}_{args.scene}_{mode}_h{args.horizon}"
        traj = np.load(f"{d}/poses.npy")
        frames = len(traj)
        img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
        mp4 = f"{args.out_dir}/{args.kind}_{args.scene}_{mode}_h{args.horizon}_s{args.seed}.mp4"
        if not os.path.exists(mp4):
            t0 = time.perf_counter()
            video = pipe.generate(PROMPT, img_pil, action_path=d, chunk_size=1,
                                  max_area=W * H, frame_num=frames,
                                  timesteps_index=timesteps, shift=args.shift,
                                  seed=args.seed, offload_model=False)
            save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                       nrow=1, normalize=True, value_range=(-1, 1))
            del video
            gc.collect(); torch.cuda.empty_cache()
            print(f"[pr] {mode} generated in {time.perf_counter()-t0:.0f}s", flush=True)
        results[mode] = mp4
    del pipe
    gc.collect(); torch.cuda.empty_cache()

    # ---- evaluate ----
    frames0 = read_frames(results["A_current"])
    h, w = frames0[0].shape[:2]
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"examples/{args.scene}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].numpy()
    P = np.load(f"examples/pred_{args.kind}_{args.scene}_intended.npy")
    # intended smooth trajectory at output-frame resolution
    intended = interpolate_camera_poses(
        np.linspace(0, len(P) - 1, len(P)),
        torch.from_numpy(P[:, :3, :3]).float(),
        torch.from_numpy(P[:, :3, 3]).float(),
        np.linspace(0, len(P) - 1, len(frames0))).numpy()
    a = analytic_flow(intended, Ks, h, w)

    print(f"\n[pr] ===== {args.kind} / scene {args.scene} / h={args.horizon} =====")
    print(f"{'mode':11s} {'corr@lag0':>10s} {'corr@lag+4':>11s} {'best_lag':>9s} "
          f"{'best_corr':>10s}")
    rows = {}
    for mode in MODES:
        g = gen_flow(read_frames(results[mode]))
        n = min(len(a), len(g))
        c0 = corr_at(a, g, 0)
        c4 = corr_at(a, g, 4)
        best_l, best_c = 0, -9
        for L in range(-8, 9):
            c = corr_at(a, g, L)
            if not np.isnan(c) and c > best_c:
                best_c, best_l = c, L
        rows[mode] = dict(corr0=c0, corr4=c4, best_lag=best_l, best_corr=best_c)
        print(f"{mode:11s} {c0:10.3f} {c4:11.3f} {best_l:9d} {best_c:10.3f}")

    with open(f"{args.out_dir}/{args.kind}_{args.scene}_h{args.horizon}_s{args.seed}.json", "w") as fh:
        json.dump(dict(kind=args.kind, scene=args.scene, horizon=args.horizon,
                       seed=args.seed, rows=rows, marks=marks), fh, indent=1)
    print(f"\n[pr] corr@lag+4 is the metric that matters: it asks whether the "
          f"generated motion matches the control at DISPLAY time (+4 frames).")


if __name__ == "__main__":
    main()
