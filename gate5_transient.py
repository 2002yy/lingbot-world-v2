#!/usr/bin/env python
"""Gate 5: changepoint transient metrics.

One trajectory per phase (25/50/75% of a chunk) containing four change types:
    W -> release | W -> S | yaw+ -> yaw- | strafe L -> R
Conditioned with A_current / B_cv / E_scaled / D_oracle.

Per change we measure, from the signed projection of the generated flow onto the
intended flow:
    wrong_dir_dur   frames with projection < 0 (still moving the old way)
    peak_overshoot  max projection after the change
    recovery        frames until projection first exceeds 0.5
    ite             integrated tracking error (mean |g-a| over the window)

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python gate5_transient.py --scene 04
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
from cam_controller import CameraController
from gen_predictive import predict, cv_gain, write_traj
from run_predictive import read_frames, gen_flow, analytic_flow

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
PHASES = [1, 2, 3]
METHODS = ["A_current", "B_cv", "E_scaled", "D_oracle"]

# (name, pre_input, post_input)
CHANGES = [
    ("W_release", dict(fwd=0.7), dict()),
    ("W_to_S", dict(fwd=0.7), dict(fwd=-0.7)),
    ("yaw_swap", dict(yaw=0.7, fwd=0.4), dict(yaw=-0.7, fwd=0.4)),
    ("strafe_rev", dict(strafe=0.8), dict(strafe=-0.8)),
]
PRE_CHUNKS, POST_CHUNKS, SETTLE = 3, 3, 2


def build_states(scene, phase):
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max = 6.0
    ctl.cfg.pitch_rate_max = 2.0
    ctl.cfg.v_max = 1.0
    chunk_states, frame_poses, events = [], [], []
    ctl.set_input()
    for _ in range(SETTLE):
        chunk_states.append(ctl.clone())
        for _ in range(4):
            ctl.step(dt=0.25); frame_poses.append(ctl.pose.copy())
    for name, pre, post in CHANGES:
        ctl.set_input(**pre)
        for _ in range(PRE_CHUNKS):
            chunk_states.append(ctl.clone())
            for _ in range(4):
                ctl.step(dt=0.25); frame_poses.append(ctl.pose.copy())
        ctl.set_input(**post)
        # change lands at `phase` frames into the next chunk
        for j in range(4):
            ctl.step(dt=0.25); frame_poses.append(ctl.pose.copy())
            if j == phase:
                chunk_states.append(ctl.clone())   # chunk whose pose is post-change
        for _ in range(POST_CHUNKS - 1):
            chunk_states.append(ctl.clone())
            for _ in range(4):
                ctl.step(dt=0.25); frame_poses.append(ctl.pose.copy())
        change_frame = len(frame_poses) - 4 * POST_CHUNKS + phase
        events.append(dict(name=name, frame=change_frame))
    return chunk_states, np.stack(frame_poses), events


def projection_metrics(flow_g, flow_a, frame, win=14):
    """Direction via bounded cosine similarity + magnitude ratio.

    A raw projection <fg,fa>/|fa|^2 blows up when the intended flow is near zero,
    so use cos = <fg,fa>/(|fg||fa|) for direction and |fg|/|fa| for amplitude.
    """
    cos, mag_g, mag_a, err = [], [], [], []
    # reference scale: intended flow magnitude over the window
    mags = [float(np.sqrt(np.sum(flow_a[k] * flow_a[k])))
            for k in range(frame, min(frame + win, len(flow_g)))]
    ref = float(np.mean(mags)) if mags else 0.0
    thr = 0.15 * ref
    for k in range(frame, min(frame + win, len(flow_g))):
        fg, fa = flow_g[k], flow_a[k]
        ng = float(np.sqrt(np.sum(fg * fg)))
        na = float(np.sqrt(np.sum(fa * fa)))
        mag_g.append(ng); mag_a.append(na)
        err.append(float(np.mean(np.linalg.norm(fg - fa, axis=2))))
        if na > thr and ng > 1e-6:
            cos.append(float(np.sum(fg * fa) / (ng * na)))
    if not cos:
        return dict(wrong=0, overshoot=float("nan"), recovery=float("nan"),
                    ite=float(np.mean(err)) if err else float("nan"), n=0)
    cos = np.array(cos)
    wrong = int((cos < 0).sum())
    rec = next((i for i, v in enumerate(cos) if v > 0.5), len(cos))
    return dict(wrong=wrong, overshoot=float(np.mean(mag_g)),
                recovery=int(rec), ite=float(np.mean(err)), n=len(cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="output/gate5")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    timesteps = [int(x) for x in args.ts.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[g5] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    meta = {}
    for phase in PHASES:
        states, frame_poses, events = build_states(args.scene, phase)
        np.save(f"examples/g5_{args.scene}_p{phase}_intended.npy", frame_poses)
        gain = cv_gain(states, 1.0)
        meta[phase] = dict(events=events, n_chunks=len(states), gain=gain)
        for mode in METHODS:
            lats = [predict(states, c, mode, 1.0, gain=gain) for c in range(len(states))]
            d = f"examples/g5_{args.scene}_p{phase}_{mode}"
            write_traj(None, lats, d, f"examples/{args.scene}")
            mp4 = f"{args.out_dir}/{args.scene}_p{phase}_{mode}_s{args.seed}.mp4"
            if not os.path.exists(mp4):
                frames = 4 * (len(states) - 1) + 1
                img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
                t0 = time.perf_counter()
                video = pipe.generate(PROMPT, img_pil, action_path=d, chunk_size=1,
                                      max_area=W * H, frame_num=frames,
                                      timesteps_index=timesteps, shift=args.shift,
                                      seed=args.seed, offload_model=False)
                save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                           nrow=1, normalize=True, value_range=(-1, 1))
                del video
                gc.collect(); torch.cuda.empty_cache()
                print(f"[g5] {args.scene} p{phase} {mode} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    del pipe
    gc.collect(); torch.cuda.empty_cache()

    # ---- evaluate ----
    print(f"\n[g5] ===== transient metrics (scene {args.scene}) =====")
    print(f"{'phase':>5s} {'change':11s} {'method':10s} {'wrong':>6s} "
          f"{'overshoot':>10s} {'recover':>8s} {'ite':>7s}")
    rows = []
    for phase in PHASES:
        intended = np.load(f"examples/g5_{args.scene}_p{phase}_intended.npy")
        for mode in METHODS:
            mp4 = f"{args.out_dir}/{args.scene}_p{phase}_{mode}_s{args.seed}.mp4"
            frames = read_frames(mp4)
            h, w = frames[0].shape[:2]
            Ks = get_Ks_transformed(
                torch.from_numpy(np.load(f"examples/{args.scene}/intrinsics.npy")).float(),
                480, 832, h, w, h, w)[0].numpy()
            P = interpolate_camera_poses(
                np.linspace(0, len(intended) - 1, len(intended)),
                torch.from_numpy(intended[:, :3, :3]).float(),
                torch.from_numpy(intended[:, :3, 3]).float(),
                np.linspace(0, len(intended) - 1, len(frames))).numpy()
            flow_a = analytic_flow_field(P, Ks, h, w)
            flow_g = gen_flow_field(frames)
            n = min(len(flow_a), len(flow_g))
            for ev in meta[phase]["events"]:
                m = projection_metrics(flow_g[:n], flow_a[:n], ev["frame"])
                rows.append(dict(phase=phase, change=ev["name"], method=mode, **m))
                print(f"{phase:5d} {ev['name']:11s} {mode:10s} {m['wrong']:6.2f} "
                      f"{m['overshoot']:10.3f} {m['recovery']:8.2f} {m['ite']:7.3f}")
    with open(f"{args.out_dir}/transient.json", "w") as fh:
        json.dump(dict(meta={str(k): v for k, v in meta.items()}, rows=rows), fh, indent=1)

    print("\n[g5] ===== summary vs A_current =====")
    print(f"{'method':10s} {'wrong':>7s} {'overshoot':>10s} {'recover':>8s} {'ite':>7s}")
    base = {}
    for mode in METHODS:
        sub = [r for r in rows if r["method"] == mode]
        agg = dict(wrong=np.mean([r["wrong"] for r in sub]),
                   overshoot=np.nanmean([r["overshoot"] for r in sub]),
                   recovery=np.nanmean([r["recovery"] for r in sub]),
                   ite=np.nanmean([r["ite"] for r in sub]))
        base[mode] = agg
        print(f"{mode:10s} {agg['wrong']:7.2f} {agg['overshoot']:10.3f} "
              f"{agg['recovery']:8.2f} {agg['ite']:7.3f}")
    a = base["A_current"]
    print()
    for mode in METHODS:
        if mode == "A_current":
            continue
        b = base[mode]
        print(f"  {mode:10s} dWrong={b['wrong']-a['wrong']:+.2f}f "
              f"dOver={100*(b['overshoot']/a['overshoot']-1):+.1f}% "
              f"dITE={100*(b['ite']/a['ite']-1):+.1f}%")


def gen_flow_field(frames):
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    out = [None]
    for i in range(1, len(g)):
        out.append(cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None,
                                                0.5, 3, 15, 3, 5, 1.2, 0))
    return out


def analytic_flow_field(poses, Ks, h, w):
    fx, fy, cx, cy = [float(x) for x in Ks]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 0).astype(np.float64)
    out = [None]
    for k in range(1, len(poses)):
        R = poses[k][:3, :3] @ poses[k - 1][:3, :3].T
        H = K @ R @ Ki
        q = H @ pts
        q = q[:2] / q[2]
        d = (q - pts[:2]).T.reshape(h, w, 2)
        out.append(d.astype(np.float32))
    return out


if __name__ == "__main__":
    main()
